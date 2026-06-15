"""Streaming source + pre-filters."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import regex as re
from datasets import load_dataset

from .config import Settings

logger = logging.getLogger(__name__)

_MIN_CHARS = 200
_MAX_CHARS = 20_000

# Rows materialized per ``collect()`` in the polars fallback, so we never load a
# whole parquet file into memory at once.
_PARQUET_BATCH_ROWS = 10_000

_GREEK_LETTER_RATIO = 0.6
_MAX_URL_RATIO = 0.05  # > 5% of chars belonging to URLs marks a link farm
_MAX_UPPER_RATIO = 0.40
_MAX_PUNCT_RATIO = 0.25

_GREEK_CHAR = re.compile(r"\p{Script=Greek}")
_LETTER = re.compile(r"\p{L}")
_UPPER = re.compile(r"\p{Lu}")
_PUNCT = re.compile(r"\p{P}")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def iter_source(settings: Settings) -> Iterator[dict]:
    """Yield raw documents from the source dataset.

    Tries ``datasets.load_dataset`` first (streaming). If iteration fails
    (e.g. pyarrow ``Couldn't deserialize thrift`` on column chunks >2 GB,
    which trips on the single-row-group ``finepdfs_el`` shard), falls back to
    a polars read of the local parquet files under
    ``{source_name}/{source_config}/*.parquet``.
    """
    if not settings.source_name:
        raise ValueError(
            "settings.source_name is empty — set SOURCE_NAME (e.g. HuggingFaceFW/fineweb-2)"
        )

    try:
        yield from _iter_via_datasets(settings)
    except Exception as e:
        logger.warning(
            "datasets iteration failed (%s: %s); falling back to polars",
            type(e).__name__,
            e,
        )
        yield from _iter_via_polars(settings)


def _iter_via_datasets(settings: Settings) -> Iterator[dict]:
    kwargs: dict[str, Any] = {"streaming": True}
    if settings.source_config:
        kwargs["name"] = settings.source_config
    kwargs["split"] = settings.source_split or "train"

    ds = load_dataset(settings.source_name, **kwargs)
    yield from ds


def _iter_via_polars(settings: Settings) -> Iterator[dict]:
    base = Path(settings.source_name)
    src_dir = base / settings.source_config if settings.source_config else base
    files = sorted(src_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"polars fallback found no parquet files under {src_dir}"
        )
    for f in files:
        lf = pl.scan_parquet(f)
        offset = 0
        while True:
            batch = lf.slice(offset, _PARQUET_BATCH_ROWS).collect()
            if batch.is_empty():
                break
            for row in batch.iter_rows(named=True):
                yield row
            offset += _PARQUET_BATCH_ROWS


def pre_filter(doc: dict[str, Any]) -> bool:
    """Return True if the source document should be rephrased.

    Minimal pre-filters:
      - length in [_MIN_CHARS, _MAX_CHARS]
      - Greek-letter ratio ≥ 0.6 of letter characters
      - URL / uppercase / punctuation ratios within sane bounds
    """
    text = doc.get("text") or ""
    n = len(text)
    if n < _MIN_CHARS or n > _MAX_CHARS:
        return False

    letters = len(_LETTER.findall(text))
    if letters == 0:
        return False
    if len(_GREEK_CHAR.findall(text)) / letters < _GREEK_LETTER_RATIO:
        return False

    url_chars = sum(len(m.group(0)) for m in _URL.finditer(text))
    if url_chars / n > _MAX_URL_RATIO:
        return False

    if len(_UPPER.findall(text)) / letters > _MAX_UPPER_RATIO:
        return False

    if len(_PUNCT.findall(text)) / n > _MAX_PUNCT_RATIO:
        return False

    return True


def truncate_at_boundary(text: str, max_chars: int) -> str:
    """Cut at the last newline before ``max_chars``.

    If the slice has no newline we return it as-is rather than emptying the doc.
    """
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    if "\n" in head:
        return head.rsplit("\n", 1)[0]
    return head
