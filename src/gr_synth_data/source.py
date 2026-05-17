"""Streaming source + pre-filters (guide §2)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import regex as re
from datasets import load_dataset

from .config import Settings

_MIN_CHARS = 200
_MAX_CHARS = 20_000

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
    """Yield raw documents from the source dataset in streaming mode."""
    if not settings.source_name:
        raise ValueError(
            "settings.source_name is empty — set SOURCE_NAME (e.g. HuggingFaceFW/fineweb-2)"
        )

    kwargs: dict[str, Any] = {"streaming": True}
    if settings.source_config:
        kwargs["name"] = settings.source_config
    kwargs["split"] = settings.source_split or "train"

    ds = load_dataset(settings.source_name, **kwargs)
    yield from ds


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
    """Cut at the last newline before ``max_chars`` (guide §5.2 pattern).

    If the slice has no newline we return it as-is rather than emptying the doc.
    """
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    if "\n" in head:
        return head.rsplit("\n", 1)[0]
    return head
