"""Per-prompt buffered shard writer. Flushes to parquet + Hub every N rows."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from .config import Settings
from .prompts import PROMPTS
from .types import Record
from .upload import HubUploader

_PROMPT_NAMES: tuple[str, ...] = tuple(PROMPTS.keys())
_PART_RE = re.compile(r"part-(\d+)\.parquet$")

_PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "text": pl.String,
    "source_id": pl.String,
    "source_data": pl.String,
    "prompt": pl.String,
    "model": pl.String,
    "language_confidence": pl.Float32,
}

_FAILED_SCHEMA: dict[str, pl.DataType] = {
    **_PARQUET_SCHEMA,
    "failure_reason": pl.String,
}


class ShardManager:
    """Buffers filtered records per prompt and flushes a parquet shard every
    ``settings.rows_per_flush`` records.

    Shard layout: ``{local_shard_dir}/{prompt}/part-{NNNNN}.parquet``.
    Hub layout:   ``{prompt}/part-{NNNNN}.parquet`` in ``settings.hf_repo_id``.

    Resume: at init, the per-prompt ``source_id`` set is loaded from both the
    success dir (``{prompt}/part-*.parquet``) and the failure dir
    (``{prompt}_failed/part-*.parquet``), filtered by ``settings.source_config``.
    The producer skips any (doc, prompt) whose source_id is in that union, so
    neither previously-kept nor previously-failed records are re-attempted.
    The next shard index is derived from the highest existing ``part-{NNNNN}``
    in the success dir.
    """

    def __init__(
        self,
        settings: Settings,
        uploader: HubUploader | None,
        *,
        dry_run: bool = False,
    ) -> None:
        self._settings = settings
        self._uploader = None if dry_run else uploader
        self._dry_run = dry_run

        self._buffers: dict[str, list[Record]] = {p: [] for p in _PROMPT_NAMES}
        self._buffers_failed: dict[str, list[Record]] = {p: [] for p in _PROMPT_NAMES}

        self._locks: dict[str, asyncio.Lock] = {
            p: asyncio.Lock() for p in _PROMPT_NAMES
        }

        settings.local_shard_dir.mkdir(parents=True, exist_ok=True)
        self._source_data = settings.source_config
        self._seen: dict[str, set[str]] = {}
        self._next_shard: dict[str, int] = {}
        for p in _PROMPT_NAMES:
            prompt_dir = settings.local_shard_dir / p
            failed_prompt_dir = settings.local_shard_dir / f"{p}_failed"

            prompt_dir.mkdir(parents=True, exist_ok=True)
            failed_prompt_dir.mkdir(parents=True, exist_ok=True)

            ids, next_idx = _load_existing_source_ids(prompt_dir, self._source_data)
            failed_ids, _ = _load_existing_source_ids(
                failed_prompt_dir, self._source_data
            )

            self._seen[p] = ids | failed_ids
            self._next_shard[p] = next_idx

    def is_seen(self, prompt: str, source_id: str) -> bool:
        """Return True if ``source_id`` is already in a flushed shard for ``prompt``."""
        return source_id in self._seen[prompt]

    def seen_counts(self) -> dict[str, int]:
        """Per-prompt count of source_ids already on disk (for startup logging)."""
        return {p: len(self._seen[p]) for p in _PROMPT_NAMES}

    def iter_existing_texts(self, prompt: str) -> Iterator[str]:
        """Yield ``text`` values for the current ``source_data`` across all
        ``part-*.parquet`` files for ``prompt``. Used to rehydrate the
        ``MinHashDeduper`` so cross-run near-duplicate detection works."""
        prompt_dir = self._settings.local_shard_dir / prompt
        yield from _iter_existing_texts(prompt_dir, self._source_data)

    async def add(self, record: Record, status: str) -> None:
        """Append the record to its prompt's buffer; trigger a flush if over budget."""
        prompt = record.prompt

        if status:
            buffers = self._buffers_failed
            flush = self.flush_failed
            record.failure_reason = status
        else:
            buffers = self._buffers
            flush = self.flush

        async with self._locks[prompt]:
            buffers[prompt].append(record)
            self._seen[prompt].add(record.source_id)
            over_budget = len(buffers[prompt]) >= self._settings.rows_per_flush

        if over_budget:
            await flush(prompt)

    async def flush_failed(self, prompt: str) -> None:
        async with self._locks[prompt]:
            buf = self._buffers_failed[prompt]
            shard_idx = self._next_shard[prompt]
            self._buffers_failed[prompt] = []
            self._next_shard[prompt] = shard_idx + 1

        filename = f"part-{shard_idx:05d}-failed.parquet"
        local_path = self._settings.local_shard_dir / f"{prompt}_failed" / filename
        await asyncio.to_thread(_write_parquet, buf, local_path, _FAILED_SCHEMA)

    async def flush(self, prompt: str) -> None:
        """Write the buffer for ``prompt`` to parquet, upload, clear."""
        async with self._locks[prompt]:
            buf = self._buffers[prompt]
            shard_idx = self._next_shard[prompt]
            self._buffers[prompt] = []
            self._next_shard[prompt] = shard_idx + 1

        filename = f"part-{shard_idx:05d}.parquet"
        local_path = self._settings.local_shard_dir / prompt / filename
        await asyncio.to_thread(_write_parquet, buf, local_path, _PARQUET_SCHEMA)

        if self._uploader is not None:
            repo_path = f"{prompt}/{filename}"
            await asyncio.to_thread(self._uploader.upload, local_path, repo_path)

    async def close(self) -> None:
        """Flush every non-empty buffer at shutdown so we don't lose tail data."""
        for prompt in _PROMPT_NAMES:
            if self._buffers[prompt]:
                await self.flush(prompt)
            if self._buffers_failed[prompt]:
                await self.flush_failed(prompt)


def _load_existing_source_ids(
    prompt_dir: Path, source_data: str
) -> tuple[set[str], int]:
    """Scan ``prompt_dir`` for ``part-NNNNN.parquet`` files and return
    ``(source_ids whose source_data == source_data, next shard index)``.

    Uses polars' lazy scan with column pruning + predicate pushdown.
    Files not matching ``part-{int}.parquet`` are ignored. Every shard is
    expected to carry a ``source_data`` column — legacy shards without it
    will raise during the polars filter.
    """
    ids: set[str] = set()
    max_idx = -1
    for path in sorted(prompt_dir.glob("part-*.parquet")):
        m = _PART_RE.match(path.name)
        if not m:
            continue
        max_idx = max(max_idx, int(m.group(1)))
        lf = pl.scan_parquet(path)
        sids = (
            lf.filter(pl.col("source_data") == source_data)
            .select("source_id")
            .collect()
            .get_column("source_id")
            .drop_nulls()
            .to_list()
        )
        ids.update(sids)

    return ids, max_idx + 1


def _iter_existing_texts(prompt_dir: Path, source_data: str) -> Iterator[str]:
    """Stream ``text`` values for the matching ``source_data`` across all
    ``part-*.parquet`` files under ``prompt_dir``.

    Reads one parquet at a time so the full text set never lives in memory.
    """
    for path in sorted(prompt_dir.glob("part-*.parquet")):
        if not _PART_RE.match(path.name):
            continue
        col = (
            pl.scan_parquet(path)
            .filter(pl.col("source_data") == source_data)
            .select("text")
            .collect()
            .get_column("text")
            .drop_nulls()
        )
        for v in col:
            yield v


def _write_parquet(
    records: list[Record],
    local_path: Path,
    schema: dict[str, pl.DataType] = _PARQUET_SCHEMA,
) -> None:
    """Materialise ``records`` to a parquet file under ``local_path``.

    All columns in ``schema`` are filled; missing keys become NULL.
    ``source_id`` is coerced to string so int/str sources don't break the schema.
    """
    columns: dict[str, list] = {name: [] for name in schema}
    for rec in records:
        columns["text"].append(rec.text)
        columns["source_id"].append(rec.source_id)
        columns["source_data"].append(rec.source_data)
        columns["prompt"].append(rec.prompt)
        columns["model"].append(rec.model)
        conf = rec.language_confidence
        columns["language_confidence"].append(None if conf is None else float(conf))
        if "failure_reason" in columns:
            columns["failure_reason"].append(rec.failure_reason)

    df = pl.DataFrame(columns, schema=schema)
    df.write_parquet(local_path, compression="zstd")
