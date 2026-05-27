"""Async fan-out pipeline: source → 4 prompts × N concurrent vLLM calls → filters → shards."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic_ai import Agent

from .agent import build_agent
from .config import Settings
from .filters import (
    FilterStats,
    MinHashDeduper,
    apply_all,
    load_lid_model,
)
from .prompts import PROMPTS
from .shard import ShardManager
from .source import iter_source, pre_filter, truncate_at_boundary
from .upload import HubUploader
from .types import Record

_PROMPT_NAMES: tuple[str, ...] = tuple(PROMPTS.keys())


log = logging.getLogger(__name__)
# Dedicated logger so the progress heartbeat shows up even when --verbose is off.
# cli.py attaches a handler and pins its level to INFO with propagate=False.
progress_log = logging.getLogger("gr_synth.progress")



async def rephrase(
    agent: Agent,
    prompt_name: str,
    doc_text: str,
    source_id: str | int,
    settings: Settings,
) -> Record:
    """Run a single rephrasing call against the agent."""
    result = await agent.run(PROMPTS[prompt_name].format(doc=doc_text))

    return Record(
        prompt=prompt_name,
        source_id=source_id,
        model=settings.vllm_model_id,
        text=result.output,
        language_confidence=None,
        source_data=settings.source_config
    )


async def _process_one(
    *,
    agent: Agent[None, str],
    sem: asyncio.Semaphore,
    prompt_name: str,
    doc_text: str,
    source_id: str | int,
    settings: Settings,
    lid_model: Any,
    deduper: MinHashDeduper,
    stats: FilterStats,
    shard_mgr: ShardManager,
) -> None:
    """One unit of work: rephrase → filter → (maybe) hand off to the shard manager.

    Exceptions are logged and swallowed so a single bad doc can't sink the run.
    """
    try:
        async with sem:
            log.info(f"Rephrasing {source_id} using {prompt_name} ...")
            record = await rephrase(agent, prompt_name, doc_text, source_id, settings)
    except Exception:
        log.exception(
            "rephrase failed (source_id=%s, prompt=%s)", source_id, prompt_name
        )
        return

    record, status = apply_all(
        record,
        lid_model=lid_model,
        deduper=deduper,
        stats=stats,
        settings=settings,
    )
    if stats.seen % settings.progress_every == 0:
        progress_log.info(
            "processed %d records (kept=%d, dropped lang=%d length=%d format=%d dup=%d)",
            stats.seen,
            stats.kept,
            stats.dropped_lang,
            stats.dropped_length,
            stats.dropped_format,
            stats.dropped_dup,
        )
    # if kept is None:
    #     return
    try:
        await shard_mgr.add(record, status)
    except Exception:
        log.exception(
            "shard add failed (source_id=%s, prompt=%s)", source_id, prompt_name
        )


async def run_pipeline(
    settings: Settings,
    *,
    max_docs: int | None = None,
    prompts: tuple[str, ...] | None = None,
    dry_run: bool = False,
) -> FilterStats:
    """End-to-end run. Returns the aggregate ``FilterStats`` so the CLI can summarise.

    ``prompts=None`` means every key of ``PROMPTS``. ``dry_run=True`` skips the
    Hub upload but still writes parquet locally.
    """
    chosen = prompts if prompts is not None else _PROMPT_NAMES
    unknown = set(chosen) - set(_PROMPT_NAMES)
    if unknown:
        raise ValueError(f"Unknown prompt names: {sorted(unknown)}")

    agent = build_agent(settings)
    sem = asyncio.Semaphore(settings.concurrency)
    lid_model = load_lid_model(settings)

    uploader = None if dry_run else HubUploader(settings)
    if uploader is not None:
        uploader.ensure_readme()

    shard_mgr = ShardManager(settings, uploader, dry_run=dry_run)
    dedupers = {
        p: MinHashDeduper(settings.minhash_threshold, settings.minhash_perm)
        for p in chosen
    }
    stats = FilterStats()

    # Per-prompt resume: each prompt skips any source_id already present in
    # its on-disk parquet shards. The set is built at ShardManager init.
    seen_counts = shard_mgr.seen_counts()
    progress_log.info("seen source_ids per prompt at startup: %s", seen_counts)

    # Rehydrate each prompt's MinHashDeduper from texts already on disk for
    # the current source_data, so near-duplicates across runs are caught.
    for prompt_name in chosen:
        if seen_counts.get(prompt_name, 0) == 0:
            continue
        rehydrated = 0
        for text in shard_mgr.iter_existing_texts(prompt_name):
            dedupers[prompt_name].add_and_check(text)
            rehydrated += 1
        progress_log.info(
            "rehydrated dedup state for %s from %d existing records",
            prompt_name,
            rehydrated,
        )

    pending: set[asyncio.Task[None]] = set()
    # Cap in-flight tasks so the producer doesn't outrun the workers and balloon memory.
    max_pending = max(settings.concurrency * 4, 16)

    # iter_source is a sync generator pulling over the network; offload its
    # next() to a thread so a slow HTTP fetch doesn't stall the event loop.
    src_iter = iter(iter_source(settings))

    def _next_doc() -> dict[str, Any] | None:
        try:
            return next(src_iter)
        except StopIteration:
            return None

    docs_seen = 0
    docs_streamed = 0
    skipped: dict[str, int] = {p: 0 for p in chosen}
    while True:
        if max_docs is not None and docs_seen >= max_docs:
            break

        doc = await asyncio.to_thread(_next_doc)
        if doc is None:
            break

        if not pre_filter(doc):
            continue

        doc_text = truncate_at_boundary(doc["text"], settings.max_input_chars)
        # alexliap/high-quality-gr-text has no `id`; `url` is the stable identifier.
        source_id = doc.get("url")

        # Per-prompt resume gate: skip prompts whose on-disk shards already
        # contain this source_id. Count skips per prompt for periodic logging.
        to_schedule: list[str] = []
        for p in chosen:
            if shard_mgr.is_seen(p, source_id):
                skipped[p] += 1
            else:
                to_schedule.append(p)

        docs_streamed += 1
        if docs_streamed % settings.skip_log_every == 0:
            progress_log.info(
                "resume skips per prompt after %d source docs: %s",
                docs_streamed,
                skipped,
            )

        if not to_schedule:
            continue

        docs_seen += 1

        for prompt_name in to_schedule:
            task = asyncio.create_task(
                _process_one(
                    agent=agent,
                    sem=sem,
                    prompt_name=prompt_name,
                    doc_text=doc_text,
                    source_id=source_id,
                    settings=settings,
                    lid_model=lid_model,
                    deduper=dedupers[prompt_name],
                    stats=stats,
                    shard_mgr=shard_mgr,
                )
            )
            pending.add(task)
            task.add_done_callback(pending.discard)

        while len(pending) >= max_pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if (exc := t.exception()) is not None:
                    log.error("task crashed: %r", exc)

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    await shard_mgr.close()
    progress_log.info(
        "final resume skips per prompt after %d source docs: %s",
        docs_streamed,
        skipped,
    )
    return stats
