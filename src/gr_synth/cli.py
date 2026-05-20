"""Typer CLI entry-point."""

from __future__ import annotations

import asyncio
import logging
import random
import statistics
from pathlib import Path

import pyarrow.parquet as pq
import typer

from .config import load_settings
from .filters import FilterStats
from .generate import run_pipeline
from .prompts import PROMPTS

app = typer.Typer(add_completion=False, no_args_is_help=True)

_PROMPT_NAMES: tuple[str, ...] = tuple(PROMPTS.keys())


@app.command()
def run(
    max_docs: int | None = typer.Option(
        None, help="Cap source docs (None = stream forever)."
    ),
    dry_run: bool = typer.Option(
        False, help="Write shards locally; skip Hub upload."
    ),
    prompts: str | None = typer.Option(
        None, help="Comma-separated prompt names (default: all four)."
    ),
    verbose: bool = typer.Option(False, help="Enable INFO logging."),
) -> None:
    """Run the generation pipeline."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    # Progress heartbeat: independent handler so it prints every N records
    # regardless of --verbose.
    progress_log = logging.getLogger("gr_synth_data.progress")
    progress_log.setLevel(logging.INFO)
    progress_log.propagate = False
    if not progress_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s PROGRESS | %(message)s"))
        progress_log.addHandler(handler)

    settings = load_settings()
    chosen: tuple[str, ...] | None = None
    if prompts:
        chosen = tuple(p.strip() for p in prompts.split(",") if p.strip())

    stats = asyncio.run(
        run_pipeline(settings, max_docs=max_docs, prompts=chosen, dry_run=dry_run)
    )
    _print_stats(stats)


@app.command("spot-check")
def spot_check(
    n: int = typer.Option(50, help="Number of random records to display."),
) -> None:
    """Sample N records from the latest local shards and print filter stats.

    Read a fresh slice every few hours to catch
    English drift, refusals, and template collapse.
    """
    settings = load_settings()

    per_prompt = max(1, n // len(_PROMPT_NAMES))
    rng = random.Random(0)

    for prompt in _PROMPT_NAMES:
        prompt_dir = settings.local_shard_dir / prompt
        shards = sorted(prompt_dir.glob("part-*.parquet"))
        if not shards:
            typer.echo(f"\n=== {prompt}: no shards yet ===")
            continue
        latest = shards[-1]
        table = pq.read_table(latest)
        rows = table.num_rows
        typer.echo(
            f"\n=== {prompt}: {latest.name} ({rows} rows, {_size_str(latest)}) ==="
        )
        if rows == 0:
            continue
        sample_idx = rng.sample(range(rows), k=min(per_prompt, rows))
        sample = table.take(sample_idx).to_pylist()
        confs = [r.get("language_confidence") for r in sample if r.get("language_confidence") is not None]
        if confs:
            typer.echo(
                f"  language_confidence: min={min(confs):.3f} "
                f"mean={statistics.fmean(confs):.3f} max={max(confs):.3f}"
            )
        for rec in sample:
            head = (rec.get("text") or "").strip().splitlines()[:6]
            typer.echo(f"  --- source_id={rec.get('source_id')} ---")
            for line in head:
                typer.echo(f"    {line}")


def _print_stats(stats: FilterStats) -> None:
    seen = max(stats.seen, 1)
    typer.echo("=== filter stats ===")
    typer.echo(f"  seen           : {stats.seen}")
    typer.echo(f"  dropped_lang   : {stats.dropped_lang}  ({stats.dropped_lang / seen:.1%})")
    typer.echo(f"  dropped_length : {stats.dropped_length}  ({stats.dropped_length / seen:.1%})")
    typer.echo(f"  dropped_format : {stats.dropped_format}  ({stats.dropped_format / seen:.1%})")
    typer.echo(f"  dropped_dup    : {stats.dropped_dup}  ({stats.dropped_dup / seen:.1%})")
    typer.echo(f"  kept           : {stats.kept}  ({stats.kept / seen:.1%})")
    if stats.lid_confidences:
        typer.echo(
            "  lid_confidence : "
            f"min={min(stats.lid_confidences):.3f} "
            f"mean={statistics.fmean(stats.lid_confidences):.3f} "
            f"max={max(stats.lid_confidences):.3f}"
        )


def _size_str(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


if __name__ == "__main__":
    app()
