import asyncio
import logging
import random
import statistics

import polars as pl
import typer

from .config import load_settings
from .filters import FilterStats
from .generate import run_pipeline
from .prompts import PROMPTS
from .upload import HubUploader

app = typer.Typer(add_completion=False, no_args_is_help=True)

_PROMPT_NAMES: tuple[str, ...] = tuple(PROMPTS.keys())


@app.command()
def run(
    max_docs: int | None = typer.Option(
        None,
        help=(
            "Cap source docs with new work — i.e. docs that have at least one "
            "prompt not yet present in the on-disk shards (None = stream forever)."
        ),
    ),
    dry_run: bool = typer.Option(True, help="Write shards locally; skip Hub upload."),
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
    progress_log = logging.getLogger("gr_synth.progress")
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
    """Sample N records from all local shards for each prompt and print filter stats.

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
        df = pl.read_parquet([str(s) for s in shards])
        rows = df.height
        total_size = sum(s.stat().st_size for s in shards)
        typer.echo(
            f"\n=== {prompt}: {len(shards)} shards "
            f"({rows} rows, {_size_str(total_size)}) ==="
        )
        if rows == 0:
            continue
        sample_idx = rng.sample(range(rows), k=min(per_prompt, rows))
        sample = df[sample_idx].to_dicts()
        confs = [
            r.get("language_confidence")
            for r in sample
            if r.get("language_confidence") is not None
        ]
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


@app.command("refresh-readme")
def refresh_readme(
    message: str = typer.Option(
        "refresh dataset card",
        "--message",
        "-m",
        help="Commit message for the README update.",
    ),
) -> None:
    """Force-overwrite the dataset README on the Hub with the current template."""
    settings = load_settings()
    uploader = HubUploader(settings)
    uploader.refresh_readme(commit_message=message)
    typer.echo(f"refreshed README on {settings.hf_repo_id}")


def _print_stats(stats: FilterStats) -> None:
    seen = max(stats.seen, 1)
    typer.echo("=== filter stats ===")
    typer.echo(f"  seen           : {stats.seen}")
    typer.echo(
        f"  dropped_lang   : {stats.dropped_lang}  ({stats.dropped_lang / seen:.1%})"
    )
    typer.echo(
        f"  dropped_length : {stats.dropped_length}  ({stats.dropped_length / seen:.1%})"
    )
    typer.echo(
        f"  dropped_format : {stats.dropped_format}  ({stats.dropped_format / seen:.1%})"
    )
    typer.echo(
        f"  dropped_dup    : {stats.dropped_dup}  ({stats.dropped_dup / seen:.1%})"
    )
    typer.echo(f"  kept           : {stats.kept}  ({stats.kept / seen:.1%})")
    if stats.lid_confidences:
        typer.echo(
            "  lid_confidence : "
            f"min={min(stats.lid_confidences):.3f} "
            f"mean={statistics.fmean(stats.lid_confidences):.3f} "
            f"max={max(stats.lid_confidences):.3f}"
        )


def _size_str(size: int) -> str:
    s = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if s < 1024:
            return f"{s:.1f}{unit}"
        s /= 1024
    return f"{s:.1f}TB"


if __name__ == "__main__":
    app()
