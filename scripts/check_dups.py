"""Per-prompt near-duplicate audit.

Reads a prompt's success shards and runs them through the same
``MinHashDeduper`` used during generation, reporting how many records are
near-duplicates. The check is scoped per prompt (one deduper instance over all
of that prompt's rows), matching how dedup runs in-process during a run.

Examples:
    uv run python scripts/check_dups.py faq
    uv run python scripts/check_dups.py math --threshold 0.85
"""

import argparse
from pathlib import Path

import polars as pl

from gr_synth.config import load_settings
from gr_synth.filters import MinHashDeduper

_PROMPTS = ("faq", "math", "table", "tutorial")


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", choices=_PROMPTS, help="Which prompt's shards to check.")
    ap.add_argument(
        "--root", type=Path, default=Path("data/shards"), help="Shard root."
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=settings.minhash_threshold,
        help=f"MinHash LSH threshold (default: {settings.minhash_threshold}).",
    )
    ap.add_argument(
        "--num-perm",
        type=int,
        default=settings.minhash_perm,
        help=f"MinHash permutations (default: {settings.minhash_perm}).",
    )
    ap.add_argument(
        "--report-every",
        type=int,
        default=50_000,
        help="Print progress every N rows (0 to disable).",
    )
    args = ap.parse_args()

    prompt_dir = args.root / args.prompt
    shards = sorted(prompt_dir.glob("part-*.parquet"))
    if not shards:
        print(f"{prompt_dir}: no shards")
        return

    texts = pl.read_parquet([str(s) for s in shards]).get_column("text").to_list()
    total = len(texts)
    print(
        f"{args.prompt}: {total:,} rows across {len(shards)} shard(s) | "
        f"deduper threshold={args.threshold}, num_perm={args.num_perm}"
    )

    deduper = MinHashDeduper(args.threshold, args.num_perm)
    dups = 0
    for i, text in enumerate(texts, start=1):
        if not deduper.add_and_check(text):
            dups += 1
        if args.report_every and i % args.report_every == 0:
            print(f"  {i:,}/{total:,} scanned, {dups:,} near-dups so far")

    unique = total - dups
    pct = (dups / total * 100) if total else 0.0
    print(
        f"=== {args.prompt}: {unique:,} unique | "
        f"{dups:,} near-duplicates ({pct:.2f}%) of {total:,} rows ==="
    )


if __name__ == "__main__":
    main()
