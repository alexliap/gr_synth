import argparse
from pathlib import Path

import polars as pl

_PROMPTS = ("faq", "math", "table", "tutorial")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", choices=_PROMPTS, help="Which prompt's shards to read.")
    ap.add_argument("--root", default="data/shards", help="Shard root (default: data/shards).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, write the merged DataFrame to this path as a single parquet.",
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        help="Compression for the merged output (default: zstd).",
    )
    args = ap.parse_args()

    prompt_dir = Path(args.root) / args.prompt
    shards = sorted(prompt_dir.glob("part-*.parquet"))
    if not shards:
        print(f"{prompt_dir}: no shards")
        return

    df = pl.read_parquet([str(s) for s in shards])
    total_bytes = sum(s.stat().st_size for s in shards)
    print(
        f"=== {args.prompt}: {len(shards)} shard(s), "
        f"{df.height:,} rows, {df.width} cols, {total_bytes / 1024:.1f} KB on disk ==="
    )
    print(f"columns: {df.columns}")

    by_source = df.group_by("source_data").len().sort("len", descending=True)
    print("by source_data:")
    for row in by_source.iter_rows(named=True):
        print(f"  {row['source_data']}: {row['len']:,}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.out, compression=args.compression)
        size = args.out.stat().st_size
        print(
            f"merged {len(shards)} shard(s) → {args.out} "
            f"({df.height:,} rows, {size / 1024 / 1024:.1f} MB)"
        )


if __name__ == "__main__":
    main()
