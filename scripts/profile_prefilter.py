"""Count how many docs / tokens each pre_filter step drops, for one or more
source configs under tmp/data/.

Uses Polars for parquet reading — pyarrow's iter_batches hangs on the single-
row-group files under tmp/data/.

Usage:
    uv run --with polars python scripts/profile_prefilter.py finepdfs_el wikipedia_el [--limit 100]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import polars as pl
import regex as re

# Mirror the constants from src/gr_synth_data/source.py
_MIN_CHARS = 200
_MAX_CHARS = 20_000
_GREEK_LETTER_RATIO = 0.6
_MAX_URL_RATIO = 0.05
_MAX_UPPER_RATIO = 0.40
_MAX_PUNCT_RATIO = 0.25

_GREEK_CHAR = re.compile(r"\p{Script=Greek}")
_LETTER = re.compile(r"\p{L}")
_UPPER = re.compile(r"\p{Lu}")
_PUNCT = re.compile(r"\p{P}")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)

_VERDICTS = (
    "kept",
    "too_short",
    "too_long",
    "no_letters",
    "not_greek",
    "too_many_urls",
    "too_much_upper",
    "too_much_punct",
)


def classify(text: str) -> str:
    n = len(text)
    if n < _MIN_CHARS:
        return "too_short"
    if n > _MAX_CHARS:
        return "too_long"
    letters = len(_LETTER.findall(text))
    if letters == 0:
        return "no_letters"
    if len(_GREEK_CHAR.findall(text)) / letters < _GREEK_LETTER_RATIO:
        return "not_greek"
    url_chars = sum(len(m.group(0)) for m in _URL.finditer(text))
    if url_chars / n > _MAX_URL_RATIO:
        return "too_many_urls"
    if len(_UPPER.findall(text)) / letters > _MAX_UPPER_RATIO:
        return "too_much_upper"
    if len(_PUNCT.findall(text)) / n > _MAX_PUNCT_RATIO:
        return "too_much_punct"
    return "kept"


def profile_config(config_dir: Path, limit: int | None) -> None:
    files = sorted(config_dir.glob("*.parquet"))
    if not files:
        print(f"\n=== {config_dir.name}: no parquet files ===")
        return

    # Lazy scan + head(limit) lets Polars push the row-limit down into the
    # parquet reader so we don't materialise the whole row group.
    lf = pl.scan_parquet([str(f) for f in files]).select(["text", "token_count"])
    if limit is not None:
        lf = lf.head(limit)
    df = lf.collect()

    docs: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for text, tk in zip(df["text"].to_list(), df["token_count"].to_list()):
        verdict = classify(text or "")
        docs[verdict] += 1
        tokens[verdict] += int(tk or 0)

    total_docs = sum(docs.values())
    total_tokens = sum(tokens.values())
    print(f"\n=== {config_dir.name} (scanned {total_docs:,} docs, {total_tokens:,} tokens) ===")
    print(f"  {'verdict':<16} {'docs':>10} {'doc%':>7}   {'tokens':>15} {'tok%':>7}")
    for v in _VERDICTS:
        d = docs.get(v, 0)
        t = tokens.get(v, 0)
        dp = 100.0 * d / max(total_docs, 1)
        tp = 100.0 * t / max(total_tokens, 1)
        print(f"  {v:<16} {d:>10,} {dp:>6.1f}%   {t:>15,} {tp:>6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="Config names under tmp/data/.")
    ap.add_argument("--root", default="tmp/data")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max docs to scan per config (default: 100).")
    args = ap.parse_args()

    for cfg in args.configs:
        profile_config(Path(args.root) / cfg, args.limit)


if __name__ == "__main__":
    main()
