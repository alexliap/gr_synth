# gr-synth

Greek synthetic pre-training data pipeline. Streams an `alexliap/high-quality-gr-text`
config (`fineweb_hq_el`, `finepdfs_el`, `finewiki_el`, or `wikipedia_el`), rephrases each
document through four pedagogically rich prompts (FAQ, Math, Table, Tutorial) against a
deployed vLLM endpoint via Pydantic AI, filters the output, and pushes parquet shards
(flushed every `ROWS_PER_FLUSH` records) to the Hugging Face Hub.

## Quick start

```bash
uv sync
cp .env.example .env       # fill VLLM_BASE_URL, HF_TOKEN, HF_REPO_ID
mkdir -p models
curl -L -o models/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Smoke test, no Hub upload:
uv run gr-synth run --max-docs 25 --dry-run

# Read back a few samples + filter drop-rates:
uv run gr-synth spot-check --n 10
```

## Layout

- [src/gr_synth/prompts.py](src/gr_synth/prompts.py) — the four Greek prompts.
- [src/gr_synth/agent.py](src/gr_synth/agent.py) — Pydantic AI agent on the vLLM endpoint.
- [src/gr_synth/filters.py](src/gr_synth/filters.py) — language-ID, format, dedup.
- [src/gr_synth/shard.py](src/gr_synth/shard.py) — row-count-buffered shard writer (flushes every `ROWS_PER_FLUSH` records).
- [src/gr_synth/cli.py](src/gr_synth/cli.py) — entry point.
