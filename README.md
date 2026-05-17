# gr-synth-data

Greek synthetic pre-training data pipeline. Streams FineWeb-2 `ell_Grek`, rephrases each
document through four pedagogically rich prompts (FAQ, Math, Table, Tutorial) against a
deployed vLLM endpoint via Pydantic AI, filters the output, and pushes 5 GB parquet shards
to the Hugging Face Hub.

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

- [src/gr_synth_data/prompts.py](src/gr_synth_data/prompts.py) — the four Greek prompts.
- [src/gr_synth_data/agent.py](src/gr_synth_data/agent.py) — Pydantic AI agent on the vLLM endpoint.
- [src/gr_synth_data/filters.py](src/gr_synth_data/filters.py) — language-ID, format, dedup.
- [src/gr_synth_data/shard.py](src/gr_synth_data/shard.py) — 5 GB buffered shard writer.
- [src/gr_synth_data/cli.py](src/gr_synth_data/cli.py) — entry point.

See `greek_synthetic_data_guide.md` for the recipe this pipeline implements.