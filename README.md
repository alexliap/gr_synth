# gr-synth

Greek synthetic pre-training data pipeline. Streams an `alexliap/high-quality-gr-text`
config (`fineweb_hq_el`, `finepdfs_el`, `finewiki_el`, or `wikipedia_el`), rephrases each
document through four pedagogically rich prompts (FAQ, Math, Table, Tutorial) against a
deployed vLLM endpoint via Pydantic AI, filters the output, and pushes parquet shards
(flushed every `ROWS_PER_FLUSH` records) to the Hugging Face Hub.

## Output

Generated dataset: [`alexliap/greek-synth-v1`](https://huggingface.co/datasets/alexliap/greek-synth-v1) (default `HF_REPO_ID`).
Each prompt is its own HF dataset config: `faq`, `math`, `table`, `tutorial`. Schema:
`text`, `source_id`, `source_data`, `prompt`, `model`, `language_confidence`.

## Quick start

```bash
uv sync
cp .env.example .env       # fill VLLM_BASE_URL, VLLM_PORTS, HF_TOKEN, HF_REPO_ID
mkdir -p models
curl -L -o models/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Smoke test, no Hub upload:
uv run gr-synth run --max-docs 25 --dry-run
```

`VLLM_BASE_URL` is the shared host (no port, no path); `VLLM_PORTS` is a
comma-separated list of ports, one per deployment of the same model. Rephrase
calls are round-robined across them, so adding a second port roughly doubles
LLM throughput — just bump `CONCURRENCY` proportionally to keep both endpoints
busy.

```bash

# Real run with verbose logs and a subset of prompts:
uv run gr-synth run --verbose --prompts faq,math

# Read back a few samples + filter drop-rates:
uv run gr-synth spot-check --n 10

# Force-overwrite the dataset README on the Hub (otherwise it's only written on first push):
uv run gr-synth refresh-readme
```

## Resume

Re-running picks up where the previous run left off. At startup `ShardManager` scans
`data/shards/<prompt>/part-*.parquet`, builds a per-prompt set of `source_id`s already
on disk (filtered by the current `SOURCE_CONFIG`), and rehydrates each prompt's
`MinHashDeduper` from those existing texts. The producer then skips any
`(source_id, prompt)` pair already in a flushed shard — no `state.json` involved.

## Layout

Package:

- [src/gr_synth/cli.py](src/gr_synth/cli.py) — `gr-synth run / spot-check / refresh-readme` entry points.
- [src/gr_synth/generate.py](src/gr_synth/generate.py) — async fan-out pipeline: source → 4 prompts × N concurrent vLLM calls → filters → shards.
- [src/gr_synth/agent.py](src/gr_synth/agent.py) — Pydantic AI agent on the vLLM endpoint.
- [src/gr_synth/prompts.py](src/gr_synth/prompts.py) — the four Greek prompt templates.
- [src/gr_synth/source.py](src/gr_synth/source.py) — streaming source loader + pre-filters.
- [src/gr_synth/filters.py](src/gr_synth/filters.py) — language-ID, format validation, near-dup MinHash LSH.
- [src/gr_synth/shard.py](src/gr_synth/shard.py) — row-count-buffered shard writer (flushes every `ROWS_PER_FLUSH` records), parquet-based resume.
- [src/gr_synth/upload.py](src/gr_synth/upload.py) — Hub uploader + dataset README template.
- [src/gr_synth/config.py](src/gr_synth/config.py) — `pydantic-settings` env binding.
- [src/gr_synth/types.py](src/gr_synth/types.py) — `Record` schema.

Helper scripts (one-off utilities, run via `uv run python scripts/<file>`):

- [scripts/read_shards.py](scripts/read_shards.py) — read all `part-*.parquet` under a prompt dir, print a summary, and optionally merge them into one consolidated parquet (`--out`).
- [scripts/upload_merged.py](scripts/upload_merged.py) — push merged per-prompt parquets (anything matching `<prompt>*.parquet` under `--input`) to the configured HF dataset repo.
- [scripts/profile_prefilter.py](scripts/profile_prefilter.py) — count how many source docs each pre-filter step drops, per `SOURCE_CONFIG`.
