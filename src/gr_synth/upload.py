"""Hugging Face Hub uploader for finished parquet shards."""

import io
import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError

from .config import Settings

log = logging.getLogger(__name__)

_README_TEMPLATE = """\
---
license: apache-2.0
language:
  - el
task_categories:
  - text-generation
tags:
  - synthetic
  - greek
  - pretraining
  - nlp
configs:
  - config_name: faq
    data_files: faq/*.parquet
  - config_name: math
    data_files: math/*.parquet
  - config_name: table
    data_files: table/*.parquet
  - config_name: tutorial
    data_files: tutorial/*.parquet
---

# Greek Synthetic Data

Greek synthetic pre-training data produced by rephrasing source documents through four
pedagogically rich prompt formats (FAQ, Math, Table, Tutorial) — see
[The Synthetic Data Playbook: Generating Trillions of the Finest Tokens](https://huggingface.co/spaces/HuggingFaceFW/finephrase) for the methodology.

## Coverage status

Source corpus: [`alexliap/high-quality-gr-text`](https://huggingface.co/datasets/alexliap/high-quality-gr-text).
Each config is rephrased separately and merged into the same prompt subdirectories; the
originating config is preserved in the `source_data` column of every row.

- [x] `wikipedia_el`
- [x] `finewiki_el`
- [x] `finepdfs_el`
- [ ] `fineweb_hq_el` (in progress)

## Infrastructure

Generation runs on [Lightning AI](https://lightning.ai/) — the rephrasing model,
[`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), is served through a vLLM
endpoint hosted on a Lightning Studio, and the pipeline streams the source corpus into
it from a separate machine.

Each subdirectory holds parquet shards for a single prompt format:

| split    | path                  |
|----------|-----------------------|
| faq      | `faq/part-*.parquet`      |
| math     | `math/part-*.parquet`     |
| table    | `table/part-*.parquet`    |
| tutorial | `tutorial/part-*.parquet` |

## Schema

| column                | type    | notes                                          |
|-----------------------|---------|------------------------------------------------|
| `text`                | string  | synthetic output                               |
| `source_id`           | string  | id of the source document                      |
| `source_data`         | string  | source dataset config the document came from   |
| `prompt`              | string  | one of `faq`, `math`, `table`, `tutorial`      |
| `model`               | string  | rephrasing model + revision                    |
| `language_confidence` | float32 | fastText `lid.176` Greek probability post-filter |

Code used for generating the synthetic data: [`gr-synth`](https://github.com/alexliap/gr_synth).
"""


class HubUploader:
    """Thin wrapper around ``huggingface_hub.HfApi``.

    All methods are synchronous — ``ShardManager`` runs them via ``asyncio.to_thread``
    so the event loop stays responsive during a slow Hub PUT.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._repo_id = settings.hf_repo_id
        self._api = HfApi(token=settings.hf_token)
        self._repo_ready = False

    # ---- public API ------------------------------------------------------

    def upload(self, local_path: Path, repo_path: str) -> None:
        """Upload a single file. ``repo_path`` is relative to the dataset repo root."""
        self._ensure_repo()
        log.info("uploading %s → %s:%s", local_path, self._repo_id, repo_path)
        self._api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=self._repo_id,
            repo_type="dataset",
            commit_message=f"upload {repo_path}",
        )

    def ensure_readme(self) -> None:
        """Write the dataset card if it doesn't exist on the Hub yet.

        Only the first run pays the cost; subsequent calls hit ``EntryNotFoundError``
        only if the file truly is missing.
        """
        self._ensure_repo()
        try:
            self._api.hf_hub_download(
                repo_id=self._repo_id,
                repo_type="dataset",
                filename="README.md",
            )
            return
        except EntryNotFoundError:
            pass
        except Exception as exc:
            # Network hiccup, auth scope, etc. — log and bail rather than overwrite.
            log.warning("could not check README on Hub: %r", exc)
            return

        self._upload_readme(commit_message="add dataset card")

    def refresh_readme(self, commit_message: str | None = None) -> None:
        """Force-overwrite the dataset card on the Hub with the current template."""
        self._ensure_repo()
        self._upload_readme(commit_message=commit_message or "refresh dataset card")

    def _upload_readme(self, *, commit_message: str) -> None:
        readme = _README_TEMPLATE.format(repo_id=self._repo_id).encode("utf-8")
        self._api.upload_file(
            path_or_fileobj=io.BytesIO(readme),
            path_in_repo="README.md",
            repo_id=self._repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    # ---- internal --------------------------------------------------------

    def _ensure_repo(self) -> None:
        if self._repo_ready:
            return
        self._api.create_repo(
            repo_id=self._repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        self._repo_ready = True
