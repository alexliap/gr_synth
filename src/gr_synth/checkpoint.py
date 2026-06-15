import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .config import Settings

log = logging.getLogger(__name__)

# CHECKPOINT_VERSION = 1

# # Success shards are ``part-NNNNN.parquet``; failed shards are ``part-NNNNN-failed.parquet``.
# # Both consume the same per-prompt counter, so both dirs must be scanned for the next index.
# _SUCCESS_RE = re.compile(r"part-(\d+)\.parquet$")
# _FAILED_RE = re.compile(r"part-(\d+)-failed\.parquet$")


class ConfigCheckpoint(BaseModel):
    source_name: str
    prompt: str
    seen_row_index: int


