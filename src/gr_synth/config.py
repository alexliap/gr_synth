from pathlib import Path
from typing import Annotated

from pydantic import model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Bare host for the vLLM endpoint(s) — no port, no path. Full URLs are
    # composed at agent-build time as f"{vllm_base_url}:{port}/v1" for each
    # entry in ``vllm_ports`` (round-robin across them).
    vllm_base_url: str = "http://localhost"
    # ``NoDecode`` keeps the raw env string ("8005,8006") from going through
    # pydantic-settings' default JSON decoder; the model_validator below splits it.
    vllm_ports: Annotated[list[int], NoDecode] = [8000]
    vllm_api_key: str = "EMPTY"
    vllm_model_id: str = "Qwen/Qwen3.5-2B"

    hf_token: str | None = None
    hf_repo_id: str = "alexliap/greek-synth-v1"

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    repetition_penalty: float = 1.05

    concurrency: int = 192
    rows_per_flush: int = 1000
    local_shard_dir: Path = Path("./data/shards")

    source_name: str = ""
    source_config: str = ""
    source_split: str = ""
    max_input_chars: int = 12_000

    lid_model_path: Path = Path("./models/lid.176.bin")
    lid_threshold: float = 0.9
    min_output_chars: int = 150
    max_output_chars: int = 8_000

    minhash_threshold: float = 0.8
    minhash_perm: int = 128

    progress_every: int = 2000
    skip_log_every: int = 1000

    @model_validator(mode="before")
    @classmethod
    def _parse_vllm_ports(cls, data):
        # Accept comma-separated env strings ("8005,8006") for VLLM_PORTS in
        # addition to native lists. ``pydantic-settings`` otherwise tries JSON.
        if isinstance(data, dict):
            v = data.get("vllm_ports") or data.get("VLLM_PORTS")
            if isinstance(v, str):
                ports = [int(p.strip()) for p in v.split(",") if p.strip()]
                data["vllm_ports"] = ports
                data.pop("VLLM_PORTS", None)
        return data


def load_settings() -> Settings:
    return Settings()
