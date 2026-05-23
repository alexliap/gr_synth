from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model_id: str = "ilsp/Llama-Krikri-8B-Instruct"

    hf_token: str | None = None
    hf_repo_id: str = "alexliap/greek-synth-v1"

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    repetition_penalty: float = 1.05

    concurrency: int = 128*6
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

    progress_every: int = 1000
    skip_log_every: int = 500


def load_settings() -> Settings:
    return Settings()
