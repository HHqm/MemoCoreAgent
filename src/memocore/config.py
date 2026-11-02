"""Runtime settings, read from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _env_or(name: str, default: str) -> str:
    return _env(name) or default


def _local_or_hub(root: Path, folder: str, hub_id: str) -> str:
    # Prefer a copy placed in models/ (Windows without symlink rights cannot use the HF cache).
    local = root / "models" / folder
    return str(local) if (local / "config.json").exists() else hub_id


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    answer_model: str
    judge_model: str
    embedding_model: str
    embedding_device: str | None
    reranker_model: str
    data_dir: Path
    runs_dir: Path
    cache_path: Path
    embedding_cache_path: Path
    rerank_cache_path: Path

    @classmethod
    def from_env(cls, root: Path = PROJECT_ROOT) -> "Settings":
        load_dotenv(root / ".env", override=False)
        answer_model = _env_or("MEMOCORE_ANSWER_MODEL", "deepseek-chat")
        return cls(
            api_key=_env("MEMOCORE_API_KEY") or _env("DEEPSEEK_API_KEY"),
            base_url=_env_or("MEMOCORE_BASE_URL", "https://api.deepseek.com"),
            answer_model=answer_model,
            judge_model=_env_or("MEMOCORE_JUDGE_MODEL", answer_model),
            embedding_model=_env_or("MEMOCORE_EMBEDDING_MODEL", _local_or_hub(root, "bge-m3", "BAAI/bge-m3")),
            embedding_device=_env("MEMOCORE_EMBEDDING_DEVICE"),
            reranker_model=_env_or(
                "MEMOCORE_RERANKER_MODEL", _local_or_hub(root, "bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3")
            ),
            data_dir=Path(_env_or("MEMOCORE_DATA_DIR", str(root / "data"))),
            runs_dir=Path(_env_or("MEMOCORE_RUNS_DIR", str(root / "runs"))),
            cache_path=Path(_env_or("MEMOCORE_CACHE_PATH", str(root / ".cache" / "llm_cache.sqlite"))),
            embedding_cache_path=Path(
                _env_or("MEMOCORE_EMBEDDING_CACHE_PATH", str(root / ".cache" / "embeddings.sqlite"))
            ),
            rerank_cache_path=Path(_env_or("MEMOCORE_RERANK_CACHE_PATH", str(root / ".cache" / "rerank.sqlite"))),
        )

    @property
    def locomo_path(self) -> Path:
        return self.data_dir / "locomo" / "locomo10.json"
