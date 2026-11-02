from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from memocore.config import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_locomo.json"


def temp_settings(test_case) -> Settings:
    """Settings rooted in a throwaway directory holding the tiny LoCoMo fixture."""
    root = Path(tempfile.mkdtemp(prefix="memocore-test-"))
    test_case.addCleanup(shutil.rmtree, root, True)
    data_dir = root / "data"
    (data_dir / "locomo").mkdir(parents=True)
    shutil.copy(FIXTURE, data_dir / "locomo" / "locomo10.json")
    return Settings(
        api_key=None,
        base_url="http://unused.invalid",
        answer_model="unused",
        judge_model="unused",
        embedding_model="unused",
        embedding_device=None,
        data_dir=data_dir,
        runs_dir=root / "runs",
        cache_path=root / "cache.sqlite",
        embedding_cache_path=root / "embeddings.sqlite",
    )
