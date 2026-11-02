from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from memocore.kvcache import BlobCache
from memocore.retrieval.bm25 import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """bge-reranker style cross-encoder, half precision on GPU, with a disk cache of pair scores."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        cache_path: Path | None = None,
    ):
        # Load torch now, before anything can import scikit-learn (see metrics._stemmer).
        import torch  # noqa: F401

        self.name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._cache = BlobCache(cache_path, "rerank_scores") if cache_path else None
        self._model = None
        # Model time and pair count, so throughput can be reported without the cache's help.
        self.model_seconds = 0.0
        self.pairs_scored = 0

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else None
            self._model = CrossEncoder(self.name, device=device, max_length=self._max_length, model_kwargs=kwargs)
        return self._model

    def _key(self, query: str, passage: str) -> str:
        return hashlib.sha256(f"{self.name}\0{query}\0{passage}".encode("utf-8")).hexdigest()

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        keys = [self._key(query, passage) for passage in passages]
        found = (
            {key: float(np.frombuffer(blob, dtype=np.float32)[0]) for key, blob in self._cache.get_many(keys).items()}
            if self._cache
            else {}
        )
        missing = [i for i, key in enumerate(keys) if key not in found]
        if missing:
            model = self._load()
            start = time.perf_counter()
            fresh = model.predict(
                [(query, passages[i]) for i in missing],
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self.model_seconds += time.perf_counter() - start
            self.pairs_scored += len(missing)
            fresh = np.asarray(fresh, dtype=np.float32).reshape(-1)
            if self._cache:
                self._cache.put_many((keys[i], np.float32(value).tobytes()) for i, value in zip(missing, fresh))
            found.update((keys[i], float(value)) for i, value in zip(missing, fresh))
        return np.array([found[key] for key in keys], dtype=np.float32)


class LexicalReranker:
    """Offline stand-in for tests and --mock runs: scores by shared words, no model."""

    name = "lexical"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        words = set(tokenize(query))
        return np.array([len(words & set(tokenize(passage))) for passage in passages], dtype=np.float32)
