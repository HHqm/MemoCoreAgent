from __future__ import annotations

import hashlib
import queue
import threading
import time
from concurrent.futures import Future
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
        return self.score_many([(query, passages)])[0]

    def score_many(self, requests: Sequence[tuple[str, Sequence[str]]]) -> list[np.ndarray]:
        """Score several (query, passages) requests with a single model call for every uncached pair."""
        keys = [[self._key(query, passage) for passage in passages] for query, passages in requests]
        flat = [key for request_keys in keys for key in request_keys]
        found = (
            {key: float(np.frombuffer(blob, dtype=np.float32)[0]) for key, blob in self._cache.get_many(flat).items()}
            if self._cache
            else {}
        )
        pairs = {}  # key -> (query, passage), once per distinct pair
        for (query, passages), request_keys in zip(requests, keys):
            for passage, key in zip(passages, request_keys):
                if key not in found:
                    pairs.setdefault(key, (query, passage))
        if pairs:
            model = self._load()
            start = time.perf_counter()
            fresh = model.predict(
                list(pairs.values()), batch_size=self._batch_size, show_progress_bar=False, convert_to_numpy=True
            )
            self.model_seconds += time.perf_counter() - start
            self.pairs_scored += len(pairs)
            fresh = np.asarray(fresh, dtype=np.float32).reshape(-1)
            if self._cache:
                self._cache.put_many((key, np.float32(value).tobytes()) for key, value in zip(pairs, fresh))
            found.update((key, float(value)) for key, value in zip(pairs, fresh))
        return [np.array([found[key] for key in request_keys], dtype=np.float32) for request_keys in keys]


class BatchingReranker:
    """Dynamic batching for a served store: score() calls that arrive within `window_ms` of each other,
    from concurrent searches, go to the GPU as one model call. One request alone waits at most the window.

    Merging eight searches' pairs into one call measured about 28% less GPU time on a laptop RTX 5060.
    """

    def __init__(self, inner: CrossEncoderReranker, *, window_ms: float = 3.0, max_pairs: int = 1024):
        self.inner = inner
        self.name = inner.name
        self._window = window_ms / 1000
        self._max_pairs = max_pairs
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._in_flight = 0  # requests submitted and not yet answered
        self.calls = self.requests = 0  # model calls made, requests served: how much merging happened
        threading.Thread(target=self._serve, name="rerank-batcher", daemon=True).start()

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        done: Future = Future()
        with self._lock:
            self._in_flight += 1
        try:
            self._queue.put((query, list(passages), done))
            return done.result()
        finally:
            with self._lock:
                self._in_flight -= 1

    def _serve(self) -> None:
        while True:
            batch = [self._queue.get()]
            pairs = len(batch[0][1])
            # Wait for company only when another search is in flight; a lone request goes straight to the GPU.
            with self._lock:
                alone = self._in_flight <= 1
            deadline = time.perf_counter() + (0.0 if alone else self._window)
            while pairs < self._max_pairs:
                try:
                    item = self._queue.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                batch.append(item)
                pairs += len(item[1])
            try:
                results = self.inner.score_many([(query, passages) for query, passages, _ in batch])
            except Exception as error:  # every waiting search gets the error rather than hanging
                for *_, done in batch:
                    done.set_exception(error)
                continue
            self.calls += 1
            self.requests += len(batch)
            for (*_, done), scores in zip(batch, results):
                done.set_result(scores)


class LexicalReranker:
    """Offline stand-in for tests and --mock runs: scores by shared words, no model."""

    name = "lexical"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        words = set(tokenize(query))
        return np.array([len(words & set(tokenize(passage))) for passage in passages], dtype=np.float32)
