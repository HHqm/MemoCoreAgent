"""Text embedders. Vectors are L2-normalised, so a dot product is a cosine similarity."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, device: str | None = None, batch_size: int = 32):
        # Load torch now, before anything can import scikit-learn (see metrics._stemmer).
        import torch  # noqa: F401

        self.name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.name, device=self._device)
            if model.device.type == "cuda":
                # Half precision halves GPU memory and roughly doubles throughput; rankings barely move.
                model.half()
            self._model = model
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class CachedEmbedder:
    """Content-addressed vector cache on disk: re-running an evaluation never re-encodes the same text."""

    def __init__(self, inner: Embedder, path: Path):
        self.inner = inner
        self.name = inner.name
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self._conn.commit()
        self.hits = self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}\0{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(text) for text in texts]
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's bound-parameter limit
            batch = keys[start : start + 500]
            rows = self._conn.execute(
                f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(batch))})", batch
            ).fetchall()
            found.update((key, np.frombuffer(blob, dtype=np.float32)) for key, blob in rows)
        missing = [i for i, key in enumerate(keys) if key not in found]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = self.inner.encode([texts[i] for i in missing])
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)",
                [(keys[i], vector.astype(np.float32).tobytes()) for i, vector in zip(missing, fresh)],
            )
            self._conn.commit()
            found.update((keys[i], vector.astype(np.float32)) for i, vector in zip(missing, fresh))
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[key] for key in keys])


class HashingEmbedder:
    """Deterministic bag-of-words vectors for offline tests; it has no notion of meaning."""

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                out[row, int.from_bytes(digest, "little") % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
