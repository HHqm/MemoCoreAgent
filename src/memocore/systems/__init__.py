from __future__ import annotations

from memocore.embeddings import Embedder
from memocore.systems.base import MemorySystem, RetrievedContext
from memocore.systems.full_context import FullContextSystem
from memocore.systems.naive_rag import NaiveRAGSystem

SYSTEM_NAMES = ("full_context", "naive_rag")
NEEDS_EMBEDDER = frozenset({"naive_rag"})


def build_system(name: str, *, embedder: Embedder | None = None, top_k: int = 10, chunk_turns: int = 1) -> MemorySystem:
    if name == "full_context":
        return FullContextSystem()
    if name == "naive_rag":
        if embedder is None:
            raise ValueError("naive_rag needs an embedder")
        return NaiveRAGSystem(embedder, top_k=top_k, chunk_turns=chunk_turns)
    raise ValueError(f"unknown system {name!r}; choose from {', '.join(SYSTEM_NAMES)}")


__all__ = [
    "MemorySystem",
    "RetrievedContext",
    "FullContextSystem",
    "NaiveRAGSystem",
    "SYSTEM_NAMES",
    "NEEDS_EMBEDDER",
    "build_system",
]
