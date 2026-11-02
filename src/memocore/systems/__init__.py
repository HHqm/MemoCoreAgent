from __future__ import annotations

from memocore.embeddings import Embedder
from memocore.retrieval.rerank import Reranker
from memocore.systems.base import MemorySystem, NoMemorySystem, RetrievedContext, render_turns
from memocore.systems.full_context import FullContextSystem
from memocore.systems.hybrid_rag import HybridOptions, HybridRAGSystem
from memocore.systems.naive_rag import NaiveRAGSystem

SYSTEM_NAMES = ("no_memory", "full_context", "naive_rag", "hybrid_rag")
NEEDS_EMBEDDER = frozenset({"naive_rag", "hybrid_rag"})


def build_system(
    name: str,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    top_k: int = 20,
    chunk_turns: int = 1,
    hybrid: HybridOptions | None = None,
    cite: bool = False,
) -> MemorySystem:
    system: MemorySystem
    if name == "no_memory":
        system = NoMemorySystem()
    elif name == "full_context":
        system = FullContextSystem()
    elif name == "naive_rag":
        if embedder is None:
            raise ValueError("naive_rag needs an embedder")
        system = NaiveRAGSystem(embedder, top_k=top_k, chunk_turns=chunk_turns)
    elif name == "hybrid_rag":
        system = HybridRAGSystem(embedder, reranker, top_k=top_k, options=hybrid or HybridOptions())
    else:
        raise ValueError(f"unknown system {name!r}; choose from {', '.join(SYSTEM_NAMES)}")
    system.cite = cite
    return system


__all__ = [
    "MemorySystem",
    "RetrievedContext",
    "render_turns",
    "NoMemorySystem",
    "FullContextSystem",
    "NaiveRAGSystem",
    "HybridOptions",
    "HybridRAGSystem",
    "SYSTEM_NAMES",
    "NEEDS_EMBEDDER",
    "build_system",
]
