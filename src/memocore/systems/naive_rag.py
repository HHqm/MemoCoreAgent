from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memocore.datasets.locomo import Conversation, Session, Turn
from memocore.embeddings import Embedder
from memocore.systems.base import MemorySystem, RetrievedContext, render_turns


@dataclass(frozen=True)
class _Chunk:
    session: Session
    turns: tuple[Turn, ...]

    def render(self) -> str:
        # The session date stays attached so relative times ("yesterday") remain resolvable.
        return self.session.render(self.turns)


class NaiveRAGSystem(MemorySystem):
    """Dense retrieval over fixed windows of raw turns: the baseline a memory engine has to beat."""

    name = "naive_rag"

    def __init__(self, embedder: Embedder, *, top_k: int = 10, chunk_turns: int = 1):
        if top_k < 1 or chunk_turns < 1:
            raise ValueError("top_k and chunk_turns must be at least 1")
        self.top_k = top_k
        self.chunk_turns = chunk_turns
        self._embedder = embedder
        self._chunks: list[_Chunk] = []
        self._matrix: np.ndarray | None = None

    def ingest(self, conversation: Conversation) -> None:
        chunks = []
        for session in conversation.sessions:
            # Windows never cross a session boundary, so every chunk has exactly one date.
            for start in range(0, len(session.turns), self.chunk_turns):
                chunks.append(_Chunk(session, session.turns[start : start + self.chunk_turns]))
        self._chunks = chunks
        self._matrix = self._embedder.encode([chunk.render() for chunk in chunks]) if chunks else None

    def ranked_chunks(self, questions: list[str], k: int) -> list[list[tuple[str, ...]]]:
        """For each question, the turn ids of its k best chunks, best first. Encodes all questions in one batch."""
        if self._matrix is None:
            return [[] for _ in questions]
        scores = self._embedder.encode(questions) @ self._matrix.T
        k = min(k, len(self._chunks))
        ranked = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            best_first = sorted((int(i) for i in top), key=lambda i: (-row[i], i))
            ranked.append([tuple(turn.dia_id for turn in self._chunks[i].turns) for i in best_first])
        return ranked

    def retrieve(self, question: str) -> RetrievedContext:
        if self._matrix is None:
            return RetrievedContext("", ())
        scores = self._matrix @ self._embedder.encode([question])[0]
        k = min(self.top_k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        # Present hits in conversation order rather than score order, so the timeline reads naturally.
        ordered = sorted(int(i) for i in top)
        return render_turns([(self._chunks[i].session, turn) for i in ordered for turn in self._chunks[i].turns])
