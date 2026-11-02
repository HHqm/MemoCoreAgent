from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from memocore.datasets.locomo import Conversation, Session, Turn
from memocore.embeddings import Embedder
from memocore.retrieval.bm25 import BM25
from memocore.retrieval.fusion import rrf, top_indices
from memocore.retrieval.rerank import Reranker
from memocore.systems.base import MemorySystem, RetrievedContext, render_turns
from memocore.temporal import anchor_conversation


@dataclass(frozen=True)
class HybridOptions:
    use_dense: bool = True
    use_bm25: bool = True
    # "rrf" fuses the dense and BM25 rankings; "pool" keeps the dense ranking and only lets BM25 add
    # its top `bm25_pool` hits to the reranker's candidates, so lexical matches cannot push out
    # semantic ones (RRF fusion lost recall on multi-hop and open-domain questions).
    fusion: str = "rrf"
    bm25_pool: int = 10
    rerank_candidates: int = 0  # how many candidates the cross-encoder re-scores; 0 disables it
    neighbors: int = 0  # turns added on each side of a hit, within the same session
    max_per_session: int | None = None  # cap on hits per session, to spread a budget across sessions
    time_anchors: bool = False  # resolve relative times at write time (memocore.temporal)
    pool: int = 100  # depth of each first-stage ranking
    # Pseudo-relevance feedback: a second dense hop, from the question plus its top `feedback_turns`
    # reranked hits, adds up to `feedback_candidates` new turns for the reranker. It reaches evidence
    # that looks nothing like the question itself, such as the second hop of a multi-hop question.
    feedback_turns: int = 0
    feedback_weight: float = 0.5
    feedback_candidates: int = 20


class HybridRAGSystem(MemorySystem):
    """The memory engine's read path, applied to raw turns.

    Dense and BM25 recall are fused with RRF, a cross-encoder re-scores the head of the list, and a
    turn budget is spent on the result. Every stage can be switched off, which is how
    `memocore retrieval-ablation` measures what each one contributes at the same budget.
    """

    name = "hybrid_rag"

    def __init__(
        self,
        embedder: Embedder | None,
        reranker: Reranker | None = None,
        *,
        top_k: int = 20,
        options: HybridOptions = HybridOptions(),
    ):
        if not (options.use_dense or options.use_bm25):
            raise ValueError("enable dense retrieval, BM25 or both")
        if options.use_dense and embedder is None:
            raise ValueError("dense retrieval needs an embedder")
        if options.rerank_candidates and reranker is None:
            raise ValueError("rerank_candidates > 0 needs a reranker")
        if options.fusion not in ("rrf", "pool"):
            raise ValueError(f"unknown fusion {options.fusion!r}; use 'rrf' or 'pool'")
        if options.fusion == "pool" and not (options.use_dense and options.use_bm25 and options.rerank_candidates):
            raise ValueError("fusion='pool' needs dense retrieval, BM25 and a reranker")
        if options.feedback_turns and not (options.use_dense and options.rerank_candidates):
            raise ValueError("feedback_turns > 0 needs dense retrieval and a reranker")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k
        self.options = options
        self._embedder = embedder
        self._reranker = reranker
        self._pairs: list[tuple[Session, Turn]] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None
        self._bm25: BM25 | None = None

    def ingest(self, conversation: Conversation) -> None:
        if self.options.time_anchors:
            conversation = anchor_conversation(conversation)
        # Conversation order, so sorting indices restores the timeline.
        self._pairs = [(session, turn) for session in conversation.sessions for turn in session.turns]
        self._texts = [session.render([turn]) for session, turn in self._pairs]
        self._matrix = self._embedder.encode(self._texts) if self.options.use_dense and self._texts else None
        self._bm25 = BM25(self._texts) if self.options.use_bm25 else None

    def rank(self, questions: Sequence[str]) -> list[list[int]]:
        """Turn indices for each question, best first."""
        if not self._pairs:
            return [[] for _ in questions]
        vectors = self._embedder.encode(list(questions)) if self._matrix is not None else None
        dense = vectors @ self._matrix.T if vectors is not None else None
        ranked_all = []
        for row, question in enumerate(questions):
            dense_ranking = top_indices(dense[row], self.options.pool) if dense is not None else None
            bm25_ranking = (
                top_indices(self._bm25.scores(question), self.options.pool, positive_only=True)
                if self._bm25 is not None
                else None
            )
            if self.options.fusion == "pool":
                head = dense_ranking[: self.options.rerank_candidates]
                in_head = set(head)
                head += [i for i in bm25_ranking[: self.options.bm25_pool] if i not in in_head]
                in_head.update(head)
                ranked = head + [i for i in dense_ranking if i not in in_head]
            else:
                rankings = [r for r in (dense_ranking, bm25_ranking) if r is not None]
                ranked = rrf(rankings) if len(rankings) > 1 else rankings[0]
                head = ranked[: self.options.rerank_candidates]
            if self._reranker is not None and self.options.rerank_candidates:
                scores = self._reranker.score(question, [self._texts[i] for i in head])
                if self.options.feedback_turns:
                    head, scores = self._feedback(question, vectors[row], head, scores)
                # The un-reranked tail stays behind the head, so a capped budget can still be filled.
                in_head = set(head)
                ranked = [head[i] for i in np.argsort(-scores, kind="stable")] + [i for i in ranked if i not in in_head]
            ranked_all.append(ranked)
        return ranked_all

    def _feedback(
        self, question: str, vector: np.ndarray, head: list[int], scores: np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        top = [head[i] for i in np.argsort(-scores, kind="stable")[: self.options.feedback_turns]]
        probe = vector + self.options.feedback_weight * self._matrix[top].mean(axis=0)
        known = set(head)
        new = [i for i in top_indices(self._matrix @ probe, self.options.pool) if i not in known]
        new = new[: self.options.feedback_candidates]
        if not new:
            return head, scores
        # Scored against the original question, so the second hop only adds candidates, never bias.
        return head + new, np.concatenate([scores, self._reranker.score(question, [self._texts[i] for i in new])])

    def select(self, ranked: Sequence[int], budget: int) -> list[int]:
        """Spend a budget of turns on a ranking: each hit brings its neighbours; sessions can be capped."""
        chosen: list[int] = []
        seen: set[int] = set()
        hits_per_session: Counter[int] = Counter()
        for hit in ranked:
            if len(chosen) >= budget:
                break
            if hit in seen:
                continue
            session = self._pairs[hit][0].index
            if self.options.max_per_session and hits_per_session[session] >= self.options.max_per_session:
                continue
            hits_per_session[session] += 1
            for index in self._with_neighbours(hit):
                if len(chosen) >= budget:
                    break
                if index not in seen:
                    seen.add(index)
                    chosen.append(index)
        return chosen

    def _with_neighbours(self, hit: int) -> Iterator[int]:
        yield hit
        session = self._pairs[hit][0]
        for offset in range(1, self.options.neighbors + 1):
            for index in (hit - offset, hit + offset):
                if 0 <= index < len(self._pairs) and self._pairs[index][0] is session:
                    yield index

    def turn_ids(self, indices: Sequence[int]) -> tuple[str, ...]:
        return tuple(self._pairs[i][1].dia_id for i in indices)

    def render(self, indices: Sequence[int]) -> RetrievedContext:
        return render_turns([self._pairs[i] for i in sorted(indices)], self.cite)

    def retrieve(self, question: str) -> RetrievedContext:
        return self.render(self.select(self.rank([question])[0], self.top_k))
