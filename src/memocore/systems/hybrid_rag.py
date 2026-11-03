from __future__ import annotations

import re
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
from memocore.temporal import anchor_conversation, parse_session_date, question_window, turn_spans


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
    # Contextual indexing: each turn is embedded and keyword-indexed together with the turns around it
    # in its session, so a reply ("That was so inspiring!") can be found by what it answers. Only the
    # turn itself is returned and counted against the budget.
    context_before: int = 0
    context_after: int = 0
    context_in_rerank: bool = False  # let the cross-encoder see the same context
    # Time-aware reading: when a question names a period ("in October 2023", "on 3 June, 2023"), the turns
    # said in it or resolved to it at write time join the reranker's candidates (up to `time_candidates`,
    # best dense score first) and get `time_boost` added to their rerank score. Nothing is filtered out.
    time_candidates: int = 0
    time_boost: float = 0.0
    # Speaker prior: when a question names exactly one speaker, that speaker's turns get `speaker_boost`
    # added to their dense similarity, so more of them reach the reranker. In LoCoMo 96% of the evidence
    # for such questions is said by the person named. Nothing is filtered out.
    speaker_boost: float = 0.0


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
        if (options.time_candidates or options.time_boost) and not (options.use_dense and options.rerank_candidates):
            raise ValueError("time_candidates and time_boost need dense retrieval and a reranker")
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

    def ingest(self, conversation: Conversation, *, anchored: bool = False) -> None:
        """Index a conversation. `anchored` means its turns already carry their time notes, as the
        memory store's turns do (resolved once, at write time)."""
        if self.options.time_anchors and not anchored:
            conversation = anchor_conversation(conversation)
        self._pairs, self._texts, self._index_texts, self._rerank_texts, self._spans = [], [], [], [], []
        self._matrix = None
        self._bm25 = BM25([]) if self.options.use_bm25 else None
        self._by_speaker = {}
        self._append([(session, turn) for session in conversation.sessions for turn in session.turns])

    def extended(self, pairs: Sequence[tuple[Session, Turn]]) -> HybridRAGSystem:
        """A copy that also holds turns said after everything indexed so far, indexing only those, so
        a store that keeps growing does not rebuild its whole index after every write. The turns must
        carry their time notes already. This system is left untouched, so a search running on it
        meanwhile is unaffected. Rankings equal those of a system that ingested everything at once."""
        if self.options.context_after:
            raise ValueError("context_after changes the indexed text of earlier turns; ingest everything again")
        last = self._pairs[-1][0] if self._pairs else None
        if pairs and last is not None and pairs[0][0].index < last.index:
            raise ValueError("extended() only appends turns after the last indexed session")
        # One Session object per session index, the last indexed one included: rendering and neighbour
        # lookups tell sessions apart by identity.
        canonical = {last.index: last} if last is not None else {}
        pairs = [(canonical.setdefault(session.index, session), turn) for session, turn in pairs]
        copy = object.__new__(HybridRAGSystem)
        copy.__dict__.update(self.__dict__)
        copy._append(pairs)
        return copy

    def _append(self, pairs: list[tuple[Session, Turn]]) -> None:
        # Every container is rebound, never changed in place: a system made by extended() shares them
        # with the one it came from. Conversation order, so sorting indices restores the timeline.
        start = len(self._pairs)
        self._pairs = self._pairs + pairs
        texts = [session.render([turn]) for session, turn in pairs]
        contextual = self.options.context_before or self.options.context_after
        index_texts = [self._in_context(i) for i in range(start, len(self._pairs))] if contextual else texts
        self._texts = self._texts + texts
        self._index_texts = self._index_texts + index_texts
        self._rerank_texts = self._rerank_texts + (index_texts if self.options.context_in_rerank else texts)
        if self.options.use_dense and pairs:
            vectors = self._embedder.encode(index_texts)
            self._matrix = vectors if self._matrix is None else np.concatenate([self._matrix, vectors])
        if self._bm25 is not None:
            self._bm25 = self._bm25.extended(index_texts)
        dates = {stamp: parse_session_date(stamp) for stamp in {session.date_time for session, _ in pairs}}
        self._spans = self._spans + [turn_spans(dates[session.date_time], turn.time_note) for session, turn in pairs]
        speakers = sorted(set(self._by_speaker) | {turn.speaker for _, turn in pairs})
        self._speaker_patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in speakers if name}
        self._by_speaker = {
            name: np.concatenate([
                self._by_speaker.get(name, np.zeros(start, dtype=np.float32)),
                np.array([turn.speaker == name for _, turn in pairs], dtype=np.float32),
            ])
            for name in speakers
        }

    def _speaker_prior(self, question: str) -> np.ndarray | None:
        if not self.options.speaker_boost:
            return None
        named = [name for name, pattern in self._speaker_patterns.items() if pattern.search(question)]
        return self.options.speaker_boost * self._by_speaker[named[0]] if len(named) == 1 else None

    def _in_window(self, question: str) -> set[int]:
        if not (self.options.time_candidates or self.options.time_boost):
            return set()
        window = question_window(question)
        if window is None:
            return set()
        return {i for i, spans in enumerate(self._spans) if any(window.overlaps(start, end) for start, end in spans)}

    def _in_context(self, index: int) -> str:
        session = self._pairs[index][0]
        window = range(index - self.options.context_before, index + self.options.context_after + 1)
        turns = [self._pairs[i][1] for i in window if 0 <= i < len(self._pairs) and self._pairs[i][0] is session]
        return session.render(turns)

    def rank(self, questions: Sequence[str]) -> list[list[int]]:
        """Turn indices for each question, best first."""
        if not self._pairs:
            return [[] for _ in questions]
        vectors = self._embedder.encode(list(questions)) if self._matrix is not None else None
        dense = vectors @ self._matrix.T if vectors is not None else None
        ranked_all = []
        for row, question in enumerate(questions):
            prior = self._speaker_prior(question) if dense is not None else None
            if prior is not None:
                dense[row] = dense[row] + prior
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
            in_window = self._in_window(question)
            if in_window and self.options.time_candidates and dense is not None:
                known = set(head)
                dated = sorted((i for i in in_window if i not in known), key=lambda i: -dense[row][i])
                head = head + dated[: self.options.time_candidates]
            if self._reranker is not None and self.options.rerank_candidates:
                scores = self._reranker.score(question, [self._rerank_texts[i] for i in head])
                if self.options.feedback_turns:
                    head, scores = self._feedback(question, vectors[row], head, scores)
                if in_window and self.options.time_boost:
                    scores = scores + self.options.time_boost * np.array([i in in_window for i in head], dtype=np.float32)
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
        return head + new, np.concatenate([scores, self._reranker.score(question, [self._rerank_texts[i] for i in new])])

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

    def turns(self, indices: Sequence[int]) -> list[tuple[Session, Turn]]:
        return [self._pairs[i] for i in indices]

    def render(self, indices: Sequence[int]) -> RetrievedContext:
        return render_turns([self._pairs[i] for i in sorted(indices)], self.cite)

    def retrieve(self, question: str) -> RetrievedContext:
        return self.render(self.select(self.rank([question])[0], self.top_k))
