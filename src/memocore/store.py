"""Persistent long-term memory for an agent, read through the same path the benchmarks measure.

Each turn is stored once, verbatim, in SQLite, beside what is derived from it at write time: its
resolved-time note and its vector. Nothing is rewritten or summarised, so every recalled line can be
traced to the words that were said, and forgetting a turn deletes the text and everything derived
from it in one row. Reads build a HybridRAGSystem over the stored turns, so the evidence recall that
`memocore retrieval-ablation` measures is the recall this store delivers (`memocore memory-replay`
checks that, question by question).
"""

from __future__ import annotations

import calendar
import hashlib
import re
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from memocore.datasets.locomo import Conversation, Session, Turn
from memocore.embeddings import Embedder
from memocore.retrieval.rerank import Reranker
from memocore.systems.hybrid_rag import HybridOptions, HybridRAGSystem
from memocore.temporal import time_note

# The evaluated default read path (docs/results/retrieval_ablation.md, docs/results/optimization_rounds.md);
# RunConfig's defaults must match it, which tests/test_store.py checks.
DEFAULT_OPTIONS = HybridOptions(
    fusion="pool", rerank_candidates=50, time_anchors=True, feedback_turns=2, time_candidates=20, speaker_boost=0.1
)
DEFAULT_BUDGET = 20
# Without an explicit session key, a message more than this long after the previous one opens a new session.
SESSION_GAP = timedelta(minutes=30)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user TEXT NOT NULL,
    number INTEGER NOT NULL,
    key TEXT,
    started_at TEXT NOT NULL,
    last_at TEXT NOT NULL,
    turns_added INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user, number)
);
CREATE TABLE IF NOT EXISTS turns (
    user TEXT NOT NULL,
    session INTEGER NOT NULL,
    position INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    image TEXT,
    time_note TEXT,
    said_at TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    vector BLOB NOT NULL,
    vector_key TEXT,  -- hash of the exact text the vector was computed from
    PRIMARY KEY (user, session, position)
);
CREATE TABLE IF NOT EXISTS revisions (
    user TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,  -- bumped by every write
    forgets INTEGER NOT NULL DEFAULT 0  -- bumped by deletions only; between two of them, writes only append
);
"""
_TURN_ID = re.compile(r"^D(\d+):(\d+)$")
_SESSION_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.IGNORECASE)
_MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}


def format_session_time(moment: datetime) -> str:
    """The stamp LoCoMo puts on a session, such as "1:56 pm on 8 May, 2023"."""
    hour = moment.hour % 12 or 12
    meridiem = "am" if moment.hour < 12 else "pm"
    return f"{hour}:{moment.minute:02d} {meridiem} on {moment.day} {calendar.month_name[moment.month]}, {moment.year}"


def parse_session_time(text: str) -> datetime:
    match = _SESSION_TIME.search(text)
    if not match:
        raise ValueError(f"not a session time: {text!r}")
    hour, minute, meridiem, day, month, year = match.groups()
    hour = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
    return datetime(int(year), _MONTHS[month.lower()], int(day), hour, int(minute))


@dataclass(frozen=True)
class Utterance:
    """One message to remember."""

    text: str
    speaker: str = "user"
    at: datetime | None = None  # when it was said; now if omitted
    session: str | None = None  # caller's conversation key; omitted, sessions split on SESSION_GAP
    image: str | None = None  # description of an image shared with the message


@dataclass(frozen=True)
class Memory:
    id: str  # "D<session>:<position>", the id the answer prompt cites
    speaker: str
    text: str
    session_time: str
    image: str | None = None
    time_note: str | None = None


@dataclass(frozen=True)
class Recall:
    memories: tuple[Memory, ...]  # in conversation order
    context: str  # the memories rendered with ids and dates, in the format the answer prompt expects


def _vector_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _StoredVectors:
    """Serves the vectors saved at write time, so a read never re-encodes stored turns.

    Vectors are found by a hash of the exact text they were computed from. When that text no longer
    matches, as when a forgotten turn was part of the next turn's indexed context, the turn is encoded
    afresh instead of being served a stale vector.
    """

    def __init__(self, inner: Embedder, vectors: dict[str, np.ndarray]):
        self.inner = inner
        self.name = inner.name
        self._vectors = vectors

    def add(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors.update(vectors)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [_vector_key(text) for text in texts]
        missing = {key: text for key, text in zip(keys, texts) if key not in self._vectors}
        fresh = dict(zip(missing, self.inner.encode(list(missing.values())))) if missing else {}
        return np.stack([self._vectors.get(key, fresh.get(key)) for key in keys])


class _Index(NamedTuple):
    revision: int
    forgets: int
    last: tuple[int, int]  # (session, position) of the newest turn indexed
    system: HybridRAGSystem
    vectors: _StoredVectors


class MemoryStore:
    def __init__(
        self,
        path: Path | str,
        embedder: Embedder,
        reranker: Reranker | None,
        *,
        options: HybridOptions = DEFAULT_OPTIONS,
        budget: int = DEFAULT_BUDGET,
    ):
        if options.context_after:
            # Each vector is computed once, when its turn is written; a later turn cannot change it.
            raise ValueError("MemoryStore indexes turns as they arrive, so context_after must be 0")
        self.embedder = embedder
        self.reranker = reranker
        self.options = options
        self.budget = budget
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        # Deleted rows are overwritten on disk, not just unlinked, so forgotten text cannot be recovered from the file.
        self._db.execute("PRAGMA secure_delete = ON")
        self._db.executescript(_SCHEMA)
        if "vector_key" not in {row[1] for row in self._db.execute("PRAGMA table_info(turns)")}:
            self._db.execute("ALTER TABLE turns ADD COLUMN vector_key TEXT")  # stores created before the column
        if "forgets" not in {row[1] for row in self._db.execute("PRAGMA table_info(revisions)")}:
            self._db.execute("ALTER TABLE revisions ADD COLUMN forgets INTEGER NOT NULL DEFAULT 0")
        self._lock = threading.Lock()
        self._indexes: dict[str, _Index] = {}

    def close(self) -> None:
        self._db.close()

    # --- writing -------------------------------------------------------------------------------

    def add(self, user: str, text: str, **fields) -> Memory:
        return self.add_many(user, [Utterance(text, **fields)])[0]

    def add_many(self, user: str, utterances: Iterable[Utterance]) -> list[Memory]:
        """Store messages verbatim, with their time notes and vectors, in one transaction."""
        utterances = list(utterances)
        with self._lock, self._db:
            staged = []
            recent: dict[int, list[Turn]] = {}  # per session, the turns before the one being staged
            for utterance in utterances:
                if not utterance.text.strip() and not utterance.image:
                    raise ValueError("cannot remember an empty message")
                at = utterance.at or datetime.now().replace(microsecond=0)
                number, started_at = self._session_for(user, utterance.session, at)
                position = self._db.execute(
                    "UPDATE sessions SET turns_added = turns_added + 1, last_at = ? WHERE user = ? AND number = ? "
                    "RETURNING turns_added",
                    (at.isoformat(), user, number),
                ).fetchone()[0]
                stamp = format_session_time(started_at)
                # Times are resolved against the session date, as in the evaluated write path.
                turn = Turn(f"D{number}:{position}", utterance.speaker, utterance.text, utterance.image)
                turn = replace(turn, time_note=time_note(turn.text, started_at.date()))
                if number not in recent:
                    recent[number] = self._last_turns(user, number)
                # The same text HybridRAGSystem indexes: the turn with its `context_before` predecessors.
                window = recent[number][-self.options.context_before:] if self.options.context_before else []
                indexed = Session(number, stamp, ()).render([*window, turn])
                recent[number].append(turn)
                staged.append((number, position, at, stamp, turn, indexed))
            vectors = self.embedder.encode([indexed for *_, indexed in staged])
            self._db.executemany(
                "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (user, number, position, turn.speaker, turn.text, turn.image_caption, turn.time_note,
                     at.isoformat(), self.embedder.name, np.asarray(vector, dtype=np.float32).tobytes(),
                     _vector_key(indexed))
                    for (number, position, at, _, turn, indexed), vector in zip(staged, vectors)
                ],
            )
            self._bump(user)
        return [Memory(turn.dia_id, turn.speaker, turn.text, stamp, turn.image_caption, turn.time_note)
                for _, _, _, stamp, turn, _ in staged]

    def _last_turns(self, user: str, session: int) -> list[Turn]:
        if not self.options.context_before:
            return []
        rows = self._db.execute(
            "SELECT position, speaker, text, image, time_note FROM turns WHERE user = ? AND session = ? "
            "ORDER BY position DESC LIMIT ?",
            (user, session, self.options.context_before),
        ).fetchall()
        return [Turn(f"D{session}:{position}", speaker, text, image, note) for position, speaker, text, image, note in reversed(rows)]

    def _session_for(self, user: str, key: str | None, at: datetime) -> tuple[int, datetime]:
        last = self._db.execute(
            "SELECT number, key, started_at, last_at FROM sessions WHERE user = ? ORDER BY number DESC LIMIT 1", (user,)
        ).fetchone()
        if last is not None:
            number, last_key, started_at, last_at = last
            same = last_key == key if key is not None else (
                last_key is None and at - datetime.fromisoformat(last_at) <= SESSION_GAP
            )
            if same:
                return number, datetime.fromisoformat(started_at)
        number = last[0] + 1 if last is not None else 1
        self._db.execute(
            "INSERT INTO sessions (user, number, key, started_at, last_at) VALUES (?, ?, ?, ?, ?)",
            (user, number, key, at.isoformat(), at.isoformat()),
        )
        return number, at

    def _bump(self, user: str, *, forgot: bool = False) -> None:
        self._db.execute(
            "INSERT INTO revisions VALUES (?, 1, ?) ON CONFLICT(user) DO UPDATE SET revision = revision + 1, "
            "forgets = forgets + excluded.forgets",
            (user, int(forgot)),
        )

    def forget(self, user: str, ids: Sequence[str] | None = None) -> int:
        """Delete the given turns, or everything stored for the user when ids is None. Returns turns deleted."""
        with self._lock, self._db:
            if ids is None:
                deleted = self._db.execute("DELETE FROM turns WHERE user = ?", (user,)).rowcount
                self._db.execute("DELETE FROM sessions WHERE user = ?", (user,))
            else:
                keys = [_parse_id(turn_id) for turn_id in ids]
                deleted = sum(
                    self._db.execute(
                        "DELETE FROM turns WHERE user = ? AND session = ? AND position = ?", (user, *key)
                    ).rowcount
                    for key in keys
                )
            self._bump(user, forgot=True)
            self._indexes.pop(user, None)
        return deleted

    # --- reading -------------------------------------------------------------------------------

    def search(self, user: str, query: str, *, budget: int | None = None) -> Recall:
        """The turns the evaluated read path picks for this query, within a budget of turns."""
        with self._lock:
            system = self._system(user)
        if system is None:
            return Recall((), "")
        chosen = system.select(system.rank([query])[0], budget or self.budget)
        context = system.render(chosen)
        return Recall(tuple(_memory(*pair) for pair in system.turns(sorted(chosen))), context.text)

    def get(self, user: str, ids: Sequence[str]) -> list[Memory]:
        """The stored turns behind cited ids, in the order asked; unknown ids are skipped."""
        with self._lock:
            found = []
            for turn_id in ids:
                session, position = _parse_id(turn_id)
                row = self._db.execute(
                    "SELECT t.speaker, t.text, t.image, t.time_note, s.started_at FROM turns t JOIN sessions s "
                    "ON s.user = t.user AND s.number = t.session WHERE t.user = ? AND t.session = ? AND t.position = ?",
                    (user, session, position),
                ).fetchone()
                if row:
                    speaker, text, image, note, started_at = row
                    found.append(Memory(turn_id, speaker, text, format_session_time(datetime.fromisoformat(started_at)),
                                        image, note))
            return found

    def count(self, user: str) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM turns WHERE user = ?", (user,)).fetchone()[0]

    def _system(self, user: str) -> HybridRAGSystem | None:
        row = self._db.execute("SELECT revision, forgets FROM revisions WHERE user = ?", (user,)).fetchone()
        revision, forgets = row if row else (0, 0)
        cached = self._indexes.get(user)
        if cached and cached.revision == revision:
            return cached.system
        if cached and cached.forgets == forgets:
            # Nothing was deleted since the index was built, so the writes since then only appended
            # turns: index just those. A search still running on the old index is unaffected.
            sessions, vectors, last = self._load(user, after=cached.last)
            cached.vectors.add(vectors)
            pairs = [(session, turn) for session in sessions for turn in session.turns]
            system = cached.system.extended(pairs) if pairs else cached.system
            self._indexes[user] = _Index(revision, forgets, last or cached.last, system, cached.vectors)
            return system
        sessions, vectors, last = self._load(user)
        if not sessions:
            self._indexes.pop(user, None)
            return None
        # A vector saved under another embedding model is re-encoded rather than mixed in.
        stored = _StoredVectors(self.embedder, vectors)
        system = HybridRAGSystem(stored, self.reranker, options=self.options)
        system.cite = True
        system.ingest(Conversation(user, "", "", tuple(sessions)), anchored=True)
        self._indexes[user] = _Index(revision, forgets, last, system, stored)
        return system

    def _load(
        self, user: str, after: tuple[int, int] = (0, 0)
    ) -> tuple[list[Session], dict[str, np.ndarray], tuple[int, int] | None]:
        """The user's turns after (session, position) `after`, grouped into sessions, with their stored vectors."""
        rows = self._db.execute(
            "SELECT s.number, s.started_at, t.position, t.speaker, t.text, t.image, t.time_note, t.embedding_model, "
            "t.vector, t.vector_key FROM turns t JOIN sessions s ON s.user = t.user AND s.number = t.session "
            "WHERE t.user = ? AND (t.session, t.position) > (?, ?) ORDER BY t.session, t.position",
            (user, *after),
        ).fetchall()
        sessions: dict[int, tuple[str, list[Turn]]] = {}
        vectors: dict[str, np.ndarray] = {}
        for number, started_at, position, speaker, text, image, note, model, blob, key in rows:
            stamp = format_session_time(datetime.fromisoformat(started_at))
            # The note resolved at write time, the one the stored vector was computed with.
            note = note if self.options.time_anchors else None
            sessions.setdefault(number, (stamp, []))[1].append(Turn(f"D{number}:{position}", speaker, text, image, note))
            if model == self.embedder.name and key:
                vectors[key] = np.frombuffer(blob, dtype=np.float32)
        last = (rows[-1][0], rows[-1][2]) if rows else None
        return [Session(number, stamp, tuple(turns)) for number, (stamp, turns) in sessions.items()], vectors, last


def _parse_id(turn_id: str) -> tuple[int, int]:
    match = _TURN_ID.match(turn_id.strip())
    if not match:
        raise ValueError(f"not a memory id: {turn_id!r} (expected something like D3:12)")
    return int(match.group(1)), int(match.group(2))


def _memory(session: Session, turn: Turn) -> Memory:
    return Memory(turn.dia_id, turn.speaker, turn.text, session.date_time, turn.image_caption, turn.time_note)
