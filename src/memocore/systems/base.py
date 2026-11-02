from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from memocore.datasets.locomo import Conversation, Session, Turn


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    # LoCoMo turn ids the context was built from; scored against the gold evidence.
    turn_ids: tuple[str, ...]


class MemorySystem(ABC):
    """Ingests a whole conversation, then supplies the context for each question.

    Answer generation is deliberately not part of a system: every system feeds the same answer
    model and prompt, so a score difference can only come from the context it supplied.
    """

    name: str
    # Prefix each context line with its turn id, so the answer can cite where it came from.
    cite: bool = False

    @abstractmethod
    def ingest(self, conversation: Conversation) -> None: ...

    @abstractmethod
    def retrieve(self, question: str) -> RetrievedContext: ...


def render_turns(pairs: Sequence[tuple[Session, Turn]], with_ids: bool = False) -> RetrievedContext:
    """Context from (session, turn) pairs already in conversation order, one date header per run of a session."""
    blocks: list[tuple[Session, list[Turn]]] = []
    for session, turn in pairs:
        if blocks and blocks[-1][0] is session:
            blocks[-1][1].append(turn)
        else:
            blocks.append((session, [turn]))
    text = "\n\n".join(session.render(turns, with_ids) for session, turns in blocks)
    return RetrievedContext(text, tuple(turn.dia_id for _, turn in pairs))


class NoMemorySystem(MemorySystem):
    """Gives the model no memory at all: the floor that shows how much guessing and common sense score."""

    name = "no_memory"

    def ingest(self, conversation: Conversation) -> None:
        pass

    def retrieve(self, question: str) -> RetrievedContext:
        return RetrievedContext("(no memory available)", ())
