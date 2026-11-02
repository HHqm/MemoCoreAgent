from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from memocore.datasets.locomo import Conversation


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

    @abstractmethod
    def ingest(self, conversation: Conversation) -> None: ...

    @abstractmethod
    def retrieve(self, question: str) -> RetrievedContext: ...
