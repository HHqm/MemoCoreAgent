from __future__ import annotations

from memocore.datasets.locomo import Conversation
from memocore.systems.base import MemorySystem, RetrievedContext


class FullContextSystem(MemorySystem):
    """Puts the entire conversation in the prompt: the accuracy ceiling and the cost ceiling."""

    name = "full_context"

    def __init__(self) -> None:
        self._context = RetrievedContext("", ())

    def ingest(self, conversation: Conversation) -> None:
        self._context = RetrievedContext(conversation.render(self.cite), conversation.turn_ids)

    def retrieve(self, question: str) -> RetrievedContext:
        return self._context
