"""Prompts shared by every memory system, so systems differ only in the context they supply."""

from __future__ import annotations

import hashlib

from memocore.datasets.locomo import ABSTAIN_ANSWER, Conversation
from memocore.llm import Message

ANSWER_MAX_TOKENS = 200

ANSWER_SYSTEM = "You answer questions about long-running conversations between two people, using only the memory you are given."

# The memory comes first and the question last: with full context every question of a conversation
# shares one long prompt prefix, which the provider's prefix cache can serve cheaply.
ANSWER_TEMPLATE = """Below is memory from conversations between {speaker_a} and {speaker_b}. Each block starts with the date and time that conversation took place.

<memory>
{context}
</memory>

Instructions:
- Answer using only the memory above.
- Relative times such as "yesterday" or "last week" are relative to the date of the conversation they were said in; convert them to an absolute date or period.
- If the question asks what is likely or probable, reason from the memory and give your best answer.
- Reply with a short phrase (ideally under 10 words), not a full sentence.
- If the memory does not contain the information, reply exactly: {abstain}.

Question: {question}
Answer:"""

PROMPT_VERSION = hashlib.sha256((ANSWER_SYSTEM + ANSWER_TEMPLATE).encode("utf-8")).hexdigest()[:12]


def answer_messages(conversation: Conversation, context: str, question: str) -> list[Message]:
    user = ANSWER_TEMPLATE.format(
        speaker_a=conversation.speaker_a,
        speaker_b=conversation.speaker_b,
        context=context,
        question=question,
        abstain=ABSTAIN_ANSWER,
    )
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
