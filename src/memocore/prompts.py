"""Prompts shared by every memory system, so systems differ only in the context they supply."""

from __future__ import annotations

import hashlib
import re

from memocore.datasets.locomo import ABSTAIN_ANSWER, Conversation
from memocore.llm import Message

# Generous because DeepSeek's current models think before answering and the cap covers both.
ANSWER_MAX_TOKENS = 2048

ANSWER_SYSTEM = "You answer questions about long-running conversations between two people, using only the memory you are given."

# The memory comes first and the question last: with full context every question of a conversation
# shares one long prompt prefix, which the provider's prefix cache can serve cheaply.
_HEAD = """Below is memory from conversations between {speaker_a} and {speaker_b}. Each block starts with the date and time that conversation took place.

<memory>
{context}
</memory>

Instructions:
- Answer using only the memory above. Bracketed notes, such as a shared image's description or a resolved time, are part of the memory.
- Relative times such as "yesterday" or "last week" are relative to the date of the conversation they were said in; convert them to an absolute date or period, or state them relative to that date (for example "the week before 9 June 2023"). Never include the clock time of a conversation.
- If the question asks what is likely or probable, reason from the memory and give your best answer.
"""

_TAIL = """
Question: {question}
Answer:"""

ANSWER_TEMPLATES = {
    # The first end-to-end run (docs/results/end_to_end_s300.md).
    "v1": _HEAD
    + """- Reply with a short phrase (ideally under 10 words), not a full sentence. Answer only what is asked: no reasons, examples or extra details the question does not ask for.
- If the memory does not contain the information, reply exactly: {abstain}.
"""
    + _TAIL,
    # From the v1 error analysis: answers were too vague (dropped names and qualifiers) or padded with
    # extra details and loosely related list items. Rules alone made the model write full sentences or
    # quote chat filler, so the form is shown by examples about made-up people instead.
    "v2": _HEAD
    + """- Reply with a short phrase (ideally under 10 words), not a full sentence.
- Be specific about the thing asked: keep the names, numbers and qualifiers the memory gives for it, and prefer the memory's own key word to a synonym, without quoting chat filler.
- Add nothing beyond the thing asked: no reasons, places, descriptions, feelings or times of day unless the question asks for them. For a list, give only the items the memory states as answers to this question.
- If the memory does not contain the information, reply exactly: {abstain}.

Examples of the expected form, from other conversations:
Q: What instrument did Sam buy? A: An electric guitar
Q: What does Lee do on weekends? A: Rock climbing, baking bread
Q: How did Ana describe the concert? A: Unforgettable
Q: When did Ben move to Denver? A: The week before 3 May 2022
Q: Would Kim enjoy a cooking class? A: Likely yes
"""
    + _TAIL,
    # v2 plus grounding: the context shows turn ids and every part of the answer must cite one. Aimed at
    # the largest error class (padding from loosely related turns), and it makes answers traceable.
    "v3": _HEAD
    + """- Each memory line starts with its id in square brackets, such as [D3:12].
- Reply with a short phrase (ideally under 10 words), not a full sentence, followed by the ids of the lines it comes from in square brackets.
- Every part of the answer must come from a line you cite; leave out anything you cannot cite. For a list, cite a line for each item, and include every item the memory states as an answer to this question.
- Be specific about the thing asked: keep the names, numbers and qualifiers the memory gives for it, and prefer the memory's own key word to a synonym, without quoting chat filler.
- Add nothing beyond the thing asked: no reasons, places, descriptions, feelings or times of day unless the question asks for them.
- If the memory does not contain the information, reply exactly: {abstain}.

Examples of the expected form, from other conversations:
Q: What instrument did Sam buy? A: An electric guitar [D4:7]
Q: What does Lee do on weekends? A: Rock climbing, baking bread [D2:3, D9:14]
Q: How did Ana describe the concert? A: Unforgettable [D6:2]
Q: When did Ben move to Denver? A: The week before 3 May 2022 [D11:5]
Q: Would Kim enjoy a cooking class? A: Likely yes [D3:8, D7:1]
"""
    + _TAIL,
}
DEFAULT_PROMPT = "v3"  # same accuracy as v1 on the paired test, and every answer is traceable
# Prompts whose context carries turn ids and whose replies end with cited ids.
CITING_PROMPTS = frozenset({"v3"})

_CITATION = re.compile(r"\s*[\[(]\s*(D\d+:\d+(?:\s*[,;]\s*D\d+:\d+)*)\s*[\])]")
_TURN_ID = re.compile(r"D\d+:\d+")


def split_citations(reply: str) -> tuple[str, tuple[str, ...]]:
    """Separate a citing reply into the bare answer (what gets graded) and the turn ids it cites."""
    cited = [turn_id for group in _CITATION.findall(reply) for turn_id in _TURN_ID.findall(group)]
    answer = _CITATION.sub("", reply).strip().rstrip(",;").strip()
    return answer, tuple(dict.fromkeys(cited))


def prompt_version(name: str = DEFAULT_PROMPT) -> str:
    """Content hash recorded with each run, so editing a prompt can never silently mix two versions."""
    return hashlib.sha256((ANSWER_SYSTEM + _template(name)).encode("utf-8")).hexdigest()[:12]


def _template(name: str) -> str:
    if name not in ANSWER_TEMPLATES:
        raise ValueError(f"unknown prompt {name!r}; choose from {', '.join(ANSWER_TEMPLATES)}")
    return ANSWER_TEMPLATES[name]


def answer_messages(
    conversation: Conversation, context: str, question: str, prompt: str = DEFAULT_PROMPT
) -> list[Message]:
    user = _template(prompt).format(
        speaker_a=conversation.speaker_a,
        speaker_b=conversation.speaker_b,
        context=context,
        question=question,
        abstain=ABSTAIN_ANSWER,
    )
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
