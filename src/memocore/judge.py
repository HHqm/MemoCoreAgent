"""LLM-as-judge grading with a lenient and a strict rubric.

The lenient rubric follows common practice in memory papers (same fact, any wording). The strict
rubric applies the principles LoCoMo-Refined reports as much closer to human labels: full coverage,
no contradiction, no unsupported additions, matching time granularity. Reporting both shows how
much of a score depends on the judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from memocore.datasets.locomo import ABSTAIN_ANSWER
from memocore.llm import Message
from memocore.metrics import normalize_answer

JUDGE_KINDS = ("strict", "lenient")
JUDGE_MAX_TOKENS = 200

_HEADER = """You are grading an answer to a question about a long conversation between two people.

Question: {question}
Gold answer: {gold}
Generated answer: {prediction}
"""

_LENIENT = """Be generous with wording. Mark CORRECT if the generated answer states the same fact as the gold answer, even if it is longer, phrased differently, or writes the same date or period in another format. Mark WRONG if it states a different fact, misses the point of the gold answer, or says the information is unavailable when the gold answer provides it."""

_STRICT = """Mark CORRECT only if every rule holds:
1. Coverage: it contains every piece of information the gold answer requires; for a list, every required item.
2. No contradiction: nothing in it conflicts with the gold answer.
3. No overreach: it adds no specific claims (extra list items, names, dates, reasons) that the gold answer does not support. Harmless rephrasing is fine.
4. Time granularity: a time must match at the gold answer's granularity. If the gold answer is a specific day, a month or year alone is WRONG; another expression of the same day (for example "the Sunday before 25 May 2023" for 21 May 2023) is fine.
Otherwise mark WRONG."""

_FOOTER = """
If the gold answer is "{abstain}", the generated answer is CORRECT only if it also says the information is not available, and WRONG if it gives a substantive answer.

Respond with JSON only: {{"reasoning": "<one sentence>", "label": "CORRECT" or "WRONG"}}"""

_LABEL_FIELD = re.compile(r'"label"\s*:\s*"(CORRECT|WRONG)"', re.IGNORECASE)
_BARE_LABEL = re.compile(r"\b(CORRECT|WRONG)\b")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Verdict:
    label: str | None  # None when the judge output could not be parsed
    reasoning: str

    @property
    def correct(self) -> bool:
        return self.label == "CORRECT"


def judge_messages(kind: str, question: str, gold: str, prediction: str) -> list[Message]:
    if kind not in JUDGE_KINDS:
        raise ValueError(f"unknown judge {kind!r}; choose from {', '.join(JUDGE_KINDS)}")
    rubric = _STRICT if kind == "strict" else _LENIENT
    prompt = (
        _HEADER.format(question=question, gold=gold, prediction=prediction or "(empty)")
        + "\n"
        + rubric
        + "\n"
        + _FOOTER.format(abstain=ABSTAIN_ANSWER)
    )
    return [{"role": "user", "content": prompt}]


def parse_verdict(text: str) -> Verdict:
    stripped = _FENCE.sub("", text.strip())
    try:
        data = json.loads(stripped)
        label = str(data.get("label", "")).upper()
        if label in ("CORRECT", "WRONG"):
            return Verdict(label, str(data.get("reasoning", "")))
    except (json.JSONDecodeError, AttributeError):
        pass
    field = _LABEL_FIELD.search(text)
    if field:
        return Verdict(field.group(1).upper(), text.strip()[:500])
    bare = _BARE_LABEL.findall(text.upper())
    return Verdict(bare[-1] if bare else None, text.strip()[:500])


_MOCK_GOLD = re.compile(r"^Gold answer: (.*)$", re.MULTILINE)
_MOCK_PREDICTION = re.compile(r"^Generated answer: (.*)$", re.MULTILINE)


def mock_judge_responder(messages: list[Message]) -> str:
    """Offline judge for --mock runs: CORRECT when the normalised gold appears in the prediction."""
    prompt = messages[-1]["content"]
    gold = _MOCK_GOLD.search(prompt)
    prediction = _MOCK_PREDICTION.search(prompt)
    hit = bool(gold and prediction) and normalize_answer(gold.group(1)) in normalize_answer(prediction.group(1))
    return json.dumps({"reasoning": "mock judge: substring match", "label": "CORRECT" if hit else "WRONG"})
