"""Loader for the LoCoMo long-conversation memory benchmark (snap-research/locomo, locomo10.json)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Category ids follow the official scorer (task_eval/evaluation.py in snap-research/locomo).
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
# Adversarial questions are usually excluded from reported scores; opt in with --categories.
DEFAULT_CATEGORIES = (1, 2, 3, 4)
ABSTAIN_ANSWER = "Not mentioned in the conversation"

_SESSION_KEY = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class Turn:
    dia_id: str
    speaker: str
    text: str
    image_caption: str | None = None
    # Derived at write time (see memocore.temporal); the raw text above is never rewritten.
    time_note: str | None = None

    def render(self) -> str:
        line = f"{self.speaker}: {self.text}"
        if self.image_caption:
            line += f" [shares an image: {self.image_caption}]"
        if self.time_note:
            line += f" [{self.time_note}]"
        return line


@dataclass(frozen=True)
class Session:
    index: int
    date_time: str
    turns: tuple[Turn, ...]

    def render(self, turns: Sequence[Turn] | None = None) -> str:
        body = "\n".join(turn.render() for turn in (self.turns if turns is None else turns))
        return f"--- Conversation at {self.date_time} ---\n{body}"


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]

    @property
    def turn_ids(self) -> tuple[str, ...]:
        return tuple(turn.dia_id for session in self.sessions for turn in session.turns)

    def render(self) -> str:
        return "\n\n".join(session.render() for session in self.sessions)


@dataclass(frozen=True)
class QAItem:
    qa_id: str
    sample_id: str
    index: int
    question: str
    answer: str | None
    category: int
    evidence: tuple[str, ...]
    adversarial_answer: str | None = None

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category-{self.category}")

    @property
    def gold(self) -> str:
        """Reference answer. Adversarial questions are unanswerable, so abstaining is the right answer."""
        if self.category == 5 or self.answer is None:
            return ABSTAIN_ANSWER
        return self.answer


@dataclass(frozen=True)
class LocomoSample:
    conversation: Conversation
    qa: tuple[QAItem, ...]


def load_locomo(path: Path) -> list[LocomoSample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_parse_sample(item) for item in raw]


def _parse_sample(item: dict) -> LocomoSample:
    sample_id = str(item["sample_id"])
    conv = item["conversation"]
    sessions = []
    for key, value in conv.items():
        match = _SESSION_KEY.match(key)
        if not match or not isinstance(value, list):
            continue
        sessions.append(
            Session(
                index=int(match.group(1)),
                date_time=str(conv.get(f"{key}_date_time", "")),
                turns=tuple(_parse_turn(turn) for turn in value),
            )
        )
    sessions.sort(key=lambda session: session.index)
    conversation = Conversation(sample_id, str(conv["speaker_a"]), str(conv["speaker_b"]), tuple(sessions))
    qa = tuple(_parse_qa(sample_id, index, raw_qa) for index, raw_qa in enumerate(item.get("qa", [])))
    return LocomoSample(conversation, qa)


def _parse_turn(raw: dict) -> Turn:
    return Turn(
        dia_id=str(raw["dia_id"]),
        speaker=str(raw["speaker"]),
        text=str(raw.get("text", "")),
        image_caption=raw.get("blip_caption") or None,
    )


def _parse_qa(sample_id: str, index: int, raw: dict) -> QAItem:
    answer = raw.get("answer")
    adversarial = raw.get("adversarial_answer")
    return QAItem(
        # Same id scheme as LoCoMo-Refined (<sample_id>#q<index>), so predictions can be cross-scored.
        qa_id=f"{sample_id}#q{index:04d}",
        sample_id=sample_id,
        index=index,
        question=str(raw["question"]),
        answer=None if answer is None else str(answer),
        category=int(raw["category"]),
        evidence=_split_evidence(raw.get("evidence", [])),
        adversarial_answer=None if adversarial is None else str(adversarial),
    )


def _split_evidence(values: Sequence) -> tuple[str, ...]:
    # A few annotations pack several turn ids into one string, e.g. "D8:6; D9:17".
    ids: list[str] = []
    for value in values:
        ids.extend(part.strip() for part in re.split(r"[;,]", str(value)) if part.strip())
    return tuple(ids)
