"""Lexical metrics, replicating the official LoCoMo scorer (snap-research/locomo task_eval/evaluation.py)."""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache
from statistics import fmean
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCTUATION = set(string.punctuation)
# The official scorer counts an adversarial answer as correct when it contains one of these.
ABSTAIN_MARKERS = ("not mentioned", "no information available")


def normalize_answer(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


@lru_cache(maxsize=1)
def _stemmer():
    # Imported lazily: nltk pulls in scikit-learn, and on Windows torch fails to load its DLLs
    # (WinError 1114) if scikit-learn was loaded first. Embedders import torch before this runs.
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def _stems(text: str) -> list[str]:
    stemmer = _stemmer()
    return [stemmer.stem(word) for word in normalize_answer(text).split()]


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = _stems(prediction), _stems(gold)
    same = sum((Counter(predicted) & Counter(expected)).values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, gold: str) -> float:
    """Multi-hop gold answers list several facts separated by commas; each is scored on its own."""
    predictions = [part.strip() for part in prediction.split(",")]
    golds = [part.strip() for part in gold.split(",")]
    return fmean(max(token_f1(p, g) for p in predictions) for g in golds)


def locomo_f1(prediction: str, gold: str, category: int) -> float:
    if category == 5:
        lowered = prediction.lower()
        return 1.0 if any(marker in lowered for marker in ABSTAIN_MARKERS) else 0.0
    if category == 3:
        # Open-domain golds append a rationale after ";" ("Likely no; she wants to be a counselor").
        gold = gold.split(";")[0].strip()
    if category == 1:
        return multi_answer_f1(prediction, gold)
    return token_f1(prediction, gold)


def answer_visible(gold: str, context: str, category: int) -> bool | None:
    """Whether every word of the gold answer appears somewhere in the context.

    A cheap, LLM-free proxy for "could a model read the answer off this context". It is how
    write-time time anchoring is measured: "yesterday" in a turn never contains "7 May 2023",
    the anchored note does.
    """
    if category == 5:
        return None
    if category == 3:
        gold = gold.split(";")[0]
    needed = set(normalize_answer(gold).split())
    if not needed:
        return None
    return needed <= set(normalize_answer(context).split())


def evidence_recall(evidence: Sequence[str], context_ids: Sequence[str]) -> float | None:
    """Share of gold evidence turns present in the context: separates retrieval misses from reasoning misses."""
    if not evidence:
        return None
    context = set(context_ids)
    return sum(turn_id in context for turn_id in evidence) / len(evidence)


def citation_scores(cited: Sequence[str], evidence: Sequence[str], context_ids: Sequence[str]) -> dict:
    """How well an answer's cited turns match the annotated evidence, and whether they were in its context.

    LoCoMo's evidence lists are sometimes incomplete, so precision here is a lower bound.
    """
    cited_set, gold = set(cited), set(evidence)
    hits = len(cited_set & gold)
    return {
        "cited_ids": list(cited),
        "citation_precision": hits / len(cited_set) if cited_set and gold else None,
        "citation_recall": hits / len(gold) if gold else None,
        "citation_in_context": len(cited_set & set(context_ids)) / len(cited_set) if cited_set else None,
    }
