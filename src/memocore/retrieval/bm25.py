"""Okapi BM25 for a few thousand short conversation turns."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
STOPWORDS = frozenset(
    """a an the and or but if of to in on at for with from by is are was were be been being am i you he she it
    we they me him her them my your his its our their this that these those what which who whom when where why
    how do does did doing have has had will would can could should shall may might must not no so as about into
    than then there here just also very too up out over again""".split()
)


def _stem(token: str) -> str:
    # A light suffix strip; nltk's Porter stemmer is avoided here because importing nltk drags in
    # scikit-learn, which breaks torch's DLL loading on Windows when it comes first.
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """English words (stopwords dropped, light stemming) plus Chinese character bigrams.

    Chinese has no spaces, and a dictionary segmenter would be one more dependency; overlapping
    bigrams are the usual dictionary-free choice for BM25 and match most two-character words.
    """
    tokens = [_stem(token) for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]
    for run in _CJK.findall(text):
        tokens.extend([run] if len(run) == 1 else (run[i : i + 2] for i in range(len(run) - 1)))
    return tokens


class BM25:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._size = 0
        self._lengths = np.zeros(0, dtype=np.float32)
        self._df: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._add(documents)

    def extended(self, documents: Sequence[str]) -> BM25:
        """A copy with `documents` appended, indexing only those. This index is left untouched, so a
        search running on it meanwhile is unaffected; scores equal those of an index built in one go."""
        copy = object.__new__(BM25)
        copy.__dict__.update(self.__dict__)
        copy._df, copy._postings = dict(self._df), dict(self._postings)
        copy._add(documents)
        return copy

    def _add(self, documents: Sequence[str]) -> None:
        counts = [Counter(tokenize(document)) for document in documents]
        added: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, c in enumerate(counts, start=self._size):
            for term, tf in c.items():
                added[term].append((index, tf))
        for term, postings in added.items():
            # A new list rather than an in-place append: the index this one was extended from shares the old one.
            self._postings[term] = self._postings.get(term, []) + postings
            self._df[term] = self._df.get(term, 0) + len(postings)
        self._lengths = np.concatenate([self._lengths, np.array([sum(c.values()) for c in counts], dtype=np.float32)])
        self._size += len(counts)
        average = float(self._lengths.mean()) if self._size and self._lengths.mean() > 0 else 1.0
        self._norm = self.k1 * (1 - self.b + self.b * self._lengths / average)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self._size, dtype=np.float32)
        for term in set(tokenize(query)):
            df = self._df.get(term)
            if df is None:
                continue
            idf = math.log(1 + (self._size - df + 0.5) / (df + 0.5))
            for index, tf in self._postings[term]:
                out[index] += idf * tf * (self.k1 + 1) / (tf + self._norm[index])
        return out
