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
        counts = [Counter(tokenize(document)) for document in documents]
        lengths = np.array([sum(c.values()) for c in counts], dtype=np.float32)
        average = float(lengths.mean()) if len(lengths) and lengths.mean() > 0 else 1.0
        self._norm = k1 * (1 - b + b * lengths / average)
        frequency = Counter(term for c in counts for term in c)
        n = len(documents)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, c in enumerate(counts):
            for term, tf in c.items():
                self._postings[term].append((index, tf))
        self._size = n

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self._size, dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, tf in self._postings[term]:
                out[index] += idf * tf * (self.k1 + 1) / (tf + self._norm[index])
        return out
