from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def top_indices(scores: np.ndarray, n: int, *, positive_only: bool = False) -> list[int]:
    """Indices of the n highest scores, best first; ties go to the earlier index."""
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(scores))
    if len(candidates) == 0 or n <= 0:
        return []
    n = min(n, len(candidates))
    top = candidates[np.argpartition(-scores[candidates], n - 1)[:n]]
    return sorted((int(i) for i in top), key=lambda i: (-scores[i], i))


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: needs no score calibration between dense and lexical retrievers."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] += 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda item: (-fused[item], item))
