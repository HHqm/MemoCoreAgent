"""Bootstrap confidence intervals, so a one-point difference is not mistaken for an improvement."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

BOOTSTRAP_SAMPLES = 2000


def bootstrap_ci(
    values: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    picks = np.random.default_rng(seed).integers(0, len(data), size=(samples, len(data)))
    means = data[picks].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(
    baseline: Sequence[float], candidate: Sequence[float], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0
) -> dict[str, Any]:
    """Difference candidate - baseline over the same questions, resampling questions rather than runs."""
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same questions in the same order")
    diff = b - a
    result: dict[str, Any] = {
        "n": len(diff),
        "baseline": float(a.mean()) if len(a) else None,
        "candidate": float(b.mean()) if len(b) else None,
        "diff": float(diff.mean()) if len(diff) else None,
        "ci": None,
        "p_value": None,
    }
    if len(diff) < 2:
        return result
    picks = np.random.default_rng(seed).integers(0, len(diff), size=(samples, len(diff)))
    means = diff[picks].mean(axis=1)
    result["ci"] = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
    # Two-sided: how often a resample lands on the other side of zero.
    result["p_value"] = float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))
    return result
