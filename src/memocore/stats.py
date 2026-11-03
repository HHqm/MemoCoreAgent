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


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    """Interval for a proportion that stays inside [0, 1] even when it is near 0 or 1, as agreement rates are."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Agreement between two raters beyond what their label frequencies alone would produce by chance."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n) for label in labels
    )
    return None if expected == 1 else (observed - expected) / (1 - expected)
