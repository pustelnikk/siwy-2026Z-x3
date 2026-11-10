"""Map normalisations, similarity measures and bootstrap intervals.

The randomization verdict is known to depend on both normalisation and similarity measure
(Binder et al., 2021), so every check reports every (normalisation, measure) combination.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity

EPS = 1e-12


def normalize(m: np.ndarray, mode: str) -> np.ndarray:
    """Normalise one map.

    raw      - signed values, unchanged
    abs      - absolute values
    abs_max  - absolute values divided by their maximum (Adebayo et al., 2018)
    max_abs  - signed values divided by the maximum absolute value
    """
    m = np.asarray(m, dtype=np.float64)
    if mode == "raw":
        return m
    if mode == "abs":
        return np.abs(m)
    if mode == "abs_max":
        a = np.abs(m)
        return a / (a.max() + EPS)
    if mode == "max_abs":
        return m / (np.abs(m).max() + EPS)
    raise ValueError(f"unknown normalisation {mode!r}")


def _flat_corr(fn):
    def measure(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.ravel(), b.ravel()
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(fn(a, b)[0])

    return measure


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on 2-D maps; undefined (NaN) for maps with fewer than 7 frequency bins."""
    if min(a.shape) < 7:
        return float("nan")
    data_range = max(a.max(), b.max()) - min(a.min(), b.min())
    if data_range == 0:
        return float("nan")
    return float(structural_similarity(a, b, data_range=data_range))


def topk_intersection(a: np.ndarray, b: np.ndarray, fraction: float = 0.1) -> float:
    """Fraction of the top-``fraction`` cells of ``a`` that are also in the top cells of ``b``."""
    a, b = a.ravel(), b.ravel()
    k = max(1, int(round(fraction * a.size)))
    top_a = np.argpartition(-a, k - 1)[:k]
    top_b = np.argpartition(-b, k - 1)[:k]
    return len(np.intersect1d(top_a, top_b)) / k


MEASURES: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "spearman": _flat_corr(spearmanr),
    "pearson": _flat_corr(pearsonr),
    "ssim": ssim,
    "topk": topk_intersection,
}


def similarity(a: np.ndarray, b: np.ndarray, normalization: str, measure: str) -> float:
    return MEASURES[measure](normalize(a, normalization), normalize(b, normalization))


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval, NaNs ignored."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
