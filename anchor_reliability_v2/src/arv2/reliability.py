"""Agreement statistics, problem-clustered bootstrap and the three-way decision rule."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr


def seed_from(material: str) -> int:
    return int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def icc_2_1(x, y) -> float:
    """Shrout & Fleiss ICC(2,1): two-way random effects, absolute agreement, single rater."""
    data = np.column_stack([x, y]).astype(float)
    n, k = data.shape
    if n < 3:
        return float("nan")
    grand = data.mean()
    msr = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)
    resid = data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand
    mse = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else float("nan")


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def spearman_brown(r: float, factor: float) -> float:
    """Projected reliability when the measurement is lengthened by ``factor``."""
    return factor * r / (1 + (factor - 1) * r)


def cluster_bootstrap(groups: np.ndarray, statistic: Callable[[np.ndarray], float],
                      reps: int = 2000, seed: int = 0) -> np.ndarray:
    """Resample whole clusters (problems) with replacement; ``statistic`` gets row indices."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    out = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        out[r] = statistic(rows)
    return out


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float, int]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return float("nan"), float("nan"), len(draws)
    lo, hi = np.quantile(finite, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi), int(len(draws) - len(finite))


def three_way(lo: float, hi: float, threshold: float) -> str:
    """PASS if the whole interval clears the threshold, FAIL if it lies wholly below,
    INCONCLUSIVE otherwise (or if the interval is undefined)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "INCONCLUSIVE"
    if lo >= threshold:
        return "PASS"
    if hi < threshold:
        return "FAIL"
    return "INCONCLUSIVE"
