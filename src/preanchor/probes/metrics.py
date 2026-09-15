"""Train-CV selection metrics (SPEC §3.10-3.11)."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.ptp(prediction) == 0 or np.ptp(target) == 0:
        return math.nan
    return float(spearmanr(target, prediction).statistic)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))
