"""Stage 9 calibration: the SPEC §3.10 Platt scaler fitted on validation rows only.

``calibration_decision`` is the explicit, prespecified skip rule (items 4-5). ``fit_platt``
is the classic Platt (1999) sigmoid ``p = 1 / (1 + exp(A * s + B))`` on the raw decision
score, fitted by maximum likelihood against the prior-corrected targets of the original
method. Nothing here reads a partition: callers pass validation arrays only, and the test
lock (``probes/stage9.py``) is the only place that touches test rows.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from preanchor.config import ExperimentConfig

METHOD = "platt_sigmoid_prior_corrected_mle"


def calibration_decision(n_positive: int, n_classes: int, config: ExperimentConfig) -> dict:
    minimum = int(config.probes["calibration_minimum_positives"])
    classes = int(config.probes["calibration_minimum_classes"])
    if n_classes < classes:
        return {"calibrate": False, "reason": f"validation has {n_classes} class(es)"}
    if n_positive < minimum:
        return {"calibrate": False,
                "reason": f"validation has {n_positive} positives, fewer than {minimum}"}
    return {"calibrate": True, "reason": None}


def platt_targets(y: np.ndarray) -> np.ndarray:
    """Platt's prior-corrected targets: (N+ + 1) / (N+ + 2) and 1 / (N- + 2)."""
    y = np.asarray(y, dtype=np.int64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))


def fit_platt(score: np.ndarray, y: np.ndarray) -> dict:
    """Return ``{"A": a, "B": b, ...}`` such that ``sigmoid(-(A * score + B))`` is the
    calibrated probability. Deterministic (BFGS from the standard start)."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if s.ndim != 1 or s.shape != y.shape or len(s) == 0:
        raise ValueError("Calibration needs aligned one-dimensional scores and labels")
    if not np.isfinite(s).all() or set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Calibration needs finite scores and binary labels")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration needs both classes")
    t = platt_targets(y)

    def objective(params: np.ndarray) -> float:
        a, b = params
        z = a * s + b
        # log(1 + exp(z)) computed stably; negative log-likelihood of the targets.
        log1pexp = np.logaddexp(0.0, z)
        return float(np.sum(log1pexp - t * z))

    def gradient(params: np.ndarray) -> np.ndarray:
        a, b = params
        p = 1.0 / (1.0 + np.exp(-(a * s + b)))
        d = p - t
        return np.array([np.sum(d * s), np.sum(d)])

    n_pos = int(y.sum())
    start = np.array([0.0, math.log((len(y) - n_pos + 1.0) / (n_pos + 1.0))])
    result = minimize(objective, start, jac=gradient, method="BFGS",
                      options={"gtol": 1e-8, "maxiter": 10000})
    a, b = (float(v) for v in result.x)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("Platt fit diverged")
    return {"method": METHOD, "A": -a, "B": -b, "converged": bool(result.success),
            "iterations": int(result.nit), "negative_log_likelihood": float(result.fun),
            "n_rows": int(len(y)), "n_positive": n_pos, "fitted_on": "validation"}


def apply_platt(parameters: dict, score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(float(parameters["A"]) * s + float(parameters["B"])))
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Calibrated probability outside [0, 1]")
    return p
