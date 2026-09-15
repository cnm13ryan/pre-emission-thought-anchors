"""SPEC §3.11 metric definitions for a scored partition (validation or test).

Binary metrics rank by ``score_raw`` (AUCPR, AUROC, the per-trajectory top-10 % metrics)
and score probabilities by ``probability`` or ``probability_calibrated`` (Brier, log loss,
the 10-bin equal-frequency ECE and the reliability curve). Continuous metrics are on the
``log1p`` scale the Ridge probe is fitted on. Undefined values (a single class) are ``None``.
"""

from __future__ import annotations

import math

import numpy as np

from preanchor.probes.metrics import auroc, average_precision, rmse, spearman

TOP_FRACTION = 0.10
LOG_LOSS_EPS = 1e-15


def _finite_or_none(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(np.asarray(p, dtype=np.float64), LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> dict:
    """Equal-frequency bins by predicted probability (stable sort, ties by row order)."""
    order = np.argsort(np.asarray(p, dtype=np.float64), kind="stable")
    chunks = [c for c in np.array_split(order, bins) if len(c)]
    rows, ece = [], 0.0
    n = len(order)
    for k, idx in enumerate(chunks):
        mean_p, mean_y = float(np.mean(p[idx])), float(np.mean(y[idx]))
        ece += len(idx) / n * abs(mean_y - mean_p)
        rows.append({"bin": k, "count": int(len(idx)), "mean_probability": mean_p,
                     "observed_rate": mean_y, "positives": int(y[idx].sum())})
    return {"bins": bins, "binning": "equal_frequency", "ece": float(ece), "curve": rows}


def top_k(n: int) -> int:
    return max(1, int(math.ceil(TOP_FRACTION * n)))


def per_trajectory_ranking(y: np.ndarray, score: np.ndarray,
                           trajectory: list[str]) -> dict[str, dict]:
    """Precision@top-10 % and NDCG@top-10 % inside each trajectory (binary gains, log2
    discount). NDCG is undefined for a trajectory without a positive."""
    out = {}
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(trajectory):
        groups.setdefault(t, []).append(i)
    for t, idx in groups.items():
        idx = np.asarray(idx)
        k = top_k(len(idx))
        order = idx[np.argsort(-score[idx], kind="stable")]
        top = y[order[:k]]
        precision = float(np.mean(top))
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(top * discount))
        ideal_top = np.sort(y[idx])[::-1][:k]
        ideal = float(np.sum(ideal_top * discount))
        out[t] = {"n": int(len(idx)), "k": k, "positives": int(y[idx].sum()),
                  "precision_top10_percent": precision,
                  "ndcg_top10_percent": (dcg / ideal) if ideal > 0 else None}
    return out


def ranking_summary(per_trajectory: dict[str, dict]) -> dict:
    precision = [v["precision_top10_percent"] for v in per_trajectory.values()]
    ndcg = [v["ndcg_top10_percent"] for v in per_trajectory.values()
            if v["ndcg_top10_percent"] is not None]
    return {"precision_top10_percent": _finite_or_none(np.mean(precision)) if precision else None,
            "ndcg_top10_percent": _finite_or_none(np.mean(ndcg)) if ndcg else None,
            "n_trajectories": len(per_trajectory), "n_trajectories_with_positive": len(ndcg)}


def binary_metrics(y: np.ndarray, score: np.ndarray, probability: np.ndarray,
                   trajectory: list[str], bins: int,
                   probability_calibrated: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=np.int64)
    two_classes = len(np.unique(y)) == 2
    per_traj = per_trajectory_ranking(y, score, trajectory)
    out = {"n": int(len(y)), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
           "average_precision": _finite_or_none(average_precision(y, score))
           if two_classes else None,
           "auroc": _finite_or_none(auroc(y, score)) if two_classes else None,
           "brier": brier(y, probability), "log_loss": log_loss(y, probability),
           "reliability": reliability(y, probability, bins),
           **ranking_summary(per_traj), "per_trajectory": per_traj}
    out["ece"] = out["reliability"]["ece"]
    if probability_calibrated is not None:
        rel = reliability(y, probability_calibrated, bins)
        out["calibrated"] = {"brier": brier(y, probability_calibrated),
                             "log_loss": log_loss(y, probability_calibrated),
                             "ece": rel["ece"], "reliability": rel}
    return out


def continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    return {"n": int(len(target)), "spearman": _finite_or_none(spearman(target, prediction)),
            "rmse": _finite_or_none(rmse(target, prediction))}
