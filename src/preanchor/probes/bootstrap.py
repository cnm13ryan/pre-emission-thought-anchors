"""SPEC §3.11 clustered paired bootstrap.

Resampling unit = ``problem_id`` with replacement; a sampled problem brings both
trajectories and every sentence, with multiplicity. Every model is evaluated on the same
replicate, so paired differences are differences within a replicate. Seeds derive from
``seed_root`` (``derived_seed(seed_root, "bootstrap", split)``) and the procedure is
deterministic. Intervals are percentile intervals and, with four test problems,
descriptive only (AGENTS §5.5).
"""

from __future__ import annotations

import numpy as np

from preanchor.config import ExperimentConfig, config_hash
from preanchor.ids import derived_seed
from preanchor.probes.evaluation import brier, log_loss, per_trajectory_ranking
from preanchor.probes.metrics import auroc, average_precision, rmse, spearman

BOOTSTRAP_PURPOSE = "bootstrap"
BINARY_STATISTICS = ("average_precision", "auroc", "brier", "log_loss",
                     "precision_top10_percent", "ndcg_top10_percent")
CONTINUOUS_STATISTICS = ("spearman", "rmse")
PAIRED_STATISTICS = ("average_precision", "auroc")


def bootstrap_seed(config: ExperimentConfig, split: str) -> int:
    return derived_seed(config.seed_root, BOOTSTRAP_PURPOSE, split)


def check_spec(config: ExperimentConfig) -> tuple[int, str, bool]:
    e = config.evaluation
    replicates, unit, paired = int(e["bootstrap_replicates"]), e["bootstrap_unit"], \
        bool(e["bootstrap_paired"])
    if unit != "problem_id" or not paired or replicates < 1:
        raise ValueError("Frozen bootstrap design differs from the configuration")
    return replicates, unit, paired


def sample_problems(rng: np.random.Generator, problems: np.ndarray) -> np.ndarray:
    return rng.choice(problems, size=len(problems), replace=True)


def replicate_indices(sampled: np.ndarray, by_problem: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([by_problem[int(p)] for p in sampled])


def _binary_statistics(y: np.ndarray, score: np.ndarray, prob: np.ndarray,
                       traj_values: dict[str, dict], traj_ids: np.ndarray, idx: np.ndarray,
                       traj_multiplicity: dict[str, int]) -> dict[str, float]:
    yy, ss, pp = y[idx], score[idx], prob[idx]
    two = len(np.unique(yy)) == 2
    precision = [traj_values[t]["precision_top10_percent"] for t, m in traj_multiplicity.items()
                 for _ in range(m)]
    ndcg = [traj_values[t]["ndcg_top10_percent"] for t, m in traj_multiplicity.items()
            for _ in range(m) if traj_values[t]["ndcg_top10_percent"] is not None]
    return {"average_precision": average_precision(yy, ss) if two else np.nan,
            "auroc": auroc(yy, ss) if two else np.nan,
            "brier": brier(yy, pp), "log_loss": log_loss(yy, pp),
            "precision_top10_percent": float(np.mean(precision)) if precision else np.nan,
            "ndcg_top10_percent": float(np.mean(ndcg)) if ndcg else np.nan}


def _interval(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"low": None, "high": None, "mean": None, "n_valid": 0}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "mean": float(np.mean(finite)),
            "n_valid": int(len(finite))}


def clustered_bootstrap(*, problem_id: np.ndarray, trajectory_id: list[str], y: np.ndarray,
                        importance: np.ndarray, binary: dict[str, dict[str, np.ndarray]],
                        continuous: dict[str, np.ndarray], primary: str, baseline: str,
                        config: ExperimentConfig, split: str,
                        replicates: int | None = None) -> dict:
    """``binary[model] = {"score": ..., "probability": ...}``; ``continuous[model]`` is the
    Ridge prediction. Returns point estimates, percentile intervals, paired differences of
    ``primary`` minus every other binary model, and the exact sampling record."""
    n_replicates, unit, paired = check_spec(config)
    n_replicates = replicates or n_replicates
    problem_id = np.asarray(problem_id)
    problems = np.array(sorted(set(int(p) for p in problem_id)))
    by_problem = {int(p): np.flatnonzero(problem_id == p) for p in problems}
    traj_ids = np.asarray(trajectory_id)
    traj_problem = {t: int(problem_id[np.flatnonzero(traj_ids == t)[0]])
                    for t in set(trajectory_id)}
    y = np.asarray(y, dtype=np.int64)
    seed = bootstrap_seed(config, split)
    rng = np.random.default_rng(seed)
    traj_values = {name: per_trajectory_ranking(y, m["score"], trajectory_id)
                   for name, m in binary.items()}
    stats = {name: {s: np.full(n_replicates, np.nan) for s in BINARY_STATISTICS}
             for name in binary}
    cont = {name: {s: np.full(n_replicates, np.nan) for s in CONTINUOUS_STATISTICS}
            for name in continuous}
    sampled_record = []
    for r in range(n_replicates):
        sampled = sample_problems(rng, problems)
        sampled_record.append([int(p) for p in sampled])
        idx = replicate_indices(sampled, by_problem)
        counts = {int(p): int(c) for p, c in zip(*np.unique(sampled, return_counts=True),
                                                    strict=True)}
        multiplicity = {t: counts[p] for t, p in traj_problem.items() if p in counts}
        for name, m in binary.items():
            values = _binary_statistics(y, m["score"], m["probability"], traj_values[name],
                                        traj_ids, idx, multiplicity)
            for s in BINARY_STATISTICS:
                stats[name][s][r] = values[s]
        for name, pred in continuous.items():
            cont[name]["spearman"][r] = spearman(importance[idx], pred[idx])
            cont[name]["rmse"][r] = rmse(importance[idx], pred[idx])
    result = {"schema_version": 1, "replicates_requested": n_replicates,
              "sampling_unit": unit, "paired": paired, "with_replacement": True,
              "cluster": "both trajectories and every eligible sentence of a sampled problem",
              "seed": seed, "seed_purpose": [BOOTSTRAP_PURPOSE, split],
              "n_problems": int(len(problems)), "problems": [int(p) for p in problems],
              "interval": "percentile_2.5_97.5", "descriptive_only": True,
              "binary": {name: {s: _interval(v[s]) for s in BINARY_STATISTICS}
                         for name, v in stats.items()},
              "continuous": {name: {s: _interval(v[s]) for s in CONTINUOUS_STATISTICS}
                             for name, v in cont.items()},
              "paired_differences": {}, "sampled_problems_sha256": None}
    for name in binary:
        if name == primary:
            continue
        result["paired_differences"][f"{primary}-minus-{name}"] = {
            s: _interval(stats[primary][s] - stats[name][s]) for s in PAIRED_STATISTICS}
    if baseline in binary and primary in binary:
        result["aucpr_gain_over_baseline"] = result["paired_differences"].get(
            f"{primary}-minus-{baseline}", {}).get("average_precision")
    result["sampled_problems_sha256"] = config_hash(sampled_record)
    return result
