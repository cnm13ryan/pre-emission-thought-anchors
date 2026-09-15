"""SPEC §10.6 probe sanity: prevalence, separable recovery, shuffled labels, probability
range, explicit calibration skip."""

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.probes import models as M
from preanchor.probes.calibrate import calibration_decision
from preanchor.probes.dataset import Partition
from preanchor.probes.fit import _fold_job, run_unit
from preanchor.probes.folds import grouped_folds

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def make(n=400, groups=8, seed=3, separable=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    obs = rng.standard_normal((n, 12))
    site = rng.standard_normal((n, 6)).astype(np.float32)
    if separable:
        site[:, 2] = y * 4 + rng.standard_normal(n) * 0.1
    p = Partition("train", [f"s{i}" for i in range(n)], ["t"] * n,
                  np.repeat(np.arange(groups), n // groups), y, rng.random(n) + y,
                  rng.random(n), obs, ["u/k/s"] * n, np.arange(n), 0.5)
    return p, site


def test_prevalence_aucpr_equals_prevalence(config):
    p, _ = make()
    folds = grouped_folds(p.y, p.problem_id, config)
    cv, pred, _ = run_unit("prevalence", None, p, None, folds, config, workers=1)
    for r in cv.to_pylist():
        assert abs(r["average_precision"] - r["val_pos"] / r["n_val"]) < 1e-12
    probs = pred.column("probability").to_pylist()
    assert min(probs) >= 0 and max(probs) <= 1


def test_separable_synthetic_feature_is_recovered(config):
    p, site = make(separable=True)
    folds = grouped_folds(p.y, p.problem_id, config)
    cv, _, _ = _fold_job("activation_only", M.BINARY, [1.0], M.design("activation_only", p, site),
                      p, folds[0], config)
    assert cv[0]["average_precision"] > 0.99 and cv[0]["auroc"] > 0.99 and cv[0]["converged"]


def test_shuffled_labels_return_chance(config):
    p, site = make(separable=True, n=800)
    rng = np.random.default_rng(7)
    p.y = rng.permutation(p.y)
    folds = grouped_folds(p.y, p.problem_id, config)
    aps, aucs = [], []
    for fold in folds:
        cv, _, _ = _fold_job("activation_only", M.BINARY, [1e-2],
                          M.design("activation_only", p, site), p, fold, config)
        aps.append(cv[0]["average_precision"] - fold["val_pos"] / fold["n_val"])
        aucs.append(cv[0]["auroc"])
    assert abs(np.mean(aps)) < 0.1 and abs(np.mean(aucs) - 0.5) < 0.1


def test_probabilities_in_range_and_calibration_skip_is_explicit(config):
    p, site = make()
    folds = grouped_folds(p.y, p.problem_id, config)
    _, pred, _ = _fold_job("activation_plus_observables", M.BINARY, [100.0],
                        M.design("activation_plus_observables", p, site), p, folds[1], config)
    probs = [r["probability"] for r in pred]
    assert min(probs) >= 0 and max(probs) <= 1
    assert calibration_decision(54, 2, config) == {"calibrate": True, "reason": None}
    assert calibration_decision(19, 2, config)["calibrate"] is False
    assert "19 positives" in calibration_decision(19, 2, config)["reason"]
    assert calibration_decision(50, 1, config)["calibrate"] is False
