"""Platt calibration: recovers a known sigmoid, probabilities in [0, 1], explicit skips."""

from pathlib import Path

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.probes.calibrate import apply_platt, calibration_decision, fit_platt, platt_targets

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def test_platt_recovers_a_known_sigmoid():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(4000) * 2
    p_true = 1 / (1 + np.exp(-(1.5 * s - 0.7)))
    y = (rng.random(4000) < p_true).astype(int)
    fitted = fit_platt(s, y)
    assert fitted["converged"] and fitted["fitted_on"] == "validation"
    assert abs(fitted["A"] + 1.5) < 0.15 and abs(fitted["B"] - 0.7) < 0.15
    p = apply_platt(fitted, s)
    assert p.min() >= 0 and p.max() <= 1
    assert np.corrcoef(p, p_true)[0, 1] > 0.99


def test_targets_are_prior_corrected():
    t = platt_targets(np.array([1, 1, 0, 0, 0]))
    assert np.allclose(t, [3 / 4, 3 / 4, 1 / 5, 1 / 5, 1 / 5])


def test_fit_refuses_one_class_or_misaligned():
    with pytest.raises(ValueError, match="both classes"):
        fit_platt(np.arange(5.0), np.ones(5, dtype=int))
    with pytest.raises(ValueError, match="aligned"):
        fit_platt(np.arange(5.0), np.array([0, 1]))


def test_skip_rule_is_explicit(config):
    assert calibration_decision(20, 2, config) == {"calibrate": True, "reason": None}
    assert calibration_decision(19, 2, config) == {
        "calibrate": False, "reason": "validation has 19 positives, fewer than 20"}
    assert calibration_decision(500, 1, config) == {
        "calibrate": False, "reason": "validation has 1 class(es)"}
