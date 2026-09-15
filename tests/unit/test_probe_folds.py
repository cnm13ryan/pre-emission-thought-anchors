"""Grouped folds, seeds and the model table (SPEC §3.10, §10.5)."""

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.features.observables import PRIMARY_FEATURES
from preanchor.ids import derived_seed
from preanchor.probes import models as M
from preanchor.probes.dataset import OBSERVABLE_BLOCKS, Partition
from preanchor.probes.folds import assert_disjoint, fold_seed, grouped_folds
from preanchor.probes.metrics import average_precision, spearman

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def partition(n=120, seed=0, groups=6):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    return Partition("train", [f"p:{i}" for i in range(n)], ["p:c"] * n,
                     np.repeat(np.arange(groups), n // groups), y, rng.random(n),
                     rng.random(n), rng.standard_normal((n, 12)), ["u/k/s"] * n,
                     np.arange(n), 0.5)


def test_split_and_seed(config):
    assert_disjoint(config)
    assert fold_seed(config) == derived_seed(config.seed_root, "cv_folds")
    assert fold_seed(config) != derived_seed(config.seed_root + 1, "cv_folds")


def test_grouped_folds_keep_problems_apart_and_are_deterministic(config):
    p = partition()
    folds = grouped_folds(p.y, p.problem_id, config)
    again = grouped_folds(p.y, p.problem_id, config)
    assert len(folds) == 4
    covered = []
    for f, g in zip(folds, again, strict=True):
        assert not set(p.problem_id[f["train_idx"]]) & set(p.problem_id[f["val_idx"]])
        assert f["train_pos"] > 0 and f["val_pos"] > 0
        assert np.array_equal(f["val_idx"], g["val_idx"]) and f["seed"] == g["seed"]
        covered += f["val_problems"]
    assert sorted(covered) == list(range(6))


def test_single_class_fold_fails_loudly(config):
    p = partition()
    p.y[:] = 0
    p.y[0] = 1
    with pytest.raises(ValueError, match="single class"):
        grouped_folds(p.y, p.problem_id, config)


def test_observable_blocks_cover_the_frozen_allow_list_once(config):
    assert sorted(OBSERVABLE_BLOCKS["observable_combined"]) == sorted(PRIMARY_FEATURES)
    assert OBSERVABLE_BLOCKS["observable_combined"] == list(config.features["observables"]) \
        or sorted(OBSERVABLE_BLOCKS["observable_combined"]) == sorted(
            config.features["observables"])
    for name in ("position", "context", "uncertainty"):
        assert all(c in PRIMARY_FEATURES for c in OBSERVABLE_BLOCKS[name])
    assert not any(c.startswith("diagnostic__") for c in OBSERVABLE_BLOCKS["observable_combined"])
    M.check_frozen(config)


def test_design_shapes(config):
    p = partition()
    site = np.zeros((len(p), 7), dtype=np.float32)
    assert M.design("prevalence", p, None).shape == (len(p), 0)
    assert M.design("position", p, None).shape == (len(p), 3)
    assert M.design("uncertainty", p, None).shape == (len(p), 6)
    assert M.design("observable_combined", p, None).shape == (len(p), 12)
    assert M.design("activation_only", p, site).shape == (len(p), 7)
    assert M.design("activation_plus_observables", p, site).shape == (len(p), 19)
    with pytest.raises(ValueError):
        M.design("activation_only", p, None)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([1.0, 1.0])))
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 9.0])) == 1.0
