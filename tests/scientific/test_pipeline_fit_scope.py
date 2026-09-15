"""SPEC §10.5-10.6: scalers, residualisation and hyperparameters are fitted inside training
folds only; Stage 8 never sees validation or test rows."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from preanchor.config import load_config
from preanchor.probes import models as M
from preanchor.probes.dataset import Partition
from preanchor.probes.fit import _fold_job
from preanchor.probes.folds import grouped_folds

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def make(n=160, groups=8, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    obs = rng.standard_normal((n, 12)) * np.arange(1, 13)
    return Partition("train", [f"s{i}" for i in range(n)], ["t"] * n,
                     np.repeat(np.arange(groups), n // groups), y, rng.random(n),
                     rng.random(n), obs, ["u/k/s"] * n, np.arange(n), 0.5)


def test_fold_local_scaler_differs_from_leaked_global_scaler(config, monkeypatch):
    p = make()
    folds = grouped_folds(p.y, p.problem_id, config)
    fitted_sizes = []
    original = StandardScaler.fit

    def spy(self, X, y=None, **kw):
        fitted_sizes.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(StandardScaler, "fit", spy)
    _fold_job("observable_combined", M.BINARY, [1.0], M.design("observable_combined", p, None),
              p, folds[0], config)
    assert fitted_sizes and all(s == folds[0]["n_train"] for s in fitted_sizes)
    local = StandardScaler().fit(p.observables[folds[0]["train_idx"]])
    leaked = StandardScaler().fit(p.observables)
    assert not np.allclose(local.mean_, leaked.mean_)


def test_residualizer_is_fitted_on_training_rows_only():
    p = make()
    site = np.random.default_rng(2).standard_normal((len(p), 5)).astype(np.float32)
    X = M.design(M.RESIDUALIZED, p, site)
    tr = np.arange(100)
    r = M.Residualizer(12).fit(X[tr])
    X_val = X[100:].copy()
    X_val[:, :5] += 100  # changing validation activations must not change the projection
    r2 = M.Residualizer(12).fit(X[tr])
    assert np.array_equal(r.coef_, r2.coef_)
    out = r.transform(X_val)
    assert out.shape == (60, 5)
    # Residual of training rows is orthogonal to the fitted observable projection.
    res = r.transform(X[tr]).astype(np.float64)
    z = np.concatenate([r.scaler_.transform(X[tr][:, -12:]), np.ones((100, 1))], axis=1)
    assert np.abs(z.T @ res).max() < 1e-6


def test_pipeline_families_and_grids_are_frozen(config):
    assert M.hyperparameters(M.BINARY, config) == [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    assert M.hyperparameters(M.CONTINUOUS, config) == [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100,
                                                       1e3, 1e4]
    pipe = M.make_pipeline("activation_plus_observables", M.BINARY, 0.1, config)
    model = pipe.named_steps["model"]
    assert (model.penalty, model.class_weight, model.max_iter, model.C) == (
        "l2", "balanced", 5000, 0.1)
    assert type(pipe.named_steps["scale"]).__name__ == "StandardScaler"
    with pytest.raises(ValueError):
        M.make_pipeline("prevalence", M.CONTINUOUS, None, config)
