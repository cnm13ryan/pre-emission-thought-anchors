"""The SPEC §3.10 model table as sklearn pipelines; every fit is fold-local by construction.

``design`` builds the input matrix for a model from a partition's observables and one
activation site. ``Residualizer`` removes the fold-fitted linear projection of the
observables from the activation columns before scaling and probing.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preanchor.config import ExperimentConfig
from preanchor.probes.dataset import OBSERVABLE_BLOCKS, Partition

BASELINES = ["prevalence", "position", "context", "uncertainty", "observable_combined"]
ACTIVATION_MODELS = ["activation_only", "activation_plus_observables"]
RESIDUALIZED = "residualized_activation"
BINARY, CONTINUOUS = "binary", "continuous"


def check_frozen(config: ExperimentConfig) -> None:
    p = config.probes
    if (p["scaler"], p["binary"], p["class_weight"], p["continuous"]) != (
            "StandardScaler", "logistic_l2", "balanced", "ridge"):
        raise ValueError("Frozen probe families differ from the configuration")
    if list(p["baselines"]) != BASELINES or list(p["activation_models"]) != ACTIVATION_MODELS:
        raise ValueError("Frozen model table differs from the configuration")
    if p["residualization"] != "fold_local_linear_observables":
        raise ValueError("Frozen residualization control differs from the configuration")


def uses_activation(name: str) -> bool:
    return name in ACTIVATION_MODELS or name == RESIDUALIZED


def design(name: str, partition: Partition, site: np.ndarray | None) -> np.ndarray:
    """Model input matrix (float32). Observables come last so ``Residualizer`` finds them."""
    if name == "prevalence":
        return np.zeros((len(partition), 0), dtype=np.float32)
    if name in OBSERVABLE_BLOCKS:
        return partition.observable_block(name).astype(np.float32)
    if site is None or site.shape[0] != len(partition):
        raise ValueError("Activation models need one site aligned to the partition")
    if name == "activation_only":
        return site
    if name in (RESIDUALIZED, "activation_plus_observables"):
        return np.concatenate([site, partition.observable_block("observable_combined")
                               .astype(np.float32)], axis=1)
    raise ValueError(f"Unknown model {name}")


class Residualizer(BaseEstimator, TransformerMixin):
    """Fold-local linear removal of the trailing ``n_obs`` observable columns."""

    def __init__(self, n_obs: int = 12):
        self.n_obs = n_obs

    def fit(self, X, y=None):
        act, obs = X[:, :-self.n_obs], X[:, -self.n_obs:]
        self.scaler_ = StandardScaler().fit(obs)
        z = np.concatenate([self.scaler_.transform(obs), np.ones((len(X), 1))], axis=1)
        self.coef_, *_ = np.linalg.lstsq(z.astype(np.float64), act.astype(np.float64),
                                         rcond=None)
        return self

    def transform(self, X):
        act, obs = X[:, :-self.n_obs], X[:, -self.n_obs:]
        z = np.concatenate([self.scaler_.transform(obs), np.ones((len(X), 1))], axis=1)
        return (act.astype(np.float64) - z @ self.coef_).astype(np.float32)


class Prevalence(BaseEstimator):
    """Constant training anchor rate (SPEC §3.10 prevalence baseline)."""

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.full(len(X), self.rate_)

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.stack([1 - p, p], axis=1)


def make_pipeline(name: str, task: str, hyperparameter: float | None,
                  config: ExperimentConfig) -> Pipeline:
    steps = []
    if name == RESIDUALIZED:
        steps.append(("residualize", Residualizer(len(OBSERVABLE_BLOCKS["observable_combined"]))))
    if name == "prevalence":
        if task != BINARY:
            raise ValueError("The prevalence baseline is binary only")
        return Pipeline([("model", Prevalence())])
    steps.append(("scale", StandardScaler()))
    if task == BINARY:
        steps.append(("model", LogisticRegression(
            C=float(hyperparameter), penalty="l2", class_weight="balanced",
            max_iter=int(config.probes["max_iter"]), solver="lbfgs")))
    elif task == CONTINUOUS:
        steps.append(("model", Ridge(alpha=float(hyperparameter))))
    else:
        raise ValueError(task)
    return Pipeline(steps)


def hyperparameters(task: str, config: ExperimentConfig) -> list[float]:
    key = "logistic_C" if task == BINARY else "ridge_alpha"
    return [float(v) for v in config.probes[key]]


def scores(pipeline: Pipeline, task: str, X: np.ndarray) -> dict[str, np.ndarray]:
    if task == BINARY:
        raw = np.asarray(pipeline.decision_function(X), dtype=np.float64)
        prob = np.asarray(pipeline.predict_proba(X)[:, 1], dtype=np.float64)
        if not ((prob >= 0) & (prob <= 1)).all():
            raise ValueError("Probability outside [0, 1]")
        return {"score_raw": raw, "probability": prob}
    pred = np.asarray(pipeline.predict(X), dtype=np.float64)
    return {"score_raw": pred, "importance_predicted": pred}


def converged(pipeline: Pipeline) -> bool:
    model = pipeline.named_steps["model"]
    n_iter = getattr(model, "n_iter_", None)
    return True if n_iter is None else bool(np.max(n_iter) < model.max_iter)
