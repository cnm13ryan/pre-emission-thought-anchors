"""SPEC §3.11 bootstrap: problem clusters stay intact, every model sees the same replicate,
2,000 replicates under the mvp configuration, deterministic seeds."""

from pathlib import Path

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.ids import derived_seed
from preanchor.probes import bootstrap as B

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def synthetic(n_problems=4, per_traj=12, seed=0):
    rng = np.random.default_rng(seed)
    problem, traj, y, imp = [], [], [], []
    for p in range(n_problems):
        for kind in ("correct", "incorrect"):
            for i in range(per_traj):
                problem.append(1000 + p)
                traj.append(f"{1000 + p}:{kind}")
                y.append(int(i % 4 == 0))
                imp.append(rng.random())
    y = np.array(y)
    score = y + rng.standard_normal(len(y)) * 0.5
    prob = 1 / (1 + np.exp(-score))
    binary = {"activation_plus_observables": {"score": score, "probability": prob},
              "observable_combined": {"score": rng.standard_normal(len(y)),
                                      "probability": rng.random(len(y))}}
    continuous = {"activation_plus_observables": np.array(imp) + rng.standard_normal(len(y)) * .1}
    return dict(problem_id=np.array(problem), trajectory_id=traj, y=y, importance=np.array(imp),
                binary=binary, continuous=continuous, primary="activation_plus_observables",
                baseline="observable_combined")


def test_replicate_keeps_both_trajectories_and_all_sentences_of_a_problem():
    d = synthetic()
    by_problem = {int(p): np.flatnonzero(d["problem_id"] == p)
                  for p in np.unique(d["problem_id"])}
    idx = B.replicate_indices(np.array([1001, 1001, 1003, 1000]), by_problem)
    traj = np.asarray(d["trajectory_id"])[idx]
    assert len(idx) == 4 * 24
    assert sorted(set(traj)) == ["1000:correct", "1000:incorrect", "1001:correct",
                                 "1001:incorrect", "1003:correct", "1003:incorrect"]
    assert (np.asarray(d["problem_id"])[idx] == 1001).sum() == 48   # multiplicity two
    for t in set(traj):
        assert (traj == t).sum() % 12 == 0                            # whole trajectories


def test_samples_problem_ids_not_sentences(config, monkeypatch):
    d = synthetic()
    calls = []
    original = B.replicate_indices

    def spy(sampled, by_problem):
        calls.append(np.array(sampled))
        return original(sampled, by_problem)

    monkeypatch.setattr(B, "replicate_indices", spy)
    out = B.clustered_bootstrap(**d, config=config, split="validation", replicates=25)
    assert len(calls) == 25                       # one draw per replicate, shared by all models
    assert all(len(c) == 4 and set(c) <= {1000, 1001, 1002, 1003} for c in calls)
    assert out["sampling_unit"] == "problem_id" and out["paired"] and out["with_replacement"]
    assert out["n_problems"] == 4 and out["replicates_requested"] == 25
    diff = out["paired_differences"]["activation_plus_observables-minus-observable_combined"]
    assert set(diff) == {"average_precision", "auroc"}
    assert 0 < diff["average_precision"]["n_valid"] <= 25
    assert out["aucpr_gain_over_baseline"] == diff["average_precision"]


def test_default_replicates_are_2000_and_seed_is_derived(config):
    assert B.check_spec(config)[0] == 2000
    assert B.bootstrap_seed(config, "test") == derived_seed(config.seed_root, "bootstrap", "test")
    assert B.bootstrap_seed(config, "test") != B.bootstrap_seed(config, "validation")


def test_deterministic_replay(config):
    d = synthetic()
    a = B.clustered_bootstrap(**d, config=config, split="test", replicates=40)
    b = B.clustered_bootstrap(**d, config=config, split="test", replicates=40)
    assert a == b
    c = B.clustered_bootstrap(**d, config=config, split="validation", replicates=40)
    assert c["sampled_problems_sha256"] != a["sampled_problems_sha256"]
    ap = a["binary"]["activation_plus_observables"]["average_precision"]
    assert ap["low"] <= ap["mean"] <= ap["high"]
