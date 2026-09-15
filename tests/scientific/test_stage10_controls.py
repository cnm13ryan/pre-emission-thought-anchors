"""SPEC Stage 10 scientific failure modes, unit level: thresholds from training only,
K=4/K=5 semantics, the replacement-support rule, deterministic quintiles, no forbidden
field in a refit design, the lead-one mapping, stratified permutations and the absence of
any selection path in the Stage 10 code."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.features.observables import PRIMARY_FEATURES
from preanchor.labels.convergence import convergence_idx, eligible
from preanchor.probes import models as M
from preanchor.probes import stage10 as S
from preanchor.probes.dataset import Partition

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


def make_partition(split, problems, n_per_problem=20, seed=0, importance=None):
    rng = np.random.default_rng(seed)
    n = n_per_problem * len(problems)
    problem_id = np.repeat(np.array(problems), n_per_problem)
    if importance is None:
        importance = rng.gamma(1.0, 1.0, n)
    y = (importance >= 1.0).astype(int)
    obs = rng.standard_normal((n, 12))
    ids = [f"{p}:correct:{i}" for i, p in enumerate(problem_id)]
    traj = [f"{p}:correct" for p in problem_id]
    return Partition(split, ids, traj, problem_id, y, np.log1p(importance), importance, obs,
                     ["units/k/x.safetensors"] * n, np.arange(n), 1.0)


# --------------------------------------------------------------------------- thresholds


def test_alternate_thresholds_come_from_training_rows_only(config):
    train = make_partition("train", config.splits.train, seed=1)
    qs = [0.8, 0.9, 0.95]
    thresholds = S.train_thresholds(train, config, qs)
    assert set(thresholds) == {"q80", "q90", "q95"}
    for q in qs:
        assert thresholds[f"q{int(round(q * 100))}"] == pytest.approx(
            float(np.quantile(train.importance_primary, q)))
    assert thresholds["q80"] < thresholds["q90"] < thresholds["q95"]
    # Validation and test importance values cannot influence any threshold: they are never
    # passed. A partition holding a non-training problem is refused outright.
    with pytest.raises(ValueError, match="non-training"):
        S.train_thresholds(make_partition("test", config.splits.test), config, qs)
    mixed = make_partition("train", config.splits.train[:6] + config.splits.test[:1], seed=2)
    with pytest.raises(ValueError, match="non-training"):
        S.train_thresholds(mixed, config, qs)


def test_thresholds_apply_numerically_to_other_splits_without_recomputation(config):
    train = make_partition("train", config.splits.train, seed=3)
    thresholds = S.train_thresholds(train, config, [0.8, 0.9, 0.95])
    test = make_partition("test", config.splits.test, seed=4)
    for name, value in thresholds.items():
        relabelled = S.relabel(test, value)
        assert relabelled.anchor_threshold == value
        assert np.array_equal(relabelled.y, (test.importance_primary >= value).astype(int))
        # Not the test set's own quantile.
        q = float(name[1:]) / 100
        assert value != pytest.approx(float(np.quantile(test.importance_primary, q)))
        # Nothing else changes.
        assert relabelled.sentence_id == test.sentence_id
        assert np.array_equal(relabelled.importance_primary, test.importance_primary)


# --------------------------------------------------------------------------- convergence


def test_k4_and_k5_share_the_extreme_accuracy_definition():
    acc = [0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    k5, k4 = convergence_idx(acc, 5), convergence_idx(acc, 4)
    assert (k5, k4) == (7, 2)          # four extreme forks converge earlier; five need the zeros
    assert all(eligible(i, k4) <= eligible(i, k5) for i in range(len(acc)))  # K=4 is a subset
    boundary = [0.5, 0.995, 0.99, 1.0, 1.0, 0.5]
    assert convergence_idx(boundary, 4) == 1 and convergence_idx(boundary, 5) is None
    # 0.98 is not extreme under either rule.
    assert convergence_idx([0.98] * 8, 4) is None and convergence_idx([0.98] * 8, 5) is None
    with pytest.raises(ValueError):
        convergence_idx(acc, 0)


# --------------------------------------------------------------------------- replacement


def make_rows(dissimilar, sample, kinds=None, idx=None, entropy=None):
    n = len(dissimilar)
    p = make_partition("validation", [330], n_per_problem=n, seed=5)
    return S.Rows(p, np.array(kinds or ["correct"] * n), np.array(idx or list(range(n))),
                  np.arange(n) * 3, np.arange(n) * 3 + 1,
                  np.array(entropy or np.linspace(0, 2, n)), np.ones(n, dtype=bool),
                  np.array(dissimilar), np.array(sample))


def test_replacement_support_rule_is_fraction_or_count():
    spec = {"replacement_minimum_fraction": 0.10, "replacement_minimum_count": 10}
    rows = make_rows([10, 9, 3, 0, 10, 100], [100, 100, 20, 100, 200, 100])
    supported, reasons = S.replacement_support(rows, spec)
    # 10/100 passes both; 9/100 fails both; 3/20 = 0.15 passes by fraction; 0 fails;
    # 10/200 = 0.05 passes by count; 100/100 passes.
    assert supported.tolist() == [True, False, True, False, True, True]
    assert reasons["supported"] == 4 and reasons["excluded"] == 2
    assert reasons["excluded_both_below_minimum"] == 2
    assert reasons["excluded_fraction_below_minimum"] == 3   # 9/100, 0/100, 10/200
    assert reasons["excluded_count_below_minimum"] == 3      # 9, 3, 0
    with pytest.raises(ValueError, match="sample_count"):
        S.replacement_support(make_rows([1, 1], [100, 0]), spec)


# --------------------------------------------------------------------------- strata


def test_quintile_assignment_is_deterministic_and_train_derived():
    train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    cuts = S.quintile_cuts(train, 5)
    assert cuts == [float(np.quantile(train, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    assigned = S.assign_quintile(np.array([-1, 0, 1.7, 1.9, 5.3, 5.5, 7.3, 100.0]), cuts)
    assert assigned.tolist() == [1, 1, 1, 2, 3, 4, 5, 5]
    assert S.assign_quintile(np.array(cuts), cuts).tolist() == [1, 2, 3, 4]   # cut belongs below
    assert np.array_equal(S.assign_quintile(train, cuts), S.assign_quintile(train, cuts))
    # Values outside the training range still land in an end bin; cut points ignore them.
    assert S.quintile_cuts(train, 5) == S.quintile_cuts(np.concatenate([train]), 5)


def test_subgroups_partition_rows_without_duplication_or_loss():
    rows = make_rows([50] * 6, [100] * 6, kinds=["correct", "incorrect", "correct",
                                                  "incorrect", "incorrect", "correct"])
    correct = np.flatnonzero(rows.trace_kind == "correct")
    incorrect = np.flatnonzero(rows.trace_kind == "incorrect")
    assert sorted(correct.tolist() + incorrect.tolist()) == list(range(6))
    assert not set(correct) & set(incorrect)
    empty = S.take(rows.partition, np.array([], dtype=int))
    out = S.evaluate_group(analysis="trajectory_subgroup", variant="incorrect", split="test",
                           source=S.FROZEN, partition=empty, binary={}, meta={},
                           config=load_config(ROOT / "configs/mvp.yaml"), bootstrap=False)
    assert len(out) == 1 and out[0]["metric"] == "empty_group" and out[0]["value"] is None
    assert '"undefined": "empty_group"' in out[0]["detail"]


def test_single_class_group_stays_explicitly_undefined(config):
    p = make_partition("test", [6998], n_per_problem=10, seed=6)
    p = S.relabel(p, 1e9)   # no anchors at all
    score = np.random.default_rng(0).random(10)
    binary = {S.PRIMARY_MODEL: {"score": score, "probability": score},
              S.BASELINE: {"score": score[::-1], "probability": score[::-1]}}
    meta = {k: {"layer": 44 if k == S.PRIMARY_MODEL else None, "hyperparameter": 0.1}
            for k in binary}
    out = S.evaluate_group(analysis="x", variant="y", split="test", source=S.FROZEN,
                           partition=p, binary=binary, meta=meta, config=config)
    ap = [r for r in out if r["metric"] == "average_precision" and r["comparison"] is None]
    assert ap and all(r["value"] is None and "single_class" in r["detail"] for r in ap)
    paired = [r for r in out if r["comparison"]]
    assert paired and all(r["value"] is None for r in paired)
    assert all(r["boot_low"] is None for r in out)


def test_no_forbidden_field_enters_a_refit_design(config):
    excluded = set(config.features["excluded"])
    for block in M.OBSERVABLE_BLOCKS.values():
        assert not set(block) & excluded
    assert sorted(M.OBSERVABLE_BLOCKS["observable_combined"]) == sorted(PRIMARY_FEATURES)
    p = make_partition("train", [1591], n_per_problem=8, seed=7)
    site = np.zeros((8, 5), dtype=np.float32)
    for family in S.REFIT_FAMILIES:
        X = M.design(family, p, site if M.uses_activation(family) else None)
        expected = (5 if M.uses_activation(family) else 0) + (
            12 if family != "activation_only" else 0)
        assert X.shape == (8, expected)
    # The stratifiers (trace_kind, eligible_k4, support counts, entropy bins) are not
    # attributes of the Partition the design is built from.
    assert not {"trace_kind", "eligible_k4", "dissimilar_count", "position_quintile"} & set(
        Partition.__dataclass_fields__)


# --------------------------------------------------------------------------- lead one


def lead_fixture(n=5, trajectory="330:correct"):
    p = make_partition("validation", [330], n_per_problem=n, seed=8)
    p.sentence_id[:] = [f"{trajectory}:{i}" for i in range(n)]
    p.trajectory_id[:] = [trajectory] * n
    starts = np.array([3, 10, 17, 25, 31])[:n]
    rows = S.Rows(p, np.array(["correct"] * n), np.arange(n), starts - 1, starts,
                  np.linspace(0, 1, n), np.ones(n, dtype=bool), np.full(n, 50), np.full(n, 100))
    table = {}
    for i in range(n):
        table[f"{trajectory}:{i}"] = {
            "sentence_id": f"{trajectory}:{i}", "trajectory_id": trajectory, "sentence_idx": i,
            "boundary_token_idx": int(starts[i] - 1), "target_token_start": int(starts[i]),
            "shard_path": "units/k/330_correct.safetensors", "row_idx": i,
            **{c: float(i + k) for k, c in enumerate(PRIMARY_FEATURES)}}
    return rows, table


def test_lead_one_source_is_the_boundary_before_the_previous_sentence():
    rows, table = lead_fixture()
    available, source, mapping = S.lead_one_source(rows, table)
    assert available.tolist() == [False, True, True, True, True]
    assert mapping[0]["unavailable_reason"] == "first_sentence_has_no_predecessor"
    assert mapping[0]["feature_sentence_id"] is None
    for k, m in enumerate(mapping[1:], start=1):
        assert m["feature_sentence_id"] == f"330:correct:{k - 1}"
        assert m["feature_sentence_idx"] == k - 1
        src = table[m["feature_sentence_id"]]
        # No token of S_(i-1) or S_i is visible: boundary < start(S_(i-1)) <= start(S_i).
        assert m["feature_boundary_token_idx"] == src["boundary_token_idx"]
        assert src["boundary_token_idx"] < src["target_token_start"] <= \
            int(rows.target_token_start[k])
        assert src["boundary_token_idx"] < int(rows.boundary_token_idx[k])
    # Source rows are S_(i-1); labels stay those of S_i; identity is unchanged.
    assert source.sentence_id == [f"330:correct:{i}" for i in range(4)]
    assert np.array_equal(source.row_idx, np.arange(4))
    assert source.trajectory_id == ["330:correct"] * 4
    target = S.take(rows.partition, np.flatnonzero(available))
    assert target.sentence_id == [f"330:correct:{i}" for i in range(1, 5)]
    assert np.array_equal(target.y, rows.partition.y[1:])
    # Observables at the source boundary are those of S_(i-1), not of S_i.
    assert source.observables[0, 0] == table["330:correct:0"]["sentence_idx"]


def test_lead_one_never_crosses_trajectories_and_fails_on_a_late_boundary():
    rows, table = lead_fixture()
    # The predecessor row of sentence 2 is removed; a same-index row in another trajectory
    # must not be used.
    other = dict(table["330:correct:1"], trajectory_id="330:incorrect",
                 sentence_id="330:incorrect:1")
    table.pop("330:correct:1")
    table["330:incorrect:1"] = other
    available, source, mapping = S.lead_one_source(rows, table)
    assert not available[2] and mapping[2]["unavailable_reason"] == "predecessor_row_missing"
    assert available[1] and mapping[1]["feature_sentence_id"] == "330:correct:0"
    assert all(s.startswith("330:correct:") for s in source.sentence_id)
    # A stored predecessor whose boundary is not before its own target start is refused.
    rows, table = lead_fixture()
    table["330:correct:1"]["boundary_token_idx"] = table["330:correct:1"]["target_token_start"]
    with pytest.raises(ValueError, match="does not precede"):
        S.lead_one_source(rows, table)
    rows, table = lead_fixture()
    table["330:correct:1"]["boundary_token_idx"] = int(rows.boundary_token_idx[2])
    table["330:correct:1"]["target_token_start"] = int(rows.boundary_token_idx[2]) + 1
    with pytest.raises(ValueError, match="not earlier"):
        S.lead_one_source(rows, table)


# --------------------------------------------------------------------------- permutations


def test_permutations_stay_inside_strata_and_keep_single_class_strata(config):
    rng = np.random.default_rng(0)
    problem = np.repeat([1591, 2137, 4682], 30)
    kind = np.tile(np.repeat(["correct", "incorrect"], 15), 3)
    quintile = np.tile(np.arange(1, 6), 18)
    strata = S.strata_of(problem, kind, quintile)
    y = (rng.random(90) < 0.3).astype(int)
    single = strata == "1591:correct:1"
    y[single] = 0                                   # one single-class stratum
    permuted, counts = S.permute_within_strata(y, strata, np.random.default_rng(1))
    assert counts["n_strata"] == 30
    assert counts["n_single_class_strata"] >= 1 and counts["n_permutable_strata"] == \
        counts["n_strata"] - counts["n_single_class_strata"]
    assert counts["n_rows_permuted"] + counts["n_rows_fixed"] == 90
    for key in set(strata):
        idx = strata == key
        assert permuted[idx].sum() == y[idx].sum()      # class counts preserved per stratum
        if len(np.unique(y[idx])) < 2:
            assert np.array_equal(permuted[idx], y[idx])
    assert not np.array_equal(permuted, y)
    again, _ = S.permute_within_strata(y, strata, np.random.default_rng(1))
    assert np.array_equal(permuted, again)              # deterministic given the seed
    seeds = [S.permutation_seed(config, k) for k in range(config.evaluation["permutation_count"])]
    assert len(seeds) == 100 and len(set(seeds)) == 100
    assert seeds == [S.permutation_seed(config, k) for k in range(100)]


# --------------------------------------------------------------------------- selection


def test_stage10_has_no_selection_or_test_driven_choice_path():
    source = (ROOT / "src/preanchor/probes/stage10.py").read_text()
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    assert not {"selection_table", "mean_cv", "_best", "layer_curve", "hyperparameters",
                "fit_platt", "freeze", "evaluate_test"} & imported
    assert "frozen_selection.pending" not in source
    # Refits take the hyperparameter from the frozen record only.
    assert "hyperparameters(" not in source
    assert "M.make_pipeline(family, M.BINARY, meta[\"hyperparameter\"], config)" in source
    # Every unit key is bound to the Stage 9 inputs and the code commit.
    assert "input_hashes=dict(manifest[\"input_hashes\"])" in source
    assert math.isclose(S.THRESHOLD_QUANTILE, 0.9)
