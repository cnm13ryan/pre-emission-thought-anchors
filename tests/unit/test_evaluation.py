"""SPEC §3.11 metric definitions on hand-computable inputs."""

import numpy as np

from preanchor.probes import evaluation as E


def test_reliability_equal_frequency_and_ece():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    rel = E.reliability(y, p, 2)
    assert [b["count"] for b in rel["curve"]] == [4, 4]
    assert rel["curve"][0]["observed_rate"] == 0 and rel["curve"][1]["observed_rate"] == 1
    assert abs(rel["ece"] - (0.5 * 0.25 + 0.5 * 0.25)) < 1e-12


def test_top10_percent_ranking_per_trajectory():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    score = np.arange(20, dtype=float)             # highest score = last row
    traj = ["a"] * 10 + ["b"] * 10
    per = E.per_trajectory_ranking(y, score, traj)
    assert per["a"]["k"] == 1 and per["a"]["precision_top10_percent"] == 0.0
    assert per["b"]["k"] == 1 and per["b"]["precision_top10_percent"] == 1.0
    assert per["b"]["ndcg_top10_percent"] == 1.0 and per["a"]["ndcg_top10_percent"] == 0.0
    summary = E.ranking_summary(per)
    assert summary["precision_top10_percent"] == 0.5 and summary["n_trajectories"] == 2


def test_binary_metrics_handle_a_single_class():
    y = np.zeros(6, dtype=int)
    m = E.binary_metrics(y, np.arange(6.0), np.full(6, 0.2), ["t"] * 6, 3)
    assert m["average_precision"] is None and m["auroc"] is None
    assert abs(m["brier"] - 0.04) < 1e-12 and m["ndcg_top10_percent"] is None
    assert m["n_trajectories_with_positive"] == 0


def test_log_loss_is_clipped():
    assert np.isfinite(E.log_loss(np.array([1, 0]), np.array([0.0, 1.0])))
