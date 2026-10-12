"""Unit tests: answer parsing, score mathematics, agreement statistics, decision rule."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2 import reliability as R  # noqa: E402
from arv2 import scores as S  # noqa: E402

# ------------------------------------------------------------------ parsing


def test_v1_takes_first_box_and_accepts_unterminated():
    assert A.extract_boxed_answers("x \\boxed{3} then \\boxed{5}")[0] == "3"
    assert A.extract_boxed_answers("trunc \\boxed{12")[0] == "12"     # documented defect
    assert A.answer_v1_from_text("no box") == "<NO_ANSWER>"


def test_v2_takes_last_box_after_think():
    text = "<think>try \\boxed{3}</think> so the answer is \\boxed{5}"
    assert A.parse_v2(text) == A.Parsed("answered", "5")
    assert A.parse_v2("a \\boxed{3} b \\boxed{\\frac{1}{2}}").answer == "\\frac12"


def test_v2_ignores_box_inside_think_when_answer_region_exists():
    text = "\\boxed{7} reasoning</think> final \\boxed{8}"
    assert A.parse_v2(text).answer == "8"


def test_v2_unterminated_and_truncation():
    assert A.parse_v2("work \\boxed{12").status == "unterminated_box"
    assert A.parse_v2("work work", finish_reason="length").status == "length_truncated"
    assert A.parse_v2("work work", token_count=16384).status == "length_truncated"
    # a complete answer wins over the truncation flag
    assert A.parse_v2("\\boxed{4}", finish_reason="length").status == "answered"


def test_v2_unboxed_stated_and_no_answer():
    assert A.parse_v2("... so\n\n**Answer:** 6.40").answer == "6.4"
    assert A.parse_v2("**Answer:** 30.8 units.").answer == "30.8"
    assert A.parse_v2("I am still thinking about it and").status == "no_answer"
    assert A.parse_v2("").status == "empty" and A.parse_v2(None).status == "empty"


def test_normalize_v2_is_idempotent_and_canonical():
    for raw in ["6.40", "1,000", "\\dfrac{1}{2}", "{{3}}", "  12. ", "+5", "0.50"]:
        once = A.normalize_v2(raw)
        assert A.normalize_v2(once) == once
    assert A.normalize_v2("1,000") == "1000"
    assert A.normalize_latex("1,000") == "1.000"         # the v1 defect being repaired
    assert A.normalize_v2("6.40") == "6.4"
    assert A.normalize_v2("{{3}}") == "3"


def test_category_v2_handling():
    assert A.category_v2(A.Parsed("length_truncated", None)) == A.TRUNCATED
    assert A.category_v2(A.Parsed("no_answer", None)) == A.NON_ANSWER
    assert A.category_v2(A.Parsed("unboxed_stated", "4")) == "4"
    assert A.category_v2(A.Parsed("unboxed_stated", "4"), include_unboxed=False) == A.NON_ANSWER


# ------------------------------------------------------------------ scores


def test_kl_reproduces_mvp_rare_answer_blowup():
    without = ["a"] * 94 + ["b"] * 6           # b unseen on the other side
    with_ = ["a"] * 100
    value = S.kl_smoothed(without, with_, 1e-9)
    contrib = S.kl_contributions(without, with_, 1e-9)
    assert contrib["b"] > 1.3                   # the review's ~1.35 nats from 6% unseen
    assert value > 1.25 and contrib["b"] / value > 0.95
    assert S.kl_smoothed(without, with_, 0.5) < 0.2


def test_tv_jsd_bounds_and_identities():
    a, b = ["x"] * 50 + ["y"] * 50, ["x"] * 100
    assert S.total_variation(a, a) == 0
    assert S.total_variation(["x"], ["y"]) == 1
    assert math.isclose(S.total_variation(a, b), 0.5)
    assert S.jensen_shannon(a, a) == 0
    assert math.isclose(S.jensen_shannon(["x"], ["y"]), 1.0)
    assert 0 < S.jensen_shannon(a, b) < 1
    # rare unseen answer: TV contribution is bounded by its own mass
    assert math.isclose(S.total_variation(["a"] * 94 + ["b"] * 6, ["a"] * 100), 0.06)


def test_delta_accuracy_and_null_floor():
    assert S.delta_accuracy([True, False], [True, True]) == 0.5
    floor = S.tv_null_expectation({"a": 0.9, "b": 0.1}, 50, 100, draws=4000, seed=1)
    assert 0.02 < floor < 0.08                 # finite-sample floor is far from zero
    with pytest.raises(ValueError):
        S.total_variation([], ["a"])


# ------------------------------------------------------------------ reliability


def test_icc_kappa_spearman_reference_values():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    assert math.isclose(R.icc_2_1(x, x), 1.0)
    assert abs(R.icc_2_1(x, rng.normal(size=500))) < 0.1
    assert R.icc_2_1(x, x + 1.0) < R.spearman(x, x + 1.0)   # absolute agreement penalises bias
    assert R.cohen_kappa(x > 1, x > 1) == 1.0
    assert abs(R.cohen_kappa(x > 1, rng.normal(size=500) > 1)) < 0.15
    assert math.isnan(R.spearman([1, 1, 1], [1, 2, 3]))


def test_icc_matches_shrout_fleiss_textbook_example():
    # Shrout & Fleiss (1979) Table 2, judges 1 and 2 only; ICC(2,1) by direct formula.
    j1 = np.array([9, 6, 8, 7, 10, 6.])
    j2 = np.array([2, 1, 4, 1, 5, 2.])
    data = np.column_stack([j1, j2])
    n, k = data.shape
    g = data.mean()
    msr = k * ((data.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(0) - g) ** 2).sum() / (k - 1)
    mse = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + g) ** 2).sum() \
        / ((n - 1) * (k - 1))
    expected = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    assert math.isclose(R.icc_2_1(j1, j2), expected)


def test_spearman_brown():
    assert math.isclose(R.spearman_brown(0.5, 2), 2 / 3)
    assert math.isclose(R.spearman_brown(0.6, 1), 0.6)


def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(5), 10)
    seen = []
    R.cluster_bootstrap(groups, lambda rows: seen.append(rows) or 0.0, reps=20, seed=3)
    for rows in seen:
        assert len(rows) == 50 and all(np.sum(rows == r) for r in rows)
        # every resampled group appears in full
        picked = groups[rows]
        for g in np.unique(picked):
            assert np.sum(picked == g) % 10 == 0


@pytest.mark.parametrize("lo,hi,expected", [(0.8, 0.9, "PASS"), (0.5, 0.7, "FAIL"),
                                            (0.7, 0.8, "INCONCLUSIVE"),
                                            (float("nan"), 0.9, "INCONCLUSIVE")])
def test_three_way(lo, hi, expected):
    assert R.three_way(lo, hi, 0.75) == expected
