"""SPEC §10.3: hand-computed KL, alpha sensitivities, dissimilar-only selection, fork i+1."""

import math

import pytest

from preanchor.labels.importance import (
    NO_ANSWER,
    canonical_from_forks,
    dissimilar_indices,
    normalize_answer,
    smoothed_kl,
)


def spec_kl(without, with_sentence, alpha):
    """Independent transcription of SPEC §3.6 with K = |union ∪ {<NO_ANSWER>}|."""
    categories = set(without) | set(with_sentence) | {NO_ANSWER}
    k = len(categories)
    total = 0.0
    for a in categories:
        p = (without.count(a) + alpha) / (len(without) + alpha * k)
        q = (with_sentence.count(a) + alpha) / (len(with_sentence) + alpha * k)
        total += p * math.log(p / q)
    return total


def test_two_category_hand_calculation():
    # without = {1: 3, 2: 1}, with = {1: 1, 2: 3}; K = 3 (1, 2, <NO_ANSWER>), alpha = 1.
    without, with_sentence, alpha = ["1", "1", "1", "2"], ["1", "2", "2", "2"], 1.0
    p = {"1": 4 / 7, "2": 2 / 7, NO_ANSWER: 1 / 7}
    q = {"1": 2 / 7, "2": 4 / 7, NO_ANSWER: 1 / 7}
    expected = sum(p[a] * math.log(p[a] / q[a]) for a in p)
    assert smoothed_kl(without, with_sentence, alpha) == pytest.approx(expected)
    assert expected == pytest.approx((4 / 7) * math.log(2) + (2 / 7) * math.log(0.5))


def test_three_category_hand_calculation_with_no_answer():
    # without = {a: 2, NO_ANSWER: 1}, with = {b: 2}; K = 3, alpha = 0.5.
    without, with_sentence, alpha = ["a", "a", NO_ANSWER], ["b", "b"], 0.5
    p = {"a": 2.5 / 4.5, "b": 0.5 / 4.5, NO_ANSWER: 1.5 / 4.5}
    q = {"a": 0.5 / 3.5, "b": 2.5 / 3.5, NO_ANSWER: 0.5 / 3.5}
    expected = sum(p[a] * math.log(p[a] / q[a]) for a in p)
    assert smoothed_kl(without, with_sentence, alpha) == pytest.approx(expected)
    assert abs(sum(p.values()) - 1) < 1e-12 and abs(sum(q.values()) - 1) < 1e-12


@pytest.mark.parametrize("alpha", [1e-9, 0.5, 1.0])
def test_alpha_sensitivities_match_specification(alpha):
    without = ["1", "1", "2", NO_ANSWER, "3"]
    with_sentence = ["1", "3", "3", "4"]
    assert smoothed_kl(without, with_sentence, alpha) == pytest.approx(
        spec_kl(without, with_sentence, alpha), rel=1e-12)


def test_kl_finite_nonnegative_and_zero_for_identical():
    assert smoothed_kl(["a", "a"], ["a", "a"]) == 0
    value = smoothed_kl(["a"] * 100, ["b"] * 100)
    assert math.isfinite(value) and value > 0
    assert smoothed_kl(["a", "b"], ["b", "a"]) == 0


def test_kl_rejects_empty_or_bad_alpha():
    with pytest.raises(ValueError):
        smoothed_kl([], ["a"])
    with pytest.raises(ValueError):
        smoothed_kl(["a"], ["a"], 0)


def test_dissimilar_indices_strictly_below_threshold():
    assert dissimilar_indices([0.79, 0.8, 0.81, 0.0]) == [0, 3]
    with pytest.raises(ValueError, match="Non-finite"):
        dissimilar_indices([0.1, float("nan")])


def test_canonical_uses_only_dissimilar_without_samples():
    forks = {0: [{"answer": "1"}, {"answer": "2"}, {"answer": "2"}],
             1: [{"answer": "1"}]}
    # Only sample 0 (similarity 0.1) is dissimilar; samples 1-2 are similar (0.95).
    result = canonical_from_forks(forks, 0, [0.1, 0.95, 0.95])
    assert (result["dissimilar_count"], result["similar_count"]) == (1, 2)
    assert result["importance_canonical"] == pytest.approx(smoothed_kl(["1"], ["1"]))
    assert result["importance_canonical"] == 0
    # Including the similar samples would make the distributions differ.
    assert smoothed_kl(["1", "2", "2"], ["1"]) > 0


def test_canonical_with_distribution_is_fork_i_plus_one_not_similar_fork_i():
    forks = {0: [{"answer": "x"}, {"answer": "y"}], 1: [{"answer": "z"}, {"answer": "z"}],
             2: [{"answer": "q"}]}
    result = canonical_from_forks(forks, 0, [0.0, 0.99])
    assert result["with_count"] == 2
    assert result["importance_canonical"] == pytest.approx(smoothed_kl(["x"], ["z", "z"]))
    # Pooling the similar fork-0 sample "y" would change the value.
    assert result["importance_canonical"] != pytest.approx(smoothed_kl(["x"], ["z", "z", "y"]))


def test_canonical_without_dissimilar_support_is_null_not_zero():
    forks = {0: [{"answer": "1"}, {"answer": "2"}], 1: [{"answer": "1"}]}
    result = canonical_from_forks(forks, 0, [0.9, 0.85])
    assert result["importance_canonical"] is None
    assert result["support"] == "no_dissimilar_samples"


def test_canonical_requires_aligned_similarities_and_adjacent_fork():
    forks = {0: [{"answer": "1"}], 1: [{"answer": "1"}]}
    with pytest.raises(ValueError, match="One similarity"):
        canonical_from_forks(forks, 0, [])
    with pytest.raises(ValueError, match="Missing adjacent"):
        canonical_from_forks({0: [{"answer": "1"}]}, 0, [0.0])


def test_missing_answers_become_explicit_no_answer_category():
    forks = {0: [{"answer": ""}, {"answer": None}, {"answer": "None"}, {"answer": "7"}],
             1: [{"answer": "7"}]}
    result = canonical_from_forks(forks, 0, [0.0, 0.0, 0.0, 0.0])
    assert result["no_answer_dissimilar_count"] == 3
    assert normalize_answer("None") == NO_ANSWER
