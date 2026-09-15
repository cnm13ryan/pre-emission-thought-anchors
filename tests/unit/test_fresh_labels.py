import math

import pytest

from preanchor.labels.fresh import answer_category, canonical_kl, legacy_kl


def test_answer_extraction_uses_first_box_not_whole_reasoning():
    assert answer_category(r"Reasoning. Answer: \boxed{\frac{1}{2}} and \boxed{3}") == r"\frac12"
    assert answer_category("No boxed answer.") == "<NO_ANSWER>"
    assert answer_category(r"Partial \boxed{42") == "42"  # Pinned upstream behavior.


def test_union_normalization_hand_calculation():
    # Union {a,b,NO_ANSWER}; disjoint supports and unequal sample counts.
    p = [3 / 5, 1 / 5, 1 / 5]
    q = [1 / 6, 4 / 6, 1 / 6]
    expected = sum(a * math.log(a / b) for a, b in zip(p, q, strict=True))
    assert canonical_kl(["a", "a"], ["b", "b", "b"], 1) == pytest.approx(expected)
    assert canonical_kl(["a"], ["a"]) == 0
    assert canonical_kl(["<NO_ANSWER>"], ["b"]) > 0
    assert legacy_kl(["<NO_ANSWER>"], ["b"]) == 0
