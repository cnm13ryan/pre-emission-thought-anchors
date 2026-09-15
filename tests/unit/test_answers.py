"""Golden answer-normalization fixtures against the pinned upstream behaviour."""

import pytest

from preanchor.labels.importance import NO_ANSWER, legacy_category, normalize_answer
from preanchor.labels.upstream_answers import extract_boxed_answers, normalize_latex

GOLDEN = [
    # (raw answer, canonical category)
    (r"\frac{1}{2}", r"\frac12"),
    (r"\dfrac{1}{2}", r"\frac12"),
    (r"\tfrac{1}{2}", r"\frac12"),
    (r" 42 ", "42"),
    ("42", "42"),
    (r"x^{2}", "x^2"),
    (r"3 \times 4", "3*4"),
    (r"3 \cdot 4", "3*4"),
    (r"50\%", "50"),
    (r"1{,}000", "1000"),
    (r"\pi", "pi"),
    (r"\text{October 30}", "october30"),
    (r"\mathrm{cm}", r"\mathrmcm"),  # Pinned quirk: brace removal precedes the \mathrm rule.
    ("3,14", "3.14"),
    ("ABC", "abc"),
]


@pytest.mark.parametrize("raw,expected", GOLDEN)
def test_golden_normalization(raw, expected):
    assert normalize_answer(raw) == expected
    assert normalize_answer(raw) == normalize_latex(raw)


def test_equivalent_algebraic_forms_normalize_consistently():
    assert normalize_answer(r"\frac{1}{2}") == normalize_answer(r"\dfrac{ 1 }{ 2 }")
    assert normalize_answer(r"x^{2}") == normalize_answer("x^2")
    assert normalize_answer("3 * 4") == normalize_answer(r"3\times4")


def test_braces_are_removed_in_pairs_not_only_closing_braces():
    value = normalize_answer(r"\frac{1}{2}")
    assert "{" not in value and "}" not in value


def test_no_answer_is_explicit():
    assert normalize_answer(None) == NO_ANSWER
    assert normalize_answer("") == NO_ANSWER
    assert normalize_answer("None") == NO_ANSWER
    assert normalize_answer("   ") == NO_ANSWER


def test_boxed_extraction_golden_fixtures():
    assert extract_boxed_answers(r"so \boxed{\frac{1}{2}} done") == [r"\frac{1}{2}"]
    assert extract_boxed_answers(r"\boxed{a} and \boxed{b}") == ["a", "b"]
    assert extract_boxed_answers("no box") == [""]
    assert extract_boxed_answers(r"\boxed{x^{2}}") == ["x^{2}"]


def test_legacy_category_reproduces_triple_normalization():
    # normalize_latex strips one brace level per pass; the pinned path applies three.
    nested = r"\frac{{a}}{{b}}"
    assert legacy_category(nested) == normalize_latex(normalize_latex(normalize_latex(nested)))
    assert legacy_category("") == ""
