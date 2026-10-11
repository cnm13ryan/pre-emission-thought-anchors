"""Tests for the shared v3 follow-up canonicaliser (EXPLORATORY)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2.canon import canonical_answer as C  # noqa: E402
from arv2.canon import category_canonical, raw_surface  # noqa: E402


@pytest.mark.parametrize("a,b", [
    ("0.5", "\\frac{1}{2}"), ("\\dfrac{1}{2}", "1/2"), ("\\frac12", ".5"),
    ("1,000", "1000"), ("1{,}000", "1000"), ("6.40", "6.4"), ("+3", "3"),
    ("-\\frac{5}{12}", "\\frac{-5}{12}"), ("-\\frac{5}{12}", "-5/12"),
    ("6.17\\%", "6.17"), ("50%", "50"), ("30^\\circ", "30"), ("30^{\\circ}", "30"),
    ("5 \\text{ cm}", "5"), ("12\\text{ units}", "12"), ("x = 5", "5"),
    ("\\$120", "120"), ("$120$", "120"), ("\\frac{10}{4}", "\\frac{5}{2}"),
    ("2\\sqrt{3}", "\\sqrt{12}"), ("\\frac{\\sqrt{3}}{2}", "\\frac{1}{2}\\sqrt{3}"),
    ("\\left(\\frac{1}{5}\\right)", "\\frac15"), ("4.", "4"),
])
def test_equivalent(a, b):
    assert C(a) == C(b)


@pytest.mark.parametrize("a,b", [
    ("6.17", "6.18"), ("\\frac{5}{12}", "-\\frac{5}{12}"), ("\\frac{53}{703125}", "\\frac{5}{3703125}"),
    ("429", "429.624"), ("120", "384"), ("\\sqrt{2}", "1.41421356"),
])
def test_distinct(a, b):
    assert C(a) != C(b)


def test_keys_and_specials():
    assert C("\\frac{1}{2}") == "q:1/2"
    assert C("88572") == "q:88572"
    assert C(A.TRUNCATED) == A.TRUNCATED and C(A.NON_ANSWER) == A.NON_ANSWER
    assert C("\\pi").startswith("r:3.14159")
    assert C("6(c+\\frac1c)^6").startswith("e:")
    assert C("\\sin x") != C("x")          # unit stripping does not eat "in"


def test_raw_surface_matches_parse_v2():
    text = "think \\boxed{3}</think> So \\boxed{\\dfrac{1}{2}} done"
    st, raw = raw_surface(text, finish_reason="stop")
    assert (st, raw) == ("answered", "\\dfrac{1}{2}")
    assert A.normalize_v2(raw) == A.parse_v2(text).answer
    assert category_canonical(st, raw) == "q:1/2"
    st, raw = raw_surface("no answer here", finish_reason="length")
    assert st == "length_truncated" and category_canonical(st, raw) == A.TRUNCATED
    st, raw = raw_surface("blah\n**Final Answer:** 12 units", finish_reason="stop")
    assert st == "unboxed_stated" and category_canonical(st, raw) == "q:12"


@pytest.mark.parametrize("a,b", [
    ("-\\frac{1}{2^{2013}\\cdot 4027}", "-\\frac{1}{4027\\cdot 2^{2013}}"),
    ("\\frac{1}{2014!}", "\\frac{1}{2014 \\cdot 2013!}"),
    ("\\left(\\frac{9}{8}\\right)^{2007}", "(9/8)^{2007}"),
    ("\\frac{1}{2014\\cdot4027}", "\\frac{1}{8110378}"),
])
def test_nested_and_large(a, b):
    assert C(a) == C(b) and not C(a).startswith("s:")
