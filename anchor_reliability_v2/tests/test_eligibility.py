"""Eligibility: v2 rules must not depend on anything at or after S_i (leakage tests)."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import eligibility as E  # noqa: E402


def test_converged_pre_needs_full_past_window():
    acc = [0.5, 1, 1, 1, 1, 1, 0.5, 0.5]
    assert E.converged_pre(acc) == [False] * 5 + [True] * 3
    # MVP rule fires 4 sentences earlier because it looks ahead
    assert E.converged_mvp(acc) == [False] + [True] * 7


def test_converged_pre_missing_fork_breaks_window():
    assert not any(E.converged_pre([1, 1, None, 1, 1, 1, 1]))


def test_boxed_before_vs_mvp_final_answer():
    texts = ["a", "so \\boxed{3}", "check", "\\boxed{3} again", "done"]
    assert E.boxed_before(texts) == [False, False, True, True, True]
    assert E.final_answer_mvp(texts) == [False, False, False, True, True]


def test_v2_rules_invariant_to_future_mutation():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(6, 30)
        acc = [rng.choice([0.0, 1.0, 0.5, None]) for _ in range(n)]
        texts = [rng.choice(["x", "\\boxed{1}"]) for _ in range(n)]
        i = rng.randrange(n)
        fut_acc = acc[: i + 1] + [rng.choice([0.0, 1.0, 0.5, None]) for _ in range(n - i - 1)]
        fut_txt = texts[:i] + [rng.choice(["x", "\\boxed{1}"]) for _ in range(n - i)]
        assert E.converged_pre(acc)[i] == E.converged_pre(fut_acc)[i]
        assert E.boxed_before(texts)[i] == E.boxed_before(fut_txt)[i]


def test_mvp_rules_do_leak_future_information():
    # documents why the MVP rules were replaced: a future change flips sentence 0
    assert E.converged_mvp([1, 1, 1, 1, 1])[0] is True
    assert E.converged_mvp([1, 1, 1, 1, 0.5])[0] is False
    assert E.final_answer_mvp(["x", "\\boxed{2}"])[0] is False
    assert E.final_answer_mvp(["\\boxed{2}", "x"])[0] is True
