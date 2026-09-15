"""Legacy adapter parity with the pinned repository on a frozen fixture.

The oracle is the pinned source itself: the two functions are extracted by name from
``vendor/thought-anchors/analyze_rollouts.py`` at the locked commit and executed with
their real dependencies (``math``, ``defaultdict``, ``numpy`` and the pinned
``normalize_answer``). Nothing in the submodule is modified.
"""

import ast
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from preanchor.labels.importance import legacy_from_forks, legacy_kl_divergence
from preanchor.labels.upstream_answers import normalize_latex

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "vendor/thought-anchors/analyze_rollouts.py"


def pinned_functions():
    tree = ast.parse(UPSTREAM.read_text())
    wanted = {"calculate_kl_divergence", "calculate_counterfactual_importance_kl"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name in wanted], type_ignores=[])
    namespace = {"math": math, "defaultdict": defaultdict, "np": np,
                 "normalize_answer": lambda a, use_sympy=False: normalize_latex(a)}
    exec(compile(module, str(UPSTREAM), "exec"), namespace)  # noqa: S102 - pinned source
    return namespace["calculate_kl_divergence"], namespace["calculate_counterfactual_importance_kl"]


def test_pinned_module_defaults_are_the_ones_the_adapter_reproduces():
    source = UPSTREAM.read_text()
    # The last module-level assignment wins: alpha 1.0 (line 72 overrides line 70).
    assignments = [line.strip() for line in source.splitlines()
                   if line.startswith("smoothing = ")]
    assert assignments[-1] == "smoothing = 1.0"
    assert "use_similar_chunks=True" in source


FIXTURE = {
    # fork 0 samples: (chunk_resampled similarity to chunk_removed, answer)
    0: [(0.10, "1"), (0.20, "2"), (0.95, "1"), (0.30, ""), (0.99, "3"), (0.50, "None"),
        (0.79, r"\frac{1}{2}"), (0.81, "2"), (0.05, "2")],
    1: [(1.0, "1"), (1.0, "2"), (1.0, "2"), (1.0, ""), (1.0, "x")],
}


def make_forks():
    forks, sims = {}, {}
    for idx, rows in FIXTURE.items():
        forks[idx] = [{"chunk_removed": "orig", "chunk_resampled": f"r{idx}_{i}",
                       "answer": answer, "is_correct": answer == "1"}
                      for i, (_, answer) in enumerate(rows)]
        sims[idx] = [s for s, _ in rows]
    return forks, sims


def pinned_oracle(alpha=1.0, use_similar_chunks=True):
    _, cf_kl = pinned_functions()
    forks, sims = make_forks()
    # Reproduce analyze_problem's chunk_info construction: drop ""/"None" answers,
    # normalize once at load time, and build the text-keyed embedding cache.
    chunk_info, cache = {}, {}
    for idx, rows in forks.items():
        chunk_info[idx] = []
        for i, sol in enumerate(rows):
            if sol["answer"] != "" and sol["answer"] != "None":
                chunk_info[idx].append({**sol, "answer": normalize_latex(sol["answer"])})
            value = sims[idx][i]
            cache[sol["chunk_resampled"]] = np.array([value, math.sqrt(1 - value ** 2)])
    cache["orig"] = np.array([1.0, 0.0])  # cosine(orig, r) == sims value by construction
    args = types.SimpleNamespace(similarity_threshold=0.8, use_similar_chunks=use_similar_chunks,
                                 use_prob_true=False, global_vocab=None, laplace_alpha=alpha)
    return cf_kl(0, chunk_info, cache, {}, None, args)


@pytest.mark.parametrize("alpha", [1.0, 1e-9, 0.5])
def test_legacy_adapter_matches_pinned_repository(alpha):
    forks, sims = make_forks()
    ours = legacy_from_forks(forks, 0, sims[0], alpha=alpha)
    theirs = pinned_oracle(alpha=alpha)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-15)
    assert ours > 0


def test_legacy_adapter_reproduces_similar_chunk_pooling_flag():
    forks, sims = make_forks()
    pooled = legacy_from_forks(forks, 0, sims[0], alpha=1.0, use_similar_chunks=True)
    unpooled = legacy_from_forks(forks, 0, sims[0], alpha=1.0, use_similar_chunks=False)
    assert pooled == pytest.approx(pinned_oracle(use_similar_chunks=True))
    assert unpooled == pytest.approx(pinned_oracle(use_similar_chunks=False))
    assert pooled != pytest.approx(unpooled)


def test_legacy_kl_matches_pinned_kl_on_frozen_lists():
    kl, _ = pinned_functions()
    a = ["1", "1", "2", r"\frac{1}{2}", ""]
    b = ["1", "2", "2", "2", "x"]
    for alpha in (1.0, 0.5, 1e-9):
        theirs = kl([{"answer": x} for x in a], [{"answer": x} for x in b], alpha=alpha)
        assert legacy_kl_divergence(a, b, alpha) == pytest.approx(theirs, rel=1e-12)
    # Frozen value for alpha = 1.0 so a future regression is visible without the oracle.
    assert legacy_kl_divergence(a, b, 1.0) == pytest.approx(0.3205155897104657, rel=1e-9)


def test_legacy_returns_zero_without_support_unlike_canonical_null():
    forks, sims = make_forks()
    assert legacy_from_forks({0: forks[0]}, 0, sims[0]) == 0.0
    assert legacy_from_forks(forks, 0, [0.99] * len(forks[0])) == 0.0
