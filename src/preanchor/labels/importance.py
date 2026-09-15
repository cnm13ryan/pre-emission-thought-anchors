"""Counterfactual importance: frozen canonical estimand and the pinned legacy adapter.

Canonical (SPEC §3.6): the "without" distribution uses only fork-i samples whose
replacement sentence is dissimilar to S_i (cosine < 0.8, pinned MiniLM); the "with"
distribution uses every fork-(i+1) sample. Categories are the union of normalized
answers including ``<NO_ANSWER>``; both distributions are smoothed with the same
category count K and I = sum_a p(a) log(p(a)/q(a)).

Legacy: the exact arithmetic of the pinned ``analyze_rollouts.py`` at
b53ed8c75d3f6112f68adfaec9a13d4d708c442e (``calculate_counterfactual_importance_kl``
with its module defaults: ``smoothing = 1.0``, ``use_similar_chunks=True``,
``use_prob_true=False``, local vocabulary).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from preanchor.labels.upstream_answers import normalize_latex

NO_ANSWER = "<NO_ANSWER>"
# The pinned repository drops samples whose answer field is "" or the string "None".
LEGACY_MISSING = {"", "None"}


def normalize_answer(value: object) -> str:
    """Canonical answer category: pinned LaTeX normalization; missing -> <NO_ANSWER>."""
    if value is None or not isinstance(value, str) or value in LEGACY_MISSING:
        return NO_ANSWER
    return normalize_latex(value) or NO_ANSWER


def smoothed_kl(without: Sequence[str], with_sentence: Sequence[str],
                alpha: float = 1e-9) -> float:
    """SPEC §3.6 smoothed KL over the union of categories including <NO_ANSWER>.

    Inputs are already-normalized category strings (see ``normalize_answer``).
    """
    if not without or not with_sentence:
        raise ValueError("Both answer distributions must be nonempty")
    if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0):
        raise ValueError("Smoothing alpha must be finite and positive")
    p_counts, q_counts = Counter(without), Counter(with_sentence)
    categories = sorted(set(p_counts) | set(q_counts) | {NO_ANSWER})
    k = len(categories)
    n_without, n_with = len(without), len(with_sentence)
    value = 0.0
    for category in categories:
        p = (p_counts[category] + alpha) / (n_without + alpha * k)
        q = (q_counts[category] + alpha) / (n_with + alpha * k)
        value += p * math.log(p / q)
    if not math.isfinite(value):
        raise ValueError("Non-finite KL")
    # KL is nonnegative; only floating-point rounding can produce a tiny negative.
    return value if value > 0 else 0.0


def dissimilar_indices(similarities: Sequence[float], threshold: float = 0.8) -> list[int]:
    """Indices whose replacement similarity is strictly below the threshold."""
    out = []
    for index, value in enumerate(similarities):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            raise ValueError(f"Non-finite similarity at sample {index}")
        if value < threshold:
            out.append(index)
    return out


def canonical_from_forks(forks: dict[int, list[dict]], sentence_idx: int,
                         similarities: Sequence[float], *, alpha: float = 1e-9,
                         threshold: float = 0.8) -> dict:
    """Canonical importance for S_i from fork i (filtered) and fork i+1 (all samples).

    ``similarities`` aligns with ``forks[sentence_idx]``: cosine similarity between each
    sample's ``chunk_removed`` (the original S_i) and its ``chunk_resampled``.
    """
    current = forks.get(sentence_idx) or []
    following = forks.get(sentence_idx + 1) or []
    if len(similarities) != len(current):
        raise ValueError("One similarity per fork-i sample is required")
    if not current or not following:
        raise ValueError(f"Missing adjacent fork support at sentence {sentence_idx}")
    keep = dissimilar_indices(similarities, threshold)
    without = [normalize_answer(current[i].get("answer")) for i in keep]
    with_sentence = [normalize_answer(row.get("answer")) for row in following]
    result = {"sample_count": len(current), "dissimilar_count": len(keep),
              "similar_count": len(current) - len(keep),
              "no_answer_dissimilar_count": sum(a == NO_ANSWER for a in without),
              "with_count": len(with_sentence)}
    if not without:
        result.update({"importance_canonical": None, "support": "no_dissimilar_samples"})
        return result
    result.update({"importance_canonical": smoothed_kl(without, with_sentence, alpha),
                   "support": "ok"})
    return result


def legacy_category(answer: str) -> str:
    """Pinned normalization is applied three times along the legacy call path.

    ``analyze_problem`` normalizes when loading ``chunk_info``; the counterfactual
    routine normalizes again (dropping empties); ``calculate_kl_divergence`` normalizes
    a third time (dropping empties). ``normalize_latex`` is not idempotent for nested
    braces, so the repetition is reproduced exactly. Returns "" when dropped.
    """
    first = normalize_latex(answer)
    second = normalize_latex(first)
    if not second:
        return ""
    return normalize_latex(second)


def legacy_kl_divergence(answers_1: Sequence[str], answers_2: Sequence[str],
                         alpha: float) -> float:
    """Pinned ``calculate_kl_divergence`` (answer distributions, local vocabulary)."""
    counts_1 = Counter(a for a in (legacy_category(x) for x in answers_1) if a)
    counts_2 = Counter(a for a in (legacy_category(x) for x in answers_2) if a)
    if not counts_1 or not counts_2:
        return 0.0
    vocabulary = set(counts_1) | set(counts_2)
    total_1, total_2 = sum(counts_1.values()), sum(counts_2.values())
    smoothed_1, smoothed_2 = total_1 + alpha * len(vocabulary), total_2 + alpha * len(vocabulary)
    value = 0.0
    for answer in vocabulary:
        p = (counts_1[answer] + alpha) / smoothed_1
        q = (counts_2[answer] + alpha) / smoothed_2
        value += (counts_1[answer] / total_1) * math.log(p / q)
    return max(0.0, value)


def legacy_from_forks(forks: dict[int, list[dict]], sentence_idx: int,
                      similarities: Sequence[float], *, alpha: float = 1.0,
                      threshold: float = 0.8, use_similar_chunks: bool = True) -> float:
    """Pinned ``calculate_counterfactual_importance_kl`` arithmetic on raw fork samples.

    Differences from canonical, all inherited from the pinned code: samples with a
    missing answer are dropped before similarity splitting (so ``similarities`` must
    align with the unfiltered fork-i list and are subset here); the "with" side pools
    fork i+1 with the similar fork-i samples; there is no <NO_ANSWER> category; the
    default smoothing is 1.0; and missing support returns 0.0 rather than null.
    """
    current = forks.get(sentence_idx) or []
    if len(similarities) != len(current):
        raise ValueError("One similarity per fork-i sample is required")
    later = sorted(k for k in forks if k > sentence_idx and forks[k])
    if not current or not later:
        return 0.0
    following = forks[later[0]]
    dissimilar, similar = [], []
    for row, similarity in zip(current, similarities, strict=True):
        answer = row.get("answer", "")
        if answer in LEGACY_MISSING or not isinstance(answer, str):
            continue
        if not isinstance(row.get("chunk_removed", ""), str) \
                or not isinstance(row.get("chunk_resampled", ""), str):
            continue
        (dissimilar if similarity < threshold else similar).append(answer)
    following_answers = [row.get("answer", "") for row in following
                         if isinstance(row.get("answer", ""), str)
                         and row.get("answer", "") not in LEGACY_MISSING]
    if not dissimilar or not following:
        return 0.0
    with_side = following_answers + similar if use_similar_chunks else following_answers
    return legacy_kl_divergence(dissimilar, with_side, alpha)
