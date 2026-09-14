"""Canonical smoothed counterfactual importance."""

from __future__ import annotations

import math
import re


def normalize_answer(value: str | None) -> str:
    if not value:
        return "<NO_ANSWER>"
    value = re.sub(r"\s+", "", value).strip().lower()
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\boxed{", "").replace("}", "")
    return value or "<NO_ANSWER>"


def distribution(values: list[str], alpha: float = 1e-9) -> dict[str, float]:
    normalized = [normalize_answer(v) for v in values]
    categories = sorted(set(normalized) | {"<NO_ANSWER>"})
    total = len(normalized)
    return {category: (normalized.count(category) + alpha) / (total + alpha * len(categories))
            for category in categories}


def smoothed_kl(without: list[str], with_sentence: list[str], alpha: float = 1e-9) -> float:
    p = distribution(without, alpha)
    q = distribution(with_sentence, alpha)
    for category in set(p) | set(q):
        p.setdefault(category, alpha / (len(without) + alpha * len(p)))
        q.setdefault(category, alpha / (len(with_sentence) + alpha * len(q)))
    value = sum(p[c] * math.log(p[c] / q[c]) for c in p)
    return max(0.0, value) if value > -1e-12 else 0.0


def canonical_from_forks(forks: dict[int, list[dict]], sentence_idx: int,
                         alpha: float = 1e-9) -> float:
    without = [row.get("answer", "") for row in forks.get(sentence_idx, [])
               if row.get("chunk_removed")]
    with_sentence = [row.get("answer", "") for row in forks.get(sentence_idx + 1, [])
                     if row.get("chunk_removed")]
    if not without or not with_sentence:
        raise ValueError(f"Missing adjacent fork support at sentence {sentence_idx}")
    return smoothed_kl(without, with_sentence, alpha)
