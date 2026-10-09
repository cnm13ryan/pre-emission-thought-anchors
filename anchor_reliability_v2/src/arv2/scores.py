"""Counterfactual importance scores between two answer samples.

All functions take two lists of outcome categories: ``without`` (resamples at fork i
whose first sentence is dissimilar to S_i) and ``with_`` (resamples at fork i+1).

- ``kl_smoothed``: the MVP estimand (reference only). Additive smoothing α over the
  union support, so its value is unbounded and dominated by categories absent from one
  side when α is small.
- ``total_variation``: ½ Σ|p − q| on raw frequencies. Bounded [0, 1], needs no
  smoothing, and an absent category contributes at most its own mass. PROVISIONAL
  primary.
- ``jensen_shannon``: JSD in bits, bounded [0, 1], no smoothing. Alternative.
- ``delta_accuracy``: |acc(without) − acc(with)| against the ground-truth answer.
  Alternative, and sign-free.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def _freqs(xs: Sequence[str]) -> dict[str, float]:
    if not xs:
        raise ValueError("empty sample")
    c = Counter(xs)
    n = len(xs)
    return {k: v / n for k, v in c.items()}


def kl_smoothed(without: Sequence[str], with_: Sequence[str], alpha: float = 1e-9,
                extra_categories: Sequence[str] = ("<NO_ANSWER>",)) -> float:
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    if not without or not with_:
        raise ValueError("empty sample")
    p, q = Counter(without), Counter(with_)
    cats = set(p) | set(q) | set(extra_categories)
    k = len(cats)
    value = 0.0
    for a in cats:
        pp = (p[a] + alpha) / (len(without) + alpha * k)
        qq = (q[a] + alpha) / (len(with_) + alpha * k)
        value += pp * math.log(pp / qq)
    return max(value, 0.0)


def kl_contributions(without: Sequence[str], with_: Sequence[str],
                     alpha: float = 1e-9) -> dict[str, float]:
    p, q = Counter(without), Counter(with_)
    cats = set(p) | set(q) | {"<NO_ANSWER>"}
    k = len(cats)
    return {a: ((p[a] + alpha) / (len(without) + alpha * k))
            * math.log(((p[a] + alpha) / (len(without) + alpha * k))
                       / ((q[a] + alpha) / (len(with_) + alpha * k))) for a in cats}


def total_variation(without: Sequence[str], with_: Sequence[str]) -> float:
    p, q = _freqs(without), _freqs(with_)
    return 0.5 * sum(abs(p.get(a, 0.0) - q.get(a, 0.0)) for a in set(p) | set(q))


def jensen_shannon(without: Sequence[str], with_: Sequence[str]) -> float:
    p, q = _freqs(without), _freqs(with_)
    value = 0.0
    for a in set(p) | set(q):
        pa, qa = p.get(a, 0.0), q.get(a, 0.0)
        m = 0.5 * (pa + qa)
        if pa:
            value += 0.5 * pa * math.log2(pa / m)
        if qa:
            value += 0.5 * qa * math.log2(qa / m)
    return min(max(value, 0.0), 1.0)


def delta_accuracy(without_correct: Sequence[bool], with_correct: Sequence[bool]) -> float:
    if not without_correct or not with_correct:
        raise ValueError("empty sample")
    return abs(sum(map(bool, without_correct)) / len(without_correct)
               - sum(map(bool, with_correct)) / len(with_correct))


def tv_null_expectation(dist: dict[str, float], n1: int, n2: int, draws: int = 2000,
                        seed: int = 0) -> float:
    """Expected TV between two independent samples of sizes n1, n2 from ONE distribution:
    the finite-sample floor of the score when the true effect is zero."""
    import numpy as np

    rng = np.random.default_rng(seed)
    cats = list(dist)
    prob = np.asarray([dist[c] for c in cats], dtype=float)
    prob = prob / prob.sum()
    a = rng.multinomial(n1, prob, size=draws) / n1
    b = rng.multinomial(n2, prob, size=draws) / n2
    return float((0.5 * np.abs(a - b).sum(1)).mean())
