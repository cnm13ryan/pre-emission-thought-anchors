"""Fresh answer categories and the SPEC §3.6 union-support KL estimand."""

import math
from collections import Counter

from preanchor.labels.upstream_answers import extract_boxed_answers, normalize_latex


def answer_category(continuation):
    # Upstream generation selects the first boxed answer, including its documented
    # unterminated-box behavior; never classify the whole reasoning string.
    answer = extract_boxed_answers(continuation)[0]
    return normalize_latex(answer) or "<NO_ANSWER>"


def canonical_kl(without, with_sentence, alpha=1e-9):
    if not without or not with_sentence or alpha <= 0:
        raise ValueError("Nonempty distributions and positive smoothing required")
    p, q = Counter(without), Counter(with_sentence)
    categories = set(p) | set(q) | {"<NO_ANSWER>"}
    k = len(categories)
    value = 0.0
    for category in sorted(categories):
        pp = (p[category] + alpha) / (len(without) + alpha * k)
        qq = (q[category] + alpha) / (len(with_sentence) + alpha * k)
        value += pp * math.log(pp / qq)
    return value


def legacy_kl(without, with_sentence, alpha=1e-9):
    # Pinned legacy excludes missing answers and smooths over observed union only.
    p = Counter(a for a in without if a != "<NO_ANSWER>")
    q = Counter(a for a in with_sentence if a != "<NO_ANSWER>")
    if not p or not q:
        return 0.0
    categories = set(p) | set(q)
    value = 0.0
    for category in sorted(categories):
        pp = (p[category] + alpha) / (sum(p.values()) + alpha * len(categories))
        qq = (q[category] + alpha) / (sum(q.values()) + alpha * len(categories))
        value += pp * math.log(pp / qq)
    return max(0.0, value)
