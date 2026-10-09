"""Pre-emission eligibility (plan §4) and the MVP rules it replaces, for comparison.

Every v2 rule uses only the trace prefix before S_i or forks j <= i (which are resampled
from prefixes ending before S_i). The MVP rules are reimplemented here only to count how
many sentences each one removes; they are not used for inclusion.
"""

from __future__ import annotations

from collections.abc import Sequence


def _extreme_low(a: float) -> bool:
    return 0.0 <= a <= 0.01


def _extreme_high(a: float) -> bool:
    return 0.99 <= a <= 1.0


def converged_pre(accuracy: Sequence[float | None], k: int = 5) -> list[bool]:
    """True at i when forks i-k+1..i all exist and are all <=0.01 or all >=0.99 accurate.
    Once true, every later sentence is also excluded (the run has converged)."""
    out, latched = [], False
    for i in range(len(accuracy)):
        window = accuracy[max(0, i - k + 1): i + 1]
        if not latched and i >= k - 1 and all(a is not None for a in window):
            if all(_extreme_low(a) for a in window) or all(_extreme_high(a) for a in window):
                latched = True
        out.append(latched)
    return out


def converged_mvp(accuracy: Sequence[float | None], k: int = 5) -> list[bool]:
    """MVP rule: first j where forks j..j+k-1 are all extreme; exclude i >= j (uses up to
    k-1 FUTURE forks). Missing forks count as 0.5 (never extreme), as in the MVP."""
    seq = [0.5 if a is None else a for a in accuracy]
    start = None
    for j in range(len(seq) - k + 1):
        block = seq[j: j + k]
        if all(_extreme_low(a) for a in block) or all(_extreme_high(a) for a in block):
            start = j
            break
    return [start is not None and i >= start for i in range(len(seq))]


def boxed_before(texts: Sequence[str]) -> list[bool]:
    """True at i when any sentence strictly before i contains ``\\boxed{``."""
    out, seen = [], False
    for t in texts:
        out.append(seen)
        seen = seen or "\\boxed{" in t
    return out


def final_answer_mvp(texts: Sequence[str]) -> list[bool]:
    """MVP rule: exclude i >= index of the LAST sentence containing ``\\boxed{`` (uses
    future sentences)."""
    idx = [i for i, t in enumerate(texts) if "\\boxed{" in t]
    last = max(idx) if idx else None
    return [last is not None and i >= last for i in range(len(texts))]
