"""Frozen convergence masks for raw answer resamples."""
def convergence_idx(accuracy_by_fork: list[float], consecutive: int = 5) -> int | None:
    if consecutive < 1: raise ValueError('consecutive must be positive')
    for i in range(0, len(accuracy_by_fork)-consecutive+1):
        block=accuracy_by_fork[i:i+consecutive]
        if all(0 <= x <= .01 for x in block) or all(.99 <= x <= 1 for x in block): return i
    return None

def eligible(sentence_idx: int, convergence: int | None, final_answer_idx: int | None = None) -> bool:
    return (final_answer_idx is None or sentence_idx < final_answer_idx) and (convergence is None or sentence_idx < convergence)
