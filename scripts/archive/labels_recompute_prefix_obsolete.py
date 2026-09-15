"""Canonical labels from downloaded raw counterfactual resamples."""
from __future__ import annotations
import json
from pathlib import Path
from .importance import canonical_from_forks
from .convergence import convergence_idx, eligible

def load_answers(path: Path) -> list[dict]:
    value=json.loads(path.read_bytes())
    if isinstance(value, dict):
        value=value.get('solutions', value.get('samples', []))
    if not isinstance(value,list): raise ValueError('raw resamples must be a list')
    return value

def recompute_sentence(sentence_idx: int, paths: dict[int, Path], *, convergence: int|None=None, final_answer_idx: int|None=None, alpha: float=1e-9) -> dict:
    if not eligible(sentence_idx, convergence, final_answer_idx):
        return {'eligible':False,'exclusion_reason':'convergence_or_final_answer'}
    if sentence_idx not in paths or sentence_idx+1 not in paths:
        return {'eligible':False,'exclusion_reason':'missing_adjacent_fork'}
    value=canonical_from_forks({sentence_idx:load_answers(paths[sentence_idx]), sentence_idx+1:load_answers(paths[sentence_idx+1])}, sentence_idx, alpha)
    return {'eligible':True,'importance_canonical':value,'importance_primary':value,'label_source':'canonical_fallback'}

def recompute_trajectory(sentence_count: int, paths: dict[int,Path], accuracies: list[float]|None=None, *, alpha: float=1e-9) -> list[dict]:
    conv=convergence_idx(accuracies) if accuracies is not None else None
    return [recompute_sentence(i, paths, convergence=conv, alpha=alpha) for i in range(sentence_count)]
