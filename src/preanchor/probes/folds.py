"""Problem-grouped, anchor-stratified folds seeded from ``seed_root`` (SPEC §3.10)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from preanchor.config import ExperimentConfig
from preanchor.ids import derived_seed

FOLD_PURPOSE = "cv_folds"


def assert_disjoint(config: ExperimentConfig) -> None:
    sets = [set(config.splits.train), set(config.splits.validation), set(config.splits.test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Problem splits overlap")


def fold_seed(config: ExperimentConfig) -> int:
    return derived_seed(config.seed_root, FOLD_PURPOSE)


def grouped_folds(y: np.ndarray, groups: np.ndarray, config: ExperimentConfig) -> list[dict]:
    """Every problem in exactly one validation fold; both classes on both sides."""
    if config.probes["group_column"] != "problem_id" or config.probes["cv_splitter"] != \
            "StratifiedGroupKFold":
        raise ValueError("Frozen CV design differs from the configuration")
    n_splits, seed = int(config.probes["cv_folds"]), fold_seed(config)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=bool(config.probes["cv_shuffle"]),
                                    random_state=seed)
    folds = []
    seen: list[int] = []
    for k, (tr, va) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
        tr, va = np.sort(tr), np.sort(va)
        if set(groups[tr]) & set(groups[va]):
            raise ValueError("A problem appears on both sides of a fold")
        if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
            raise ValueError(f"Fold {k} has a single class; failing the split (SPEC Stage 8)")
        seen.extend(groups[va].tolist())
        folds.append({"fold": k, "seed": seed, "train_idx": tr, "val_idx": va,
                      "val_problems": sorted(set(int(g) for g in groups[va])),
                      "n_train": int(len(tr)), "n_val": int(len(va)),
                      "train_pos": int(y[tr].sum()), "val_pos": int(y[va].sum())})
    if sorted(set(seen)) != sorted(set(groups.tolist())):
        raise ValueError("Folds do not cover every problem exactly once")
    return folds
