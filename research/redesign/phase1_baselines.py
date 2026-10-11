"""Phase 1 (redesign) item 4: word-level baselines on the candidate labels. Exploratory.

Frozen probe protocol (SPEC §3.10) applied to new feature sets. Vocabularies are fitted
inside each training fold (no held-out problem contributes to a vocabulary). Selection
uses training CV only. Validation is reported once per (baseline, label); **test rows are
never loaded**.

Usage: python research/redesign/phase1_baselines.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "preanchor-root"
DATA = ROOT / "data/processed/redesign_phase1"
TABLES = Path(__file__).resolve().parent / "tables"
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
RIDGE_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
BOW_K = [4, 16, 64]
MIN_ROWS = 5
POSITION = ["sentence_idx", "generated_token_count", "prefix_context_fraction"]
OBSERVABLES = POSITION + ["prefix_token_count", "log1p_prefix_token_count",
                          "previous_sentence_tokens", "previous_emitted_token_logprob",
                          "next_token_entropy", "next_token_max_probability",
                          "next_token_probability_margin", "next_token_top5_mass",
                          "next_token_top10_mass"]
BINARY = {"a_kl_1e-9": "label_a_kl_anchor", "b_kl_kt": "label_b_kl_kt_anchor",
          "c_fork_tv": "label_c_fork_tv_anchor"}
CONTINUOUS = {"d_log1p_kl": "label_d_log1p", "b_log1p_kl_kt": "log1p_b",
              "c_fork_tv_cont": "label_c_fork_tv"}


def fold_seed() -> int:
    encoded = json.dumps([20260914, "cv_folds"], sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(encoded.encode()).hexdigest()[:8], 16)


class Sparse(BaseEstimator, TransformerMixin):
    """Fold-local vocabulary over token-id features; values are 1 or a probability."""

    def __init__(self, kind: str, k: int = 16):
        self.kind, self.k = kind, k

    def _items(self, row) -> dict[int, float]:
        if self.kind == "prev_token":
            return {int(row["prev_token"]): 1.0}
        if self.kind == "bow":
            return {int(t): 1.0 for t in row["prefix_ids"][-self.k:]}
        return {int(t): float(p) for t, p in zip(row["top_ids"], row["top_probs"],
                                                 strict=True)}

    def fit(self, X, y=None):
        counts: dict[int, int] = {}
        for _, row in X.iterrows():
            for t in self._items(row):
                counts[t] = counts.get(t, 0) + 1
        vocab = sorted(t for t, c in counts.items() if c >= MIN_ROWS)
        self.index_ = {t: i for i, t in enumerate(vocab)}
        return self

    def transform(self, X):
        n, m = len(X), len(self.index_) + 1   # last column = out-of-vocabulary mass
        rows, cols, vals = [], [], []
        for r, (_, row) in enumerate(X.iterrows()):
            oov = 0.0
            for t, v in self._items(row).items():
                j = self.index_.get(t)
                if j is None:
                    oov += v
                else:
                    rows.append(r); cols.append(j); vals.append(v)
            rows.append(r); cols.append(m - 1); vals.append(oov)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, m))


class Dense(BaseEstimator, TransformerMixin):
    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sp.csr_matrix(X[list(self.columns)].to_numpy(dtype=float))


class Union(BaseEstimator, TransformerMixin):
    def __init__(self, parts: tuple):
        self.parts = parts

    def fit(self, X, y=None):
        for p in self.parts:
            p.fit(X, y)
        return self

    def transform(self, X):
        return sp.hstack([p.transform(X) for p in self.parts]).tocsr()


def baselines(bow_k: int) -> dict:
    return {
        "position": lambda: Dense(tuple(POSITION)),
        "observable_combined": lambda: Dense(tuple(OBSERVABLES)),
        "prev_token": lambda: Sparse("prev_token"),
        "bow_k": lambda: Sparse("bow", bow_k),
        "next_top20": lambda: Sparse("topk"),
        "observables+word": lambda: Union((Dense(tuple(OBSERVABLES)), Sparse("prev_token"),
                                           Sparse("bow", bow_k), Sparse("topk"))),
    }


def binary_model(features, c):
    return make_pipeline(features, StandardScaler(with_mean=False),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def ridge_model(features, a):
    return make_pipeline(features, StandardScaler(with_mean=False), Ridge(alpha=a))


def main() -> None:
    labels = pd.read_parquet(DATA / "candidate_labels.parquet")
    labels["log1p_b"] = np.log1p(labels.label_b_kl_kt)
    feats = pd.read_parquet(ROOT / "data/processed/stage7/features.parquet",
                            columns=["sentence_id"] + [c for c in OBSERVABLES
                                                       if c != "sentence_idx"])
    words = pd.read_parquet(DATA / "word_features.parquet")
    frame = labels.merge(feats, on="sentence_id", validate="one_to_one") \
        .merge(words, on="sentence_id", validate="one_to_one")
    frame = frame[frame.eligible & frame.split.isin(["train", "validation"])].copy()
    assert not (frame.split == "test").any()
    train_all, val_all = frame[frame.split == "train"], frame[frame.split == "validation"]
    folds = None
    results = []

    def cv_folds(train: pd.DataFrame, strat: np.ndarray):
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=fold_seed())
        return list(splitter.split(np.zeros((len(train), 1)), strat, train.problem_id))

    for kind, targets in (("binary", BINARY), ("continuous", CONTINUOUS)):
        for label_name, column in targets.items():
            train = train_all[train_all[column].notna()]
            val = val_all[val_all[column].notna()]
            strat_col = column if kind == "binary" else "label_a_kl_anchor"
            folds = cv_folds(train, train[strat_col].astype(int).to_numpy())
            y_tr = train[column].astype(float).to_numpy()
            y_va = val[column].astype(float).to_numpy()
            grid = C_GRID if kind == "binary" else RIDGE_GRID
            # bow k selected once per label by its own CV (frozen grid × k)
            candidates = []
            for name in ["position", "observable_combined", "prev_token", "bow_k",
                         "next_top20", "observables+word"]:
                ks = BOW_K if name in ("bow_k", "observables+word") else [None]
                for k in ks:
                    for h in grid:
                        scores = []
                        for tr, va in folds:
                            feat = baselines(k or 16)[name]()
                            model = (binary_model if kind == "binary" else ridge_model)(feat, h)
                            model.fit(train.iloc[tr], y_tr[tr].astype(int) if kind == "binary"
                                      else y_tr[tr])
                            if kind == "binary":
                                s = model.decision_function(train.iloc[va])
                                scores.append(average_precision_score(y_tr[va], s))
                            else:
                                s = model.predict(train.iloc[va])
                                scores.append(spearmanr(y_tr[va], s).statistic)
                        candidates.append({"baseline": name, "k": k, "h": h,
                                           "cv_mean": float(np.mean(scores)),
                                           "cv_folds": [float(x) for x in scores]})
            cand = pd.DataFrame(candidates)
            for name, group in cand.groupby("baseline", sort=False):
                best = group.sort_values(["cv_mean", "h"], ascending=[False, True]).iloc[0]
                feat = baselines(int(best.k) if pd.notna(best.k) else 16)[name]()
                model = (binary_model if kind == "binary" else ridge_model)(feat, best.h)
                model.fit(train, y_tr.astype(int) if kind == "binary" else y_tr)
                row = {"kind": kind, "label": label_name, "baseline": name,
                       "selected_h": float(best.h),
                       "selected_k": None if pd.isna(best.k) else int(best.k),
                       "train_rows": len(train), "train_pos": int(y_tr.sum())
                       if kind == "binary" else None,
                       "cv_mean": best.cv_mean, "cv_min": min(best.cv_folds),
                       "cv_max": max(best.cv_folds), "val_rows": len(val)}
                if kind == "binary":
                    s = model.decision_function(val)
                    row.update({"val_pos": int(y_va.sum()),
                                "val_prevalence": float(y_va.mean()),
                                "val_ap": float(average_precision_score(y_va, s))
                                if y_va.sum() else None,
                                "val_auroc": float(roc_auc_score(y_va, s))
                                if 0 < y_va.sum() < len(y_va) else None,
                                "status": "OK" if 0 < y_va.sum() < len(y_va)
                                else "SINGLE_CLASS"})
                else:
                    s = model.predict(val)
                    row.update({"val_spearman": float(spearmanr(y_va, s).statistic),
                                "status": "OK"})
                results.append(row)
                print(label_name, name, row.get("cv_mean"), row.get("val_ap",
                                                                    row.get("val_spearman")),
                      flush=True)
            if kind == "binary":
                results.append({"kind": kind, "label": label_name, "baseline": "prevalence",
                                "cv_mean": float(np.mean([y_tr[va].mean() for _, va in folds])),
                                "val_ap": float(y_va.mean()), "val_rows": len(val),
                                "val_pos": int(y_va.sum()), "status": "OK"})
    out = pd.DataFrame(results)
    out.to_csv(TABLES / "item4_word_baselines.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
