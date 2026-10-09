"""Leakage prevention and baseline correctness for the prediction evaluation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("s11", ROOT / "scripts/s11_evaluate.py")
s11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s11)


def test_group_folds_disjoint_and_cover():
    groups = np.repeat(np.arange(20), 7)
    folds = s11.group_folds(groups, 5, seed=1)
    seen = []
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(len(groups)))


def test_vocabulary_uses_training_rows_only():
    frame = pd.DataFrame({"prev_token": [1] * 10 + [999] * 10,
                          "prefix_ids": [[1]] * 20, "top_ids": [[1]] * 20,
                          "top_probs": [[1.0]] * 20})
    tr = np.arange(10)                       # token 999 appears only in test rows
    X = s11.sparse_tokens(frame, "prev_token", tr)
    assert X.shape[1] == 2                    # {1} + OOV column; 999 is not a feature
    assert X[15, 1] == 1.0 and X[15, 0] == 0  # test row maps to OOV


def test_recovers_known_signal_and_respects_groups():
    rng = np.random.default_rng(0)
    n = 400
    groups = np.repeat(np.arange(20), 20)
    x = rng.normal(size=n)
    y = (x + 0.3 * rng.normal(size=n) > 1).astype(int)
    frame = pd.DataFrame({c: rng.normal(size=n) for c in s11.OBS})
    frame["sentence_idx"] = x                 # signal lives in a position feature
    for c in s11.MARK:
        frame[c] = rng.normal(size=n)
    folds = s11.group_folds(groups, 5, seed=2)
    oof = np.full(n, np.nan)
    for tr, te in folds:
        oof[te] = s11.run_spec(frame, y, tr, te, "binary", {"blocks": ["position"]}, 1.0, {})
    assert s11.metric(y, oof, "binary") > 0.8
    noise = np.full(n, np.nan)
    for tr, te in folds:
        noise[te] = s11.run_spec(frame, y, tr, te, "binary", {"blocks": ["markers"]}, 1.0, {})
    assert s11.metric(y, noise, "binary") < 0.5


def test_activation_projection_fitted_on_training_rows_only():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(60, 20)).astype(np.float32)
    acts = {(0, 0): a}
    tr = np.arange(40)
    z1 = s11.act_projection(acts, 0, 0, tr).copy()
    a2 = a.copy()
    a2[40:] += 100.0                          # mutate test rows only
    s11._PCA_CACHE.clear()
    z2 = s11.act_projection({(0, 0): a2}, 0, 0, tr)
    assert np.allclose(z1[:40], z2[:40])      # training projection unaffected by test rows
