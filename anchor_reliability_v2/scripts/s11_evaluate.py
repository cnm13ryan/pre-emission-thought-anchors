"""Prediction evaluation with strict problem grouping (plan §8). Exploratory.

Outer: 5-fold GroupKFold over all 20 MVP problems (seeded problem order); every sentence
gets exactly one out-of-fold (OOF) prediction. Inner: 4-fold GroupKFold over the outer
training problems selects hyperparameters, activation layer and (for the "strongest
baseline") the baseline family. Scalers, vocabularies, PCA and the flag threshold (q90 of
the training-fold score) are fitted on training rows only.

Targets: the primary score (continuous, Ridge → Spearman) and its q90 flag
(balanced L2 logistic → AP); the MVP KL anchor flag (≥ 1.355) for continuity.
Nested comparisons: baseline vs baseline + activations (the inner-selected layer),
compared by paired problem-cluster bootstrap of the OOF metric difference.
Lead time: activations from the boundary before S_{i-1}, on rows matched to lead zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, sha256_file, write_json  # noqa: E402

LAYERS = [0, 8, 16, 24, 32, 40, 44, 48]
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
A_GRID = [1.0, 10.0, 100.0, 1e3, 1e4]
PCA_DIM, MIN_ROWS = 256, 5
POSITION = ["sentence_idx", "generated_token_count", "prefix_context_fraction"]
OBS = POSITION + ["prefix_token_count", "log1p_prefix_token_count", "previous_sentence_tokens",
                  "previous_emitted_token_logprob", "next_token_entropy",
                  "next_token_max_probability", "next_token_probability_margin",
                  "next_token_top5_mass", "next_token_top10_mass"]
MARK = ["marker_backtrack", "marker_plan", "marker_conclude", "marker_check", "entropy_v2"]
BASELINES = ["position", "observables", "markers", "next_top20", "prev_token", "bow16"]
BOOT = 2000


# ------------------------------------------------------------------ features
def dense(frame, cols):
    return frame[cols].to_numpy(float)


def sparse_tokens(frame, kind, train_idx):
    def items(r):
        if kind == "prev_token":
            return {r.prev_token: 1.0}
        if kind == "bow16":
            return {t: 1.0 for t in r.prefix_ids[-16:]}
        return dict(zip(r.top_ids, r.top_probs, strict=True))
    its = [items(r) for r in frame.itertuples()]
    counts = {}
    for i in train_idx:
        for t in its[i]:
            counts[t] = counts.get(t, 0) + 1
    vocab = {t: j for j, t in enumerate(sorted(t for t, c in counts.items() if c >= MIN_ROWS))}
    rows, cols, vals = [], [], []
    for i, d in enumerate(its):
        oov = 0.0
        for t, v in d.items():
            j = vocab.get(t)
            if j is None:
                oov += v
            else:
                rows.append(i); cols.append(j); vals.append(v)
        rows.append(i); cols.append(len(vocab)); vals.append(oov)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(its), len(vocab) + 1))


def block(frame, name, train_idx, acts, layer=None, lead=0):
    if name == "position":
        return dense(frame, POSITION)
    if name == "observables":
        return dense(frame, OBS)
    if name == "markers":
        return dense(frame, MARK)
    if name in ("next_top20", "prev_token", "bow16"):
        return sparse_tokens(frame, name, train_idx)
    if name == "act":
        return acts[(layer, lead)]
    raise KeyError(name)


def fit_predict(X, y, tr, te, kind, h):
    """Training-only scaling; returns test scores."""
    Xtr, Xte = X[tr], X[te]
    if sp.issparse(Xtr):
        sc = StandardScaler(with_mean=False).fit(Xtr)
    else:
        sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "binary":
        m = LogisticRegression(C=h, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
        return m.decision_function(Xte)
    return Ridge(alpha=h).fit(Xtr, y[tr]).predict(Xte)


def metric(y, s, kind):
    if kind == "binary":
        return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan
    return R.spearman(y, s)


def group_folds(groups, k, seed):
    uniq = np.array(sorted(set(groups)))
    order = np.random.default_rng(seed).permutation(uniq)
    assign = {g: i % k for i, g in enumerate(order)}
    f = np.array([assign[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


# ------------------------------------------------------------------ model spec
_PCA_CACHE: dict = {}


def act_projection(acts, layer, lead, tr):
    """Training-only standardise + PCA of one activation layer; cached per training set."""
    key = (layer, lead, hash(tr.tobytes()), len(acts[(layer, lead)]))
    if key not in _PCA_CACHE:
        Xa = acts[(layer, lead)]
        sc = StandardScaler().fit(Xa[tr])
        p = PCA(n_components=min(PCA_DIM, len(tr) - 1, Xa.shape[1]), random_state=0).fit(
            sc.transform(Xa[tr]))
        _PCA_CACHE[key] = p.transform(sc.transform(Xa)).astype(np.float32)
        if len(_PCA_CACHE) > 400:
            _PCA_CACHE.pop(next(iter(_PCA_CACHE)))
    return _PCA_CACHE[key]


def run_spec(frame, y, tr, te, kind, spec, h, acts):
    parts = []
    for name in spec["blocks"]:
        if name == "act":
            parts.append(act_projection(acts, spec["layer"], spec.get("lead", 0), tr))
        else:
            parts.append(block(frame, name, tr, acts))
    if len(parts) == 1:
        X = parts[0]
    elif any(sp.issparse(p) for p in parts):
        X = sp.hstack([sp.csr_matrix(p) for p in parts]).tocsr()
    else:
        X = np.hstack(parts)
    return fit_predict(X, y, tr, te, kind, h)


def inner_select(frame, y, tr, kind, specs, groups, seed, acts):
    """Choose (spec, h) by mean inner-fold metric on training problems only."""
    folds = group_folds(groups[tr], 4, seed)
    grid = C_GRID if kind == "binary" else A_GRID
    best, best_score = None, -np.inf
    for spec in specs:
        for h in grid:
            vals = []
            for itr, ite in folds:
                a, b = tr[itr], tr[ite]
                yy = y if kind != "binary" else y
                s = run_spec(frame, yy, a, b, kind, spec, h, acts)
                vals.append(metric(yy[b], s, kind))
            m = np.nanmean(vals)
            if m > best_score:
                best, best_score = (spec, h), m
    return best, best_score


def evaluate(frame, target, kind, acts, groups, seed, specs_by_model, flag_q=None):
    """OOF predictions per model; the binary flag threshold is the training-fold q90."""
    n = len(frame)
    raw = frame[target].to_numpy(float)
    outer = group_folds(groups, 5, seed)
    oof = {m: np.full(n, np.nan) for m in specs_by_model}
    ylab = np.full(n, np.nan)
    chosen = {m: [] for m in specs_by_model}
    for k, (tr, te) in enumerate(outer):
        if flag_q is not None:
            thr = np.quantile(raw[tr], flag_q) if isinstance(flag_q, float) else flag_q(raw[tr])
            y = (raw >= thr).astype(int)
        else:
            y = raw
        ylab[te] = y[te]
        picked = {}  # this fold's selected spec per model, for dependent searches
        for m, specs in specs_by_model.items():
            if callable(specs):  # e.g. augment the baseline this fold selected
                specs = specs(picked)
            (spec, h), _ = inner_select(frame, y, tr, kind, specs, groups, seed + k, acts)
            picked[m] = spec
            oof[m][te] = run_spec(frame, y, tr, te, kind, spec, h, acts)
            chosen[m].append({"fold": k, "spec": {kk: vv for kk, vv in spec.items()
                                                  if kk != "blocks"} | {"blocks": spec["blocks"]},
                              "h": h})
    return ylab, oof, chosen


def summarise(ylab, oof, kind, groups, pairs, seed):
    out = {}
    for m, s in oof.items():
        val = metric(ylab, s, kind)
        draws = R.cluster_bootstrap(groups, lambda r, s=s: metric(ylab[r], s[r], kind), BOOT,
                                    seed)
        lo, hi, bad = R.interval(draws)
        out[m] = {"metric": float(val), "lo": lo, "hi": hi, "nonfinite_draws": bad}
    comps = {}
    for a, b in pairs:
        if a in oof and b in oof:
            d0 = metric(ylab, oof[b], kind) - metric(ylab, oof[a], kind)
            draws = R.cluster_bootstrap(
                groups, lambda r: metric(ylab[r], oof[b][r], kind) - metric(ylab[r], oof[a][r],
                                                                              kind),
                BOOT, seed + 7)
            lo, hi, bad = R.interval(draws)
            comps[f"{b} - {a}"] = {"delta": float(d0), "lo": lo, "hi": hi,
                                   "nonfinite_draws": bad}
    return out, comps


def load_acts(frame_all, manifest):
    """Layer subset for lead 0 (own boundary) and lead 1 (previous sentence's boundary)."""
    root = Path(manifest["referenced_in_place"]["activation_root"])
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").set_index("sentence_id")
    acts = {(L, 0): np.zeros((len(frame_all), 5120), np.float32) for L in LAYERS}
    acts.update({(L, 1): np.full((len(frame_all), 5120), np.nan, np.float32) for L in LAYERS})
    pos = {s: i for i, s in enumerate(frame_all.sentence_id)}
    for rel, digest in manifest["referenced_in_place"]["shards"].items():
        if sha256_file(root / rel) != digest:
            raise SystemExit(f"activation shard changed: {rel}")
        with safe_open(root / rel, "pt") as f:
            arr = f.get_tensor("resid")[:, LAYERS].float().numpy()
        sub = feats[feats.shard_path == rel]
        by_idx = {int(r.sentence_idx): int(r.row_idx) for r in sub.reset_index().itertuples()}
        for sid, r in sub.iterrows():
            if sid not in pos:
                continue
            i = pos[sid]
            for j, L in enumerate(LAYERS):
                acts[(L, 0)][i] = arr[int(r.row_idx), j]
            prev = by_idx.get(int(r.sentence_idx) - 1)
            if prev is not None:
                for j, L in enumerate(LAYERS):
                    acts[(L, 1)][i] = arr[prev, j]
    return acts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 layers, for testing")
    args = ap.parse_args()
    global LAYERS
    if args.quick:
        LAYERS = [24, 48]
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    primary = json.loads((RESULTS / "s05_reliability.json").read_text())["primary_score"]
    scores = pd.read_parquet(DERIVED / "sentence_scores.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").drop(
        columns=["eligible", "split", "problem_id", "trajectory_id", "sentence_idx"],
        errors="ignore")
    bf = pd.read_parquet(DERIVED / "boundary_features.parquet")
    frame = (scores[scores.eligible_v2 & scores[primary].notna()]
             .merge(feats, on="sentence_id", validate="one_to_one")
             .merge(bf, on="sentence_id", validate="one_to_one")
             .sort_values(["trajectory_id", "sentence_idx"]).reset_index(drop=True))
    acts = load_acts(frame, manifest)
    groups = frame.problem_id.to_numpy()
    seed = R.seed_from("20260914:arv2:evaluate")
    base_specs = {b: [{"blocks": [b]}] for b in BASELINES}
    act_specs = [{"blocks": ["act"], "layer": L} for L in LAYERS]
    obs_act = [{"blocks": ["observables", "act"], "layer": L} for L in LAYERS]
    strongest = [{"blocks": [b]} for b in BASELINES]
    # Nested by construction: in each outer fold, add activations to the baseline family that
    # fold's strongest-baseline search selected; inner CV chooses only the layer and penalty.
    # (Codex review on PR #5: the earlier fixed three-family augmented search could not pair
    # activations with next_top20, which fold 4 of fork_tv_flag_q90 selected.)
    def strongest_act(picked):
        return [{"blocks": [*picked["strongest_baseline"]["blocks"], "act"], "layer": L}
                for L in LAYERS]
    models = {**base_specs, "strongest_baseline": strongest, "activation": act_specs,
              "observables+activation": obs_act, "strongest+activation": strongest_act}
    pairs = [("observables", "observables+activation"),
             ("strongest_baseline", "strongest+activation"),
             ("position", "activation"), ("strongest_baseline", "activation")]
    report = {"stage": "exploratory", "primary_score": primary, "rows": len(frame),
              "problems": int(len(set(groups))), "layers": LAYERS, "targets": {}}
    targets = [(primary, "continuous", None), (primary + "_flag_q90", "binary", 0.90)]
    mvp_ok = frame["mvp_importance"].notna()
    for name, kind, q in targets:
        col = primary
        ylab, oof, chosen = evaluate(frame, col, kind, acts, groups, seed, models, q)
        res, comps = summarise(ylab, oof, kind, groups, pairs, seed)
        report["targets"][name] = {"kind": kind, "models": res, "nested": comps,
                                   "prevalence": float(np.nanmean(ylab)) if kind == "binary"
                                   else None, "chosen": chosen}
        print(name, json.dumps({m: round(v["metric"], 3) for m, v in res.items()}), flush=True)
    # MVP KL anchor (frozen threshold) on the rows where it is defined
    sub = frame[mvp_ok].reset_index(drop=True)
    sub_acts = {k: v[mvp_ok.to_numpy()] for k, v in acts.items()}
    ylab, oof, chosen = evaluate(sub, "mvp_importance", "binary", sub_acts,
                                 sub.problem_id.to_numpy(), seed, models,
                                 lambda _tr: 1.3549810893213545)
    res, comps = summarise(ylab, oof, "binary", sub.problem_id.to_numpy(), pairs, seed)
    report["targets"]["mvp_kl_anchor"] = {"kind": "binary", "models": res, "nested": comps,
                                          "prevalence": float(np.nanmean(ylab)), "rows": len(sub)}
    # layer curve (activation-only, fixed layer, inner-selected C) on the primary flag
    curve = {}
    for L in LAYERS:
        ylab, oof, _ = evaluate(frame, primary, "binary", acts, groups, seed,
                                {"act": [{"blocks": ["act"], "layer": L}]}, 0.90)
        curve[L] = float(metric(ylab, oof["act"], "binary"))
    report["layer_curve_primary_flag_ap"] = curve
    # lead time on matched rows (rows with a previous-sentence boundary)
    has_prev = ~np.isnan(acts[(LAYERS[0], 1)][:, 0])
    m = frame[has_prev].reset_index(drop=True)
    m_acts = {k: v[has_prev] for k, v in acts.items()}
    lead_models = {"lead0": [{"blocks": ["act"], "layer": L, "lead": 0} for L in LAYERS],
                   "lead1": [{"blocks": ["act"], "layer": L, "lead": 1} for L in LAYERS]}
    lead = {}
    for name, kind, q in targets:
        ylab, oof, _ = evaluate(m, primary, kind, m_acts, m.problem_id.to_numpy(), seed,
                                lead_models, q)
        res, comps = summarise(ylab, oof, kind, m.problem_id.to_numpy(), [("lead1", "lead0")],
                               seed)
        lead[name] = {"models": res, "lead0_minus_lead1": comps, "rows": len(m)}
    report["lead_time"] = lead
    rel = json.loads((RESULTS / "s05_reliability.json").read_text())
    row = next(r for r in rel["table"] if r["score"] == primary)
    report["noise_ceiling"] = {
        "projected_full_n_reliability": row["icc_projected_full_n"],
        "ceiling_correlation": float(np.sqrt(max(row["icc_projected_full_n"], 0))),
        "note": "max expected correlation of any predictor with a single-replicate score"}
    write_json(RESULTS / ("s11_evaluation_quick.json" if args.quick else "s11_evaluation.json"),
               report)
    print(json.dumps({"layer_curve": curve, "noise_ceiling": report["noise_ceiling"]}, indent=1))


if __name__ == "__main__":
    main()
