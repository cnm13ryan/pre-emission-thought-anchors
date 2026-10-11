# ruff: noqa: E501
"""Reasoning-stage analyses A-E (exploratory, post-hoc).

Protocol: all model fitting and hyperparameter choice uses TRAIN problems only
(grouped CV by problem_id). Validation problems are the main evaluation set.
The frozen TEST problems are scored once per pre-specified model, with no
test-driven tuning. Mirrors the MVP feature families (probes/dataset.py) and
its bootstrap scheme (problem-level, 2000 reps, percentile 2.5-97.5).

Usage: uv run python research/reasoning_stage/analyses.py
Writes: research/reasoning_stage/outputs/analysis_results.json (all numbers)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "outputs"
RNG = np.random.default_rng(20260917)
STAGES = ["orient", "plan", "compute", "verify", "backtrack", "conclude", "other"]

POSITION = ["sentence_idx", "generated_token_count", "prefix_context_fraction"]
CONTEXT = ["prefix_token_count", "log1p_prefix_token_count", "previous_sentence_tokens"]
UNCERTAINTY = ["previous_emitted_token_logprob", "next_token_entropy",
               "next_token_max_probability", "next_token_probability_margin",
               "next_token_top5_mass", "next_token_top10_mass"]
OBS12 = POSITION + CONTEXT + UNCERTAINTY

df = pd.read_parquet(OUT / "dataset.parquet")
A44 = np.load(OUT / "activations_l44.npy")
A0 = np.load(OUT / "activations_l0.npy")
df = df.reset_index(drop=True)
stage_dum = pd.get_dummies(df["stage"]).reindex(columns=STAGES, fill_value=0).astype(float)

# previous-sentence stage and transition indicator (within trajectory)
df["prev_stage"] = df.groupby("trajectory_id")["stage"].shift(1)
df["transition"] = (df["stage"] != df["prev_stage"]).astype(float)
df.loc[df["prev_stage"].isna(), "transition"] = np.nan
# lead-one activation row: activation captured before sentence i-1 (within trajectory)
df["row"] = np.arange(len(df))
df["prev_row"] = df.groupby("trajectory_id")["row"].shift(1)

results = {}


def design(rows, blocks, act=None):
    mats = []
    if blocks:
        mats.append(df.loc[rows, blocks].to_numpy(dtype=float))
    if act is not None:
        mats.append(act[rows])
    return np.hstack(mats)


def fit_eval_binary(y_col, feature_blocks, act, name, eligible_only=True,
                    Cs=(1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0),
                    splits=("validation", "test"), extra_mask=None):
    """Grouped-CV C selection on train, then fit train / evaluate splits. AP metric."""
    mask = df[y_col].notna()
    if eligible_only:
        mask &= df["eligible"]
    if extra_mask is not None:
        mask &= extra_mask
    d = df[mask]
    tr = d[d["split"] == "train"]
    ytr = tr[y_col].to_numpy(dtype=float)
    def X_of(rows):
        return design(rows, feature_blocks, act)
    # C selection by grouped CV on train
    best, best_ap = None, -1
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2845359893)
    for C in Cs:
        aps = []
        for itr, ival in sgkf.split(tr.index, ytr, groups=tr["problem_id"]):
            sc = StandardScaler().fit(X_of(tr.index[itr]))
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            m.fit(sc.transform(X_of(tr.index[itr])), ytr[itr])
            yv = ytr[ival]
            if yv.sum() in (0, len(yv)):
                continue
            aps.append(average_precision_score(yv, m.predict_proba(sc.transform(X_of(tr.index[ival])))[:, 1]))
        if np.mean(aps) > best_ap:
            best_ap, best = np.mean(aps), C
    sc = StandardScaler().fit(X_of(tr.index))
    m = LogisticRegression(C=best, max_iter=3000, class_weight="balanced")
    m.fit(sc.transform(X_of(tr.index)), ytr)
    out = {"C": best, "cv_ap": round(float(best_ap), 4), "n_train": len(tr), "pos_train": int(ytr.sum())}
    for sp in splits:
        e = d[d["split"] == sp]
        ye = e[y_col].to_numpy(dtype=float)
        pe = m.predict_proba(sc.transform(X_of(e.index)))[:, 1]
        out[sp] = {"n": len(e), "pos": int(ye.sum()),
                   "ap": round(float(average_precision_score(ye, pe)), 4),
                   "auroc": round(float(roc_auc_score(ye, pe)), 4)}
        # keep predictions for paired bootstraps
        out[sp]["_pred"] = pe.tolist()
        out[sp]["_y"] = ye.tolist()
        out[sp]["_prob"] = e["problem_id"].tolist()
    results.setdefault(name.split("/")[0], {})[name.split("/", 1)[1]] = out
    print(f"[{name}] C={best} cvAP={best_ap:.4f} " +
          " ".join(f"{sp}: AP={out[sp]['ap']} AUROC={out[sp]['auroc']}" for sp in splits))
    return out


def paired_boot(a, b, sp="test", reps=2000):
    """Percentile CI for AP(a)-AP(b) resampling problems with replacement."""
    y = np.array(a[sp]["_y"])
    pa = np.array(a[sp]["_pred"])
    pb = np.array(b[sp]["_pred"])
    prob = np.array(a[sp]["_prob"])
    uniq = np.unique(prob)
    deltas = []
    for _ in range(reps):
        samp = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(prob == p) for p in samp])
        yy = y[idx]
        if yy.sum() in (0, len(yy)):
            continue
        deltas.append(average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]))
    d = np.array(deltas)
    point = average_precision_score(y, pa) - average_precision_score(y, pb)
    return {"point": round(float(point), 4), "low": round(float(np.percentile(d, 2.5)), 4),
            "high": round(float(np.percentile(d, 97.5)), 4), "mean": round(float(d.mean()), 4),
            "n_valid": len(d)}


# ---------------- Descriptives ----------------
desc = {}
trv = df[df["split"].isin(["train", "validation"])]
desc["stage_counts_all"] = df["stage"].value_counts().to_dict()
q = pd.qcut(df["prefix_context_fraction"], 5, labels=[1, 2, 3, 4, 5])
desc["stage_by_position_quintile"] = (
    pd.crosstab(q, df["stage"], normalize="index").round(3).to_dict())
el = trv[trv["eligible"]]
desc["anchor_prevalence_by_stage_train_val"] = {
    st: {"n": int((el["stage"] == st).sum()), "anchors": int(el[el["stage"] == st]["anchor"].sum()),
         "rate": round(float(el[el["stage"] == st]["anchor"].mean()), 4) if (el["stage"] == st).any() else None}
    for st in STAGES}
desc["overall_anchor_rate_train_val"] = round(float(el["anchor"].mean()), 4)
desc["mean_importance_by_stage_train_val"] = el.groupby("stage")["importance_primary"].mean().round(3).to_dict()
results["descriptives"] = desc

# ---------------- A: stage ~ position ----------------
def fit_multiclass(feature_blocks, act, name):
    tr = df[df["split"] == "train"]
    ytr = tr["stage"].to_numpy()
    sc = StandardScaler().fit(design(tr.index, feature_blocks, act))
    m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
    m.fit(sc.transform(design(tr.index, feature_blocks, act)), ytr)
    out = {}
    for sp in ["validation", "test"]:
        e = df[df["split"] == sp]
        P = m.predict_proba(sc.transform(design(e.index, feature_blocks, act)))
        yp = m.classes_[P.argmax(1)]
        maj = tr["stage"].mode()[0]
        out[sp] = {"n": len(e),
                   "accuracy": round(float(accuracy_score(e["stage"], yp)), 4),
                   "macro_f1": round(float(f1_score(e["stage"], yp, average="macro")), 4),
                   "log_loss": round(float(log_loss(e["stage"], P, labels=list(m.classes_))), 4),
                   "majority_class_accuracy": round(float((e["stage"] == maj).mean()), 4)}
    results.setdefault("stage_prediction", {})[name] = out
    print(f"[stage/{name}] " + " ".join(
        f"{sp}: acc={out[sp]['accuracy']} f1={out[sp]['macro_f1']} ll={out[sp]['log_loss']}"
        for sp in out))
    return out


fit_multiclass(POSITION, None, "position_only")
fit_multiclass(OBS12, None, "observables12")
fit_multiclass(None, A44, "activations_l44")
fit_multiclass(None, A0, "activations_l0")
fit_multiclass(POSITION, A44, "activations_l44_plus_position")
fit_multiclass(OBS12, A44, "activations_l44_plus_observables12")

# ---------------- C: anchor ladder ----------------
sd = stage_dum.copy()
for c in sd.columns:
    df[f"stg_{c}"] = sd[c]
STG = [f"stg_{c}" for c in STAGES]
ladder = {}
ladder["position"] = fit_eval_binary("anchor", POSITION, None, "anchor_ladder/position")
ladder["stage"] = fit_eval_binary("anchor", STG, None, "anchor_ladder/stage")
ladder["position_stage"] = fit_eval_binary("anchor", POSITION + STG, None, "anchor_ladder/position_stage")
ladder["act"] = fit_eval_binary("anchor", None, A44, "anchor_ladder/act44")
ladder["act_position"] = fit_eval_binary("anchor", POSITION, A44, "anchor_ladder/act44_position")
ladder["act_stage"] = fit_eval_binary("anchor", STG, A44, "anchor_ladder/act44_stage")
ladder["act_position_stage"] = fit_eval_binary("anchor", POSITION + STG, A44, "anchor_ladder/act44_position_stage")
ladder["obs12_stage"] = fit_eval_binary("anchor", OBS12 + STG, None, "anchor_ladder/obs12_stage")
ladder["act_obs12_stage"] = fit_eval_binary("anchor", OBS12 + STG, A44, "anchor_ladder/act44_obs12_stage")

boot = {}
for key, a, b in [
    ("act_vs_position", "act", "position"),
    ("actpos_vs_pos", "act_position", "position"),
    ("actposstage_vs_posstage", "act_position_stage", "position_stage"),
    ("actobsstage_vs_obsstage", "act_obs12_stage", "obs12_stage"),
    ("posstage_vs_pos", "position_stage", "position"),
]:
    boot[key] = {sp: paired_boot(ladder[a], ladder[b], sp) for sp in ["validation", "test"]}
    print(f"[boot {key}] " + " ".join(f"{sp}: {boot[key][sp]}" for sp in boot[key]))
results["anchor_ladder_bootstrap"] = boot

# Residualized: regress A44 on position+stage (fit on train), probe residuals.
tr_idx = df.index[(df["split"] == "train") & df["eligible"]]
Z = df[POSITION + STG].to_numpy(dtype=float)
zs = StandardScaler().fit(Z[tr_idx])
reg = LinearRegression().fit(zs.transform(Z[tr_idx]), A44[tr_idx])
R44 = A44 - reg.predict(zs.transform(Z))
ladder["act_resid_posstage"] = fit_eval_binary("anchor", None, R44, "anchor_ladder/act44_resid_position_stage")

# ---------------- D: anchors vs stages / transitions ----------------
dd = df[df["eligible"] & df["transition"].notna() & df["split"].isin(["train", "validation"])]
tab = dd.groupby("transition")["anchor"].agg(["size", "sum", "mean"]).round(4)
results["transitions"] = {"anchor_rate_by_transition_train_val": tab.to_dict()}
pairs = (dd.assign(pair=dd["prev_stage"] + "->" + dd["stage"])
         .groupby("pair")["anchor"].agg(["size", "sum", "mean"]))
pairs = pairs[pairs["size"] >= 30].sort_values("mean", ascending=False).round(4)
results["transitions"]["anchor_rate_by_pair_min30_train_val"] = pairs.to_dict("index")
# problem-bootstrap for transition-vs-not difference
probs = dd["problem_id"].unique()
diffs = []
for _ in range(2000):
    samp = RNG.choice(probs, size=len(probs), replace=True)
    b = pd.concat([dd[dd["problem_id"] == p] for p in samp])
    g = b.groupby("transition")["anchor"].mean()
    if 0.0 in g.index and 1.0 in g.index:
        diffs.append(g[1.0] - g[0.0])
d = np.array(diffs)
results["transitions"]["transition_minus_stay_rate_ci"] = {
    "point": round(float(tab.loc[1.0, "mean"] - tab.loc[0.0, "mean"]), 4),
    "low": round(float(np.percentile(d, 2.5)), 4), "high": round(float(np.percentile(d, 97.5)), 4)}
print("[transitions]", json.dumps(results["transitions"]["transition_minus_stay_rate_ci"]))

# ---------------- E: transition predictability (lead0 and lead1) ----------------
tmask = df["transition"].notna()
e0 = {}
e0["position"] = fit_eval_binary("transition", POSITION, None, "transition_pred/position", eligible_only=False, extra_mask=tmask)
e0["act"] = fit_eval_binary("transition", None, A44, "transition_pred/act44", eligible_only=False, extra_mask=tmask)
e0["act_position"] = fit_eval_binary("transition", POSITION, A44, "transition_pred/act44_position", eligible_only=False, extra_mask=tmask)
e0["obs12"] = fit_eval_binary("transition", OBS12, None, "transition_pred/observables12", eligible_only=False, extra_mask=tmask)
results["transition_pred_bootstrap"] = {
    "actpos_vs_pos": {sp: paired_boot(e0["act_position"], e0["position"], sp) for sp in ["validation", "test"]}}
print("[boot transition actpos_vs_pos]", json.dumps(results["transition_pred_bootstrap"]["actpos_vs_pos"]))

# lead-one: activation before sentence i-1 predicting transition at i
lmask = tmask & df["prev_row"].notna()
prev_rows = df["prev_row"].fillna(0).astype(int).to_numpy()
A44_lead1 = A44[prev_rows]  # row i holds activation of sentence i-1's boundary
e1 = {}
e1["act_lead1"] = fit_eval_binary("transition", None, A44_lead1, "transition_pred/act44_lead1", eligible_only=False, extra_mask=lmask)
e1["act_lead1_position"] = fit_eval_binary("transition", POSITION, A44_lead1, "transition_pred/act44_lead1_position", eligible_only=False, extra_mask=lmask)
e1["position_l1rows"] = fit_eval_binary("transition", POSITION, None, "transition_pred/position_lead1_rows", eligible_only=False, extra_mask=lmask)
results["transition_pred_bootstrap"]["lead1_actpos_vs_pos"] = {
    sp: paired_boot(e1["act_lead1_position"], e1["position_l1rows"], sp) for sp in ["validation", "test"]}
print("[boot lead1]", json.dumps(results["transition_pred_bootstrap"]["lead1_actpos_vs_pos"]))

# Oracle control for lead-one: does the activation before sentence i-1 predict a
# transition AT i beyond what knowing stage(i-1) itself would give? (If not, the
# lead-one result is just "predict stage(i-1), then use base transition rates".)
prev_dum = pd.get_dummies(df["prev_stage"]).reindex(columns=STAGES, fill_value=0).astype(float)
for c in STAGES:
    df[f"pstg_{c}"] = prev_dum[c]
PSTG = [f"pstg_{c}" for c in STAGES]
e1["oracle_prevstage"] = fit_eval_binary("transition", PSTG + POSITION, None,
                                         "transition_pred/oracle_prevstage_position",
                                         eligible_only=False, extra_mask=lmask)
e1["act_lead1_oracle"] = fit_eval_binary("transition", PSTG + POSITION, A44_lead1,
                                         "transition_pred/act44_lead1_plus_oracle_prevstage",
                                         eligible_only=False, extra_mask=lmask)
results["transition_pred_bootstrap"]["lead1_act_vs_oracle_prevstage"] = {
    sp: paired_boot(e1["act_lead1_oracle"], e1["oracle_prevstage"], sp)
    for sp in ["validation", "test"]}
print("[boot lead1 vs oracle]",
      json.dumps(results["transition_pred_bootstrap"]["lead1_act_vs_oracle_prevstage"]))

# Within-stage anchor check: activations vs position among compute-stage rows only.
cmask = df["stage"] == "compute"
w = {}
w["act"] = fit_eval_binary("anchor", None, A44, "within_compute/act44", extra_mask=cmask)
w["position"] = fit_eval_binary("anchor", POSITION, None, "within_compute/position", extra_mask=cmask)
results.setdefault("anchor_ladder_bootstrap", {})["within_compute_act_vs_pos"] = {
    sp: paired_boot(w["act"], w["position"], sp) for sp in ["validation", "test"]}
print("[boot within-compute]",
      json.dumps(results["anchor_ladder_bootstrap"]["within_compute_act_vs_pos"]))

# stage prediction lead-one (upcoming stage of i from boundary of i-1)
def fit_multiclass_lead1():
    m_ = df["prev_row"].notna()
    tr = df[m_ & (df["split"] == "train")]
    Xtr = A44_lead1[tr.index]
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), tr["stage"])
    out = {}
    for sp in ["validation", "test"]:
        e = df[m_ & (df["split"] == sp)]
        P = m.predict_proba(sc.transform(A44_lead1[e.index]))
        yp = m.classes_[P.argmax(1)]
        out[sp] = {"accuracy": round(float(accuracy_score(e["stage"], yp)), 4),
                   "macro_f1": round(float(f1_score(e["stage"], yp, average="macro")), 4),
                   "log_loss": round(float(log_loss(e["stage"], P, labels=list(m.classes_))), 4)}
    results["stage_prediction"]["activations_l44_lead1"] = out
    print("[stage/act44_lead1]", json.dumps(out))


fit_multiclass_lead1()

# strip prediction vectors before saving
def strip(o):
    if isinstance(o, dict):
        return {str(k): strip(v) for k, v in o.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    return o


with open(OUT / "analysis_results.json", "w") as f:
    json.dump(strip(results), f, indent=1, default=str)
print("saved", OUT / "analysis_results.json")
