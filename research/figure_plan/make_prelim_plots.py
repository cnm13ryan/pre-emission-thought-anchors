"""Preliminary validation plots for the proposed main-text figures.

Not publication figures: these check that each proposed encoding is supported
by the actual data (discreteness, sample size, dynamic range, readability).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path.home() / "preanchor-root"
REPO = Path("/Users/cnm13ryan/git/pre-emission-thought-anchors")
OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# validated palette (dataviz reference, light mode)
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
GRAY, INK, MUT = "#9a9891", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": MUT, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6})

feat = pd.read_parquet(ROOT / "data/processed/stage7/features.parquet")
el = feat[feat.eligible].copy()
el["relrank"] = el.groupby("trajectory_id")["sentence_idx"].rank(pct=True)
pt = pd.read_parquet(ROOT / "artifacts/runs/20260915T201107Z-484989d2fe-19f1ab0d/predictions/test.parquet")
pv = pd.read_parquet(ROOT / "artifacts/runs/20260915T201107Z-484989d2fe-19f1ab0d/predictions/validation.parquet")
cv = pd.read_parquet(ROOT / "artifacts/runs/20260915T193326Z-57bcb41c84-7ebd0db6/probe_models/cv_results.parquet")
perm = pd.read_parquet(ROOT / "artifacts/runs/20260915T211903Z-b230473011-815ca2b7/metrics/permutations.parquet")
lt = pd.read_parquet(ROOT / "artifacts/runs/20260915T211903Z-b230473011-815ca2b7/predictions/lead_time.parquet")
rs = json.loads((REPO / "research/reasoning_stage/outputs/analysis_results.json").read_text())

# ------------------------------------------------------------------ P1: phenomenon + confound
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ax = axes[0]
bins = np.linspace(0, 1, 11)
for split, color, lab in [("train", GRAY, "train (265 anchors)"),
                          ("validation", ORANGE, "validation (54)"),
                          ("test", BLUE, "test (25)")]:
    d = el[el.split == split]
    idx = np.digitize(d.relrank, bins[1:-1])
    rate = d.groupby(idx).anchor.mean()
    ax.stairs(rate.values, bins, color=color, lw=2, label=lab)
ax.axhline(0.10, color=GRAY, ls=":", lw=1)
ax.text(0.99, 0.102, "train prevalence 0.10", ha="right", va="bottom", fontsize=7, color=MUT)
ax.set_xlabel("relative position within trajectory")
ax.set_ylabel("anchor rate")
ax.set_title("a. anchors concentrate early", fontsize=9)
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
d = el.dropna(subset=["importance_primary"])
ax.scatter(d.relrank, d.importance_primary, s=3, alpha=0.15, color=BLUE, edgecolors="none")
ax.axhline(1.35498, color=RED, lw=1.2, ls="--")
ax.text(0.02, 1.45, "anchor threshold (train q90)", fontsize=7, color=RED)
ax.set_yscale("symlog", linthresh=0.1)
ax.set_xlabel("relative position within trajectory")
ax.set_ylabel("counterfactual importance (symlog)")
ax.set_title("b. importance vs position (all eligible)", fontsize=9)

ax = axes[2]
hj = el[el.split.isin(["validation", "test"])]
trajs = sorted(hj.trajectory_id.unique(),
               key=lambda t: (hj[hj.trajectory_id == t].split.iloc[0], t))
for i, t in enumerate(trajs):
    d = hj[hj.trajectory_id == t]
    color = BLUE if d.split.iloc[0] == "test" else ORANGE
    ax.hlines(i, 0, 1, color="#e8e7e3", lw=4)
    a = d[d.anchor]
    ax.scatter(a.relrank, np.full(len(a), i), marker="|", s=90, color=color, lw=1.4)
ax.set_yticks(range(len(trajs)))
ax.set_yticklabels([t.replace("problem_", "")[:14] for t in trajs], fontsize=5.5)
ax.set_xlabel("relative position")
ax.set_title("c. anchor positions, held-out trajectories", fontsize=9)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "P1_phenomenon.png", dpi=150)

# ------------------------------------------------------------------ P2: cumulative gain (retrieval)
MODELS = [("activation_plus_observables", BLUE, "activations + observables"),
          ("observable_combined", ORANGE, "observables only"),
          ("position", AQUA, "position only")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
for ax, (df, split, npos) in zip(axes, [(pt, "test", 25), (pv, "validation", 54)]):
    dfb = df[df.task == "binary"]
    n = dfb.sentence_id.nunique()
    for name, color, lab in MODELS:
        d = dfb[dfb.probe_name == name].sort_values("score_raw", ascending=False)
        found = np.cumsum(d.y_true.values)
        frac = np.arange(1, len(d) + 1) / len(d)
        ax.step(frac, found, where="post", color=color, lw=1.8, label=lab)
    ax.plot([0, 1], [0, npos], color=GRAY, ls=":", lw=1.2, label="chance")
    ax.axvline(0.1, color=MUT, lw=0.7, ls="--")
    ax.text(0.105, npos * 0.95, "top 10%", fontsize=7, color=MUT)
    ax.set_xlabel("fraction of sentences flagged (ranked by score)")
    ax.set_ylabel(f"anchors found (of {npos})")
    ax.set_title(f"{split}: n={n}, positives={npos}", fontsize=9)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "P2_cumulative_gain.png", dpi=150)

# ------------------------------------------------------------------ P3: incremental info: paired AP + bootstrap + permutation
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
# (a) per-problem paired AP: act+obs vs each baseline
ax = axes[0]
pts = []
for df, split, marker in [(pv, "validation", "D"), (pt, "test", "o")]:
    dfb = df[df.task == "binary"]
    for pid, d in dfb.groupby("problem_id"):
        act = d[d.probe_name == "activation_plus_observables"]
        obs = d[d.probe_name == "observable_combined"]
        pos = d[d.probe_name == "position"]
        if act.y_true.sum() == 0:
            continue
        ap_a = average_precision_score(act.y_true, act.score_raw)
        ap_o = average_precision_score(obs.y_true, obs.score_raw)
        ap_p = average_precision_score(pos.y_true, pos.score_raw)
        prev = act.y_true.mean()
        pts.append((split, pid, ap_a, ap_o, ap_p, prev, int(act.y_true.sum())))
        ax.scatter(ap_o, ap_a, marker=marker, s=40, color=BLUE,
                   facecolors="none" if split == "test" else BLUE, lw=1.4)
        ax.annotate(f"{pid}", (ap_o, ap_a), fontsize=6, color=MUT, xytext=(3, 3), textcoords="offset points")
lim = 0.45
ax.plot([0, lim], [0, lim], color=GRAY, ls=":", lw=1)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("AP, observables only")
ax.set_ylabel("AP, activations + observables")
ax.set_title("a. per-problem paired AP (8 held-out problems)\nfilled=validation, open=test", fontsize=8.5)

# (b) pooled deltas with problem-bootstrap CIs (from stage10 sensitivities, precomputed)
ax = axes[1]
rows = [("test: vs observables", 0.036796, -0.050545, 0.081401, BLUE),
        ("val: vs observables", 0.037883, -0.020284, 0.058989, BLUE),
        ("test: vs position", -0.045239, -0.236435, 0.025501, AQUA),
        ("val: vs position", 0.031504, -0.069913, 0.085680, AQUA)]
for i, (lab, v, lo, hi, c) in enumerate(rows):
    y = len(rows) - i
    ax.plot([lo, hi], [y, y], color=c, lw=2)
    ax.scatter([v], [y], color=c, s=35, zorder=3)
    ax.text(-0.30, y, lab, ha="right", va="center", fontsize=7.5)
ax.axvline(0, color=GRAY, ls=":", lw=1)
ax.set_yticks([])
ax.set_xlim(-0.3, 0.15)
ax.set_xlabel("Δ AUCPR (activations+obs − baseline), problem bootstrap 95%")
ax.set_title("b. incremental gain, pooled", fontsize=9)
ax.grid(axis="y", visible=False)

# (c) permutation null, test
ax = axes[2]
p = perm[(perm.kind == "permutation") & (perm.split == "test")].average_precision
ax.hist(p, bins=24, color="#b7d3f6", edgecolor="#fcfcfb")
ax.axvline(0.094121, color=BLUE, lw=2)
ax.text(0.096, ax.get_ylim()[1] * 0.9, "observed 0.094\n(p≈0.17)", fontsize=7.5, color=BLUE)
ax.axvline(np.median(p), color=MUT, lw=1, ls=":")
ax.set_xlabel("AUCPR under stratified label permutation (100 draws)")
ax.set_ylabel("count")
ax.set_title("c. structured permutation null (test)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "P3_incremental.png", dpi=150)
pd.DataFrame(pts, columns=["split", "problem", "ap_act", "ap_obs", "ap_pos", "prev", "npos"]).to_csv(OUT / "per_problem_ap.csv", index=False)

# ------------------------------------------------------------------ P4: layer sweep with folds + stage contrast
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [2.4, 1]})
ax = axes[0]
act = cv[(cv.probe_name == "activation_only") & (cv.task == "binary")].dropna(subset=["layer"])
best = (act.groupby(["layer", "hyperparameter"]).average_precision.mean().reset_index()
        .sort_values("average_precision").groupby("layer").tail(1).sort_values("layer"))
for _, r in best.iterrows():
    folds = act[(act.layer == r.layer) & (act.hyperparameter == r.hyperparameter)].average_precision
    ax.scatter(np.full(len(folds), r.layer), folds, s=7, color=BLUE, alpha=0.35, edgecolors="none")
ax.plot(best.layer, best.average_precision, color=BLUE, lw=2, label="mean of 4 CV folds (best C)")
ax.axhline(0.10, color=GRAY, ls=":", lw=1.2)
ax.text(47.5, 0.102, "train prevalence", ha="right", fontsize=7, color=MUT)
ax.axvline(44, color=RED, lw=0.8, ls="--")
ax.text(43.4, 0.30, "selected site 44", rotation=90, fontsize=7, color=RED, va="top")
ax.set_xlabel("extraction site (0 = embedding output … 48 = final layer)")
ax.set_ylabel("cross-validated AUCPR (anchor task)")
ax.set_title("a. anchor probe: flat, fold-unstable layer profile", fontsize=9)
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
sp = rs["stage_prediction"]
x = [0, 44]
f1 = [sp["activations_l0"]["validation"]["macro_f1"], sp["activations_l44"]["validation"]["macro_f1"]]
ax.plot(x, f1, "-o", color=VIOLET, lw=2, ms=6, label="activations")
ax.axhline(sp["observables12"]["validation"]["macro_f1"], color=ORANGE, ls="--", lw=1.4)
ax.text(1, sp["observables12"]["validation"]["macro_f1"] + 0.01, "observables (12)", fontsize=7, color=ORANGE)
ax.axhline(sp["position_only"]["validation"]["macro_f1"], color=AQUA, ls="--", lw=1.4)
ax.text(1, sp["position_only"]["validation"]["macro_f1"] + 0.01, "position", fontsize=7, color=AQUA)
ax.set_xticks([0, 44])
ax.set_xlabel("extraction site")
ax.set_ylabel("macro-F1, upcoming-stage decoding (val)")
ax.set_title("b. stage decoding: strongly\nlayer-dependent (2 sites probed)", fontsize=9)
ax.set_ylim(0, 0.55)
fig.tight_layout()
fig.savefig(OUT / "P4_layers.png", dpi=150)

# ------------------------------------------------------------------ P5: two tasks, same state: AP ladders
fig, ax = plt.subplots(figsize=(8.4, 3.6))
tp = rs["transition_pred"]
al = rs["anchor_ladder"]
rows = [
    ("anchor  · test  (prev 0.036)", 0.0363, al["position"]["test"]["ap"], 0.0573, al["act44"]["test"]["ap"]),
    ("anchor  · val   (prev 0.079)", 0.0794, al["position"]["validation"]["ap"], 0.1266, al["act44"]["validation"]["ap"]),
    ("transition · test (base 0.601)", 0.601, tp["position"]["test"]["ap"], tp["observables12"]["test"]["ap"], tp["act44"]["test"]["ap"]),
    ("transition · val  (base 0.475)", 0.475, tp["position"]["validation"]["ap"], tp["observables12"]["validation"]["ap"], tp["act44"]["validation"]["ap"]),
]
for i, (lab, prev, pos, obs, actv) in enumerate(rows):
    y = len(rows) - i
    ax.plot([prev, max(prev, pos, obs, actv)], [y, y], color="#e8e7e3", lw=5, zorder=1, solid_capstyle="round")
    ax.scatter([prev], [y], marker="|", s=200, color=INK, zorder=3)
    ax.scatter([pos], [y], s=45, color=AQUA, zorder=3)
    ax.scatter([obs], [y], s=45, color=ORANGE, zorder=3)
    ax.scatter([actv], [y], s=55, color=BLUE, zorder=4)
    ax.text(-0.02, y, lab, ha="right", va="center", fontsize=8)
ax.set_yticks([])
ax.set_xlim(-0.02, 1.0)
ax.set_xlabel("AUCPR (| = prevalence/no-information)")
ax.set_title("Same pre-emission state, two targets: importance (anchor) vs form (stage transition)", fontsize=9.5)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
                   [(AQUA, "position"), (ORANGE, "observables"), (BLUE, "activations L44")]],
          fontsize=7.5, frameon=False, loc="center right")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "P5_two_tasks.png", dpi=150)

# ------------------------------------------------------------------ P6: lead time, both splits, both tasks
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax = axes[0]
lta = lt[lt.available & (lt.probe_name.isin(["activation_plus_observables", "observable_combined"]))]
for split, c in [("test", BLUE), ("validation", ORANGE)]:
    aps, prevs = [], []
    for lead in [0, 1]:
        pn = "activation_plus_observables"
        d = lta[(lta.split == split) & (lta.lead_sentences == lead) & (lta.probe_name == pn)]
        aps.append(average_precision_score(d.y_true, d.score_raw))
        prevs.append(d.y_true.mean())
    ax.plot([0, 1], aps, "-o", color=c, lw=2, ms=6, label=f"{split}")
    ax.plot([0, 1], prevs, ls=":", color=c, lw=1.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["lead 0\n(at boundary)", "lead 1\n(one sentence earlier)"])
ax.set_ylabel("AUCPR (anchor)")
ax.set_title("a. anchors: lead-1 collapses on test only\n(dotted = prevalence)", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
for split, c in [("test", BLUE), ("validation", ORANGE)]:
    a0 = tp["act44"][split]["ap"]
    a1 = tp["act44_lead1"][split]["ap"]
    oracle = tp["oracle_prevstage_position"][split]["ap"]
    base = 0.601 if split == "test" else 0.475
    ax.plot([0, 1], [a0, a1], "-o", color=c, lw=2, ms=6, label=split)
    ax.scatter([1], [oracle], marker="x", s=50, color=c)
    ax.axhline(base, ls=":", color=c, lw=1.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["lead 0", "lead 1"])
ax.set_ylabel("AUCPR (stage transition)")
ax.set_ylim(0.4, 0.9)
ax.set_title("b. transitions: lead-1 holds, but oracle (×)\nexplains most of it", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "P6_lead_time.png", dpi=150)

print("saved:", sorted(p.name for p in OUT.glob("*.png")))
