"""EXPLORATORY (not preregistered): answer uncertainty as an "anchor opportunity" prior, and
a two-stage targeted resampling design for Phase 3.

1. On the Phase 2 pooled samples (300 per fork), how well does fork-level uncertainty
   (answer entropy, analytic TV floor, accuracy uncertainty 1-|2p-1|) rank the 7 sentences
   with a real effect (s13: permutation test, BH 10%)? Estimated at full depth and from
   cheap pilots of n = 10, 20, 50 resamples per fork. AP, recall at the top 1/3 and 1/2,
   bootstrap CIs (sentence-level and trajectory-cluster). Held-out check: positives from two
   replicates, pilot from the third.
2. Simulation of a two-stage design calibrated to the pooled data: pilot m per fork on all
   forks, deep resampling to N per fork on forks of units above an uncertainty cutoff,
   permutation TV test + BH 10% over the selected units. Truth: real units keep their
   observed fork-i / fork-(i+1) distributions (optionally shrunk toward the mixture by
   lambda, a winner's-curse sensitivity); null units use the pooled mixture for both forks.
3. Phase 3 sizing: units, traces, generations and $ for ~100 true anchor detections, with
   the Wilson and trajectory-cluster uncertainty on prevalence propagated.

Inputs: Phase 2 completions index (--index), inputs/trajectories.parquet,
derived/rollouts_v2.parquet, results/s13_sentence_effects.csv, configs/phase2_forks.json.
Output: results/s17_targeted_sampling.json. CPU only; a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, RESULTS, STUDY, write_json  # noqa: E402
from s13_phase2_diagnostics import load_samples  # noqa: E402

DEFAULT_INDEX = (Path.home() / "preanchor-root/anchor_reliability_v2/generations/phase2_mirror/"
                 "20261009T0717Z-phase2-23f5e3e6/completions_index.parquet")
COST_PER_GEN = 108.42 / 40000          # $ per generation, Phase 2 average (~$2.71 / 1,000)
GENS_PER_GPU_HOUR = 40000 / 43.5       # one H100, Phase 2 throughput
PILOT_N = [10, 20, 50]
PILOT_DRAWS = 400
BOOT = 2000
PERMS = 2000
Q = 0.10
SIM_REPS = 300
M_GRID = [10, 20, 30]
N_DEEP = [200, 300, 500]
N_UNIFORM = [100, 200, 300, 500, 1000]
CUTOFFS = {"all": 1.0, "top50": 0.5, "top33": 1 / 3}
LAMBDAS = [1.0, 0.75, 0.5]
TARGET_TP = 100
UNITS_PER_TRACE = 19                   # Phase 2 window: 20 forks, 19 labelled sentences


# ---------------------------------------------------------------- helpers
def entropy(counts: np.ndarray) -> np.ndarray:
    """Plug-in Shannon entropy (nats) along the last axis."""
    n = counts.sum(-1, keepdims=True)
    p = np.where(n > 0, counts / np.maximum(n, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log(p), 0)).sum(-1)


def floor_score(counts: np.ndarray) -> np.ndarray:
    """Analytic E[TV] under no effect for 100 vs 100 samples (s13 tv_null_analytic)."""
    n = counts.sum(-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 0.5 * np.sqrt(2 / np.pi) * (np.sqrt(p * (1 - p) * (2 / 100))).sum(-1)


def acc_unc(correct_frac: np.ndarray) -> np.ndarray:
    return 1 - np.abs(2 * correct_frac - 1)


def rank_scores(score: np.ndarray, rng) -> np.ndarray:
    """Order (descending) with random tie-break; returns permutation of indices."""
    return np.lexsort((rng.random(score.shape[-1]), -score))


def ap_and_recall(score: np.ndarray, y: np.ndarray, rng) -> tuple[float, float, float]:
    order = rank_scores(score, rng)
    hits = y[order]
    if hits.sum() == 0:
        return np.nan, np.nan, np.nan
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    ap = float((prec * hits).sum() / hits.sum())
    n = len(y)
    r33 = float(hits[: int(np.ceil(n / 3))].sum() / hits.sum())
    r50 = float(hits[: int(np.ceil(n / 2))].sum() / hits.sum())
    return ap, r33, r50


def perm_p(ca: np.ndarray, cb: np.ndarray, rng, perms: int = PERMS) -> float:
    """Exact-in-distribution permutation p-value for TV between two equal-size count vectors
    (group-a counts under the null are multivariate hypergeometric)."""
    tot = ca + cb
    nz = tot > 0
    colors = tot[nz]
    if len(colors) < 2:
        return 1.0
    n = int(ca.sum())
    obs = np.abs(ca[nz] - cb[nz]).sum()
    x = rng.multivariate_hypergeometric(colors, n, size=perms)
    t = np.abs(2 * x - colors).sum(1)
    return float((1 + (t >= obs).sum()) / (perms + 1))


def bh(p: np.ndarray, q: float = Q) -> np.ndarray:
    passed = np.zeros(len(p), bool)
    if len(p) == 0:
        return passed
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    below = np.nonzero(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def q(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))] if len(a) else [None, None]


# ---------------------------------------------------------------- data
def build(index: Path):
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    s = load_samples(index)
    cats = sorted(s.cat.unique())
    code = {c: i for i, c in enumerate(cats)}
    s["code"] = s.cat.map(code)
    forks = sorted(s.groupby(["trajectory_id", "fork_idx"]).groups)
    fpos = {f: j for j, f in enumerate(forks)}
    k = len(cats)
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s17:order"))
    C = np.zeros((len(forks), 300), int)
    Y = np.zeros((len(forks), 300))
    REP = np.zeros((len(forks), 300), int)
    for (tid, fi), g in s.groupby(["trajectory_id", "fork_idx"]):
        perm = rng.permutation(len(g))      # shuffle so any prefix is a random subsample
        C[fpos[(tid, fi)]] = g.code.to_numpy()[perm]
        Y[fpos[(tid, fi)]] = g.correct.to_numpy(float)[perm]
        REP[fpos[(tid, fi)]] = g.replicate.to_numpy()[perm]
    ia = np.array([fpos[(t, i)] for t, _, i in units])
    ib = np.array([fpos[(t, i + 1)] for t, _, i in units])
    groups = np.array([t for t, _, _ in units])
    eff = pd.read_csv(RESULTS / "s13_sentence_effects.csv")
    lab = {(t, i): bool(v) for t, i, v in zip(eff.trajectory_id, eff.sentence_idx,
                                               eff.real_tv_bh10, strict=True)}
    y = np.array([lab[(t, i)] for t, _, i in units])
    idx = pd.read_parquet(index)
    tok = idx[idx.status == "OK"].groupby(["trajectory_id", "fork_idx"]).generated_token_count.mean()
    tokf = np.array([tok.get(f, np.nan) for f in forks])
    tokf = np.where(np.isfinite(tokf), tokf, np.nanmean(tokf))
    cost_w = tokf / tokf.mean()            # token-weighted relative cost per generation
    return dict(units=units, forks=forks, k=k, C=C, Y=Y, REP=REP, ia=ia, ib=ib, groups=groups,
                y=y, cost_w=cost_w, tokf=tokf)


def counts_of(C: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((C.shape[0], k), int)
    rows = np.repeat(np.arange(C.shape[0]), C.shape[1])
    np.add.at(out, (rows, C.ravel()), 1)
    return out


def fork_metrics(C, Y, k, ia, ib):
    """Uncertainty scores per unit from fork samples C, Y (forks x n)."""
    cnt = counts_of(C, k)
    acc = Y.mean(1)
    pair = cnt[ia] + cnt[ib]
    return {
        "entropy_i": entropy(cnt[ia]), "floor_i": floor_score(cnt[ia]),
        "accunc_i": acc_unc(acc[ia]),
        "entropy_pair": entropy(pair), "floor_pair": floor_score(pair),
        "accunc_pair": acc_unc((acc[ia] + acc[ib]) / 2),
    }


# ---------------------------------------------------------------- task 1
def task1(d) -> dict:
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s17:task1"))
    y, groups, k = d["y"], d["groups"], d["k"]
    utraj = np.unique(groups)
    gidx = [np.nonzero(groups == t)[0] for t in utraj]
    out = {"positives": int(y.sum()), "units": int(len(y)), "base_rate": float(y.mean()),
           "random_expectation": {"ap": float(y.mean()), "recall_top33": 1 / 3,
                                  "recall_top50": 0.5}}
    full = fork_metrics(d["C"], d["Y"], k, d["ia"], d["ib"])
    # Score sets: n=300 deterministic; pilots are random subsamples (rows are pre-shuffled, so
    # draw a fresh random subset each time).
    res = {}
    for n in PILOT_N + [300]:
        draws = 1 if n == 300 else PILOT_DRAWS
        per_draw = []
        for _ in range(draws):
            if n == 300:
                m = full
            else:
                sel = np.argsort(rng.random(d["C"].shape), axis=1)[:, :n]
                m = fork_metrics(np.take_along_axis(d["C"], sel, 1),
                                 np.take_along_axis(d["Y"], sel, 1), k, d["ia"], d["ib"])
            per_draw.append(m)
        res[n] = {}
        for name in full:
            pt = np.array([ap_and_recall(m[name], y, rng) for m in per_draw])
            # Bootstrap: pick a random pilot draw, resample units (or trajectories).
            b_unit, b_clu = [], []
            for _ in range(BOOT):
                m = per_draw[rng.integers(len(per_draw))][name]
                ii = rng.integers(0, len(y), len(y))
                b_unit.append(ap_and_recall(m[ii], y[ii], rng))
                jj = np.concatenate([gidx[t] for t in rng.integers(0, len(utraj), len(utraj))])
                b_clu.append(ap_and_recall(m[jj], y[jj], rng))
            b_unit, b_clu = np.array(b_unit), np.array(b_clu)
            entry = {}
            for c, lab in enumerate(["ap", "recall_top33", "recall_top50"]):
                entry[lab] = {"mean": float(np.nanmean(pt[:, c])),
                              "pilot_draw_95": q(pt[:, c]) if draws > 1 else None,
                              "ci95_sentence_boot": q(b_unit[:, c]),
                              "ci95_trajectory_boot": q(b_clu[:, c])}
            res[n][name] = entry
    out["by_pilot_n"] = {str(n): v for n, v in res.items()}

    # Permutation reference for full-depth scores (labels shuffled across units).
    ref = {}
    for name in full:
        obs = ap_and_recall(full[name], y, rng)
        null = np.array([ap_and_recall(full[name], rng.permutation(y), rng) for _ in range(5000)])
        ref[name] = {"ap": obs[0], "p_ap": float((1 + (null[:, 0] >= obs[0]).sum()) / 5001),
                     "recall_top33": obs[1],
                     "p_recall_top33": float((1 + (null[:, 1] >= obs[1] - 1e-12).sum()) / 5001),
                     "recall_top50": obs[2],
                     "p_recall_top50": float((1 + (null[:, 2] >= obs[2] - 1e-12).sum()) / 5001)}
    out["full_depth_permutation_reference"] = ref
    # Where the positives sit in entropy terciles (full depth).
    ter = pd.qcut(pd.Series(full["entropy_i"]).rank(method="first"), 3, labels=["low", "mid", "high"])
    out["positives_by_entropy_i_tercile"] = pd.Series(y).groupby(ter.to_numpy(), observed=False).sum() \
        .astype(int).to_dict()
    terp = pd.qcut(pd.Series(full["entropy_pair"]).rank(method="first"), 3, labels=["low", "mid", "high"])
    out["positives_by_entropy_pair_tercile"] = pd.Series(y).groupby(terp.to_numpy(), observed=False) \
        .sum().astype(int).to_dict()
    out["positive_units"] = [
        {"trajectory_id": d["units"][j][0], "sentence_idx": int(d["units"][j][2]),
         **{nm: float(full[nm][j]) for nm in full},
         **{f"rank_{nm}": int((full[nm] > full[nm][j]).sum() + 1) for nm in
            ("entropy_i", "entropy_pair")}}
        for j in np.nonzero(y)[0]]
    out["n_units_zero_entropy_i_full"] = int((full["entropy_i"] == 0).sum())

    # Held-out: positives from the two other replicates (200 vs 200), pilot from replicate r.
    held = {}
    for r in (0, 1, 2):
        keep = d["REP"] != r
        Ck = np.array([row[kk] for row, kk in zip(d["C"], keep, strict=True)])
        Cr = np.array([row[~kk] for row, kk in zip(d["C"], keep, strict=True)])
        Yr = np.array([row[~kk] for row, kk in zip(d["Y"], keep, strict=True)])
        cnt = counts_of(Ck, k)
        p = np.array([perm_p(cnt[a], cnt[b], rng) for a, b in zip(d["ia"], d["ib"], strict=True)])
        yy = bh(p)
        h = {"positives": int(yy.sum()), "overlap_with_s13_7": int((yy & y).sum())}
        if yy.sum():
            for n in (10, 20, 50, 100):
                vals = []
                for _ in range(100 if n < 100 else 1):
                    sel = np.argsort(rng.random(Cr.shape), axis=1)[:, :n]
                    m = fork_metrics(np.take_along_axis(Cr, sel, 1),
                                     np.take_along_axis(Yr, sel, 1), k, d["ia"], d["ib"])
                    vals.append([ap_and_recall(m[nm], yy, rng) for nm in
                                 ("entropy_i", "entropy_pair", "accunc_i")])
                v = np.nanmean(np.array(vals), 0)
                h[f"pilot_n{n}"] = {nm: {"ap": float(v[c, 0]), "recall_top33": float(v[c, 1]),
                                         "recall_top50": float(v[c, 2])}
                                    for c, nm in enumerate(("entropy_i", "entropy_pair", "accunc_i"))}
            h["base_rate"] = float(yy.mean())
        held[f"heldout_rep{r}"] = h
    out["heldout_replicate_check"] = held
    return out


# ---------------------------------------------------------------- task 2
def truth_dists(d, lam: float):
    """Per-unit truth (a, b) category distributions on a compact per-unit support."""
    k = d["k"]
    cnt = counts_of(d["C"], k) / 300.0
    pa, pb = cnt[d["ia"]], cnt[d["ib"]]
    mix = (pa + pb) / 2
    real = d["y"][:, None]
    a = np.where(real, mix + lam * (pa - mix), mix)
    b = np.where(real, mix + lam * (pb - mix), mix)
    return a, b


def compact(a, b):
    supp = (a + b) > 0
    kmax = supp.sum(1).max()
    A = np.zeros((len(a), kmax))
    B = np.zeros((len(a), kmax))
    for j in range(len(a)):
        s = np.nonzero(supp[j])[0]
        A[j, : len(s)] = a[j, s]
        B[j, : len(s)] = b[j, s]
    return A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)


def draw_onehot_cum(P: np.ndarray, n: int, rng) -> np.ndarray:
    """Cumulative category counts (units x n x K) for n categorical draws per unit."""
    cdf = np.cumsum(P, 1)
    cdf[:, -1] = 1.0
    u = rng.random((P.shape[0], n))
    codes = (u[:, :, None] > cdf[:, None, :]).sum(-1)
    oh = np.zeros((P.shape[0], n, P.shape[1]), np.int32)
    np.put_along_axis(oh, codes[:, :, None], 1, axis=2)
    return np.cumsum(oh, 1)


def select_top(score: np.ndarray, frac: float, rng) -> np.ndarray:
    if frac >= 1:
        return np.ones(len(score), bool)
    kk = int(np.ceil(frac * len(score)))
    sel = np.zeros(len(score), bool)
    sel[rank_scores(score, rng)[:kk]] = True
    return sel


def task2(d) -> dict:
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s17:sim"))
    y = d["y"]
    n_units, n_forks = len(y), len(d["forks"])
    U2F = np.zeros((n_units, n_forks), bool)
    U2F[np.arange(n_units), d["ia"]] = True
    U2F[np.arange(n_units), d["ib"]] = True
    w = d["cost_w"]
    nmax = max(N_UNIFORM + N_DEEP)
    null_idx, real_idx = np.nonzero(~y)[0], np.nonzero(y)[0]
    truths = {lam: compact(*truth_dists(d, lam)) for lam in LAMBDAS}
    A0, B0 = truths[1.0]
    all_N = sorted(set(N_UNIFORM + N_DEEP))
    configs = [("uniform", None, N, "all", None) for N in N_UNIFORM]
    configs += [("two_stage", m, N, cut, sc) for m in M_GRID for N in N_DEEP
                for cut in ("top50", "top33") for sc in ("entropy_pair", "entropy_i")]
    rec = {lam: {c: [] for c in configs} for lam in LAMBDAS}
    per_pos = {lam: {c: np.zeros(len(real_idx)) for c in configs} for lam in LAMBDAS}
    t0 = time.time()
    for rep in range(SIM_REPS):
        # Nulls: shared across lambda.
        ca_null = draw_onehot_cum(A0[null_idx], nmax, rng)
        cb_null = draw_onehot_cum(B0[null_idx], nmax, rng)
        p_null = {N: np.array([perm_p(ca_null[j, N - 1], cb_null[j, N - 1], rng)
                               for j in range(len(null_idx))]) for N in all_N}
        for lam in LAMBDAS:
            A, B = truths[lam]
            ca_r = draw_onehot_cum(A[real_idx], nmax, rng)
            cb_r = draw_onehot_cum(B[real_idx], nmax, rng)
            ca = np.zeros((n_units, nmax, A.shape[1]), np.int32)
            cb = np.zeros_like(ca)
            ca[null_idx], cb[null_idx] = ca_null, cb_null
            ca[real_idx], cb[real_idx] = ca_r, cb_r
            pv = {}
            for N in all_N:
                p = np.empty(n_units)
                p[null_idx] = p_null[N]
                p[real_idx] = [perm_p(ca_r[j, N - 1], cb_r[j, N - 1], rng)
                               for j in range(len(real_idx))]
                pv[N] = p
            scores = {m: {"entropy_i": entropy(ca[:, m - 1]),
                          "entropy_pair": entropy(ca[:, m - 1] + cb[:, m - 1])} for m in M_GRID}
            for c in configs:
                kind, m, N, cut, sc = c
                if kind == "uniform":
                    sel = np.ones(n_units, bool)
                    gens = N * n_forks
                    cost = N * w.sum()
                else:
                    sel = select_top(scores[m][sc], CUTOFFS[cut], rng)
                    deep_f = U2F[sel].any(0)
                    gens = m * n_forks + (N - m) * deep_f.sum()
                    cost = m * w.sum() + (N - m) * w[deep_f].sum()
                disc = np.zeros(n_units, bool)
                disc[sel] = bh(pv[N][sel])
                tp, fp = int((disc & y).sum()), int((disc & ~y).sum())
                rec[lam][c].append((tp, fp, tp + fp, fp / max(tp + fp, 1), int((sel & y).sum()),
                                    gens, cost))
                per_pos[lam][c] += disc[real_idx]
        if rep == 0:
            print(f"  sim rep 0 took {time.time() - t0:.1f}s", flush=True)
    rows = []
    for lam in LAMBDAS:
        for c in configs:
            a = np.array(rec[lam][c], float)
            kind, m, N, cut, sc = c
            rows.append({
                "lambda": lam, "design": kind, "m": m, "N": N, "cutoff": cut, "score": sc,
                "E_TP": a[:, 0].mean(), "sd_TP": a[:, 0].std(), "E_FP": a[:, 1].mean(),
                "E_disc": a[:, 2].mean(), "mean_FDP": a[:, 3].mean(),
                "P_anyFP": float((a[:, 1] > 0).mean()), "E_pos_selected": a[:, 4].mean(),
                "generations": a[:, 5].mean(),
                "cost_flat_usd": a[:, 5].mean() * COST_PER_GEN,
                "cost_tokw_usd": a[:, 6].mean() * COST_PER_GEN,
                "gpu_hours": a[:, 5].mean() / GENS_PER_GPU_HOUR,
                "per_positive_power": (per_pos[lam][c] / SIM_REPS).round(3).tolist()})
    df = pd.DataFrame(rows)
    df["TP_per_1000usd"] = df.E_TP / df.cost_flat_usd * 1000
    df["TP_per_1000usd_tokw"] = df.E_TP / df.cost_tokw_usd * 1000
    return {"reps": SIM_REPS, "perms": PERMS, "q": Q, "units": n_units, "forks": n_forks,
            "real_units": [f"{d['units'][j][0]}:s{d['units'][j][2]}" for j in real_idx],
            "token_weight_range": [float(w.min()), float(w.max())],
            "table": df.to_dict(orient="records")}


# ---------------------------------------------------------------- task 3
def task3(d, sim: dict) -> dict:
    y, groups = d["y"], d["groups"]
    k, n = int(y.sum()), len(y)
    p_hat = k / n
    wl = wilson(k, n)
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s17:prev"))
    utraj = np.unique(groups)
    gidx = [np.nonzero(groups == t)[0] for t in utraj]
    boot = []
    for _ in range(10000):
        jj = np.concatenate([gidx[t] for t in rng.integers(0, len(utraj), len(utraj))])
        boot.append(y[jj].mean())
    cl = q(boot)
    df = pd.DataFrame(sim["table"])
    base = df[(df.design == "uniform") & (df.N == 300)].set_index("lambda").E_TP
    rows = []
    for _, r in df.iterrows():
        rel = r.E_TP / base[r["lambda"]]          # power relative to the design that found 7/190
        cost_unit = r.cost_flat_usd / n
        gens_unit = r.generations / n
        out = {"lambda": r["lambda"], "design": r.design, "m": r.m, "N": r.N, "cutoff": r.cutoff,
               "score": r.score, "rel_power_vs_U300": rel,
               "fdp": r.mean_FDP, "cost_per_unit_usd": cost_unit}
        for lab, prev in (("point", p_hat), ("wilson_lo", wl[0]), ("wilson_hi", wl[1]),
                          ("cluster_lo", cl[0]), ("cluster_hi", cl[1])):
            yld = prev * rel                      # true detections per labelled unit
            units = TARGET_TP / yld if yld > 0 else np.inf
            out[f"units_{lab}"] = units
            out[f"traces_{lab}"] = units / UNITS_PER_TRACE
            out[f"gens_{lab}"] = units * gens_unit
            out[f"usd_{lab}"] = units * cost_unit
            out[f"gpu_h_{lab}"] = units * gens_unit / GENS_PER_GPU_HOUR
        # Direct simulation yield (treats the 7 as the complete truth).
        out["units_direct_sim"] = TARGET_TP / (r.E_TP / n) if r.E_TP > 0 else np.inf
        out["usd_direct_sim"] = out["units_direct_sim"] * cost_unit
        out["expected_false_discoveries_at_target"] = TARGET_TP * r.E_FP / max(r.E_TP, 1e-9)
        rows.append(out)
    return {"prevalence": {"k": k, "n": n, "p_hat": p_hat, "wilson95": list(wl),
                           "trajectory_cluster_boot95": cl,
                           "note": "prevalence of effects detectable at uniform N=300, BH 10%"},
            "target_true_positives": TARGET_TP, "units_per_trace": UNITS_PER_TRACE,
            "table": rows}


# ---------------------------------------------------------------- MVP context
def mvp_context(d) -> dict:
    """How much of whole MVP traces looks like the high-uncertainty forks of Phase 2?"""
    k_full = fork_metrics(d["C"], d["Y"], d["k"], d["ia"], d["ib"])["entropy_i"]
    thr = {"p2_median": float(np.median(k_full)),
           "p2_top_tercile": float(np.quantile(k_full, 2 / 3))}
    # Phase 2 entropy at n=100 for a like-for-like comparison with the MVP's 100 resamples.
    k100 = fork_metrics(d["C"][:, :100], d["Y"][:, :100], d["k"], d["ia"], d["ib"])["entropy_i"]
    thr100 = {"p2_median_n100": float(np.median(k100)),
              "p2_top_tercile_n100": float(np.quantile(k100, 2 / 3))}
    r = pd.read_parquet(DERIVED / "rollouts_v2.parquet", columns=["trajectory_id", "fork_idx", "cat"])
    ent = r.groupby(["trajectory_id", "fork_idx"]).cat.agg(
        lambda c: float(entropy(c.value_counts().to_numpy()[None, :])[0]))
    per_traj = ent.groupby(level=0).size()
    return {"mvp_forks": int(len(ent)), "mvp_traces": int(per_traj.size),
            "mvp_forks_per_trace_median": float(per_traj.median()),
            "thresholds": {**thr, **thr100},
            "frac_mvp_forks_above_p2_median_n100": float((ent > thr100["p2_median_n100"]).mean()),
            "frac_mvp_forks_above_p2_top_tercile_n100":
                float((ent > thr100["p2_top_tercile_n100"]).mean()),
            "frac_mvp_forks_zero_entropy": float((ent == 0).mean()),
            "frac_p2_forks_zero_entropy_n100": float((k100 == 0).mean())}


def main() -> None:
    global SIM_REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--reps", type=int, default=SIM_REPS)
    args = ap.parse_args()
    SIM_REPS = args.reps
    t0 = time.time()
    d = build(args.index)
    print(f"loaded {len(d['units'])} units, {len(d['forks'])} forks, {d['k']} categories, "
          f"{int(d['y'].sum())} positives ({time.time() - t0:.0f}s)", flush=True)
    out = {"stage": "exploratory_not_preregistered", "script": "s17_targeted_sampling.py",
           "cost_per_generation_usd": COST_PER_GEN, "gens_per_gpu_hour": GENS_PER_GPU_HOUR}
    out["task1_uncertainty_prior"] = task1(d)
    print(f"task1 done ({time.time() - t0:.0f}s)", flush=True)
    out["mvp_context"] = mvp_context(d)
    out["task2_simulation"] = task2(d)
    print(f"task2 done ({time.time() - t0:.0f}s)", flush=True)
    out["task3_sizing"] = task3(d, out["task2_simulation"])
    write_json(RESULTS / "s17_targeted_sampling.json", out)
    print(f"wrote results/s17_targeted_sampling.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
