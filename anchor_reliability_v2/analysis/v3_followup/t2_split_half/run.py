"""v3 follow-up T2 (item 1): split-half reliability of resampling importance. EXPLORATORY, POST HOC.

Datasets (read-only shared tables under DATA/derived/v3_followup/):
  A  released MVP: every sentence i with forks i and i+1 (6,434 sentences, 20 problems,
     40 traces), 100 samples per fork, split 50/50 within each fork.
  B  Phase 2 pooled: released r0 (window forks) + r1 + r2 = 300 per fork, 190 units
     (10 problems, one trace each), split 150/150 within each fork.

Each of 1,000 random splits partitions every fork's samples independently into halves A/B
(fork i and fork i+1 are split independently). Per half, per sentence:
  fork_tv      TV(fork i, fork i+1), canonical categories (primary)
  tv_debiased  fork_tv - s13.tv_null_analytic (computed from the same half)
  floor        s13.tv_null_analytic itself (context: the noise floor)
  tv_without   TV(fork-i samples with similarity < 0.8, fork i+1)
  abs_dacc, signed_dacc   accuracy(fork i+1) - accuracy(fork i), all fork-i samples
  kl_log1p     log1p of s12-style KL(without || fork i+1), alpha = 1e-9, k = categories seen
  acc_level    accuracy at fork i (positive control)
  fork_tv_raw  fork_tv on raw categories
Scores are rounded to 10 decimals before any rank statistic.

Statistics per split: pooled Spearman and ICC(2,1) between halves; per-problem Spearman /
ICC (both traces of a problem pooled); covariance decomposition (between/within problem).
Split-averaged values are Spearman-Brown corrected (r > 0 only; otherwise reported raw).
Intervals: problem-cluster bootstrap (2,000 draws; the draw sequence is exactly that of
R.cluster_bootstrap with the stated seed). Inside each draw the statistic is recomputed on
the resampled problems for all 1,000 splits and then averaged:
  * ICC / covariances exactly, from per-(split, problem) sufficient statistics;
  * pooled Spearman exactly, as a multiplicity-weighted Spearman (average ranks in the
    expanded multiset, then weighted Pearson) — identical to re-ranking the resample —
    on the first 200 splits (point estimate uses all 1,000);
  * within-problem mean Spearman by resampling the split-averaged per-problem values.

Usage: python analysis/v3_followup/t2_split_half/run.py [--splits 1000] [--reps 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DATA, RESULTS, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent
SHARED = DATA / "derived/v3_followup"
CACHE = SHARED / "t2_split_half"
TASK = "t2_split_half"
ALPHA = 1e-9
SIM_THR = 0.8
SCORES = ["fork_tv", "tv_debiased", "floor", "tv_without", "abs_dacc", "signed_dacc",
          "kl_log1p", "acc_level", "fork_tv_raw"]
# Fixed before running: projection targets (samples per fork) and label benchmarks.
N_TARGETS = [100, 200, 300, 500]
BENCH = {"spearman": 0.80, "icc": 0.75}  # study benchmarks (s13 projection targets)
# Scores whose true value does not depend on n (SB projection beyond full n is meaningful);
# fork_tv / floor / tv_without / kl carry an n-dependent finite-sample bias.
# Pooled-Spearman bootstrap re-ranks inside every draw on the first 200 splits only (runtime);
# split-to-split SD is ~0.01, so the 200-split mean differs from the 1,000-split mean by ~0.001.
BOOT_SPLITS_SPEARMAN = 200
SB_N_VALID = {"tv_debiased", "abs_dacc", "signed_dacc", "acc_level"}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s13 = load("s13_phase2_diagnostics")


def sb(r, m):
    """Spearman-Brown by factor m, applied only where r > 0 (else r returned unchanged)."""
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(r > 0, m * r / (1 + (m - 1) * r), r)
    return out


def fl(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ----------------------------------------------------------------------------- data


def dataset_tables():
    rel = pd.read_parquet(SHARED / "released_canon.parquet",
                          columns=["trajectory_id", "problem_id", "fork_idx", "sample_idx",
                                   "cat_canon", "cat_raw", "correct_canon", "similarity"])
    p2 = pd.read_parquet(SHARED / "phase2_canon.parquet",
                         columns=["replicate", "trajectory_id", "fork_idx", "sample_idx",
                                  "cat_canon", "cat_raw", "correct_canon", "similarity"])
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    window = {(t["trajectory_id"], f) for t in plan["trajectories"] for f in t["forks"]}
    units_b = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
               for i in t["sentence_labels"]]
    keep = [(t, int(f)) in window for t, f in zip(rel.trajectory_id, rel.fork_idx, strict=True)]
    rw = rel[keep].drop(columns="problem_id").assign(replicate=0)
    pool_b = pd.concat([rw, p2[rw.columns]], ignore_index=True)
    forks_a = set(zip(rel.trajectory_id, rel.fork_idx.astype(int)))
    prob = dict(zip(rel.trajectory_id, rel.problem_id, strict=False))
    units_a = sorted((t, int(prob[t]), i) for t, i in forks_a if (t, i + 1) in forks_a)
    sel_a = rel.drop(columns="problem_id")
    return {"A": (sel_a, units_a), "B": (pool_b, units_b)}


# ----------------------------------------------------------------------------- half scores


def traj_half_scores(g: pd.DataFrame, units_i: list[int], S: int, rng) -> dict:
    """All splits for one trajectory. Returns score arrays (2, S, U) for halves A/B."""
    g = g.sort_values(["fork_idx", "replicate", "sample_idx"])
    forks = sorted(g.fork_idx.unique())
    fpos = {f: j for j, f in enumerate(forks)}
    ccodes = {c: j for j, c in enumerate(sorted(g.cat_canon.unique()))}
    rcodes = {c: j for j, c in enumerate(sorted(g.cat_raw.unique()))}
    K, KR = len(ccodes), len(rcodes)
    F = len(forks)
    ns = g.groupby("fork_idx").size().reindex(forks).to_numpy()
    assert (ns == ns[0]).all(), "unequal fork sizes"
    n = int(ns[0])
    h = n // 2
    cc = g.cat_canon.map(ccodes).to_numpy().reshape(F, n)
    rc = g.cat_raw.map(rcodes).to_numpy().reshape(F, n)
    y = g.correct_canon.to_numpy(float).reshape(F, n)
    wo = (g.similarity.to_numpy(float) < SIM_THR).reshape(F, n).astype(float)
    eye, eyer = np.eye(K, dtype=np.float32), np.eye(KR, dtype=np.float32)
    O, OR = eye[cc], eyer[rc]                      # (F, n, K)
    OW = O * wo[:, :, None].astype(np.float32)
    # masks: for each fork independently, a uniformly random half of size h
    keys = rng.random((F, S, n))
    M = (np.argsort(np.argsort(keys, axis=2), axis=2) < h).astype(np.float32)  # (F, S, n)
    tot = (O.sum(1), OR.sum(1), OW.sum(1), y.sum(1), wo.sum(1))
    CA = np.matmul(M, O)            # (F, S, K)
    RA = np.matmul(M, OR)
    WA = np.matmul(M, OW)
    YA = np.einsum("fsn,fn->fs", M, y)
    NWA = np.einsum("fsn,fn->fs", M, wo)
    halves = {0: (CA, RA, WA, YA, NWA, h),
              1: (tot[0][:, None] - CA, tot[1][:, None] - RA, tot[2][:, None] - WA,
                  tot[3][:, None] - YA, tot[4][:, None] - NWA, n - h)}
    ii = np.array([fpos[i] for i in units_i])
    jj = np.array([fpos[i + 1] for i in units_i])
    out = {s: np.empty((2, S, len(units_i))) for s in SCORES}
    for half, (C, Rr, W, Y, NW, nh) in halves.items():
        Pi, Pj = C[ii].astype(float) / nh, C[jj].astype(float) / nh  # (U, S, K)
        tvv = 0.5 * np.abs(Pi - Pj).sum(-1)
        p = (C[ii] + C[jj]).astype(float) / (2 * nh)
        floor = 0.5 * (np.sqrt(p * (1 - p) * (2 / nh)) * np.sqrt(2 / np.pi)).sum(-1)
        nw = NW[ii]
        with np.errstate(invalid="ignore", divide="ignore"):
            Wi = W[ii].astype(float)
            tvw = 0.5 * np.abs(Wi / nw[..., None] - Pj).sum(-1)
            Gj = C[jj].astype(float)
            present = (Wi + Gj) > 0
            k = present.sum(-1, keepdims=True)
            pp = (Wi + ALPHA) / (nw[..., None] + ALPHA * k)
            qq = (Gj + ALPHA) / (nh + ALPHA * k)
            kl = np.where(present, pp * np.log(pp / qq), 0.0).sum(-1)
        kl = np.maximum(kl, 0.0)
        tvw[nw == 0] = np.nan
        kl[nw == 0] = np.nan
        ai, aj = Y[ii] / nh, Y[jj] / nh
        rtv = 0.5 * np.abs(Rr[ii].astype(float) / nh - Rr[jj].astype(float) / nh).sum(-1)
        vals = {"fork_tv": tvv, "tv_debiased": tvv - floor, "floor": floor, "tv_without": tvw,
                "abs_dacc": np.abs(aj - ai), "signed_dacc": aj - ai, "kl_log1p": np.log1p(kl),
                "acc_level": ai, "fork_tv_raw": rtv}
        for s in SCORES:
            out[s][half] = np.round(vals[s].T, 10)   # (S, U)
    return out


def all_half_scores(df: pd.DataFrame, units, S: int, seed: int, chunk: int = 250):
    rng = np.random.default_rng(seed)
    by_t: dict = {}
    for t, _, i in units:
        by_t.setdefault(t, []).append(i)
    order = sorted(by_t)
    res = {s: [] for s in SCORES}
    rows = []
    groups = dict(df.groupby("trajectory_id").__iter__())
    for t in order:
        parts = [traj_half_scores(groups[t], by_t[t], min(chunk, S - c0), rng)
                 for c0 in range(0, S, chunk)]
        for s in SCORES:
            res[s].append(np.concatenate([p[s] for p in parts], axis=1))
        rows += [(t, i) for i in by_t[t]]
    prob = {t: p for t, p, _ in units}
    meta = pd.DataFrame(rows, columns=["trajectory_id", "sentence_idx"])
    meta["problem_id"] = meta.trajectory_id.map(prob)
    out = {s: np.concatenate(res[s], axis=2) for s in SCORES}
    # order sentences by problem so group slices are contiguous
    o = np.argsort(meta.problem_id.to_numpy(), kind="stable")
    meta = meta.iloc[o].reset_index(drop=True)
    return {s: v[:, :, o] for s, v in out.items()}, meta


# ----------------------------------------------------------------------------- statistics


def wspearman(a: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Spearman of (a, b) for each weight row of W (B, N): the rank correlation of the
    multiset in which row j appears W[., j] times (average ranks). NaN pairs get weight 0."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, W = a[ok], b[ok], W[:, ok].astype(float)
    if len(a) < 3:
        return np.full(W.shape[0], np.nan)

    def ranks(x):
        o = np.argsort(x, kind="stable")
        xs = x[o]
        start = np.r_[True, xs[1:] != xs[:-1]]
        gid = np.cumsum(start) - 1
        Ws = W[:, o]
        cum = np.cumsum(Ws, axis=1)
        gtot = np.add.reduceat(Ws, np.flatnonzero(start), axis=1)
        before = np.concatenate([np.zeros((W.shape[0], 1)), cum[:, np.flatnonzero(start)[1:] - 1]],
                                axis=1)
        rg = before + (gtot + 1) / 2
        r = np.empty_like(Ws)
        r[:, o] = rg[:, gid]
        return r

    ra, rb = ranks(a), ranks(b)
    sw = W.sum(1, keepdims=True)
    ma, mb = (W * ra).sum(1, keepdims=True) / sw, (W * rb).sum(1, keepdims=True) / sw
    da, db = ra - ma, rb - mb
    cov = (W * da * db).sum(1)
    va, vb = (W * da * da).sum(1), (W * db * db).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r[(va <= 1e-12) | (vb <= 1e-12)] = np.nan
    return r


def suff(A: np.ndarray, B: np.ndarray, gidx: np.ndarray, P: int) -> np.ndarray:
    """Per-(split, problem) sufficient statistics [n, Sa, Sb, Saa, Sbb, Sab] with NaN masking."""
    ok = np.isfinite(A) & np.isfinite(B)
    a, b = np.where(ok, A, 0.0), np.where(ok, B, 0.0)
    out = np.zeros((A.shape[0], P, 6))
    for j, v in enumerate((ok.astype(float), a, b, a * a, b * b, a * b)):
        for p in range(P):
            out[:, p, j] = v[:, gidx == p].sum(1)
    return out


def icc_from(st: np.ndarray) -> np.ndarray:
    """ICC(2,1) for two raters from sufficient statistics (..., 6)."""
    n, sa, sb_, saa, sbb, sab = np.moveaxis(st, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        G = (sa + sb_) / (2 * n)
        sst = saa + sbb - 2 * n * G * G
        ssr = 2 * ((saa + 2 * sab + sbb) / 4 - n * G * G)
        ssc = n * ((sa / n - G) ** 2 + (sb_ / n - G) ** 2)
        sse = sst - ssr - ssc
        msr, msc, mse = ssr / (n - 1), ssc, sse / (n - 1)
        den = msr + mse + 2 * (msc - mse) / n
        out = np.where((den > 0) & (n >= 3), (msr - mse) / den, np.nan)
    return out


def decomp_from(st: np.ndarray) -> dict:
    """Covariance (reliable variance at half-n) split into between/within problem parts.
    st: (..., P, 6). Returns arrays over leading dims."""
    n, sa, sb_, saa, sbb, sab = np.moveaxis(st, -1, 0)
    N = n.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = sa.sum(-1) / N, sb_.sum(-1) / N
        pa, pb = sa / n, sb_ / n
        between = np.nansum(n * (pa - ma[..., None]) * (pb - mb[..., None]), -1) / N
        within = np.nansum(sab - np.where(n > 0, sa * sb_ / n, 0), -1) / N
        vb = np.nansum(n * ((pa - ma[..., None]) ** 2 + (pb - mb[..., None]) ** 2) / 2, -1) / N
        vw = np.nansum((saa - np.where(n > 0, sa * sa / n, 0))
                       + (sbb - np.where(n > 0, sb_ * sb_ / n, 0)), -1) / (2 * N)
    return {"cov_total": between + within, "cov_between": between, "cov_within": within,
            "var_between": vb, "var_within": vw}


def boot_picks(groups_sorted_unique: np.ndarray, reps: int, seed: int) -> np.ndarray:
    """Multiplicity matrix (reps, P) reproducing R.cluster_bootstrap's draw sequence."""
    rng = np.random.default_rng(seed)
    P = len(groups_sorted_unique)
    mult = np.zeros((reps, P))
    for r in range(reps):
        pick = rng.choice(groups_sorted_unique, size=P, replace=True)
        idx = np.searchsorted(groups_sorted_unique, pick)
        np.add.at(mult[r], idx, 1)
    return mult


def summarise(name: str, scores: dict, meta: pd.DataFrame, n_half: int, reps: int,
              t0: float) -> dict:
    groups = meta.problem_id.to_numpy()
    uniq = np.unique(groups)
    P = len(uniq)
    gidx = np.searchsorted(uniq, groups)
    m_full = 2.0
    out = {}
    for s in SCORES:
        A, B = scores[s][0], scores[s][1]       # (S, N)
        S = A.shape[0]
        seed = R.seed_from(f"20260914:arv2:v3f:{TASK}:{name}:{s}")
        mult = boot_picks(uniq, reps, seed)
        Wb = mult[:, gidx]                        # (reps, N)
        ones = np.ones((1, A.shape[1]))
        # pooled Spearman: per split point value and per bootstrap draw
        sp_split = np.empty(S)
        sp_boot = np.zeros(reps)
        for k in range(S):
            sp_split[k] = wspearman(A[k], B[k], ones)[0]
            if k < BOOT_SPLITS_SPEARMAN:
                sp_boot += wspearman(A[k], B[k], Wb)
        sp_boot /= min(S, BOOT_SPLITS_SPEARMAN)
        # ICC + decomposition via sufficient statistics
        st = suff(A, B, gidx, P)                   # (S, P, 6)
        icc_split = icc_from(st.sum(1))
        st_b = np.einsum("rp,spj->rsj", mult, st)  # (reps, S, 6)
        icc_boot = np.nanmean(icc_from(st_b), axis=1)
        dec = decomp_from(st)
        dec_b = {}
        for c0 in range(0, reps, 50):
            part = decomp_from(mult[c0:c0 + 50, None, :, None] * st[None])
            for key, v in part.items():
                dec_b.setdefault(key, []).append(v)
        dec_b = {key: np.concatenate(v) for key, v in dec_b.items()}
        # per-problem
        pp_sp = np.full((S, P), np.nan)
        for k in range(S):
            for p in range(P):
                sl = gidx == p
                pp_sp[k, p] = wspearman(A[k, sl], B[k, sl], np.ones((1, sl.sum())))[0]
        pp_icc = icc_from(st)                     # (S, P)
        pp_sp_mean = np.nanmean(pp_sp, 0)         # split-averaged per problem
        pp_icc_mean = np.nanmean(pp_icc, 0)
        wsp_boot = np.array([np.nanmean(np.repeat(pp_sp_mean, m.astype(int))) for m in mult])
        wicc_boot = np.array([np.nanmean(np.repeat(pp_icc_mean, m.astype(int))) for m in mult])

        def block(point, draws, *, label):
            lo, hi, nf = R.interval(draws)
            rec = {"split_half": fl(point), "split_half_lo": fl(lo), "split_half_hi": fl(hi),
                   "nonfinite_draws": nf, label + "_full_n": fl(sb(point, m_full)),
                   "full_n_lo": fl(sb(lo, m_full)), "full_n_hi": fl(sb(hi, m_full))}
            rec["projected"] = {str(nt): {"est": fl(sb(point, nt / n_half)),
                                          "lo": fl(sb(lo, nt / n_half)),
                                          "hi": fl(sb(hi, nt / n_half))} for nt in N_TARGETS}
            if point > 0:
                bm = BENCH["icc" if label == "icc" else "spearman"]
                rec["n_per_fork_for_benchmark"] = fl(n_half * bm * (1 - point)
                                                     / (point * (1 - bm)))
            return rec

        rec = {
            "spearman": block(np.nanmean(sp_split), sp_boot, label="spearman"),
            "icc": block(np.nanmean(icc_split), icc_boot, label="icc"),
            "within_problem_spearman_mean": block(np.nanmean(pp_sp_mean), wsp_boot,
                                                  label="spearman"),
            "within_problem_icc_mean": block(np.nanmean(pp_icc_mean), wicc_boot, label="icc"),
            "split_sd": {"spearman": fl(np.nanstd(sp_split)), "icc": fl(np.nanstd(icc_split))},
            "within_problem_spearman_median": fl(np.nanmedian(pp_sp_mean)),
            "sb_projection_beyond_full_n_valid": s in SB_N_VALID,
            "nan_pair_rate": fl(np.mean(~(np.isfinite(A) & np.isfinite(B)))),
        }
        for key in ("spearman", "icc"):
            b = rec[key]
            b["decision_full_n"] = R.three_way(
                b["full_n_lo"] if b["full_n_lo"] is not None else np.nan,
                b["full_n_hi"] if b["full_n_hi"] is not None else np.nan, BENCH[key])
        # decomposition (split-averaged, bootstrap)
        d = {}
        cov_t = np.nanmean(dec["cov_total"])
        for part in ("between", "within"):
            share = np.nanmean(dec[f"cov_{part}"]) / cov_t if cov_t != 0 else np.nan
            share_b = (np.nanmean(dec_b[f"cov_{part}"], 1) / np.nanmean(dec_b["cov_total"], 1))
            lo, hi, _ = R.interval(share_b)
            rel_p = np.nanmean(dec[f"cov_{part}"]) / np.nanmean(dec[f"var_{part}"])
            rel_pb = np.nanmean(dec_b[f"cov_{part}"], 1) / np.nanmean(dec_b[f"var_{part}"], 1)
            rlo, rhi, _ = R.interval(rel_pb)
            d[part] = {"share_of_reliable_variance": fl(share), "share_lo": fl(lo),
                       "share_hi": fl(hi), "split_half_r": fl(rel_p),
                       "r_full_n": fl(sb(rel_p, m_full)), "r_full_n_lo": fl(sb(rlo, m_full)),
                       "r_full_n_hi": fl(sb(rhi, m_full))}
        d["cov_total_half_n"] = fl(cov_t)
        d["var_total_half_n"] = fl(np.nanmean(dec["var_between"] + dec["var_within"]))
        rec["decomposition"] = d
        rec["per_problem"] = {
            str(int(u)): {"n_sentences": int((gidx == p).sum()),
                          "spearman_split_half": fl(pp_sp_mean[p]),
                          "spearman_full_n": fl(sb(pp_sp_mean[p], m_full)),
                          "icc_split_half": fl(pp_icc_mean[p]),
                          "icc_full_n": fl(sb(pp_icc_mean[p], m_full))}
            for p, u in enumerate(uniq)}
        fn = sb(pp_sp_mean, m_full)
        rec["per_problem_share_full_n_spearman_ge_bench"] = fl(np.mean(fn >= BENCH["spearman"]))
        rec["per_problem_share_full_n_icc_ge_bench"] = fl(
            np.mean(sb(pp_icc_mean, m_full) >= BENCH["icc"]))
        out[s] = rec
        print(f"  [{name}] {s:12s} sp={rec['spearman']['split_half']:.3f}"
              f"->{rec['spearman']['spearman_full_n']:.3f} icc={rec['icc']['split_half']:.3f}"
              f" within={rec['within_problem_spearman_mean']['split_half']:.3f}"
              f"  {time.time() - t0:.0f}s", flush=True)
    return out


# ----------------------------------------------------------------------------- cross-checks


def s14_reproduction(rel: pd.DataFrame, units) -> dict:
    """s14's single even/odd split on raw categories, within = mean over trajectories."""
    codes = {c: i for i, c in enumerate(sorted(rel.cat_raw.unique()))}
    k = len(codes)
    r = rel.assign(code=rel.cat_raw.map(codes)).sort_values(["trajectory_id", "fork_idx",
                                                              "sample_idx"])
    fs = {}
    for key, g in r.groupby(["trajectory_id", "fork_idx"]):
        c, y = g.code.to_numpy(), g.correct_canon.to_numpy(float)
        h = g.sample_idx.to_numpy() % 2 == 0
        fs[key] = {"A": (c[h], y[h]), "B": (c[~h], y[~h])}
    vals = {n: {"A": [], "B": []} for n in ("fork_tv", "tv_debiased", "floor", "acc_level")}
    unrounded = {"A": [], "B": []}
    traj = []
    for t, _, i in units:
        traj.append(t)
        for part in ("A", "B"):
            (ca, ya), (cb, _) = fs[(t, i)][part], fs[(t, i + 1)][part]
            tv, f = s13.tv(ca, cb, k), s13.tv_null_analytic(ca, cb, k)
            unrounded[part].append(tv)
            for n, v in (("fork_tv", tv), ("tv_debiased", tv - f), ("floor", f),
                         ("acc_level", ya.mean())):
                vals[n][part].append(round(float(v), 10))
    traj = np.array(traj)
    out = {}
    for n, v in vals.items():
        a, b = np.array(v["A"]), np.array(v["B"])
        rr = R.spearman(a, b)
        w = [R.spearman(a[traj == t], b[traj == t]) for t in np.unique(traj)]
        w = float(np.nanmean(w))
        out[n] = {"split_half_spearman": rr, "rel_n100": float(sb(rr, 2)),
                  "ceiling_corr_sqrt_rel": float(np.sqrt(max(sb(rr, 2), 0))),
                  "within_traj_mean": w, "within_rel_n100": float(sb(w, 2))}
    out["fork_tv"]["split_half_spearman_unrounded"] = R.spearman(unrounded["A"], unrounded["B"])
    out["fork_tv"]["distinct_values_unrounded_vs_rounded"] = [
        int(len(np.unique(unrounded["A"]))), int(len(np.unique(vals["fork_tv"]["A"])))]
    s14 = json.loads((RESULTS / "s14_representation_rootcause.json").read_text())["ceilings"]
    out["s14_committed"] = {n: s14[n] for n in ("fork_tv", "tv_debiased", "floor", "acc_level")}
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--datasets", default="A,B")
    ap.add_argument("--crosscheck-only", action="store_true",
                    help="recompute only the cross-check blocks into an existing results.json")
    args = ap.parse_args()
    t0 = time.time()
    data = dataset_tables()
    if args.crosscheck_only:
        res = json.loads((OUT / "results.json").read_text())
        res["cross_check_s14_even_odd"] = s14_reproduction(data["A"][0], data["A"][1])
        write_json(OUT / "results.json", res)
        return
    res: dict = {"stage": "EXPLORATORY, POST HOC", "task": "T2 (item 1) split-half reliability",
                 "splits": args.splits, "bootstrap_reps": args.reps, "scores": SCORES,
                 "benchmarks": BENCH, "n_targets": N_TARGETS, "similarity_threshold": SIM_THR,
                 "kl_alpha": ALPHA, "grouping": "canonical (arv2.canon) unless *_raw",
                 "sb_rule": "SB applied only to r > 0; r <= 0 reported unchanged",
                 "spearman_bootstrap_splits": BOOT_SPLITS_SPEARMAN,
                 "datasets": {}}
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.datasets.split(","):
        df, units = data[name]
        if "replicate" not in df:
            df = df.assign(replicate=0)
        seed = R.seed_from(f"20260914:arv2:v3f:{TASK}:splits:{name}")
        scores, meta = all_half_scores(df, units, args.splits, seed)
        n_fork = int(df.groupby(["trajectory_id", "fork_idx"]).size().iloc[0])
        print(f"[{name}] half scores ready: {len(meta)} sentences, {meta.problem_id.nunique()} "
              f"problems, n/fork={n_fork}  {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(CACHE / f"half_scores_{name}.npz", **{s: v.astype(np.float32)
                                                                 for s, v in scores.items()})
        meta.to_parquet(CACHE / f"meta_{name}.parquet")
        res["datasets"][name] = {
            "sentences": len(meta), "problems": int(meta.problem_id.nunique()),
            "trajectories": int(meta.trajectory_id.nunique()), "n_per_fork": n_fork,
            "n_per_half": n_fork // 2,
            "scores": summarise(name, scores, meta, n_fork // 2, args.reps, t0)}
        write_json(OUT / "results.json", res)
    if "A" in args.datasets:
        res["cross_check_s14_even_odd"] = s14_reproduction(data["A"][0], data["A"][1])
    s13r = json.loads((RESULTS / "s13_phase2_diagnostics.json").read_text())
    res["cross_check_s13_curve_n150_raw_categories"] = {
        n: s13r["curve"][n]["150"] for n in s13r["curve"]}
    res["runtime_s"] = round(time.time() - t0)
    write_json(OUT / "results.json", res)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
