"""v3 follow-up T6 (item 4): nested-variance ceiling on pre-emission predictability.
EXPLORATORY, POST HOC.

Question: what share of the variance in a realised sentence's counterfactual importance
could ANY function of the history h (the prefix before S_i) explain?

At fork (history) h the resamples give pairs (s_j, a_j). Sampled sentences are clustered by
MiniLM embedding (per fork, average-linkage agglomerative clustering on cosine distance,
cut at distance 1 - threshold). For estimable clusters c (size >= min_size):
    I_c   = TV(P(A|h,c), P(A|h))           P(A|h) = all samples of the fork
    w_c   = n_c / sum of n over estimable clusters of the fork
    m(h)  = sum_c w_c I_c
    Between = Var_h(m(h))                  (forks equally weighted)
    Within  = mean_h sum_c w_c (I_c - m(h))^2
    Total   = Between + Within             (weighted variance over (h, c))
    R2_max  = Between / Total
Metrics: "tv" (canonical categories, primary), "tv_raw" (raw categories), "acc"
(|acc(c) - acc(h)| = TV on the binary correct/incorrect outcome), and "acc_signed"
(acc(c) - acc(h), linear, so its estimates are unbiased and need no null correction; by the
tower property E_S[acc(h,S)] = acc(h), so its between-history variance is ~0 by construction).
Added after the first run (documented in summary.md): "acc_signed", the entropy diagnostic,
and the weak-effect synthetic settings.

Bias correction (the estimator, exactly):
1. Mean bias. E0_c is the EXACT permutation-null expectation of I_c: answers exchangeable
   within the fork, cluster sizes fixed, so each category count of cluster c is
   hypergeometric(N, K_a, n_c) and E0_c = 0.5 * sum_a E|X_a/n_c - K_a/N|, computed from the
   hypergeometric pmf (no Monte Carlo). Debiased I~_c = I_c - E0_c; m~(h) = sum_c w_c I~_c.
2. Noise variance. Each fork's samples are split into halves A/B, stratified within every
   cluster (floor(n_c/2) to one half, the odd one to a random half). In each half
   x_c = I^A_c - E0^A_c (half's own cluster and fork distributions, half-sample null),
   y_c likewise for B. Noise in x and y is independent given the true distributions, so
       Total   = mean_h S_xy - xbar*ybar          S_xy = sum_c w_c x_c y_c
       Between = mean_h S_x S_y - xbar*ybar       S_x = sum_c w_c x_c = m~^A(h)
       Within  = mean_h (S_xy - S_x S_y)          (Total = Between + Within exactly)
   with xbar = mean_h S_x. Components are averaged over N_SPLITS random splits; ratios are
   formed after averaging. Weights w_c are the full-sample shares in both halves.
   Across-problem (and across-trajectory) parts of Between: fork-weighted covariance of
   the problem (trajectory) means of S_x and S_y. Within-problem ceiling:
   (Between - across_p) / (Total - across_p); within-trajectory likewise.
3. Naive (uncorrected) quantities from the full sample are reported alongside.
Validation: (i) synthetic null - answers permuted within fork (independent of clusters);
(ii) synthetic truth with known R2_max on the real cluster structure.

Usage: python analysis/v3_followup/t6_nested_ceiling/run.py [--reps 2000] [--stage all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.special import gammaln  # noqa: E402

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DATA, write_json  # noqa: E402

TASK = "t6_nested_ceiling"
OUT = Path(__file__).resolve().parent
SHARED = DATA / "derived/v3_followup"
CACHE = SHARED / TASK
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# ---- grid and constants FIXED BEFORE RUNNING (report every cell) ----
THRESHOLDS = (0.8, 0.7, 0.9)        # similarity; primary 0.8 (study's "without" cut)
MIN_SIZES = (5, 3, 10)              # primary 5
LINKAGE = "average"
PRIMARY = (0.8, 5)
METRICS = ("tv", "tv_raw", "acc", "acc_signed")   # tv = canonical (primary)
N_SPLITS = 16
DATASETS = ("A_released", "B_r12", "B_r012")
# K/M design grid for the GPU nested arm (fixed in advance)
DESIGN_P = (20, 40)                 # problems
DESIGN_F = 20                       # positions per problem
DESIGN_K = (2, 4, 8, 16, 32)        # sentences per position
DESIGN_M = (10, 20, 50, 100, 200)   # continuations per sentence
DESIGN_SIMS = 400
TARGET_WIDTH = 0.2

LG = gammaln(np.arange(0, 2002) + 1.0)  # log k!


def seed(name: str) -> int:
    return R.seed_from(f"20260914:arv2:v3f:{TASK}:{name}")


def label(d: dict) -> dict:
    return {"status": "EXPLORATORY, POST HOC", **d}


# ----------------------------------------------------------------------------- data


def load_samples() -> dict[str, pd.DataFrame]:
    """One table per dataset: problem_id, trajectory_id, fork_idx, text, cat_canon, cat_raw,
    correct (canonical)."""
    rel = pd.read_parquet(SHARED / "released_canon.parquet",
                          columns=["trajectory_id", "problem_id", "fork_idx", "sample_idx",
                                   "chunk_resampled", "cat_canon", "cat_raw", "correct_canon"])
    rel = rel.rename(columns={"chunk_resampled": "text", "correct_canon": "correct"})
    rel["source"] = "r0"
    p2 = pd.read_parquet(SHARED / "phase2_canon.parquet",
                         columns=["replicate", "trajectory_id", "fork_idx", "sample_idx",
                                  "first_replacement_sentence", "cat_canon", "cat_raw",
                                  "correct_canon"])
    p2 = p2.rename(columns={"first_replacement_sentence": "text", "correct_canon": "correct"})
    p2["problem_id"] = p2.trajectory_id.str.split(":").str[0].astype(int)
    p2["source"] = "r" + p2.replicate.astype(str)
    cfg = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    keys = {(t["trajectory_id"], f) for t in cfg["trajectories"] for f in t["forks"]}
    have = set(zip(p2.trajectory_id, p2.fork_idx))
    assert have == keys, "phase2 table forks differ from configs/phase2_forks.json"
    cols = ["problem_id", "trajectory_id", "fork_idx", "source", "text", "cat_canon",
            "cat_raw", "correct"]
    r0w = rel[[k in keys for k in zip(rel.trajectory_id, rel.fork_idx)]]
    out = {"A_released": rel[cols], "B_r12": p2[cols],
           "B_r012": pd.concat([p2[cols], r0w[cols]], ignore_index=True)}
    return {k: v.sort_values(["trajectory_id", "fork_idx", "source"], kind="stable")
            .reset_index(drop=True) for k, v in out.items()}


def embeddings(texts: list[str]) -> dict[str, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    vf, tf = CACHE / "minilm_vectors.npy", CACHE / "minilm_texts.parquet"
    uniq = sorted(set(texts))
    if vf.exists() and tf.exists():
        cached = pd.read_parquet(tf).text.tolist()
        if cached == uniq:
            vec = np.load(vf)
            return {"texts": uniq, "vec": vec}
    from sentence_transformers import SentenceTransformer
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, revision=REVISION, device=dev)
    t0 = time.time()
    vec = model.encode(uniq, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float32)
    print(f"embedded {len(uniq)} texts on {dev} in {time.time() - t0:.0f}s", flush=True)
    np.save(vf, vec)
    pd.DataFrame({"text": uniq}).to_parquet(tf, index=False)
    return {"texts": uniq, "vec": vec}


def cluster_labels(df: pd.DataFrame, emb: dict, name: str) -> pd.DataFrame:
    """Per-fork flat cluster labels at each threshold (cached)."""
    path = CACHE / f"clusters_{name}.parquet"
    if path.exists():
        lab = pd.read_parquet(path)
        if len(lab) == len(df):
            return lab
    pos = {t: i for i, t in enumerate(emb["texts"])}
    idx = df.text.map(pos).to_numpy()
    vec = emb["vec"].astype(np.float64)
    lab = {f"c{thr}": np.zeros(len(df), dtype=np.int32) for thr in THRESHOLDS}
    for _, rows in df.groupby(["trajectory_id", "fork_idx"], sort=False).indices.items():
        v = vec[idx[rows]]
        if len(rows) < 2:
            continue
        Z = linkage(v, method=LINKAGE, metric="cosine")
        Z[:, 2] = np.clip(Z[:, 2], 0, None)
        for thr in THRESHOLDS:
            lab[f"c{thr}"][rows] = fcluster(Z, t=1 - thr, criterion="distance") - 1
    lab = pd.DataFrame(lab)
    lab.to_parquet(path, index=False)
    return lab


# ----------------------------------------------------------------------------- estimator


def e0_tv(N: int, K: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Exact permutation-null E[TV(cluster, fork)] for each cluster size.
    K: category counts of the fork (sum N). Returns array len(sizes)."""
    if len(sizes) == 0:
        return np.zeros(0)
    us, inv = np.unique(sizes, return_inverse=True)
    kmax = int(us.max())
    x = np.arange(kmax + 1)[None, None, :]
    Ka = K[:, None, None].astype(np.int64)
    k = us[None, :, None].astype(np.int64)
    valid = (x <= Ka) & (x <= k) & (k - x <= N - Ka)
    xs = np.where(valid, x, 0)
    a = np.where(valid, Ka - xs, 0)
    b = np.where(valid, k - xs, 0)
    c = np.where(valid, N - Ka - k + xs, 0)
    lp = (LG[Ka] - LG[xs] - LG[a] + LG[N - Ka] - LG[b] - LG[c] - (LG[N] - LG[k] - LG[N - k]))
    pmf = np.where(valid, np.exp(np.where(valid, lp, 0.0)), 0.0)
    dev = np.abs(x / k - Ka / N)
    e = 0.5 * (pmf * dev).sum(axis=(0, 2))
    return e[inv]


def tv_and_e0(lab: np.ndarray, codes: np.ndarray, ncl: int, ncat: int, sel: np.ndarray,
              signed: bool = False):
    """I_c and E0_c for clusters `sel` given labels/answers of (part of) a fork.
    signed=True: codes are 0/1 correctness and I_c = acc(c) - acc(h), E0_c = 0."""
    if signed:
        n = np.bincount(lab, minlength=ncl).astype(float)[sel]
        k = np.bincount(lab, weights=codes.astype(float), minlength=ncl)[sel]
        with np.errstate(invalid="ignore", divide="ignore"):
            I = np.where(n > 0, k / n - codes.mean(), np.nan)
        return I, np.where(n > 0, 0.0, np.nan)
    C = np.zeros((ncl, ncat))
    np.add.at(C, (lab, codes), 1.0)
    n = C.sum(1)
    N = int(n.sum())
    Kc = C.sum(0)
    ph = Kc / N
    Cs, ns = C[sel], n[sel]
    with np.errstate(invalid="ignore", divide="ignore"):
        I = 0.5 * np.abs(Cs / ns[:, None] - ph[None, :]).sum(1)
    I = np.where(ns > 0, I, np.nan)
    keep = Kc > 0
    e0 = np.full(len(sel), np.nan)
    ok = ns > 0
    e0[ok] = e0_tv(N, Kc[keep].astype(int), ns[ok].astype(int))
    return I, e0


def fork_cluster_rows(lab: np.ndarray, answers: dict[str, np.ndarray], rng, min_any: int):
    """Cluster-level rows for one fork: n_c, full-sample I and E0, and per split x, y."""
    lab = np.unique(lab, return_inverse=True)[1]
    ncl = lab.max() + 1
    sizes = np.bincount(lab, minlength=ncl)
    sel = np.flatnonzero(sizes >= min_any)
    out = {"n_c": sizes[sel]}
    if len(sel) == 0:
        return out, 0
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    halves = []
    for _ in range(N_SPLITS):
        # stratified split: random order within each cluster, first h_c members -> half A
        order = np.lexsort((rng.random(len(lab)), lab))
        rank = np.empty(len(lab), dtype=int)
        rank[order] = np.arange(len(lab)) - start[lab[order]]
        h = sizes // 2 + ((sizes % 2 == 1) & (rng.random(ncl) < 0.5))
        halves.append(rank < h[lab])
    for met, (codes, ncat) in answers.items():
        sg = met == "acc_signed"
        I, e0 = tv_and_e0(lab, codes, ncl, ncat, sel, sg)
        out[f"{met}_I"], out[f"{met}_E0"] = I, e0
        xs, ys = np.empty((len(sel), N_SPLITS)), np.empty((len(sel), N_SPLITS))
        for s, inA in enumerate(halves):
            for half, store in ((inA, xs), (~inA, ys)):
                Ih, eh = tv_and_e0(lab[half], codes[half], ncl, ncat, sel, sg)
                store[:, s] = Ih - eh
        out[f"{met}_x"], out[f"{met}_y"] = xs, ys
    return out, len(sel)


def build_cluster_table(df: pd.DataFrame, lab: np.ndarray, rng_name: str,
                        answers_override: dict | None = None) -> dict:
    """Stack cluster rows over all forks of a dataset for one clustering."""
    base = seed(rng_name)
    codes = {}
    for met, col in (("tv", "cat_canon"), ("tv_raw", "cat_raw"), ("acc", "correct"),
                     ("acc_signed", "correct")):
        if answers_override is not None:
            codes[met] = answers_override[met]
        elif met == "acc_signed":
            codes[met] = df[col].astype(int).to_numpy()
        else:
            codes[met] = pd.factorize(df[col])[0]
    groups = df.groupby(["trajectory_id", "fork_idx"], sort=False).indices
    rows = {k: [] for k in ["fork", "n_c"] + [f"{m}_{q}" for m in METRICS
                                              for q in ("I", "E0", "x", "y")]}
    forks = []
    for fi, ((traj, fork), r) in enumerate(groups.items()):
        rng = np.random.default_rng([base, fi])
        ans = {}
        for met in METRICS:
            if met == "acc_signed":
                ans[met] = (codes[met][r], 2)
                continue
            cc = np.unique(codes[met][r], return_inverse=True)[1]
            ans[met] = (cc, cc.max() + 1)
        out, k = fork_cluster_rows(lab[r], ans, rng, min(MIN_SIZES))
        pc = np.bincount(ans["tv"][0]) / len(r)
        pc = pc[pc > 0]
        forks.append({"fork": fi, "trajectory_id": traj, "fork_idx": int(fork),
                      "problem_id": int(df.problem_id.iat[r[0]]), "n": len(r),
                      "entropy": float(-(pc * np.log(pc)).sum())})
        if k == 0:
            continue
        rows["fork"].append(np.full(k, fi))
        for key, v in out.items():
            rows[key].append(v)
    tab = {k: np.concatenate(v) for k, v in rows.items()}
    tab["forks"] = pd.DataFrame(forks)
    return tab


def components(tab: dict, ms: int, met: str):
    """Fork-level arrays for one (min_size, metric) cell."""
    keep = tab["n_c"] >= ms
    fork = tab["fork"][keep]
    nf = len(tab["forks"])
    n_c = tab["n_c"][keep].astype(float)
    tot = np.bincount(fork, n_c, minlength=nf)
    w = n_c / tot[fork]
    has = tot > 0
    I, E0 = tab[f"{met}_I"][keep], tab[f"{met}_E0"][keep]
    x, y = tab[f"{met}_x"][keep], tab[f"{met}_y"][keep]
    Sx = np.stack([np.bincount(fork, w * x[:, s], minlength=nf) for s in range(N_SPLITS)], 1)
    Sy = np.stack([np.bincount(fork, w * y[:, s], minlength=nf) for s in range(N_SPLITS)], 1)
    Sxy = np.stack([np.bincount(fork, w * x[:, s] * y[:, s], minlength=nf)
                    for s in range(N_SPLITS)], 1)
    m_naive = np.bincount(fork, w * I, minlength=nf)
    m_deb = np.bincount(fork, w * (I - E0), minlength=nf)
    s2_naive = np.bincount(fork, w * I ** 2, minlength=nf)
    cov_frac = tot / tab["forks"].n.to_numpy()
    d2 = (x - y) ** 2 / 2  # noise variance of a half estimate (per cluster, per split)
    nh = n_c / 2
    fdf = tab["forks"][has].reset_index(drop=True)
    return {"forks": fdf, "Sx": Sx[has], "Sy": Sy[has], "Sxy": Sxy[has],
            "m_naive": m_naive[has], "m_deb": m_deb[has], "s2_naive": s2_naive[has],
            "coverage": cov_frac[has], "n_clusters": np.bincount(fork, minlength=nf)[has],
            "noise_a": float(np.sum(w * nh * d2.mean(1)) / np.sum(w)),
            "noise_a_unweighted": float(np.mean(nh * d2.mean(1))),
            "n_dropped_forks": int((~has).sum())}


def stat_vector(C: dict, rows: np.ndarray) -> np.ndarray:
    """All headline statistics for fork rows `rows` (copies of problems are distinct)."""
    prob = C["forks"].problem_id.to_numpy()[rows]
    traj = C["traj_code"][rows]
    brk = np.r_[True, (prob[1:] != prob[:-1]) | (rows[1:] <= rows[:-1])]
    copy = np.cumsum(brk) - 1
    tcopy = np.unique(copy * 4 + traj, return_inverse=True)[1]
    Sx, Sy, Sxy = C["Sx"][rows], C["Sy"][rows], C["Sxy"][rows]
    F = len(rows)
    xb, yb = Sx.mean(0), Sy.mean(0)
    total = Sxy.mean(0) - xb * yb
    between = (Sx * Sy).mean(0) - xb * yb

    def across(g):
        cnt = np.bincount(g).astype(float)
        mx = np.stack([np.bincount(g, Sx[:, s]) for s in range(N_SPLITS)], 1) / cnt[:, None]
        my = np.stack([np.bincount(g, Sy[:, s]) for s in range(N_SPLITS)], 1) / cnt[:, None]
        return ((cnt[:, None] / F) * mx * my).sum(0) - xb * yb

    ap, at = across(copy), across(tcopy)
    T, B, Ap, At = total.mean(), between.mean(), ap.mean(), at.mean()
    W = T - B
    # entropy diagnostic: variance of the projection of m(h) on [1, H, H^2] of the fork's
    # answer entropy, cross-half (beta from half A times beta from half B)
    H = C["forks"].entropy.to_numpy()[rows]
    X = np.column_stack([np.ones(F), H, H ** 2])
    bA = np.linalg.lstsq(X, Sx, rcond=None)[0]
    bB = np.linalg.lstsq(X, Sy, rcond=None)[0]
    Xc = X[:, 1:] - X[:, 1:].mean(0)
    proj = ((Xc @ bA[1:]) * (Xc @ bB[1:])).mean(0).mean()
    mn, md = C["m_naive"][rows], C["m_deb"][rows]
    tot_n = C["s2_naive"][rows].mean() - mn.mean() ** 2
    btw_n = mn.var()
    return np.array([
        B / T if T > 0 else np.nan,                       # 0 R2_max
        (B - Ap) / (T - Ap) if T - Ap > 0 else np.nan,     # 1 R2 within problem
        (B - At) / (T - At) if T - At > 0 else np.nan,     # 2 R2 within trajectory
        T, B, W, Ap, At,                                   # 3-7 components
        md.mean(), mn.mean(),                              # 8-9 mean m(h) debiased / naive
        btw_n / tot_n if tot_n > 0 else np.nan,            # 10 naive R2
        tot_n, btw_n,                                      # 11-12 naive components
        np.sqrt(max(B, 0)), np.sqrt(max(W, 0)),            # 13-14 SDs
        proj / B if B > 0 else np.nan,                     # 15 entropy share of Between
        proj / T if T > 0 else np.nan,                     # 16 R2 of entropy alone
    ])


STAT_NAMES = ["R2_max", "R2_within_problem", "R2_within_trajectory", "var_total",
              "var_between_h", "var_within_h", "var_across_problem", "var_across_trajectory",
              "mean_m_debiased", "mean_m_naive", "R2_naive", "var_total_naive",
              "var_between_naive", "sd_between_h", "sd_within_h", "entropy_share_of_between",
              "R2_entropy_only"]


def summarise_cell(C: dict, name: str, reps: int, boot: bool = True) -> dict:
    C["traj_code"] = (C["forks"].trajectory_id.str.endswith(":incorrect")).astype(int).to_numpy()
    F = len(C["forks"])
    point = stat_vector(C, np.arange(F))
    res = {"n_forks": F, "n_problems": int(C["forks"].problem_id.nunique()),
           "n_dropped_forks": C["n_dropped_forks"],
           "mean_coverage": float(C["coverage"].mean()),
           "mean_estimable_clusters": float(C["n_clusters"].mean()),
           "noise_a": C["noise_a"], "point": dict(zip(STAT_NAMES, map(float, point)))}
    if boot:
        draws = []

        def stat(rows):
            v = stat_vector(C, rows)
            draws.append(v)
            return v[0]

        R.cluster_bootstrap(C["forks"].problem_id.to_numpy(), stat, reps=reps, seed=seed(name))
        D = np.array(draws)
        res["ci95"] = {k: R.interval(D[:, j])[:2] for j, k in enumerate(STAT_NAMES)}
        res["boot_reps"] = reps
    return res


# ----------------------------------------------------------------------------- synthetic


def synthetic_truth(df: pd.DataFrame, lab: np.ndarray, setting: dict, rng) -> tuple:
    """Draw answers from a known cluster-level model on the real cluster structure.
    Returns (answer codes for tv/tv_raw/acc, true component dict per min size)."""
    groups = df.groupby(["trajectory_id", "fork_idx"], sort=False).indices
    K = 6
    codes = np.empty(len(df), dtype=int)
    truth_rows = []
    for fi, (_, r) in enumerate(groups.items()):
        l = np.unique(lab[r], return_inverse=True)[1]
        ncl = l.max() + 1
        n_c = np.bincount(l, minlength=ncl)
        p_h = rng.dirichlet(np.full(K, 0.7))
        tau = setting["tau"](rng)
        u = setting["u"](rng, ncl)
        lam = np.clip(tau * u, 0, 1)
        tgt = rng.integers(0, K, ncl)
        P = (1 - lam)[:, None] * p_h[None, :]
        P[np.arange(ncl), tgt] += lam
        for c in range(ncl):
            idx = r[l == c]
            codes[idx] = rng.choice(K, size=len(idx), p=P[c])
        w_all = n_c / n_c.sum()
        Pm = (w_all[:, None] * P).sum(0)
        I_tv = 0.5 * np.abs(P - Pm).sum(1)
        I_acc = np.abs(P[:, 0] - Pm[0])
        truth_rows.append((fi, n_c, I_tv, I_acc, P[:, 0] - Pm[0]))
    truth = {}
    for ms in MIN_SIZES:
        for met, j in (("tv", 2), ("acc", 3), ("acc_signed", 4)):
            m, s2 = [], []
            for fi, n_c, *I in truth_rows:
                keep = n_c >= ms
                if not keep.any():
                    continue
                w = n_c[keep] / n_c[keep].sum()
                Ic = I[j - 2][keep]
                m.append((w * Ic).sum())
                s2.append((w * Ic ** 2).sum())
            m, s2 = np.array(m), np.array(s2)
            T = s2.mean() - m.mean() ** 2
            B = m.var()
            truth[f"{ms}:{met}"] = {"R2_max": B / T, "var_total": T, "var_between_h": B,
                                    "var_within_h": T - B, "mean_m": m.mean()}
    acc = (codes == 0).astype(int)
    return {"tv": codes, "tv_raw": codes, "acc": acc, "acc_signed": acc}, truth


SETTINGS = {
    "high_between": {"tau": lambda g: 0.8 * g.beta(2, 2), "u": lambda g, n: g.beta(20, 20, n)},
    "mid": {"tau": lambda g: 0.8 * g.beta(2, 2), "u": lambda g, n: g.beta(1.5, 1.5, n)},
    "low_between": {"tau": lambda g: 0.4, "u": lambda g, n: g.beta(0.5, 0.5, n)},
    # weak-effect settings (added after the first run): effect scale x0.15, closer to the
    # real data, where naive m(h) barely exceeds its permutation-null value
    "weak_high_between": {"tau": lambda g: 0.12 * g.beta(2, 2),
                          "u": lambda g, n: g.beta(20, 20, n)},
    "weak_mid": {"tau": lambda g: 0.12 * g.beta(2, 2), "u": lambda g, n: g.beta(1.5, 1.5, n)},
    "weak_low_between": {"tau": lambda g: 0.06, "u": lambda g, n: g.beta(0.5, 0.5, n)},
}


# ----------------------------------------------------------------------------- design sim


def design_sim(comp: dict, a: float, P: int, K: int, M: int, rng) -> dict:
    """Gaussian nested design: I = mu + problem + position + sentence; half estimates add
    N(0, a/(M/2)) noise. Returns 95% sampling-quantile widths of the cross-half R2s.
    (Corrected after the first run: nested estimators remove the sigma_w^2/K and
    sigma_b^2/F terms that a naive plug-in leaves in Between.)"""
    sp, sb, sw = (max(comp[k], 0) for k in ("var_across_problem", "var_between_within_p",
                                            "var_within_h"))
    ne = a / (M / 2)
    F = DESIGN_F
    r2, r2w = np.empty(DESIGN_SIMS), np.empty(DESIGN_SIMS)
    for t in range(DESIGN_SIMS):
        I = (rng.normal(0, np.sqrt(sp), (P, 1, 1)) + rng.normal(0, np.sqrt(sb), (P, F, 1))
             + rng.normal(0, np.sqrt(sw), (P, F, K)))
        x = I + rng.normal(0, np.sqrt(ne), I.shape)
        y = I + rng.normal(0, np.sqrt(ne), I.shape)
        # K distinct sentences per position: a position mean of K draws carries sigma_w^2/K
        # and a problem mean of F positions carries sigma_b^2/F, so components use unbiased
        # cross-half nested estimators (divisors K-1 and F-1).
        Sx, Sy = x.mean(2), y.mean(2)
        W = (((x - Sx[..., None]) * (y - Sy[..., None])).sum(2) / (K - 1)).mean()
        Px, Py = Sx.mean(1), Sy.mean(1)
        Bwp_raw = (((Sx - Px[:, None]) * (Sy - Py[:, None])).sum(1) / (F - 1)).mean()
        sb_h = Bwp_raw - W / K
        sp_h = ((Px - Px.mean()) * (Py - Py.mean())).sum() / (P - 1) - Bwp_raw / F
        r2[t] = (sp_h + sb_h) / (sp_h + sb_h + W)
        r2w[t] = sb_h / (sb_h + W)
    q = lambda v: float(np.diff(np.nanquantile(v, [0.025, 0.975]))[0])  # noqa: E731
    return {"P": P, "K": K, "M": M, "width_R2": q(r2), "width_R2_within_problem": q(r2w),
            "median_R2": float(np.nanmedian(r2)), "median_R2_wp": float(np.nanmedian(r2w)),
            "generations": P * F * K * M}


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--stage", default="all", choices=["embed", "all"])
    args = ap.parse_args()
    t0 = time.time()
    data = load_samples()
    emb = embeddings([t for d in data.values() for t in d.text])
    if args.stage == "embed":
        return
    results = label({"task": TASK, "question": "nested-variance ceiling R2_max on importance "
                     "predictable from the pre-emission history",
                     "grid": {"thresholds": THRESHOLDS, "min_sizes": MIN_SIZES,
                              "primary": PRIMARY, "linkage": LINKAGE, "metrics": METRICS,
                              "n_splits": N_SPLITS, "model": MODEL, "revision": REVISION},
                     "datasets": {k: {"rows": len(v), "forks": int(v.groupby(
                         ["trajectory_id", "fork_idx"]).ngroups),
                         "problems": int(v.problem_id.nunique())} for k, v in data.items()},
                     "cells": {}})
    tables = {}
    for dname, df in data.items():
        lab = cluster_labels(df, emb, dname)
        for thr in THRESHOLDS:
            tk = time.time()
            tab = build_cluster_table(df, lab[f"c{thr}"].to_numpy(), f"split:{dname}:{thr}")
            tables[(dname, thr)] = tab
            for ms in MIN_SIZES:
                for met in METRICS:
                    C = components(tab, ms, met)
                    name = f"{dname}:{thr}:{ms}:{met}"
                    results["cells"][name] = summarise_cell(C, name, args.reps)
            print(f"{dname} thr={thr} done in {time.time() - tk:.0f}s", flush=True)

    # ---- validation on dataset A, primary threshold
    dfA = data["A_released"]
    labA = cluster_labels(dfA, emb, "A_released")[f"c{PRIMARY[0]}"].to_numpy()
    val = {}
    rng = np.random.default_rng(seed("synthetic_null"))
    perm_codes = {}
    groups = dfA.groupby(["trajectory_id", "fork_idx"], sort=False).indices
    order = np.arange(len(dfA))
    for _, r in groups.items():
        order[r] = rng.permutation(r)
    for met, col in (("tv", "cat_canon"), ("tv_raw", "cat_raw"), ("acc", "correct")):
        perm_codes[met] = pd.factorize(dfA[col])[0][order]
    perm_codes["acc_signed"] = dfA["correct"].astype(int).to_numpy()[order]
    tabN = build_cluster_table(dfA, labA, "split:null", answers_override=perm_codes)
    val["null"] = {}
    for ms in MIN_SIZES:
        for met in ("tv", "acc", "acc_signed"):
            C = components(tabN, ms, met)
            val["null"][f"{ms}:{met}"] = summarise_cell(C, f"null:{ms}:{met}", 0, boot=False)
    val["known"] = {}
    for sname, setting in SETTINGS.items():
        rng = np.random.default_rng(seed(f"synthetic_known:{sname}"))
        codes, truth = synthetic_truth(dfA, labA, setting, rng)
        tabK = build_cluster_table(dfA, labA, f"split:known:{sname}", answers_override=codes)
        val["known"][sname] = {}
        for ms in MIN_SIZES:
            for met in ("tv", "acc", "acc_signed"):
                C = components(tabK, ms, met)
                est = summarise_cell(C, f"known:{sname}:{ms}:{met}", 0, boot=False)["point"]
                val["known"][sname][f"{ms}:{met}"] = {"truth": truth[f"{ms}:{met}"],
                                                      "estimate": est}
    results["validation"] = val

    # ---- K/M recommendation from the primary-cell variance components
    design = {}
    for dname in ("A_released", "B_r012"):
        for met in ("tv", "acc"):
            cell = results["cells"][f"{dname}:{PRIMARY[0]}:{PRIMARY[1]}:{met}"]
            pt = cell["point"]
            comp = {"var_across_problem": pt["var_across_problem"],
                    "var_between_within_p": pt["var_between_h"] - pt["var_across_problem"],
                    "var_within_h": pt["var_within_h"]}
            rng = np.random.default_rng(seed(f"design:{dname}:{met}"))
            grid = [design_sim(comp, cell["noise_a"], P, K, M, rng)
                    for P in DESIGN_P for K in DESIGN_K for M in DESIGN_M]
            ok = [g for g in grid if g["width_R2_within_problem"] <= TARGET_WIDTH]
            ok_all = [g for g in grid if g["width_R2"] <= TARGET_WIDTH]
            design[f"{dname}:{met}"] = {
                "inputs": {**comp, "noise_a": cell["noise_a"]}, "grid": grid,
                "cheapest_within_problem_width_le_0.2":
                    min(ok, key=lambda g: g["generations"]) if ok else None,
                "cheapest_overall_width_le_0.2":
                    min(ok_all, key=lambda g: g["generations"]) if ok_all else None}
    results["design"] = design
    results["runtime_s"] = round(time.time() - t0)
    write_json(OUT / "results.json", results)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
