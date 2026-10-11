"""v3 follow-up T4 (item 8): does problem difficulty account for the probe null?
EXPLORATORY, POST HOC.

Builds on s14 / s16 (docs/representation_rootcause.md, docs/probe_capacity.md). Same data
(6,434 sentences with forks i and i+1, 40 traces, 20 problems), same stored boundary
states (layers 24/32/40), same protocol: leave-one-problem-out (LOPO) over the 20 problems
with every scaler / PCA / model fit on the training problems only.

Targets (both groupings; canonical = primary, raw = s16's grouping, reproduced exactly):
acc_level, entropy, tv_debiased, signed_dacc, p_own (s16 definitions, computed with the
s16 functions on the shared released_canon table).

1. Difficulty (fixed here, problem level):
   d1 = mean canonical resample accuracy over ALL forks of BOTH traces (released_canon);
   d2 = no-reasoning accuracy from results/s08_memorisation.csv.
   Strata (fixed rule): median split  low = d <= median(d), high = d > median(d);
   terciles  t1 = d <= q(1/3), t2 = q(1/3) < d <= q(2/3), t3 = d > q(2/3) (numpy linear
   quantiles over the 20 problem values; ties stay together, so a stratum may be empty).
   Stratum-trained probes are fit only for strata with >= MIN_TRAIN_STRATUM problems.
   Evaluation: global LOPO probe evaluated inside each stratum, and probes trained by LOPO
   inside the stratum. Difficulty as covariate: rank partial Spearman (pooled rows) of label
   and prediction given rank(d), plus eval-time problem demeaning (problem fixed effects).
2. Within-problem labels: s16-style demeaned variants (features and training targets
   demeaned within trajectory, or within problem); metrics: pooled Spearman of demeaned
   label vs demeaned prediction, within-trace / within-problem Spearman, and pairwise
   ranking accuracy (pairs in the same trace; also same problem). Baselines: position
   (index, log index, log prefix length), position + observed entropy, difficulty, and
   activations + position (+ entropy).
3. Full width vs PCA-128: exact linear ridge on all 5,120 standardised dims (dual form,
   n x n Gram, one eigendecomposition per outer fold) vs ridge on PCA-128 (primal SVD).
   For BOTH, alpha is chosen per target by exact inner leave-one-problem-out CV over the
   19 training problems (closed-form group-deletion residuals from the decomposition;
   centring is fit on the outer training fold). The s16-style RidgeCV PCA-128 probe is
   also reported.

Uncertainty: R.cluster_bootstrap over problems, 2,000 reps, one shared seed for global
stats (so paired differences use identical draws), one seed per stratum.
Scores are rounded to 10 decimals before every rank statistic.

Usage: python analysis/v3_followup/t4_probe_difficulty/run.py [--stage fit|eval|all]
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
from safetensors import safe_open
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.answers import normalize_v2  # noqa: E402
from arv2.canon import canonical_answer  # noqa: E402
from arv2.paths import DATA, INPUTS, RESULTS, write_json  # noqa: E402

TASK = "t4_probe_difficulty"
OUT = Path(__file__).resolve().parent
CACHE = DATA / "derived/v3_followup" / TASK
SHARED = DATA / "derived/v3_followup"
STATE_CACHE = Path("/private/tmp/claude-501/-Users-cnm13ryan-git-independent/"
                   "2de96dc2-71e5-4edb-ab3d-5a7112350b73/scratchpad/s16/states_24_32_40.npy")
LAYERS = [24, 32, 40]
HEADLINE = 32
PCA_DIM = 128
ALPHAS = np.logspace(-1, 7, 17)            # s16 RidgeCV grid
LOGO_ALPHAS = np.logspace(-1, 9, 21)       # inner-CV grid for full-width and matched PCA
BASE_TARGETS = ["acc_level", "entropy", "tv_debiased", "signed_dacc", "p_own"]
GROUPINGS = ("canon", "raw")
COLS = [f"{g}:{t}" for g in GROUPINGS for t in BASE_TARGETS]
REPS = 2000
MIN_TRAIN_STRATUM = 4
SEED_ROOT = f"20260914:arv2:v3f:{TASK}"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s16 = load("s16_probe_capacity")


def rnd(a):
    return np.round(np.asarray(a, float), 10)


def group_mean(X, g):
    X = np.asarray(X)
    df = pd.DataFrame(X.reshape(len(X), -1))
    return df.groupby(np.asarray(g)).transform("mean").to_numpy(X.dtype).reshape(X.shape)


# ============================================================================ data
def build_frame():
    traj = pd.read_parquet(INPUTS / "trajectories.parquet")
    roll = pd.read_parquet(SHARED / "released_canon.parquet",
                           columns=["trajectory_id", "problem_id", "fork_idx", "sample_idx",
                                    "cat_raw", "cat_canon", "correct_raw", "correct_canon"])
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")[["sentence_id", "prefix_token_count"]]
    sents = sents.merge(feats, on="sentence_id").sort_values(
        ["trajectory_id", "sentence_idx"]).reset_index(drop=True)
    stats = {}
    for g in GROUPINGS:
        norm = canonical_answer if g == "canon" else normalize_v2
        own = {t: norm(a) for t, a in zip(traj.trajectory_id, traj.normalized_final_answer)}
        r = roll.rename(columns={f"cat_{g}": "cat", f"correct_{g}": "correct_v2"})[
            ["trajectory_id", "fork_idx", "sample_idx", "cat", "correct_v2"]]
        vocab = r.groupby("trajectory_id").cat.agg(set)
        assert all(own[t] in vocab[t] for t in own), f"own answer not in vocab ({g})"
        stats[g] = s16.fork_stats(r, own)
    fs0 = stats["canon"][0]
    nxt = {(t, i) for t, i in zip(sents.trajectory_id, sents.sentence_idx)}
    sents = sents[[(t, i + 1) in nxt and (t, i) in fs0 and (t, i + 1) in fs0
                   for t, i in zip(sents.trajectory_id, sents.sentence_idx)]].reset_index(drop=True)
    rows = []
    for t, i in zip(sents.trajectory_id, sents.sentence_idx):
        rec = {}
        for g in GROUPINGS:
            fs, k = stats[g]
            for part, pre in (("all", ""), ("A", "A_"), ("B", "B_")):
                v = s16.targets_for(fs, k, t, i, part)
                rec.update({f"{pre}{g}:{n}": v[n] for n in BASE_TARGETS})
        rows.append(rec)
    frame = pd.concat([sents, pd.DataFrame(rows)], axis=1)
    frame["trace_correct"] = frame.trajectory_id.map(dict(zip(traj.trajectory_id, traj.is_correct)))
    d1 = roll.groupby("problem_id").correct_canon.mean()
    d1_raw = roll.groupby("problem_id").correct_raw.mean()
    mem = pd.read_csv(RESULTS / "s08_memorisation.csv").set_index("problem_id")
    d2 = mem.no_reasoning_accuracy
    diff = pd.DataFrame({"d1": d1, "d1_raw": d1_raw, "d2": d2.reindex(d1.index)})
    assert diff.notna().all().all() and len(diff) == 20
    return frame, diff


def check_against_s16(frame):
    ref, _ = s16.build_frame()
    m = frame[["trajectory_id", "sentence_idx"]].merge(
        ref[["trajectory_id", "sentence_idx"] + BASE_TARGETS], on=["trajectory_id", "sentence_idx"])
    assert len(m) == len(frame) == len(ref)
    mx = {n: float(np.abs(m[n].to_numpy() - frame[f"raw:{n}"].to_numpy()).max()) for n in BASE_TARGETS}
    assert max(mx.values()) < 1e-12, mx
    return mx


def strata(diff: pd.Series) -> dict:
    """Fixed rule (see module docstring). Returns {scheme: {stratum: [problem ids]}}."""
    d = diff.sort_index()
    med = float(np.median(d))
    q1, q2 = (float(x) for x in np.quantile(d, [1 / 3, 2 / 3]))
    return {"median": {"low": sorted(d.index[d <= med]), "high": sorted(d.index[d > med])},
            "tercile": {"t1": sorted(d.index[d <= q1]),
                        "t2": sorted(d.index[(d > q1) & (d <= q2)]),
                        "t3": sorted(d.index[d > q2])},
            "cuts": {"median": med, "q1": q1, "q2": q2}}


def verify_cache(feats, states):
    """Spot-check the s16 cache against two raw shards (all rows, layers 24/32/40)."""
    man = json.loads((INPUTS / "MANIFEST.json").read_text())
    root = Path(man["referenced_in_place"]["activation_root"])
    rels = sorted(feats.shard_path.unique())
    checked = []
    for rel in (rels[0], rels[-1]):
        sub = feats[feats.shard_path == rel]
        with safe_open(root / rel, "pt") as f:
            sl = f.get_slice("resid")
            arr = np.stack([sl[:, L:L + 1].float().numpy()[:, 0] for L in LAYERS], 1)
        ok = np.array_equal(np.asarray(states[sub.index.to_numpy()]), arr[sub.row_idx.to_numpy()])
        if not ok:
            raise SystemExit(f"state cache mismatch for {rel}")
        checked.append({"shard": rel, "rows": int(len(sub)), "exact_match": ok})
    return checked


# ============================================================================ ridge helpers
def ridgecv_fit_predict(Ftr, Ytr, Fte):
    ys = StandardScaler().fit(Ytr)
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ftr, ys.transform(Ytr))
    return ys.inverse_transform(m.predict(Fte)), m.alpha_


def logo_select(U, w, y, groups):
    """Exact leave-one-group-out SSE for linear ridge given H = U diag(w/(w+a)) U^T.
    U: n x r (orthonormal columns), w: r eigenvalues of the (centred) Gram, y: centred n x T."""
    B = U.T @ y
    idx = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    Ug = {g: U[i] for g, i in idx.items()}
    sse = np.zeros((len(LOGO_ALPHAS), y.shape[1]))
    for ai, a in enumerate(LOGO_ALPHAS):
        f = w / (w + a)
        e = y - U @ (f[:, None] * B)
        for g, i in idx.items():
            H = (Ug[g] * f) @ Ug[g].T
            eg = np.linalg.solve(np.eye(len(i)) - H, e[i])
            sse[ai] += (eg ** 2).sum(0)
    return sse.argmin(0), sse


def logo_ridge_dual(Ztr, Zte, Ytr, gtr):
    """Full-width ridge via the n x n linear-kernel Gram."""
    mu = Ztr.mean(0)
    A = Ztr - mu
    K = A @ A.T
    Kte = (Zte - mu) @ A.T
    del A
    w, U = np.linalg.eigh(K)
    del K
    keep = w > w.max() * 1e-10
    w, U = w[keep], U[:, keep]
    ym = Ytr.mean(0)
    yc = Ytr - ym
    best, sse = logo_select(U, w, yc, gtr)
    B = U.T @ yc
    KU = Kte @ U
    P = np.column_stack([KU @ (B[:, j] / (w + LOGO_ALPHAS[best[j]])) for j in range(yc.shape[1])])
    return P + ym, LOGO_ALPHAS[best], int(keep.sum())


def logo_ridge_primal(Ftr, Fte, Ytr, gtr):
    mu = Ftr.mean(0)
    U, s, Vt = np.linalg.svd(Ftr - mu, full_matrices=False)
    w = s ** 2
    ym = Ytr.mean(0)
    yc = Ytr - ym
    best, sse = logo_select(U, w, yc, gtr)
    B = U.T @ yc
    FV = (Fte - mu) @ Vt.T
    P = np.column_stack([FV @ (s / (w + LOGO_ALPHAS[best[j]]) * B[:, j]) for j in range(yc.shape[1])])
    return P + ym, LOGO_ALPHAS[best]


# ============================================================================ fit stage
def fit(frame, diff, groups_strata):
    t0 = time.time()
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").reset_index(drop=True)
    states = np.load(STATE_CACHE, mmap_mode="r")
    if states.shape != (len(feats), len(LAYERS), 5120):
        raise SystemExit(f"unexpected cache shape {states.shape}")
    cache_check = verify_cache(feats, states)
    pos_of = {(t, i): j for j, (t, i) in enumerate(zip(feats.trajectory_id, feats.sentence_idx))}
    ip = np.array([pos_of[(t, i)] for t, i in zip(frame.trajectory_id, frame.sentence_idx)])
    prob_all, traj_all = feats.problem_id.to_numpy(), feats.trajectory_id.to_numpy()
    groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
    uprob = np.unique(groups)
    Y = frame[COLS].to_numpy(float)
    ent = {g: frame[f"{g}:entropy"].to_numpy(float)[:, None] for g in GROUPINGS}
    posf = np.column_stack([frame.sentence_idx, np.log1p(frame.sentence_idx),
                            np.log1p(frame.prefix_token_count)]).astype(float)
    dfeat = frame.problem_id.map(diff.d1).to_numpy(float)[:, None]
    dm = {"raw": None, "dm_traj": traj, "dm_prob": groups}
    dm_all = {"raw": None, "dm_traj": traj_all, "dm_prob": prob_all}
    preds: dict[str, np.ndarray] = {}
    alphas: dict[str, list] = {}

    def put(key, te, val):
        preds.setdefault(key, np.full(Y.shape, np.nan))[te] = val

    def dmn(X, v):
        return X if dm[v] is None else X - group_mean(X, dm[v])

    # ---- baselines without activations (grouping-specific when entropy is a feature)
    for v in ("raw", "dm_traj", "dm_prob"):
        Yv = dmn(Y, v)
        for name, F_by_g in (("pos", {g: posf for g in GROUPINGS}),
                             ("pos+ent", {g: np.hstack([posf, ent[g]]) for g in GROUPINGS}),
                             ("diff", {g: dfeat for g in GROUPINGS}),
                             ("diff+pos", {g: np.hstack([posf, dfeat]) for g in GROUPINGS})):
            if v != "raw" and name.startswith("diff"):
                continue
            key = f"{name}|{v}"
            for g in uprob:
                tr, te = groups != g, groups == g
                for gi, gname in enumerate(GROUPINGS):
                    F = dmn(F_by_g[gname], v)
                    sc = StandardScaler().fit(F[tr])
                    cols = slice(gi * 5, gi * 5 + 5)
                    P, _ = ridgecv_fit_predict(sc.transform(F[tr]), Yv[tr][:, cols], sc.transform(F[te]))
                    preds.setdefault(key, np.full(Y.shape, np.nan))[np.ix_(te, np.arange(Y.shape[1])[cols])] = P
    print(f"baselines done {time.time() - t0:.0f}s", flush=True)

    for li, L in enumerate(LAYERS):
        X = np.asarray(states[:, li], dtype=np.float32)
        for g in uprob:
            tf = time.time()
            tr_s = prob_all != g
            tr, te = groups != g, groups == g
            sc = StandardScaler().fit(X[tr_s])
            Z = sc.transform(X).astype(np.float32)
            for v in ("raw", "dm_traj", "dm_prob"):
                Zv = Z if v == "raw" else Z - group_mean(Z, dm_all[v])
                T = PCA(PCA_DIM, svd_solver="randomized", random_state=0).fit(Zv[tr_s]).transform(Zv)[ip]
                Yv = dmn(Y, v)
                P, a = ridgecv_fit_predict(T[tr], Yv[tr], T[te])
                put(f"act_pca128|{v}|L{L}", te, P)
                alphas.setdefault(f"act_pca128|{v}|L{L}", []).append(a.tolist())
                pz = dmn(posf, v)
                for gi, gname in enumerate(GROUPINGS):
                    cols = np.arange(gi * 5, gi * 5 + 5)
                    for name, extra in (("act+pos", pz), ("act+pos+ent", np.hstack([pz, dmn(ent[gname], v)]))):
                        esc = StandardScaler().fit(extra[tr])
                        Ftr = np.hstack([T[tr], esc.transform(extra[tr])])
                        Fte = np.hstack([T[te], esc.transform(extra[te])])
                        P, _ = ridgecv_fit_predict(Ftr, Yv[tr][:, cols], Fte)
                        preds.setdefault(f"{name}|{v}|L{L}", np.full(Y.shape, np.nan))[np.ix_(te, cols)] = P
                if v == "raw":
                    # matched PCA-128 with inner LOGO alpha, and full-width dual ridge
                    T64 = T.astype(np.float64)
                    P, a = logo_ridge_primal(T64[tr], T64[te], Y[tr], groups[tr])
                    put(f"act_pca128_logo|raw|L{L}", te, P)
                    alphas.setdefault(f"act_pca128_logo|raw|L{L}", []).append(a.tolist())
                    Z64 = Z[ip].astype(np.float64)
                    P, a, rank = logo_ridge_dual(Z64[tr], Z64[te], Y[tr], groups[tr])
                    del Z64
                    put(f"act_full5120_logo|raw|L{L}", te, P)
                    alphas.setdefault(f"act_full5120_logo|raw|L{L}", []).append(a.tolist())
                    alphas.setdefault(f"full_rank|L{L}", []).append(rank)
            print(f"L{L} fold {g} {time.time() - tf:.0f}s total {time.time() - t0:.0f}s", flush=True)

        # ---- stratum-trained probes (PCA-128 RidgeCV, raw), LOPO inside each stratum
        for dname, sch in groups_strata.items():
            for scheme in ("median", "tercile"):
                key = f"strat:{dname}:{scheme}|raw|L{L}"
                for sname, plist in sch[scheme].items():
                    if len(plist) < MIN_TRAIN_STRATUM:
                        continue
                    in_s_all = np.isin(prob_all, plist)
                    in_s = np.isin(groups, plist)
                    for g in plist:
                        tr_s = in_s_all & (prob_all != g)
                        tr, te = in_s & (groups != g), groups == g
                        sc = StandardScaler().fit(X[tr_s])
                        Z = sc.transform(X).astype(np.float32)
                        T = PCA(PCA_DIM, svd_solver="randomized", random_state=0).fit(Z[tr_s]).transform(Z)[ip]
                        P, _ = ridgecv_fit_predict(T[tr], Y[tr], T[te])
                        put(key, te, P)
        print(f"L{L} strata done {time.time() - t0:.0f}s", flush=True)
        del X
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "preds.npz", keys=np.array(list(preds)),
                        **{f"p{j}": v for j, v in enumerate(preds.values())})
    (CACHE / "fit_meta.json").write_text(json.dumps(
        {"alphas": alphas, "cache_check": cache_check, "fit_runtime_s": time.time() - t0},
        default=str))
    return preds


def load_preds():
    z = np.load(CACHE / "preds.npz")
    return {k: z[f"p{j}"] for j, k in enumerate(z["keys"])}


# ============================================================================ metrics
def sp(a, b):
    a, b = rnd(a), rnd(b)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_sp(y, s, z):
    ry, rs, rz = rankdata(rnd(y)), rankdata(rnd(s)), rankdata(rnd(z))
    Z = np.column_stack([np.ones(len(y)), rz])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    a, b = res(ry), res(rs)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pair_acc(y, s):
    y, s = rnd(y), rnd(s)
    iu = np.triu_indices(len(y), 1)
    dy = np.sign(y[:, None] - y[None, :])[iu]
    ds = np.sign(s[:, None] - s[None, :])[iu]
    m = dy != 0
    if not m.any():
        return np.nan
    return float(np.where(ds[m] == 0, 0.5, (ds[m] == dy[m]).astype(float)).mean())


class Ctx:
    def __init__(self, frame, diff):
        self.frame = frame
        self.prob = frame.problem_id.to_numpy()
        self.traj = frame.trajectory_id.to_numpy()
        self.tkeys = np.unique(self.traj)
        self.tprob = np.array([int(t.split(":")[0]) for t in self.tkeys])
        self.pkeys = np.unique(self.prob)
        self.d1 = frame.problem_id.map(diff.d1).to_numpy(float)
        self.d2 = frame.problem_id.map(diff.d2).to_numpy(float)
        self.tidx = {t: np.flatnonzero(self.traj == t) for t in self.tkeys}
        self.pidx = {p: np.flatnonzero(self.prob == p) for p in self.pkeys}


def unit_vals(ctx, y, s, fn, level, keep_rows=None):
    keys, idx = (ctx.tkeys, ctx.tidx) if level == "trace" else (ctx.pkeys, ctx.pidx)
    out = np.full(len(keys), np.nan)
    for j, k in enumerate(keys):
        i = idx[k]
        if keep_rows is not None:
            i = i[keep_rows[i]]
        if len(i) >= 3 and np.isfinite(s[i]).all():
            out[j] = fn(y[i], s[i])
    return out


def ci_units(vals, unit_prob, seed, subset=None):
    if subset is not None:
        m = np.isin(unit_prob, subset)
        vals, unit_prob = vals[m], unit_prob[m]
    if np.isfinite(vals).sum() == 0:
        return [np.nan, np.nan]
    draws = R.cluster_bootstrap(unit_prob, lambda r: np.nanmean(vals[r]), REPS, seed)
    lo, hi, _ = R.interval(draws)
    return [lo, hi]


def ci_rows(fn, rows_prob, seed):
    draws = R.cluster_bootstrap(rows_prob, fn, REPS, seed)
    lo, hi, _ = R.interval(draws)
    return [lo, hi]


def f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def evaluate(ctx, P, Y, j, seed, rows=None, full=True, row_cis=("overall",)):
    """Metrics for column j of prediction P vs label Y. rows: boolean mask (stratum)."""
    rows = np.ones(len(Y), bool) if rows is None else rows
    y, s = Y[:, j], P[:, j]
    ok = rows & np.isfinite(s)
    plist = np.unique(ctx.prob[ok])
    out = {"n_problems": int(len(plist)), "n_rows": int(ok.sum())}
    if ok.sum() < 3:
        return out
    wt = unit_vals(ctx, y, s, sp, "trace", ok)
    pt = unit_vals(ctx, y, s, pair_acc, "trace", ok)
    out["within_trace"] = f(np.nanmean(wt))
    out["within_trace_ci"] = [f(x) for x in ci_units(wt, ctx.tprob, seed, plist)]
    out["pair_trace"] = f(np.nanmean(pt))
    out["pair_trace_ci"] = [f(x) for x in ci_units(pt, ctx.tprob, seed, plist)]
    ii = np.flatnonzero(ok)
    yo, so, po = y[ii], s[ii], ctx.prob[ii]
    yp = yo - group_mean(yo, po)
    spm = so - group_mean(so, po)
    to = ctx.traj[ii]
    yt, st = yo - group_mean(yo, to), so - group_mean(so, to)
    out["overall"] = f(sp(yo, so))
    out["pooled_dm_prob"] = f(sp(yp, spm))
    out["pooled_dm_traj"] = f(sp(yt, st))
    if full:
        wp = unit_vals(ctx, y, s, sp, "problem", ok)
        pp = unit_vals(ctx, y, s, pair_acc, "problem", ok)
        out["within_problem"] = f(np.nanmean(wp))
        out["within_problem_ci"] = [f(x) for x in ci_units(wp, ctx.pkeys, seed, plist)]
        out["pair_problem"] = f(np.nanmean(pp))
        out["pair_problem_ci"] = [f(x) for x in ci_units(pp, ctx.pkeys, seed, plist)]
        out["partial_d1"] = f(partial_sp(yo, so, ctx.d1[ii]))
        out["partial_d2"] = f(partial_sp(yo, so, ctx.d2[ii]))
        pm = pd.DataFrame({"p": po, "y": yo, "s": so, "d": ctx.d1[ii]}).groupby("p").mean()
        out["problem_level"] = f(sp(pm.y, pm.s))
        out["problem_level_partial_d1"] = f(partial_sp(pm.y.to_numpy(), pm.s.to_numpy(), pm.d.to_numpy()))
        out["problem_mean_pred_vs_d1"] = f(sp(pm.s, pm.d))
    fns = {"overall": lambda r: sp(yo[r], so[r]),
           "partial_d1": lambda r: partial_sp(yo[r], so[r], ctx.d1[ii][r]),
           "partial_d2": lambda r: partial_sp(yo[r], so[r], ctx.d2[ii][r]),
           "pooled_dm_prob": lambda r: sp(yp[r], spm[r]),
           "pooled_dm_traj": lambda r: sp(yt[r], st[r])}
    for name in row_cis:
        out[f"{name}_ci"] = [f(x) for x in ci_rows(fns[name], po, seed)]
    return out


def paired(ctx, Pa, Pb, Y, j, seed):
    """a - b, within-trace, pair-trace and overall, identical problem draws."""
    y = Y[:, j]
    res = {}
    for name, fn in (("within_trace", sp), ("pair_trace", pair_acc)):
        d = unit_vals(ctx, y, Pa[:, j], fn, "trace") - unit_vals(ctx, y, Pb[:, j], fn, "trace")
        res[name] = f(np.nanmean(d))
        res[f"{name}_ci"] = [f(x) for x in ci_units(d, ctx.tprob, seed)]
    a, b = Pa[:, j], Pb[:, j]
    res["overall"] = f(sp(y, a) - sp(y, b))
    res["overall_ci"] = [f(x) for x in ci_rows(lambda r: sp(y[r], a[r]) - sp(y[r], b[r]), ctx.prob, seed)]
    return res


# ============================================================================ eval stage
def run_eval(frame, diff, groups_strata, preds, meta):
    t0 = time.time()
    ctx = Ctx(frame, diff)
    Y = frame[COLS].to_numpy(float)
    seed = R.seed_from(f"{SEED_ROOT}:boot")
    canon_j = [COLS.index(f"canon:{t}") for t in BASE_TARGETS]
    raw_j = [COLS.index(f"raw:{t}") for t in BASE_TARGETS]
    out: dict = {}

    # ---------------- label structure: how much variance is between problems / explained by d
    struct = {}
    for c in COLS:
        y = frame[c].to_numpy(float)
        pmean = group_mean(y, ctx.prob)
        tmean = group_mean(y, ctx.traj)
        tot = ((y - y.mean()) ** 2).sum()
        struct[c] = {"frac_var_between_problems": f(((pmean - y.mean()) ** 2).sum() / tot),
                     "frac_var_between_traces": f(((tmean - y.mean()) ** 2).sum() / tot),
                     "spearman_label_vs_d1_rows": f(sp(y, ctx.d1)),
                     "spearman_problem_mean_vs_d1": f(sp(pd.Series(y).groupby(ctx.prob).mean(),
                                                         diff.d1.sort_index()))}
    out["label_structure"] = struct

    # ---------------- label split-half reliability for the pairwise metric (n = 50 per half)
    sh = {}
    for t in BASE_TARGETS:
        a, b = frame[f"A_canon:{t}"].to_numpy(float), frame[f"B_canon:{t}"].to_numpy(float)
        wt = unit_vals(ctx, a, b, sp, "trace")
        pt = unit_vals(ctx, a, b, pair_acc, "trace")
        sh[t] = {"split_half_within_trace_spearman_n50": f(np.nanmean(wt)),
                 "split_half_pair_trace_acc_n50": f(np.nanmean(pt))}
    out["label_split_half_canon"] = sh

    # ---------------- 1+2: every prediction set, canon (CIs) and raw (points)
    table = {}
    for key, P in preds.items():
        if key.startswith("strat:"):
            continue
        variant = key.split("|")[1]
        layer = key.split("|")[2] if key.count("|") == 2 else None
        headline = layer in (None, f"L{HEADLINE}")
        if variant == "raw":
            rc = ("overall", "partial_d1", "partial_d2", "pooled_dm_prob") if headline else ()
        else:
            rc = (("pooled_dm_traj",) if variant == "dm_traj" else ("pooled_dm_prob",)) if headline else ()
        table[key] = {}
        for t, j, jr in zip(BASE_TARGETS, canon_j, raw_j):
            table[key][f"canon:{t}"] = evaluate(ctx, P, Y, j, seed, row_cis=rc)
            r = evaluate(ctx, P, Y, jr, seed, row_cis=())
            table[key][f"raw:{t}"] = {k: v for k, v in r.items() if not k.endswith("_ci")}
        print(f"eval {key} {time.time() - t0:.0f}s", flush=True)
    out["probes"] = table

    # ---------------- 1: strata (global probe evaluated inside strata; stratum-trained probe)
    st = {}
    for dname, sch in groups_strata.items():
        st[dname] = {"cuts": sch["cuts"], "strata": {s: [int(p) for p in v]
                                                     for scheme in ("median", "tercile")
                                                     for s, v in sch[scheme].items()}}
        for scheme in ("median", "tercile"):
            for sname, plist in sch[scheme].items():
                rec = {"problems": [int(p) for p in plist], "n_problems": len(plist)}
                if len(plist) < 2:
                    rec["note"] = "fewer than 2 problems: not estimable"
                    st[dname][f"{scheme}:{sname}"] = rec
                    continue
                if len(plist) < 5:
                    rec["note"] = "fewer than 5 problems: bootstrap CI not meaningful"
                rows = np.isin(ctx.prob, plist)
                sseed = R.seed_from(f"{SEED_ROOT}:boot:{dname}:{scheme}:{sname}")
                for L in LAYERS:
                    for label, key in (("global_probe", f"act_pca128|raw|L{L}"),
                                       ("stratum_trained", f"strat:{dname}:{scheme}|raw|L{L}"),
                                       ("position_global", "pos|raw")):
                        if label == "position_global" and L != HEADLINE:
                            continue
                        P = preds.get(key)
                        if P is None or not np.isfinite(P[rows]).any():
                            continue
                        tag = f"{label}" + ("" if label == "position_global" else f"|L{L}")
                        rec[tag] = {}
                        for t, j, jr in zip(BASE_TARGETS, canon_j, raw_j):
                            rec[tag][f"canon:{t}"] = evaluate(ctx, P, Y, j, sseed, rows=rows,
                                                              full=False, row_cis=("overall",))
                            r = evaluate(ctx, P, Y, jr, sseed, rows=rows, full=False, row_cis=())
                            rec[tag][f"raw:{t}"] = {k: v for k, v in r.items() if not k.endswith("_ci")}
                st[dname][f"{scheme}:{sname}"] = rec
                print(f"strata {dname} {scheme} {sname} {time.time() - t0:.0f}s", flush=True)
    out["strata"] = st

    # ---------------- paired comparisons (identical draws)
    comp = {}
    for L in LAYERS:
        full, pl, s16r = (preds[f"act_full5120_logo|raw|L{L}"], preds[f"act_pca128_logo|raw|L{L}"],
                          preds[f"act_pca128|raw|L{L}"])
        for t, j in zip(BASE_TARGETS, canon_j):
            comp[f"full5120_minus_pca128logo|L{L}|canon:{t}"] = paired(ctx, full, pl, Y, j, seed)
            if L == HEADLINE:
                comp[f"full5120_minus_pca128ridgecv|L{L}|canon:{t}"] = paired(ctx, full, s16r, Y, j, seed)
    L = HEADLINE
    for t, j in zip(BASE_TARGETS, canon_j):
        for v in ("raw", "dm_traj", "dm_prob"):
            comp[f"act_minus_pos|{v}|L{L}|canon:{t}"] = paired(
                ctx, preds[f"act_pca128|{v}|L{L}"], preds[f"pos|{v}"], Y, j, seed)
            comp[f"actpos_minus_pos|{v}|L{L}|canon:{t}"] = paired(
                ctx, preds[f"act+pos|{v}|L{L}"], preds[f"pos|{v}"], Y, j, seed)
            if t != "entropy":
                comp[f"actposent_minus_posent|{v}|L{L}|canon:{t}"] = paired(
                    ctx, preds[f"act+pos+ent|{v}|L{L}"], preds[f"pos+ent|{v}"], Y, j, seed)
        comp[f"act_minus_diff|raw|L{L}|canon:{t}"] = paired(
            ctx, preds[f"act_pca128|raw|L{L}"], preds["diff|raw"], Y, j, seed)
        print(f"paired {t} {time.time() - t0:.0f}s", flush=True)
    out["paired"] = comp

    # ---------------- alphas
    al = {}
    for k, v in meta["alphas"].items():
        if k.startswith("full_rank"):
            al[k] = [int(min(v)), int(max(v))]
            continue
        a = np.array(v, float)
        grid = LOGO_ALPHAS if "logo" in k else ALPHAS
        al[k] = {"median_log10_alpha_per_target": dict(zip(COLS, np.round(np.median(np.log10(a), 0), 2).tolist())),
                 "frac_at_grid_edge": f(np.mean((a <= grid[0] * 1.0001) | (a >= grid[-1] * 0.9999)))}
    out["alphas"] = al
    out["eval_runtime_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "eval", "all"], default="all")
    args = ap.parse_args()
    t0 = time.time()
    frame, diff = build_frame()
    s16_check = check_against_s16(frame)
    groups_strata = {"d1": strata(diff.d1), "d2": strata(diff.d2)}
    print(f"frame {len(frame)} rows; s16 raw targets reproduced (max abs diff "
          f"{max(s16_check.values()):.1e}) {time.time() - t0:.0f}s", flush=True)
    if args.stage in ("fit", "all"):
        preds = fit(frame, diff, groups_strata)
    if args.stage == "fit":
        return
    preds = load_preds()
    meta = json.loads((CACHE / "fit_meta.json").read_text())
    out = {"status": "EXPLORATORY, POST HOC", "task": TASK, "layers": LAYERS,
           "headline_layer": HEADLINE, "targets": BASE_TARGETS, "groupings": list(GROUPINGS),
           "primary_grouping": "canon", "bootstrap_reps": REPS,
           "rows": len(frame), "problems": int(frame.problem_id.nunique()),
           "trajectories": int(frame.trajectory_id.nunique()),
           "s16_raw_target_reproduction_max_abs_diff": s16_check,
           "state_cache_check": meta["cache_check"], "fit_runtime_s": meta["fit_runtime_s"],
           "difficulty": {str(p): {k: f(v) for k, v in r.items()} for p, r in diff.iterrows()},
           "difficulty_spearman_d1_d2": f(sp(diff.d1, diff.d2)),
           "logo_alpha_grid": [float(LOGO_ALPHAS[0]), float(LOGO_ALPHAS[-1]), len(LOGO_ALPHAS)],
           "ridgecv_alpha_grid": [float(ALPHAS[0]), float(ALPHAS[-1]), len(ALPHAS)]}
    out.update(run_eval(frame, diff, groups_strata, preds, meta))
    write_json(OUT / "results.json", out)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
