"""v3 follow-up T5 (item 12): every headline number with and without problem 1591.
EXPLORATORY, POST HOC.

Problem 1591 has 27 answer categories in the Phase 2 pool (mostly rounding variants of a
percentage), which inflates its noise floor. This script recomputes the headline numbers of
s12 (Phase 2 confirmatory), s13 (diagnostics), s14 (root cause), s16 (probe capacity),
s17 (targeted sampling) and s18 (plan_v3 pilot) twice: on all problems ("with") and with
problem 1591 dropped everywhere ("without": its rows, its bootstrap cluster, its LOPO fold
and, for the probes, its activations in the PCA training sets).

Conventions (fixed before running):
- Groupings: canon (arv2.canon, primary) and raw (the original basis); both are reported.
- Every interval: problem-cluster bootstrap, 2,000 replicates, seeds
  R.seed_from("20260914:arv2:v3f:t5_without_1591:<name>"). The same seed material is used
  for "with" and "without" except for the condition tag.
- Scores are rounded to 10 decimals before any rank statistic.
- Real effects: per-unit permutation p-values are taken from the shared
  sentence_effects_{raw,canon}.csv (s13 procedure, 2,000 permutations, 300 vs 300); only
  BH 10% is recomputed over the 171 remaining units (BH depends on the number of tests).
- No existing file is modified; s12-s18 and Step 1 are imported. Module-level seed
  constants inside imported scripts are redirected to this task's seed material through a
  proxy for arv2.reliability (s16, s17) or the SEED_MATERIAL global (s18).

Usage: python analysis/v3_followup/t5_without_1591/run.py [--stages s12 s13 ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(STUDY / "scripts"))
from arv2 import answers as A  # noqa: E402
from arv2 import reliability as R  # noqa: E402
from arv2.canon import canonical_answer  # noqa: E402
from arv2.paths import DATA, INPUTS, RESULTS, sha256_file, write_json  # noqa: E402

TASK = "t5_without_1591"
OUT = Path(__file__).resolve().parent
SHARED = DATA / "derived/v3_followup"
INTER = SHARED / TASK
INDEX = (DATA / "generations/phase2_mirror/20261009T0717Z-phase2-23f5e3e6"
         / "completions_index.parquet")
DROP = 1591
CONDS = {"with": None, "without": DROP}
GROUPINGS = ("canon", "raw")
BOOT = 2000
STATE_LAYERS = [8, 16, 24, 32, 40, 48]          # s14 layers (s16 uses 24/32/40 of these)
SEEDBASE = f"20260914:arv2:v3f:{TASK}"


def seed(name: str) -> int:
    return R.seed_from(f"{SEEDBASE}:{name}")


def load(name: str, path: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, path or STUDY / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s12 = load("s12_phase2_score")
s13 = load("s13_phase2_diagnostics")
s14 = load("s14_representation_rootcause")
s16 = load("s16_probe_capacity")
s17 = load("s17_targeted_sampling")
s18 = load("s18_forced_answer")
step1 = load("step1_run", STUDY / "analysis/v3_followup/step1_canon/run.py")


def r10(x):
    return np.round(np.asarray(x, float), 10)


def sp(a, b) -> float:
    return R.spearman(r10(a), r10(b))


def ci(groups, fn, name, reps=BOOT):
    lo, hi, _ = R.interval(R.cluster_bootstrap(np.asarray(groups), fn, reps, seed(name)))
    return [lo, hi]


def seed_proxy(old_prefix: str, tag: str):
    """A stand-in for arv2.reliability whose seed_from maps an imported script's hardcoded
    seed material onto this task's seed material."""
    ns = types.SimpleNamespace(**{k: getattr(R, k) for k in dir(R) if not k.startswith("_")})
    ns.seed_from = lambda m: R.seed_from(m.replace(old_prefix, f"{SEEDBASE}:{tag}"))
    return ns


# ============================================================================ shared data


def units_all():
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    window = {(t["trajectory_id"], f) for t in plan["trajectories"] for f in t["forks"]}
    return units, window


def keep_units(units, drop):
    return [u for u in units if u[1] != drop]


_CACHE: dict = {}


def tables():
    if "p2" not in _CACHE:
        rel = pd.read_parquet(SHARED / "released_canon.parquet")
        p2 = pd.read_parquet(SHARED / "phase2_canon.parquet")
        _, window = units_all()
        rel_win = rel[[(t, int(f)) in window for t, f in zip(rel.trajectory_id, rel.fork_idx,
                                                           strict=True)]]
        cols = ["replicate", "trajectory_id", "fork_idx", "cat_raw", "cat_canon",
                "correct_raw", "correct_canon"]
        pool = pd.concat([p2[cols], rel_win[cols]], ignore_index=True)
        pool["problem_id"] = pool.trajectory_id.str.split(":").str[0].astype(int)
        _CACHE.update(rel=rel, p2=p2, rel_win=rel_win, pool=pool)
    return _CACHE


def effects(g: str, drop) -> pd.DataFrame:
    """Shared per-unit permutation p-values; BH 10% recomputed over the kept units."""
    eff = pd.read_csv(SHARED / f"sentence_effects_{g}.csv")
    if drop is not None:
        eff = eff[eff.problem_id != drop].reset_index(drop=True)
    eff["real_tv_bh10"] = step1.bh(eff.p_tv.to_numpy())
    eff["real_dacc_bh10"] = step1.bh(eff.p_dacc.to_numpy())
    return eff


def view(df, g):
    return df.assign(cat=df[f"cat_{g}"], correct=df[f"correct_{g}"].astype(bool))


# ============================================================================ s12


def stage_s12() -> dict:
    t = tables()
    units, _ = units_all()
    q90 = json.loads((RESULTS / "s06_power.json").read_text())["q90_thresholds"]
    keep = ("rows", "problems", "spearman", "spearman_lo", "spearman_hi", "spearman_decision",
            "icc", "icc_lo", "icc_hi", "icc_decision", "kappa", "kappa_lo", "kappa_hi",
            "kappa_decision", "flag_rate_1", "flag_rate_2", "continuous_form", "flag_form",
            "within_problem_spearman_median")
    out = {}
    for g in GROUPINGS:
        reps = {r: s12.label_table(view(t["p2"][t["p2"].replicate == r], g), units)
                for r in (1, 2)}
        reps[0] = s12.label_table(view(t["rel_win"], g), units)
        for f in reps.values():
            for s in s12.SCORES:
                if s in f:
                    f[s] = r10(f[s])
        out[g] = {}
        for cond, drop in CONDS.items():
            sub = {r: f[f.problem_id != drop] for r, f in reps.items()}
            out[g][cond] = {
                pair: {x["score"]: {k: x.get(k) for k in keep} for x in s12.agreement(
                    sub[a], sub[b], q90, f"{SEEDBASE}:s12:{g}:{cond}:{pair}")}
                for pair, a, b in (("1_vs_2", 1, 2), ("1_vs_0", 1, 0), ("2_vs_0", 2, 0))}
        print(f"s12 {g} done", flush=True)
    return out


# ============================================================================ s13


def stage_s13(draws: int) -> dict:
    t = tables()
    units0, _ = units_all()
    out = {"curve_draws": draws, "curve_n_grid": s13.N_GRID}
    for g in GROUPINGS:
        out[g] = {}
        for cond, drop in CONDS.items():
            units = keep_units(units0, drop)
            pool = view(t["pool"][t["pool"].problem_id != drop], g)
            codes = {c: i for i, c in enumerate(sorted(pool.cat.unique()))}
            k = len(codes)
            df = pool.assign(code=pool.cat.map(codes))
            groups = np.array([p for _, p, _ in units])
            halves = {r: {key: (x.code.to_numpy(), x.correct.to_numpy(float))
                          for key, x in df[df.replicate == r].groupby(["trajectory_id",
                                                                       "fork_idx"])}
                      for r in (0, 1, 2)}
            sc = {r: {n: r10(v) for n, v in s13.unit_scores(halves[r], units, k).items()}
                  for r in (1, 2)}
            res: dict = {"units": len(units), "problems": int(len(set(groups))),
                         "categories": k}
            # 1. replicate decomposition at n = 100
            dec = {}
            for name in s13.SCORES:
                a, b = sc[1][name], sc[2][name]
                ok = np.isfinite(a) & np.isfinite(b)
                sig = float(np.cov(a[ok], b[ok])[0, 1])
                noise = float(np.var(a[ok] - b[ok], ddof=1) / 2)
                means = pd.Series((a[ok] + b[ok]) / 2).groupby(groups[ok]).mean()
                dec[name] = {**s13.rel_stats(a, b, groups), "signal_var": sig,
                             "noise_var": noise,
                             "between_problem_share_of_signal":
                                 float(np.var(means, ddof=1) / sig) if sig > 0 else None}
                ga, gb, gg = a[ok], b[ok], groups[ok]
                dec[name]["spearman_ci"] = ci(gg, lambda r, x=ga, y=gb: R.spearman(x[r], y[r]),
                                              f"s13:{g}:{cond}:dec:{name}:spearman")
                dec[name]["icc_ci"] = ci(gg, lambda r, x=ga, y=gb: R.icc_2_1(x[r], y[r]),
                                         f"s13:{g}:{cond}:dec:{name}:icc")
            res["replicates_1v2_n100"] = dec
            # floor statistics
            floor = {}
            for r in (1, 2):
                f = np.full(len(units), np.nan)
                for j, (tid, _, i) in enumerate(units):
                    f[j] = s13.tv_null_analytic(halves[r][(tid, i)][0],
                                                halves[r][(tid, i + 1)][0], k)
                floor[r] = r10(f)
            f1, f2, t1, t2 = floor[1], floor[2], sc[1]["fork_tv"], sc[2]["fork_tv"]
            res["noise_floor"] = {
                "mean_floor_r1": float(np.mean(f1)), "mean_floor_r2": float(np.mean(f2)),
                "mean_fork_tv_r1": float(np.mean(t1)),
                "floor_r1_vs_floor_r2_spearman": R.spearman(f1, f2),
                "floor_r1_vs_floor_r2_ci": ci(groups, lambda r: R.spearman(f1[r], f2[r]),
                                              f"s13:{g}:{cond}:floor12"),
                "fork_tv_r1_vs_floor_r2_spearman": R.spearman(t1, f2),
                "fork_tv_r1_vs_floor_r2_ci": ci(groups, lambda r: R.spearman(t1[r], f2[r]),
                                                f"s13:{g}:{cond}:tv1floor2"),
                "fork_tv_r1_vs_fork_tv_r2_spearman": R.spearman(t1, t2),
                "share_fork_tv_below_1p5x_floor_r1": float(np.mean(t1 < 1.5 * f1)),
                "share_below_ci": ci(groups, lambda r: float(np.mean(t1[r] < 1.5 * f1[r])),
                                     f"s13:{g}:{cond}:share"),
            }
            # real effects (BH over the kept units)
            eff = effects(g, drop)
            either = eff.real_tv_bh10 | eff.real_dacc_bh10
            res["real_effects"] = {
                "tests": len(eff), "tv_bh10": int(eff.real_tv_bh10.sum()),
                "dacc_bh10": int(eff.real_dacc_bh10.sum()), "either": int(either.sum()),
                "flagged": [f"{a}:{b}" for a, b in zip(eff.trajectory_id[either],
                                                       eff.sentence_idx[either], strict=True)],
                "per_problem_either": eff[either].groupby("problem_id").size().astype(int)
                .to_dict(),
                "bh_p_threshold_tv": float(eff.p_tv[eff.real_tv_bh10].max())
                if eff.real_tv_bh10.any() else None,
                "pooled_floor_mean": float(eff.pooled_floor.mean()),
                "pooled_tv_debiased_mean": float((eff.pooled_fork_tv - eff.pooled_floor).mean()),
            }
            # reliability vs n (s13 procedure; seeds from this task)
            rng = np.random.default_rng(seed(f"s13:{g}:{cond}:curve"))
            pool_d = {key: (x.code.to_numpy(), x.correct.to_numpy(float))
                      for key, x in df.groupby(["trajectory_id", "fork_idx"])}
            curve = {name: {} for name in s13.SCORES}
            for n in s13.N_GRID:
                acc = {name: [] for name in s13.SCORES}
                for _ in range(draws):
                    ha, hb = {}, {}
                    for key, (c, y) in pool_d.items():
                        perm = rng.permutation(len(c))
                        ha[key] = (c[perm[:n]], y[perm[:n]])
                        hb[key] = (c[perm[n:2 * n]], y[perm[n:2 * n]])
                    sa, sb = s13.unit_scores(ha, units, k), s13.unit_scores(hb, units, k)
                    for name in s13.SCORES:
                        acc[name].append(s13.rel_stats(r10(sa[name]), r10(sb[name]), groups))
                for name in s13.SCORES:
                    d = pd.DataFrame(acc[name])
                    curve[name][n] = {m: float(d[m].mean())
                                      for m in ("icc", "spearman", "within_spearman")}
            proj = {}
            for name in s13.SCORES:
                proj[name] = {}
                for m, target in (("icc", 0.75), ("spearman", 0.80), ("within_spearman", 0.80)):
                    c = s13.fit_c(s13.N_GRID, [curve[name][n][m] for n in s13.N_GRID])
                    proj[name][m] = {"c": c, "n_needed": float(target / (1 - target) * c)
                                     if np.isfinite(c) else None,
                                     "at_n300": float(300 / (300 + c)) if np.isfinite(c)
                                     else None}
            res["curve"], res["projection"] = curve, proj
            out[g][cond] = res
            print(f"s13 {g} {cond} done", flush=True)
    return out


# ============================================================================ states


def load_states() -> tuple[pd.DataFrame, np.ndarray]:
    """All stored boundary states at STATE_LAYERS, in mvp_features order, hash-verified on
    first load and cached outside the repository."""
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet").reset_index(drop=True)
    cache = INTER / "states_l8_16_24_32_40_48.npy"
    if cache.exists():
        return feats, np.load(cache, mmap_mode="r")
    manifest = json.loads((INPUTS / "MANIFEST.json").read_text())
    root = Path(manifest["referenced_in_place"]["activation_root"])
    from safetensors import safe_open

    S = np.full((len(feats), len(STATE_LAYERS), 5120), np.nan, np.float32)
    for rel, digest in manifest["referenced_in_place"]["shards"].items():
        if sha256_file(root / rel) != digest:
            raise SystemExit(f"activation shard changed: {rel}")
        sub = feats[feats.shard_path == rel]
        with safe_open(root / rel, "pt") as f:
            sl = f.get_slice("resid")
            arr = np.stack([sl[:, L:L + 1].float().numpy()[:, 0] for L in STATE_LAYERS], 1)
        S[sub.index.to_numpy()] = arr[sub.row_idx.to_numpy()]
    if np.isnan(S).any():
        raise SystemExit("missing activations")
    INTER.mkdir(parents=True, exist_ok=True)
    np.save(cache, S)
    return feats, np.load(cache, mmap_mode="r")


def released_roll(g: str) -> pd.DataFrame:
    rel = tables()["rel"]
    return pd.DataFrame({"trajectory_id": rel.trajectory_id, "fork_idx": rel.fork_idx,
                         "sample_idx": rel.sample_idx, "cat": rel[f"cat_{g}"],
                         "correct_v2": rel[f"correct_{g}"].astype(bool)})


# ============================================================================ s14


def s14_frame() -> pd.DataFrame:
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")[["sentence_id", "prefix_token_count"]]
    fss = {g: s14.fork_stats(released_roll(g)) for g in GROUPINGS}
    fs0 = fss["raw"][0]
    sents = sents.merge(feats, on="sentence_id").sort_values(
        ["trajectory_id", "sentence_idx"]).reset_index(drop=True)
    nxt = {(t, i) for t, i in zip(sents.trajectory_id, sents.sentence_idx)}
    sents = sents[[(t, i + 1) in nxt and (t, i) in fs0 and (t, i + 1) in fs0
                   for t, i in zip(sents.trajectory_id, sents.sentence_idx)]].reset_index(drop=True)
    cols = {}
    for g, (fs, k) in fss.items():
        rows = []
        for t, i in zip(sents.trajectory_id, sents.sentence_idx):
            rec = {f"{g}_{n}": v for n, v in s14.targets_for(fs, k, t, i, "all").items()}
            rec.update({f"{g}_A_{n}": v for n, v in s14.targets_for(fs, k, t, i, "A").items()})
            rec.update({f"{g}_B_{n}": v for n, v in s14.targets_for(fs, k, t, i, "B").items()})
            rows.append(rec)
        cols[g] = pd.DataFrame(rows)
    frame = pd.concat([sents] + list(cols.values()), axis=1)
    num = [c for c in frame.columns if c.startswith(("raw_", "canon_"))]
    frame[num] = r10(frame[num].to_numpy())
    return frame


def within_mean(y, s, traj) -> float:
    vals = [sp(y[traj == t], s[traj == t]) for t in np.unique(traj)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def stage_s14(device: str) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    frame_all = s14_frame()
    feats, S = load_states()
    pos_of = {(t, i): j for j, (t, i) in enumerate(zip(feats.trajectory_id, feats.sentence_idx))}
    ip_all = np.array([pos_of[(t, i)] for t, i in zip(frame_all.trajectory_id,
                                                      frame_all.sentence_idx)])
    iq_all = np.array([pos_of[(t, i + 1)] for t, i in zip(frame_all.trajectory_id,
                                                          frame_all.sentence_idx)])
    txt_all = s14.text_embeddings(frame_all.text.tolist(), device)
    T = s14.TARGETS
    tcols = [f"{g}_{n}" for g in GROUPINGS for n in T]
    out: dict = {"layers": STATE_LAYERS, "pca_dim": s14.PCA_DIM,
                 "reps_run": ["position", "text+position", "pre", "post", "pre+post",
                              "pre+position", "valueprobe_dphat"],
                 "reps_not_run": ["text_S_i", "delta"]}
    oof_store = {}
    for cond, drop in CONDS.items():
        keep = (frame_all.problem_id != drop).to_numpy()
        frame = frame_all[keep].reset_index(drop=True)
        ip, iq, txt = ip_all[keep], iq_all[keep], txt_all[keep]
        groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
        Y = frame[tcols].to_numpy(float)
        res: dict = {"rows": len(frame), "problems": int(len(set(groups))),
                     "trajectories": int(len(set(traj)))}
        # A. ceilings
        ceil = {}
        sb = lambda x: 2 * x / (1 + x) if x > 0 else x  # noqa: E731
        for g in GROUPINGS:
            ceil[g] = {}
            for n in T:
                a, b = frame[f"{g}_A_{n}"].to_numpy(), frame[f"{g}_B_{n}"].to_numpy()
                r, rw = sp(a, b), within_mean(a, b, traj)
                ceil[g][n] = {"split_half_spearman": r, "rel_n100": float(sb(r)),
                              "ceiling_corr": float(np.sqrt(max(sb(r), 0))),
                              "split_half_within": rw, "within_rel_n100": float(sb(rw)),
                              "within_ceiling_corr": float(np.sqrt(max(sb(rw), 0)))}
        res["ceilings"] = ceil
        # B. representations, LOPO with per-fold cached scaler+PCA blocks (identical maths
        # to s14.lopo_predict: each block scaled and PCA'd on the training problems only).
        posf = np.column_stack([frame.sentence_idx, np.log1p(frame.sentence_idx),
                                np.log1p(frame.prefix_token_count)]).astype(float)
        folds = np.unique(groups)

        def blk(B, tr, te, pca):
            sc = StandardScaler().fit(B[tr])
            btr, bte = sc.transform(B[tr]), sc.transform(B[te])
            if pca and B.shape[1] > s14.PCA_DIM:
                p = PCA(s14.PCA_DIM, svd_solver="randomized", random_state=0).fit(btr)
                return p.transform(btr), p.transform(bte), (sc, p)
            return btr, bte, (sc, None)

        def fit(Xtr, Xte, tr):
            ys = StandardScaler().fit(Y[tr])
            m = RidgeCV(alphas=s14.ALPHAS, alpha_per_target=True).fit(Xtr, ys.transform(Y[tr]))
            return ys.inverse_transform(m.predict(Xte))

        preds = {}

        def put(key, te, val):
            preds.setdefault(key, np.full((len(frame),) + np.shape(val)[1:], np.nan))[te] = val

        acc_cols = {g: tcols.index(f"{g}_acc_level") for g in GROUPINGS}
        fold_pos = {}
        for g_ in folds:
            tr, te = groups != g_, groups == g_
            ptr, pte, _ = blk(posf, tr, te, False)
            ttr, tte, _ = blk(txt, tr, te, True)
            fold_pos[g_] = (ptr, pte)
            put(("position", None), te, fit(ptr, pte, tr))
            put(("text+position", None), te, fit(np.hstack([ttr, ptr]), np.hstack([tte, pte]), tr))
        for L in STATE_LAYERS:
            li = STATE_LAYERS.index(L)
            pre = np.asarray(S[:, li][ip])
            post = np.asarray(S[:, li][iq])
            for g_ in folds:
                tr, te = groups != g_, groups == g_
                ptr, pte = fold_pos[g_]
                atr, ate, (scA, pA) = blk(pre, tr, te, True)
                btr, bte, _ = blk(post, tr, te, True)
                put(("pre", L), te, fit(atr, ate, tr))
                put(("post", L), te, fit(btr, bte, tr))
                put(("pre+post", L), te, fit(np.hstack([atr, btr]), np.hstack([ate, bte]), tr))
                put(("pre+position", L), te, fit(np.hstack([atr, ptr]), np.hstack([ate, pte]), tr))
                # value probe (s14 D): p-hat fit on pre states, applied to the post state
                post_te = pA.transform(scA.transform(post[te]))
                for g in GROUPINGS:
                    m = RidgeCV(alphas=s14.ALPHAS).fit(atr, Y[tr, acc_cols[g]])
                    put((f"vp_post_{g}", L), te, m.predict(post_te))
            del pre, post
            print(f"s14 {cond} layer {L} {time.time() - t0:.0f}s", flush=True)
        # evaluation
        table = []
        for (rep, L), P in preds.items():
            if rep.startswith("vp_"):
                continue
            for g in GROUPINGS:
                for n in T:
                    j = tcols.index(f"{g}_{n}")
                    y, s = Y[:, j], P[:, j]
                    c = ceil[g][n]["ceiling_corr"]
                    v = sp(y, s)
                    table.append({"grouping": g, "rep": rep, "layer": L, "target": n,
                                  "spearman": v, "within": within_mean(y, s, traj),
                                  "frac_of_ceiling": v / c if c > 0 else None})
        tab = pd.DataFrame(table)
        res["predictability"] = tab.to_dict("records")
        best = (tab.sort_values("spearman", ascending=False)
                .groupby(["grouping", "rep", "target"]).head(1)
                .sort_values(["grouping", "target", "spearman"], ascending=[True, True, False]))
        res["best_layer_by_rep_target"] = best.to_dict("records")
        # C. partial correlation of the pre-emission fork_tv probe with fork_tv given floor
        dec = {}
        for g in GROUPINGS:
            bsub = best[(best.grouping == g) & (best.rep == "pre") & (best.target == "fork_tv")]
            Lb = int(bsub.layer.iloc[0])
            s_ftv = preds[("pre", Lb)][:, tcols.index(f"{g}_fork_tv")]
            y = frame[f"{g}_fork_tv"].to_numpy()
            fl, en = frame[f"{g}_floor"].to_numpy(), frame[f"{g}_entropy"].to_numpy()
            sr = r10(s_ftv)
            ps = lambda r, z=None: s14.partial_spearman(  # noqa: E731
                y[r], sr[r], [fl[r]] if z is None else [c[r] for c in z])
            dec[g] = {
                "layer": Lb,
                "corr_pred_with": {n: sp(s_ftv, frame[f"{g}_{n}"])
                                   for n in ["fork_tv", "floor", "entropy", "acc_level",
                                             "tv_debiased", "abs_dacc"]},
                "partial_given_floor": ps(np.arange(len(y))),
                "partial_given_floor_ci": ci(groups, ps, f"s14:{g}:{cond}:partial_floor"),
                "partial_given_floor_entropy_position": ps(
                    np.arange(len(y)), [fl, en, posf[:, 0]]),
                "floor_alone_vs_fork_tv": sp(fl, y),
                "floor_alone_within": within_mean(y, fl, traj),
                "pred_vs_fork_tv_ci": ci(groups, lambda r: R.spearman(sr[r], y[r]),
                                         f"s14:{g}:{cond}:pred_ftv"),
            }
        res["pre_fork_tv_decomposition"] = dec
        # value probe summary at every layer
        vp = {}
        for g in GROUPINGS:
            vp[g] = {}
            for L in STATE_LAYERS:
                ph = preds[("pre", L)][:, tcols.index(f"{g}_acc_level")]
                d = preds[(f"vp_post_{g}", L)] - ph
                preds[(f"valueprobe_dphat_{g}", L)] = d
                vp[g][L] = {"p_hat_spearman": sp(ph, frame[f"{g}_acc_level"]),
                            "p_hat_within": within_mean(frame[f"{g}_acc_level"].to_numpy(), ph,
                                                        traj),
                            "dphat_vs_signed_dacc": sp(d, frame[f"{g}_signed_dacc"]),
                            "dphat_vs_signed_dacc_within": within_mean(
                                frame[f"{g}_signed_dacc"].to_numpy(), d, traj)}
        res["value_probe"] = vp
        # OOF baselines for the s18 H4 comparisons (fork_tv target of each grouping)
        for g in GROUPINGS:
            j = tcols.index(f"{g}_fork_tv")
            oof = pd.DataFrame({
                "sentence_id": frame.sentence_id.to_numpy(),
                "position_None": preds[("position", None)][:, j],
                "text+position_None": preds[("text+position", None)][:, j],
                "pre_8": preds[("pre", 8)][:, j], "pre+post_32": preds[("pre+post", 32)][:, j],
                "valueprobe_dphat_32": preds[(f"valueprobe_dphat_{g}", 32)]})
            oof.to_parquet(INTER / f"s14_oof_{g}_{cond}.parquet")
            oof_store[(g, cond)] = oof
        out[cond] = res
    # reproduction of the committed s14 OOF columns (raw, with 1591)
    com = pd.read_parquet(RESULTS / "s14_oof_predictions.parquet")
    mine = oof_store[("raw", "with")].merge(com, on="sentence_id", suffixes=("", "_committed"))
    out["reproduction_vs_committed_oof"] = {
        c: {"max_abs_diff": float(np.abs(mine[c] - mine[f"{c}_committed"]).max()),
            "spearman": sp(mine[c], mine[f"{c}_committed"])}
        for c in ["position_None", "text+position_None", "pre_8", "pre+post_32",
                  "valueprobe_dphat_32"] if f"{c}_committed" in mine}
    out["runtime_s"] = time.time() - t0
    return out


# ============================================================================ s16


def s16_frame(g: str) -> tuple[pd.DataFrame, dict]:
    traj = pd.read_parquet(INPUTS / "trajectories.parquet")
    norm = A.normalize_v2 if g == "raw" else canonical_answer
    own = {t: norm(a) for t, a in zip(traj.trajectory_id, traj.normalized_final_answer)}
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    feats = pd.read_parquet(INPUTS / "mvp_features.parquet")[["sentence_id", "prefix_token_count"]]
    roll = released_roll(g)
    vocab = roll.groupby("trajectory_id").cat.agg(set)
    info = {t: {"own_answer": own[t], "in_rollout_vocab": bool(own[t] in vocab[t]),
                "is_correct": bool(c)} for t, c in zip(traj.trajectory_id, traj.is_correct)}
    fs, k = s16.fork_stats(roll, own)
    sents = sents.merge(feats, on="sentence_id").sort_values(
        ["trajectory_id", "sentence_idx"]).reset_index(drop=True)
    nxt = {(t, i) for t, i in zip(sents.trajectory_id, sents.sentence_idx)}
    sents = sents[[(t, i + 1) in nxt and (t, i) in fs and (t, i + 1) in fs
                   for t, i in zip(sents.trajectory_id, sents.sentence_idx)]].reset_index(drop=True)
    rows = []
    for t, i in zip(sents.trajectory_id, sents.sentence_idx):
        rec = s16.targets_for(fs, k, t, i, "all")
        rec.update({f"A_{n}": v for n, v in s16.targets_for(fs, k, t, i, "A").items()})
        rec.update({f"B_{n}": v for n, v in s16.targets_for(fs, k, t, i, "B").items()})
        rows.append(rec)
    tg = pd.DataFrame(rows)
    tg[:] = r10(tg.to_numpy())
    frame = pd.concat([sents, tg], axis=1)
    frame["trace_correct"] = frame.trajectory_id.map(lambda t: info[t]["is_correct"])
    return frame, info


def s16_ceilings(frame, traj, tkeys, tkey_correct) -> dict:
    sb = lambda x: 2 * x / (1 + x) if x > 0 else x  # noqa: E731
    root = lambda x: float(np.sqrt(max(x, 0)))  # noqa: E731
    ceil = {}
    for n in s16.TARGETS:
        a, b = frame[f"A_{n}"].to_numpy(), frame[f"B_{n}"].to_numpy()
        r = R.spearman(a, b)
        pt = s16.per_traj_spearman(a, b, traj, tkeys)
        rw, rwc, rwi = np.nanmean(pt), np.nanmean(pt[tkey_correct]), np.nanmean(pt[~tkey_correct])
        ceil[n] = {"split_half_spearman": r, "ceiling_corr": root(sb(r)),
                   "split_half_within": float(rw), "within_ceiling_corr": root(sb(rw)),
                   "within_ceiling_correct_traces": root(sb(rwc)),
                   "within_ceiling_incorrect_traces": root(sb(rwi))}
    return ceil


def stage_s16() -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    feats_all, S = load_states()
    l16 = [STATE_LAYERS.index(L) for L in s16.LAYERS]           # 24, 32, 40
    frames = {g: s16_frame(g) for g in GROUPINGS}
    out: dict = {"layer": 32, "own_answers": {g: frames[g][1] for g in GROUPINGS},
                 "baseline": "raw variant, ridge on PCA-128 (leading comps of PCA-512), pre"}
    s16.BOOT = BOOT
    for cond, drop in CONDS.items():
        fkeep = (feats_all.problem_id != drop).to_numpy()
        feats = feats_all[fkeep].reset_index(drop=True)
        states = np.asarray(S[:, l16])[fkeep]
        pos = {(t, i): j for j, (t, i) in enumerate(zip(feats.trajectory_id, feats.sentence_idx))}
        prob_all, traj_all = feats.problem_id.to_numpy(), feats.trajectory_id.to_numpy()
        res: dict = {}
        fr = {}
        for g in GROUPINGS:
            f0 = frames[g][0]
            fr[g] = f0[f0.problem_id != drop].reset_index(drop=True)
        frame = fr["canon"]
        assert (frame.sentence_id.to_numpy() == fr["raw"].sentence_id.to_numpy()).all()
        ip = np.array([pos[(t, i)] for t, i in zip(frame.trajectory_id, frame.sentence_idx)])
        iq = np.array([pos[(t, i + 1)] for t, i in zip(frame.trajectory_id, frame.sentence_idx)])
        groups, traj = frame.problem_id.to_numpy(), frame.trajectory_id.to_numpy()
        tkeys = np.unique(traj)
        info = frames["canon"][1]
        tkey_correct = np.array([info[t]["is_correct"] for t in tkeys])
        tkey_prob = np.array([int(t.split(":")[0]) for t in tkeys])
        ceils = {g: s16_ceilings(fr[g], traj, tkeys, tkey_correct) for g in GROUPINGS}
        res["rows"], res["problems"] = len(frame), int(len(set(groups)))
        res["ceilings"] = ceils
        # baseline rows (s16 sections B/C, raw variant, ridge_pca128, pre, layer 32)
        Yall = np.hstack([fr[g][s16.TARGETS].to_numpy(float) for g in GROUPINGS])
        P = np.full(Yall.shape, np.nan)
        X = states[:, s16.LAYERS.index(32)]
        for g_ in np.unique(groups):
            tr_s = prob_all != g_
            tr, te = groups != g_, groups == g_
            sc = StandardScaler().fit(X[tr_s])
            Z = sc.transform(X).astype(np.float32)
            p = PCA(s16.PCA_MAX, svd_solver="randomized", random_state=0).fit(Z[tr_s])
            F = p.transform(Z)[ip][:, :128]
            ys = StandardScaler().fit(Yall[tr])
            m = RidgeCV(alphas=s16.ALPHAS, alpha_per_target=True).fit(F[tr], ys.transform(Yall[tr]))
            P[te] = ys.inverse_transform(m.predict(F[te]))
            print(f"s16 {cond} baseline fold {g_} {time.time() - t0:.0f}s", flush=True)
        uprob = np.unique(groups)
        rng = np.random.default_rng(seed(f"s16:{cond}:boot"))
        picks = [rng.choice(uprob, size=len(uprob), replace=True) for _ in range(BOOT)]
        prow = {g: np.flatnonzero(groups == g) for g in uprob}
        ptraj = {g: np.flatnonzero(tkey_prob == g) for g in uprob}
        boot_rows = [np.concatenate([prow[g] for g in pk]) for pk in picks]
        boot_traj = [np.concatenate([ptraj[g] for g in pk]) for pk in picks]
        base = {}
        for gi, g in enumerate(GROUPINGS):
            base[g] = {}
            for j, n in enumerate(s16.TARGETS):
                col = gi * len(s16.TARGETS) + j
                y, s = Yall[:, col], r10(P[:, col])
                pt = s16.per_traj_spearman(y, s, traj, tkeys)
                ob = np.array([s16.fast_spearman(y[br], s[br]) for br in boot_rows])
                wb = np.array([np.nanmean(pt[bt]) for bt in boot_traj])
                base[g][n] = {"overall": R.spearman(y, s), "overall_ci": list(R.interval(ob)[:2]),
                              "within": float(np.nanmean(pt)),
                              "within_ci": list(R.interval(wb)[:2]),
                              "within_correct_traces": float(np.nanmean(pt[tkey_correct])),
                              "within_incorrect_traces": float(np.nanmean(pt[~tkey_correct])),
                              "ceiling": ceils[g][n]["ceiling_corr"],
                              "within_ceiling": ceils[g][n]["within_ceiling_corr"]}
        res["baseline_ridge_pca128_pre"] = base
        # section E (controls: position-only, PCA-128 pre, p_own partials, trace correctness)
        res["controls"] = {}
        for g in GROUPINGS:
            s16.R = seed_proxy("20260914:arv2:s16", f"s16:{g}:{cond}")
            try:
                res["controls"][g] = s16.section_e(
                    frame=fr[g], Y=fr[g][s16.TARGETS].to_numpy(float), states=states, ip=ip,
                    iq=iq, prob_all=prob_all, traj_all=traj_all, tkeys=tkeys,
                    tkey_correct=tkey_correct, tkey_prob=tkey_prob, ceil=ceils[g], layers=[32])
            finally:
                s16.R = R
            print(f"s16 {cond} {g} controls {time.time() - t0:.0f}s", flush=True)
        out[cond] = res
        del states
    out["runtime_s"] = time.time() - t0
    return out


# ============================================================================ s17


def s17_data(g: str, drop) -> dict:
    t = tables()
    units = keep_units(units_all()[0], drop)
    s = view(t["pool"][t["pool"].problem_id != drop], g)
    cats = sorted(s.cat.unique())
    code = {c: i for i, c in enumerate(cats)}
    s = s.assign(code=s.cat.map(code))
    forks = sorted(s.groupby(["trajectory_id", "fork_idx"]).groups)
    fpos = {f: j for j, f in enumerate(forks)}
    rng = np.random.default_rng(seed(f"s17:{g}:{drop}:order"))
    C = np.zeros((len(forks), 300), int)
    Y = np.zeros((len(forks), 300))
    REP = np.zeros((len(forks), 300), int)
    for (tid, fi), x in s.groupby(["trajectory_id", "fork_idx"]):
        assert len(x) == 300
        perm = rng.permutation(len(x))
        C[fpos[(tid, fi)]] = x.code.to_numpy()[perm]
        Y[fpos[(tid, fi)]] = x.correct.to_numpy(float)[perm]
        REP[fpos[(tid, fi)]] = x.replicate.to_numpy()[perm]
    ia = np.array([fpos[(tt, i)] for tt, _, i in units])
    ib = np.array([fpos[(tt, i + 1)] for tt, _, i in units])
    eff = effects(g, drop)
    lab = {(a, i): bool(v) for a, i, v in zip(eff.trajectory_id, eff.sentence_idx,
                                               eff.real_tv_bh10, strict=True)}
    y = np.array([lab[(tt, i)] for tt, _, i in units])
    idx = pd.read_parquet(INDEX, columns=["trajectory_id", "fork_idx", "status",
                                          "generated_token_count"])
    tok = idx[idx.status == "OK"].groupby(["trajectory_id", "fork_idx"]).generated_token_count.mean()
    tokf = np.array([tok.get(f, np.nan) for f in forks])
    tokf = np.where(np.isfinite(tokf), tokf, np.nanmean(tokf))
    return dict(units=units, forks=forks, k=len(cats), C=C, Y=Y, REP=REP, ia=ia, ib=ib,
                groups=np.array([tt for tt, _, _ in units]), y=y, cost_w=tokf / tokf.mean(),
                tokf=tokf)


def stage_s17(sim_reps: int, groupings) -> dict:
    t0 = time.time()
    # task 2 restricted (fixed before running) to uniform N=300 (the yield reference) and
    # the two-stage m=20, N=300 cells, which contain the recommended design
    # (m20 N300 top50 entropy_pair).
    s17.M_GRID, s17.N_DEEP, s17.N_UNIFORM = [20], [300], [300]
    s17.SIM_REPS = sim_reps
    out: dict = {"sim_reps": sim_reps, "configs": "uniform N300; two-stage m20 N300 x "
                 "{top50, top33} x {entropy_pair, entropy_i}; lambda 1/0.75/0.5"}
    for g in groupings:
        out[g] = {}
        for cond, drop in CONDS.items():
            d = s17_data(g, drop)
            s17.R = seed_proxy("20260914:arv2:s17", f"s17:{g}:{cond}")
            try:
                t1 = s17.task1(d)
                sim = s17.task2(d)
                t3 = s17.task3(d, sim)
            finally:
                s17.R = R
            tab = pd.DataFrame(sim["table"])
            rec3 = pd.DataFrame(t3["table"])

            def cell(df, design, cut=None, score=None):
                m = df.design == design
                if cut:
                    m &= (df.cutoff == cut) & (df.score == score)
                return df[m]

            rd = cell(rec3, "two_stage", "top50", "entropy_pair")
            u3 = cell(rec3, "uniform")
            out[g][cond] = {
                "units": len(d["units"]), "positives_tv_bh10": int(d["y"].sum()),
                "categories": d["k"],
                "entropy_ap": {
                    sc: {"ap_full_depth": t1["full_depth_permutation_reference"][sc]["ap"],
                         "p_ap": t1["full_depth_permutation_reference"][sc]["p_ap"],
                         "recall_top33": t1["full_depth_permutation_reference"][sc]["recall_top33"],
                         "ci95_trajectory_boot": t1["by_pilot_n"]["300"][sc]["ap"]
                         ["ci95_trajectory_boot"],
                         "pilot_n20_ap_mean": t1["by_pilot_n"]["20"][sc]["ap"]["mean"],
                         "pilot_n20_ap_ci95_trajectory_boot":
                             t1["by_pilot_n"]["20"][sc]["ap"]["ci95_trajectory_boot"]}
                    for sc in ("entropy_i", "entropy_pair", "floor_i", "accunc_i")},
                "base_rate": t1["base_rate"],
                "positives_by_entropy_i_tercile": t1["positives_by_entropy_i_tercile"],
                "heldout_replicate_check": t1["heldout_replicate_check"],
                "prevalence": t3["prevalence"],
                "recommended_design": rd[["lambda", "rel_power_vs_U300", "fdp", "units_point",
                                          "traces_point", "usd_point", "usd_wilson_lo",
                                          "usd_wilson_hi", "units_direct_sim",
                                          "usd_direct_sim",
                                          "expected_false_discoveries_at_target"]
                                         ].to_dict("records"),
                "recommended_design_sim": cell(tab, "two_stage", "top50", "entropy_pair")[
                    ["lambda", "E_TP", "E_FP", "mean_FDP", "generations", "cost_flat_usd"]
                ].to_dict("records"),
                "uniform_300": u3[["lambda", "units_point", "usd_point", "usd_wilson_lo",
                                   "usd_wilson_hi", "fdp"]].to_dict("records"),
                "uniform_300_sim": cell(tab, "uniform")[["lambda", "E_TP", "E_FP",
                                                         "mean_FDP"]].to_dict("records"),
                "all_cells_sizing": rec3[["lambda", "design", "cutoff", "score",
                                          "rel_power_vs_U300", "usd_point"]].to_dict("records"),
            }
            print(f"s17 {g} {cond} done {time.time() - t0:.0f}s", flush=True)
    return out


# ============================================================================ s18


def round_float_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = r10(df[c])
    return df


def stage_s18(reps: int) -> dict:
    t = tables()
    units0, _ = units_all()
    rows, forced = s18.load_build(s18.BUILD)
    cands = json.loads((s18.BUILD / "candidates.json").read_text())["candidates"]
    resp = s18.load_responses(rows, forced, s18.STORE)
    mem = pd.read_csv(RESULTS / "s08_memorisation.csv")
    nr_acc = dict(zip(mem.problem_id, mem.no_reasoning_accuracy, strict=True))
    cands_c, cmap, _ = step1.canon_candidates(cands)
    resp_c = resp.copy()
    resp_c["category"] = [cmap[tt].get(c, c) if isinstance(c, str) else c
                          for tt, c in zip(resp.trajectory_id, resp.category, strict=True)]
    keys = ["H1a_spearman_level", "H1b_within_level", "H1_primary", "S2_sampled_level",
            "S3_entropy", "S4_signed_change", "S5_tv_debiased", "S6_partial_fork_tv_given_floor",
            "S7_ap_real_effects", "H4_level_vs_position", "level_problem_constant",
            "H4_change_vs_baselines", "forks", "problems", "sentences", "real_effects",
            "ap_prevalence", "ap_abs_dp_forced", "sample_other_rate_mean",
            "sample_nonanswer_rate_mean", "R2_sampled_seed_agreement",
            "sampled_vs_scored_spearman", "H1_descriptive", "change_descriptive"]
    out: dict = {"oof_baselines": "this task's s14 rerun (fork_tv target of the same grouping "
                                  "and the same with/without condition)"}
    old_seed = s18.SEED_MATERIAL
    for g in GROUPINGS:
        targets = step1.build_targets(view(t["pool"], g))
        cd, rp = (cands_c, resp_c) if g == "canon" else (cands, resp)
        orig = s18.parse_forced_sample
        if g == "canon":
            s18.parse_forced_sample = step1.canon_forced
        out[g] = {}
        try:
            for cond, drop in CONDS.items():
                s18.SEED_MATERIAL = f"{SEEDBASE}:s18:{g}:{cond}"
                units = keep_units(units0, drop)
                eff = effects(g, drop)
                oof = pd.read_parquet(INTER / f"s14_oof_{g}_{cond}.parquet")
                tg = targets[(targets.label_set == "r12") & (targets.problem_id != drop)
                             ].drop(columns="label_set")
                fr = s18.fork_readouts(rp, cd, "primary", "main", "out").merge(
                    tg[["trajectory_id", "fork_idx", "acc_level", "entropy"]],
                    on=["trajectory_id", "fork_idx"])
                fr["problem_baseline"] = fr.problem_id.map(nr_acc)
                frr = round_float_cols(fr.drop(columns="dist")).assign(dist=fr.dist.values)
                sents = round_float_cols(s18.sentence_table(fr, tg, units, eff))
                sents["sentence_id"] = sents.trajectory_id + ":" + sents.sentence_idx.astype(str)
                sents = sents.merge(oof[["sentence_id"] + [b for b in s18.BASELINES if b in oof]],
                                    on="sentence_id", how="left")
                sampled = s18.sample_readouts(rp, cd, "out")
                sampled = sampled[sampled.problem_id != drop]
                m = s18.compute_metrics(frr, sents, sampled=sampled, baselines=oof, reps=reps)
                # gates and controls on the same subset
                rsub = rp[rp.problem_id != drop]
                sc = rsub[rsub.kind == "score"]
                span = float((~sc[sc.present].span_ok.astype(bool)).mean())
                comp = float(rsub[(rsub.kind == "score") & (rsub.suffix == "primary")
                                  & (rsub.store == "main")].present.mean())
                mass = float(fr.mass.median())
                gates = {"Q1_span_mismatch": bool(span <= s18.GATES["span_mismatch_max"]),
                         "Q2_candidate_mass": bool(mass >= s18.GATES["median_mass_min"]),
                         "Q3_completeness": bool(comp >= s18.GATES["completeness_min"])}
                rpl = s18.fork_readouts(rp, cd, "primary", "replay", "out")
                j = fr.merge(rpl, on=["trajectory_id", "fork_idx"], suffixes=("", "_replay"))
                r1 = {"forks": len(j)}
                if len(j) >= 3:
                    r1.update(spearman=sp(j.p_forced, j.p_forced_replay),
                              max_abs_diff=float(np.abs(j.p_forced - j.p_forced_replay).max()))
                fmt = {}
                for alt in ("alt1", "alt2"):
                    a = s18.fork_readouts(rp, cd, alt, "main", "out")
                    jj = fr.merge(a, on=["trajectory_id", "fork_idx"], suffixes=("", "_alt"))
                    fmt[alt] = {"forks": len(jj),
                                "spearman": sp(jj.p_forced, jj.p_forced_alt)
                                if len(jj) >= 3 else float("nan")}
                rec = {k: m.get(k) for k in keys if k in m}
                if "H1b_within_level" in rec:
                    rec["H1b_within_level"] = {k: v for k, v in rec["H1b_within_level"].items()}
                rec.update(span_mismatch_rate=span, completeness_score_primary_main=comp,
                           candidate_mass_median=mass, quality_gates=gates,
                           R1_scoring_determinism=r1, format_sensitivity=fmt,
                           conclusion=s18.classify(m, gates),
                           real_effect_units=eff.loc[eff.real_tv_bh10 | eff.real_dacc_bh10,
                                                     ["trajectory_id", "sentence_idx"]]
                           .astype(str).agg(":".join, axis=1).tolist())
                out[g][cond] = rec
                print(f"s18 {g} {cond} done", flush=True)
        finally:
            s18.parse_forced_sample = orig
            s18.SEED_MATERIAL = old_seed
    return out


# ============================================================================ assemble


def dec3(x, est="estimate"):
    if not isinstance(x, dict):
        return None
    return {"est": x.get(est), "lo": x.get("lo"), "hi": x.get("hi"), "decision": x.get("decision")}


def headline_table(parts: dict) -> list[dict]:
    rows = []

    def add(block, metric, g, w, wo, dw=None, dwo=None, note=None):
        rec = {"block": block, "metric": metric, "grouping": g, "with": w, "without": wo,
               "diff": (wo - w) if isinstance(w, (int, float)) and isinstance(wo, (int, float))
               and w is not None and wo is not None else None,
               "decision_with": dw, "decision_without": dwo,
               "decision_changes": (dw != dwo) if dw is not None else None}
        if note:
            rec["note"] = note
        rows.append(rec)

    if "s12" in parts:
        for g in GROUPINGS:
            P = parts["s12"][g]
            for pair in ("1_vs_2", "1_vs_0", "2_vs_0"):
                for s in s12.SCORES:
                    a, b = P["with"][pair][s], P["without"][pair][s]
                    for st in ("spearman", "icc", "kappa"):
                        if a.get(st) is None:
                            continue
                        add("s12", f"{pair} {s} {st} [lo,hi] with={a[st+'_lo']:.3f},"
                            f"{a[st+'_hi']:.3f} without={b[st+'_lo']:.3f},{b[st+'_hi']:.3f}",
                            g, a[st], b[st], a.get(f"{st}_decision"), b.get(f"{st}_decision"))
    if "s13" in parts:
        for g in GROUPINGS:
            P = parts["s13"][g]
            for s in ("fork_tv", "tv_debiased", "acc_level", "signed_dacc", "abs_dacc",
                      "smooth_dacc"):
                for st in ("spearman", "icc"):
                    a, b = P["with"]["replicates_1v2_n100"][s], P["without"]["replicates_1v2_n100"][s]
                    add("s13", f"n100 r1v2 {s} {st} ci with={a[st+'_ci']} without={b[st+'_ci']}",
                        g, a[st], b[st])
                a, b = P["with"]["replicates_1v2_n100"][s], P["without"]["replicates_1v2_n100"][s]
                add("s13", f"n100 r1v2 {s} within_spearman", g, a["within_spearman"],
                    b["within_spearman"])
                add("s13", f"n100 {s} between_problem_share_of_signal", g,
                    a["between_problem_share_of_signal"], b["between_problem_share_of_signal"])
            for st in ("floor_r1_vs_floor_r2_spearman", "fork_tv_r1_vs_floor_r2_spearman",
                       "fork_tv_r1_vs_fork_tv_r2_spearman", "share_fork_tv_below_1p5x_floor_r1",
                       "mean_floor_r1", "mean_fork_tv_r1"):
                add("s13", f"floor {st}", g, P["with"]["noise_floor"][st],
                    P["without"]["noise_floor"][st])
            for st in ("tv_bh10", "dacc_bh10", "either"):
                add("s13", f"real effects {st} (BH over {P['with']['real_effects']['tests']} / "
                    f"{P['without']['real_effects']['tests']} tests)", g,
                    P["with"]["real_effects"][st], P["without"]["real_effects"][st])
            for s in ("fork_tv", "tv_debiased", "acc_level"):
                for m in ("icc", "spearman", "within_spearman"):
                    a, b = P["with"]["projection"][s][m], P["without"]["projection"][s][m]
                    add("s13", f"curve {s} {m} projected at n300", g, a["at_n300"], b["at_n300"])
                    add("s13", f"curve {s} {m} n_needed", g, a["n_needed"], b["n_needed"])
                add("s13", f"curve {s} icc measured at n150", g,
                    P["with"]["curve"][s]["150"]["icc"], P["without"]["curve"][s]["150"]["icc"])
    if "s14" in parts:
        P = parts["s14"]
        for g in GROUPINGS:
            for n in s14.TARGETS:
                add("s14", f"ceiling {n} ceiling_corr", g, P["with"]["ceilings"][g][n]
                    ["ceiling_corr"], P["without"]["ceilings"][g][n]["ceiling_corr"])
                add("s14", f"ceiling {n} within_ceiling_corr", g, P["with"]["ceilings"][g][n]
                    ["within_ceiling_corr"], P["without"]["ceilings"][g][n]["within_ceiling_corr"])
            bw = pd.DataFrame(P["with"]["best_layer_by_rep_target"])
            bo = pd.DataFrame(P["without"]["best_layer_by_rep_target"])
            for rep in ("position", "pre", "post", "pre+post"):
                for n in s14.TARGETS:
                    a = bw[(bw.grouping == g) & (bw.rep == rep) & (bw.target == n)].iloc[0]
                    b = bo[(bo.grouping == g) & (bo.rep == rep) & (bo.target == n)].iloc[0]
                    lay = f" L{int(a.layer)}->L{int(b.layer)}" if rep != "position" else ""
                    add("s14", f"best {rep}{lay} {n} spearman", g, float(a.spearman),
                        float(b.spearman))
                    add("s14", f"best {rep}{lay} {n} within", g, float(a.within), float(b.within))
            a, b = P["with"]["pre_fork_tv_decomposition"][g], P["without"]["pre_fork_tv_decomposition"][g]
            add("s14", f"pre fork_tv probe partial|floor (L{a['layer']}->L{b['layer']}) ci "
                f"with={a['partial_given_floor_ci']} without={b['partial_given_floor_ci']}", g,
                a["partial_given_floor"], b["partial_given_floor"])
            add("s14", "pre fork_tv probe corr with floor", g, a["corr_pred_with"]["floor"],
                b["corr_pred_with"]["floor"])
            add("s14", "pre fork_tv probe corr with fork_tv", g, a["corr_pred_with"]["fork_tv"],
                b["corr_pred_with"]["fork_tv"])
            add("s14", "pre fork_tv probe partial|floor,entropy,position", g,
                a["partial_given_floor_entropy_position"], b["partial_given_floor_entropy_position"])
            add("s14", "floor alone vs fork_tv", g, a["floor_alone_vs_fork_tv"],
                b["floor_alone_vs_fork_tv"])
    if "s16" in parts:
        P = parts["s16"]
        for g in GROUPINGS:
            for n in ("acc_level", "signed_dacc", "tv_debiased", "entropy", "p_own"):
                a, b = P["with"]["baseline_ridge_pca128_pre"][g][n], \
                    P["without"]["baseline_ridge_pca128_pre"][g][n]
                add("s16", f"L32 ridge PCA-128 pre {n} within ci with={a['within_ci']} "
                    f"without={b['within_ci']}", g, a["within"], b["within"])
                add("s16", f"L32 ridge PCA-128 pre {n} overall", g, a["overall"], b["overall"])
                ca, cb = P["with"]["controls"][g]["position_only"][n], \
                    P["without"]["controls"][g]["position_only"][n]
                add("s16", f"position-only {n} within", g, ca["within"], cb["within"])
            for nm in ("given_position", "given_position_entropy"):
                a = P["with"]["controls"][g]["by_layer"]["32"]["p_own_partial_within"][nm]
                b = P["without"]["controls"][g]["by_layer"]["32"]["p_own_partial_within"][nm]
                add("s16", f"p_own partial {nm} (L32) ci with={a['ci']} without={b['ci']}", g,
                    a["activation_probe"], b["activation_probe"])
            a = P["with"]["controls"][g]["trace_correctness_decoding"]["32"]
            b = P["without"]["controls"][g]["trace_correctness_decoding"]["32"]
            add("s16", "trace-correctness row AUC (L32)", g, a["row_auc_overall"],
                b["row_auc_overall"])
    if "s17" in parts:
        P = parts["s17"]
        for g in [x for x in GROUPINGS if x in P]:
            a, b = P[g]["with"], P[g]["without"]
            add("s17", "positives (tv BH10)", g, a["positives_tv_bh10"], b["positives_tv_bh10"])
            for sc in ("entropy_i", "entropy_pair"):
                add("s17", f"AP {sc} full depth (traj-boot ci with="
                    f"{a['entropy_ap'][sc]['ci95_trajectory_boot']} without="
                    f"{b['entropy_ap'][sc]['ci95_trajectory_boot']})", g,
                    a["entropy_ap"][sc]["ap_full_depth"], b["entropy_ap"][sc]["ap_full_depth"])
                add("s17", f"AP {sc} permutation p", g, a["entropy_ap"][sc]["p_ap"],
                    b["entropy_ap"][sc]["p_ap"])
                add("s17", f"AP {sc} pilot n20 mean", g, a["entropy_ap"][sc]["pilot_n20_ap_mean"],
                    b["entropy_ap"][sc]["pilot_n20_ap_mean"])
                add("s17", f"AP {sc} base rate (random AP)", g, a["base_rate"], b["base_rate"])
            add("s17", "prevalence p_hat", g, a["prevalence"]["p_hat"], b["prevalence"]["p_hat"])
            for ra, rb in zip(a["recommended_design"], b["recommended_design"], strict=True):
                lam = ra["lambda"]
                add("s17", f"recommended m20 N300 top50 entropy_pair lambda={lam} usd_point "
                    f"(wilson with=[{ra['usd_wilson_lo']:.0f},{ra['usd_wilson_hi']:.0f}] without="
                    f"[{rb['usd_wilson_lo']:.0f},{rb['usd_wilson_hi']:.0f}])", g, ra["usd_point"],
                    rb["usd_point"])
                add("s17", f"recommended lambda={lam} rel_power_vs_U300", g,
                    ra["rel_power_vs_U300"], rb["rel_power_vs_U300"])
                add("s17", f"recommended lambda={lam} fdp", g, ra["fdp"], rb["fdp"])
            for ra, rb in zip(a["uniform_300"], b["uniform_300"], strict=True):
                add("s17", f"uniform N300 lambda={ra['lambda']} usd_point", g, ra["usd_point"],
                    rb["usd_point"])
    if "s18" in parts:
        P = parts["s18"]
        for g in GROUPINGS:
            a, b = P[g]["with"], P[g]["without"]
            for k in ("H1a_spearman_level", "H1b_within_level", "S2_sampled_level", "S3_entropy",
                      "S4_signed_change", "S5_tv_debiased", "S6_partial_fork_tv_given_floor",
                      "S7_ap_real_effects", "H4_level_vs_position"):
                x, y = dec3(a.get(k)), dec3(b.get(k))
                if x is None or y is None:
                    continue
                add("s18", f"{k} ci with=[{x['lo']:.3f},{x['hi']:.3f}] without=[{y['lo']:.3f},"
                    f"{y['hi']:.3f}]", g, x["est"], y["est"], x["decision"], y["decision"])
            add("s18", "H1_primary", g, None, None, a["H1_primary"], b["H1_primary"])
            add("s18", "conclusion", g, None, None, a["conclusion"]["conclusion"],
                b["conclusion"]["conclusion"])
            add("s18", "candidate_mass_median", g, a["candidate_mass_median"],
                b["candidate_mass_median"])
            add("s18", "sample_other_rate_mean", g, a.get("sample_other_rate_mean"),
                b.get("sample_other_rate_mean"))
            add("s18", "real effects used for S7", g, a.get("real_effects"), b.get("real_effects"))
            for nm in sorted(set(a.get("H4_change_vs_baselines", {}))
                             & set(b.get("H4_change_vs_baselines", {}))):
                x = dec3(a["H4_change_vs_baselines"][nm]["ap_diff_forced_minus_baseline"])
                y = dec3(b["H4_change_vs_baselines"][nm]["ap_diff_forced_minus_baseline"])
                add("s18", f"H4 AP diff forced - {nm}", g, x["est"], y["est"], x["decision"],
                    y["decision"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["s12", "s13", "s14", "s16", "s17", "s18"])
    ap.add_argument("--reps", type=int, default=BOOT)
    ap.add_argument("--curve-draws", type=int, default=s13.DRAWS)
    ap.add_argument("--sim-reps", type=int, default=300)
    ap.add_argument("--s17-groupings", nargs="*", default=["canon"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    INTER.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not args.assemble_only:
        for st in args.stages:
            print(f"=== {st}", flush=True)
            if st == "s12":
                part = stage_s12()
            elif st == "s13":
                part = stage_s13(args.curve_draws)
            elif st == "s14":
                part = stage_s14(args.device)
            elif st == "s16":
                part = stage_s16()
            elif st == "s17":
                part = stage_s17(args.sim_reps, args.s17_groupings)
            elif st == "s18":
                part = stage_s18(args.reps)
            else:
                raise SystemExit(st)
            write_json(INTER / f"part_{st}.json", part)
            print(f"=== {st} written {time.time() - t0:.0f}s", flush=True)
    parts = {st: json.loads((INTER / f"part_{st}.json").read_text())
             for st in ["s12", "s13", "s14", "s16", "s17", "s18"]
             if (INTER / f"part_{st}.json").exists()}
    res = {"status": "EXPLORATORY, POST HOC", "task": "T5 (item 12): without problem 1591",
           "dropped_problem": DROP, "groupings": list(GROUPINGS), "primary_grouping": "canon",
           "bootstrap_reps": BOOT, "seed_material": SEEDBASE, "parts": parts,
           "headline_table": headline_table(parts)}
    write_json(OUT / "results.json", res)
    print(f"results written {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
