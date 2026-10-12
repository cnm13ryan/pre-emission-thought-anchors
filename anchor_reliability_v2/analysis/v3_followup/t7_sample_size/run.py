"""v3 follow-up T7 (item 10): Phase 3 sample size. EXPLORATORY, POST HOC.

How many problems does Phase 3 need for a 95% problem-cluster bootstrap interval of width
hi - lo <= 0.30 on each primary contrast?

Contrast list (fixed here before any result was computed; see CONTRASTS below):
  (a) plan_v3 preregistered statistics (s18): H1a, H1b, S2..S7, H4_level_vs_position and
      the H4 AP differences (forced minus baseline) for every baseline s18 reports.
  (b) Phase 2 reliability of fork_tv, replicate 1 vs 2 (s12): Spearman, ICC(2,1), kappa(q90).
  (c) likely GPU-plan primaries: level-readout validity (proxy S2), within-trace change
      correlation (proxy S4), split-half reliability of tv_debiased (Phase 2 r1 vs r2).
Grouping: canonical (arv2.canon) is primary. Raw-grouping widths are read from the
committed results files for the analytic method only.

Methods (both reported):
  1. Analytic: w_10 = width of the pilot's 10-problem cluster bootstrap, recomputed by
     importing s18.compute_metrics / s12.agreement (their own seeds), and also from this
     script's own N=10 bootstrap; N = 10 * (w_10 / 0.30)^2.
  2. Bootstrap extrapolation: draw N problems with replacement from the 10 at
     N in NGRID, recompute the statistic, take the 2.5-97.5% width; report the smallest
     grid N with width <= 0.30 and a log-log interpolation between bracketing grid points
     (beyond 200: 1/sqrt(N) from the N=200 width).
Variance decomposition at N=10: problem-only bootstrap (V_c), within-problem-only row
bootstrap with the 10 problems fixed (V_w), two-stage (V_2). within share = V_w / V_c.
Model Var = s2_b/N + s2_w/(N m); N needed at m units per problem from M_GRID.
Sensitivity: everything again without problem 1591.

Usage: python analysis/v3_followup/t7_sample_size/run.py [--reps 2000]
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
CACHE = DATA / "derived/v3_followup"
TASK = "t7_sample_size"
TARGET_WIDTH = 0.30
NGRID = (10, 20, 30, 50, 80, 120, 200)
M_GRID = (5, 10, 20, 40, 80)          # units (forks / sentences) per problem
WITHIN_DOMINATED = 0.5                # within share of V_c at or above this -> flagged
DROP = 1591


def seed(name: str) -> int:
    return R.seed_from(f"20260914:arv2:v3f:{TASK}:{name}")


def load_step1():
    spec = importlib.util.spec_from_file_location(
        "step1_canon_run", STUDY / "analysis/v3_followup/step1_canon/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = load_step1()
s12, s13, s18 = S1.s12, S1.s13, S1.s18


# ----------------------------------------------------------------------------- fast stats


def rnd(x) -> np.ndarray:
    return np.round(np.asarray(x, float), 10)


def spear(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def ap(y, s) -> float:
    """sklearn average_precision_score (step-wise, ties grouped) without its overhead."""
    y = np.asarray(y, bool)
    npos = y.sum()
    if npos == 0 or npos == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def within_median(x, y, cid) -> float:
    vals = []
    for c in np.unique(cid):
        m = cid == c
        vals.append(spear(x[m], y[m]))
    vals = np.asarray(vals)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")


# ----------------------------------------------------------------------------- data


def pilot_frames(drop=None):
    """Canonical-grouping pilot inputs exactly as step1 pilot_block builds them."""
    trajs = pd.read_parquet(S1.INPUTS / "trajectories.parquet")
    gt_canon = {t: S1.canonical_answer(g) for t, g in
                zip(trajs.trajectory_id, trajs.ground_truth_answer, strict=True)}
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    window = {(t["trajectory_id"], f) for t in plan["trajectories"] for f in t["forks"]}
    rel = pd.read_parquet(CACHE / "released_canon.parquet")
    p2 = pd.read_parquet(CACHE / "phase2_canon.parquet")
    rel_win = rel[[(t, int(f)) in window for t, f in zip(rel.trajectory_id, rel.fork_idx,
                                                       strict=True)]]
    cols = ["replicate", "trajectory_id", "fork_idx", "cat_raw", "cat_canon",
            "correct_raw", "correct_canon"]
    pool = pd.concat([p2[cols], rel_win[cols]], ignore_index=True)
    targets = S1.build_targets(S1.view(pool, "canon"))
    rows, forced = s18.load_build(s18.BUILD)
    cands = json.loads((s18.BUILD / "candidates.json").read_text())["candidates"]
    resp = s18.load_responses(rows, forced, s18.STORE)
    oof = pd.read_parquet(RESULTS / "s14_oof_predictions.parquet")
    cands_c, cmap, _ = S1.canon_candidates(cands)
    resp_c = resp.copy()
    resp_c["category"] = [cmap[t].get(c, c) if isinstance(c, str) else c
                          for t, c in zip(resp.trajectory_id, resp.category, strict=True)]
    eff = pd.read_csv(CACHE / "sentence_effects_canon.csv")
    orig = s18.parse_forced_sample
    s18.parse_forced_sample = S1.canon_forced
    try:
        tg = targets[(targets.label_set == "r12") & (targets.problem_id != drop)
                     ].drop(columns="label_set")
        fr = s18.fork_readouts(resp_c, cands_c, "primary", "main", "out").merge(
            tg[["trajectory_id", "fork_idx", "acc_level", "entropy"]],
            on=["trajectory_id", "fork_idx"])
        sents = s18.sentence_table(fr, tg, units, eff)
        sents["sentence_id"] = sents.trajectory_id + ":" + sents.sentence_idx.astype(str)
        sents = sents.merge(oof[["sentence_id"] + [b for b in s18.BASELINES if b in oof]],
                            on="sentence_id", how="left")
        sampled = s18.sample_readouts(resp_c, cands_c, "out")
    finally:
        s18.parse_forced_sample = orig
    return dict(fr=fr, sents=sents, sampled=sampled, oof=oof, units=units, p2=p2,
                rel_win=rel_win, pool=pool, gt_canon=gt_canon)


def blocks_from(F) -> dict:
    """Row-level arrays per data block; scores rounded to 10 decimals."""
    out = {}
    f = F["fr"].dropna(subset=["p_forced", "acc_level"]).reset_index(drop=True)
    out["forks"] = {"g": f.problem_id.to_numpy(), "p": rnd(f.p_forced), "acc": rnd(f.acc_level),
                    "pos": rnd(f.fork_idx), "hf": rnd(f.entropy_forced),
                    "hp": rnd(f.entropy)}
    sp = F["sampled"].merge(f[["trajectory_id", "fork_idx", "acc_level", "p_forced"]],
                            on=["trajectory_id", "fork_idx"])
    s1 = sp[sp["pass"] == s18.SAMPLE_PASSES[0]].reset_index(drop=True)
    out["sampled"] = {"g": s1.problem_id.to_numpy(), "ps": rnd(s1.p_sampled),
                      "a1": rnd(s1.acc_level)}
    s = F["sents"].dropna(subset=["dp_forced"]).reset_index(drop=True)
    sb = {"g": s.problem_id.to_numpy(), "dp": rnd(s.dp_forced), "sd": rnd(s.signed_dacc),
          "ftv": rnd(s.forced_tv), "tvd": rnd(s.tv_debiased), "fork_tv": rnd(s.pooled_fork_tv),
          "floor": rnd(s.floor), "y": s.real_effect.to_numpy(bool),
          "b:position_sentence_idx": rnd(s.sentence_idx)}
    for b in s18.BASELINES:
        if b in s:
            v = s[b].to_numpy(float)
            if np.isfinite(v).all():
                sb[f"b:{b}"] = rnd(np.abs(v) if b.startswith("valueprobe") else v)
    out["sents"] = sb
    # Phase 2 fork_tv r1 vs r2 (s12 label_table on the canonical view)
    reps = {r: s12.label_table(S1.view(F["p2"][F["p2"].replicate == r], "canon"), F["units"])
            for r in (1, 2)}
    m = reps[1].merge(reps[2], on=["trajectory_id", "problem_id", "sentence_idx"],
                      suffixes=("_1", "_2"))
    m = m[m.fork_tv_1.notna() & m.fork_tv_2.notna()].reset_index(drop=True)
    q90 = json.loads((RESULTS / "s06_power.json").read_text())["q90_thresholds"]["fork_tv"]
    out["p2_fork_tv"] = {"g": m.problem_id.to_numpy(), "a": rnd(m.fork_tv_1),
                         "b": rnd(m.fork_tv_2), "q90": q90}
    # Phase 2 tv_debiased r1 vs r2 (s13 unit_scores, global canonical codes as step1 s13_block)
    pool = S1.view(F["pool"], "canon")
    codes = {c: i for i, c in enumerate(sorted(pool.cat.unique()))}
    df = pool.assign(code=pool.cat.map(codes))
    halves = {r: {key: (g.code.to_numpy(), g.correct.to_numpy(float))
                  for key, g in df[df.replicate == r].groupby(["trajectory_id", "fork_idx"])}
              for r in (1, 2)}
    sc = {r: s13.unit_scores(halves[r], F["units"], len(codes)) for r in (1, 2)}
    gu = np.array([p for _, p, _ in F["units"]])
    ok = np.isfinite(sc[1]["tv_debiased"]) & np.isfinite(sc[2]["tv_debiased"])
    out["p2_tvd"] = {"g": gu[ok], "a": rnd(sc[1]["tv_debiased"][ok]),
                     "b": rnd(sc[2]["tv_debiased"][ok])}
    return out


def drop_problem(blocks: dict, prob: int) -> dict:
    out = {}
    for k, d in blocks.items():
        keep = d["g"] != prob
        out[k] = {c: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
                  for c, v in d.items()}
    return out


# ----------------------------------------------------------------------------- contrasts
# (family, name, block, threshold, fn(D, rows, cid)). Fixed before any result was seen.


def _ap_diff(base):
    return lambda D, r, c: ap(D["y"][r], D["ftv"][r]) - ap(D["y"][r], D[base][r])


CONTRASTS = [
    ("a", "H1a_spearman_level", "forks", 0.60, lambda D, r, c: spear(D["p"][r], D["acc"][r])),
    ("a", "H1b_within_level", "forks", 0.30,
     lambda D, r, c: within_median(D["p"][r], D["acc"][r], c)),
    ("a", "S2_sampled_level", "sampled", 0.60, lambda D, r, c: spear(D["ps"][r], D["a1"][r])),
    ("a", "S3_entropy", "forks", 0.50, lambda D, r, c: spear(D["hf"][r], D["hp"][r])),
    ("a", "S4_signed_change", "sents", 0.30, lambda D, r, c: spear(D["dp"][r], D["sd"][r])),
    ("a", "S5_tv_debiased", "sents", 0.20, lambda D, r, c: spear(D["ftv"][r], D["tvd"][r])),
    ("a", "S6_partial_fork_tv_given_floor", "sents", 0.10,
     lambda D, r, c: s18.partial_spearman(D["ftv"][r], D["fork_tv"][r], D["floor"][r])),
    ("a", "S7_ap_real_effects", "sents", 0.25, lambda D, r, c: ap(D["y"][r], D["ftv"][r])),
    ("a", "H4_level_vs_position", "forks", 0.0,
     lambda D, r, c: spear(D["p"][r], D["acc"][r]) - abs(spear(D["pos"][r], D["acc"][r]))),
] + [("a", f"H4ap:{b}", "sents", 0.0, _ap_diff(f"b:{b}"))
     for b in ["position_sentence_idx"] + list(s18.BASELINES)] + [
    ("b", "phase2_fork_tv_spearman", "p2_fork_tv", 0.80,
     lambda D, r, c: spear(D["a"][r], D["b"][r])),
    ("b", "phase2_fork_tv_icc", "p2_fork_tv", 0.75,
     lambda D, r, c: R.icc_2_1(D["a"][r], D["b"][r])),
    ("b", "phase2_fork_tv_kappa_q90", "p2_fork_tv", 0.60,
     lambda D, r, c: R.cohen_kappa(D["a"][r] >= D["q90"], D["b"][r] >= D["q90"])),
    ("c", "phase2_tv_debiased_spearman", "p2_tvd", None,
     lambda D, r, c: spear(D["a"][r], D["b"][r])),
]
ALIASES = {"c:level_readout_validity": "S2_sampled_level",
           "c:within_trace_change_correlation": "S4_signed_change",
           "c:split_half_tv_debiased": "phase2_tv_debiased_spearman"}


# ----------------------------------------------------------------------------- resampling


def draws(D, fn, n_prob, reps, sd, mode="cluster"):
    """mode: cluster (problems with replacement, rows intact), within (the observed problems
    fixed, rows resampled within each), twostage (both)."""
    rng = np.random.default_rng(sd)
    uniq = np.unique(D["g"])
    idx = [np.flatnonzero(D["g"] == u) for u in uniq]
    out = np.empty(reps)
    for k in range(reps):
        pick = (rng.integers(0, len(uniq), n_prob) if mode != "within"
                else np.arange(len(uniq)))
        parts, cid = [], []
        for j, p in enumerate(pick):
            rows = idx[p]
            if mode != "cluster":
                rows = rows[rng.integers(0, len(rows), len(rows))]
            parts.append(rows)
            cid.append(np.full(len(rows), j))
        out[k] = fn(D, np.concatenate(parts), np.concatenate(cid))
    return out


def width(x):
    lo, hi, bad = R.interval(x)
    return {"lo": lo, "hi": hi, "width": hi - lo, "nonfinite": bad}


def n_from_grid(grid: dict) -> dict:
    ns = sorted(grid)
    ws = [grid[n]["width"] for n in ns]
    first = next((n for n, w in zip(ns, ws) if w <= TARGET_WIDTH), None)
    if first is None:
        interp = ns[-1] * (ws[-1] / TARGET_WIDTH) ** 2
        how = "beyond grid: 1/sqrt(N) from N=200"
    elif first == ns[0]:
        interp = ns[0] * (ws[0] / TARGET_WIDTH) ** 2
        how = "at/below N=10: 1/sqrt(N) from N=10"
    else:
        i = ns.index(first)
        n0, n1, w0, w1 = ns[i - 1], ns[i], ws[i - 1], ws[i]
        t = (np.log(TARGET_WIDTH) - np.log(w0)) / (np.log(w1) - np.log(w0))
        interp = float(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0))))
        how = "log-log interpolation"
    return {"first_grid_N": first, "N_interp": float(interp), "how": how}


def analyse(blocks, tag, reps):
    res = {}
    for fam, name, blk, thr, fn in CONTRASTS:
        D = blocks[blk]
        if blk == "sents" and name.startswith("H4ap:") and f"b:{name[5:]}" not in D:
            res[name] = {"family": fam, "skipped": "baseline not finite"}
            continue
        t0 = time.time()
        n_obs = len(np.unique(D["g"]))
        est = fn(D, np.arange(len(D["g"])), D["g"])
        grid = {n: width(draws(D, fn, n, reps, seed(f"{tag}:{name}:N{n}"))) for n in NGRID}
        if n_obs != 10:
            grid[n_obs] = width(draws(D, fn, n_obs, reps, seed(f"{tag}:{name}:N{n_obs}")))
        w_obs = grid[n_obs]["width"]
        v_c = np.nanvar(draws(D, fn, n_obs, reps, seed(f"{tag}:{name}:vc")))
        v_w = np.nanvar(draws(D, fn, n_obs, reps, seed(f"{tag}:{name}:vw"), "within"))
        v_2 = np.nanvar(draws(D, fn, n_obs, reps, seed(f"{tag}:{name}:v2"), "twostage"))
        m0 = len(D["g"]) / n_obs
        s2w = v_w * n_obs * m0
        s2b = max(n_obs * (v_c - v_w), 0.0)
        n_an = n_obs * (w_obs / TARGET_WIDTH) ** 2
        tradeoff = {}
        for m in M_GRID:
            scale = (s2b + s2w / m) / (s2b + s2w / m0) if (s2b + s2w / m0) > 0 else np.nan
            nm = n_an * scale
            tradeoff[m] = {"N_problems": float(nm), "total_units": float(nm * m)}
        res[name] = {
            "family": fam, "threshold": thr, "estimate": est, "problems_observed": n_obs,
            "rows": int(len(D["g"])), "units_per_problem": m0,
            "w_obs_own_bootstrap": w_obs, "N_analytic": float(n_an),
            "bootstrap_grid": {str(k): v for k, v in sorted(grid.items())},
            "N_bootstrap": n_from_grid({k: v for k, v in grid.items() if k in NGRID}),
            "variance": {"V_cluster": float(v_c), "V_within_only": float(v_w),
                         "V_two_stage": float(v_2),
                         "within_share": float(v_w / v_c) if v_c > 0 else None,
                         "sigma2_between": float(s2b), "sigma2_within_per_unit": float(s2w),
                         "within_dominated": bool(v_c > 0 and v_w / v_c >= WITHIN_DOMINATED)},
            "tradeoff_units_per_problem": {str(k): v for k, v in tradeoff.items()},
        }
        print(f"[{tag}] {name}: est={est:.3f} w={w_obs:.3f} N_an={n_an:.0f} "
              f"N_bs={res[name]['N_bootstrap']['N_interp']:.0f} "
              f"within={res[name]['variance']['within_share']} {time.time() - t0:.0f}s",
              flush=True)
    return res


# ----------------------------------------------------------------------------- import check


def imported_widths(F, F_no, reps):
    """Pilot and Phase 2 widths recomputed through s18.compute_metrics / s12.agreement
    (their own seeds), canonical grouping, as step1 pilot_block / s12_block."""
    out = {}
    for tag, G in (("all", F), ("without_1591", F_no)):
        m = s18.compute_metrics(G["fr"], G["sents"], sampled=G["sampled"], baselines=G["oof"],
                                reps=reps)
        rec = {}
        for k in ("H1a_spearman_level", "H1b_within_level", "S2_sampled_level", "S3_entropy",
                  "S4_signed_change", "S5_tv_debiased", "S6_partial_fork_tv_given_floor",
                  "S7_ap_real_effects", "H4_level_vs_position"):
            if k in m:
                rec[k] = {x: m[k][x] for x in ("estimate", "lo", "hi", "nonfinite")}
        for b, v in m.get("H4_change_vs_baselines", {}).items():
            d = v["ap_diff_forced_minus_baseline"]
            rec[f"H4ap:{b}"] = {x: d[x] for x in ("estimate", "lo", "hi", "nonfinite")}
        out[tag] = rec
    q90 = json.loads((RESULTS / "s06_power.json").read_text())["q90_thresholds"]
    r1 = s12.label_table(S1.view(F["p2"][F["p2"].replicate == 1], "canon"), F["units"])
    r2 = s12.label_table(S1.view(F["p2"][F["p2"].replicate == 2], "canon"), F["units"])
    for tag, (a, b, sm) in {"all": (r1, r2, "20260914:arv2:bootstrap"),
                            "without_1591": (r1[r1.problem_id != DROP], r2[r2.problem_id != DROP],
                                             "20260914:arv2:bootstrap:no1591")}.items():
        ag = next(x for x in s12.agreement(a, b, q90, sm) if x["score"] == "fork_tv")
        for k, nm in (("spearman", "phase2_fork_tv_spearman"), ("icc", "phase2_fork_tv_icc"),
                      ("kappa", "phase2_fork_tv_kappa_q90")):
            out[tag][nm] = {"estimate": ag[k], "lo": ag[f"{k}_lo"], "hi": ag[f"{k}_hi"],
                            "nonfinite": ag[f"{k}_nonfinite"]}
    for tag in out:
        for k, v in out[tag].items():
            v["width"] = v["hi"] - v["lo"]
            n0 = 10 if tag == "all" else 9
            v["N_analytic"] = n0 * (v["width"] / TARGET_WIDTH) ** 2
    return out


def committed_raw_widths():
    pc = json.loads((RESULTS / "plan_v3_pilot.json").read_text())
    p2 = json.loads((RESULTS / "phase2_reliability.json").read_text())
    out = {}
    for tag, key, n0 in (("all", "primary_candidates_out_of_window", 10),
                         ("without_1591", "sensitivity_without_1591", 9)):
        m = pc[key]
        rec = {}
        for k in ("H1a_spearman_level", "H1b_within_level", "S2_sampled_level", "S3_entropy",
                  "S4_signed_change", "S5_tv_debiased", "S6_partial_fork_tv_given_floor",
                  "S7_ap_real_effects", "H4_level_vs_position"):
            rec[k] = m[k]["hi"] - m[k]["lo"]
        for b, v in m["H4_change_vs_baselines"].items():
            d = v["ap_diff_forced_minus_baseline"]
            rec[f"H4ap:{b}"] = d["hi"] - d["lo"]
        src = p2["replicate_1_vs_2"] if tag == "all" else p2["sensitivity_without_1591"]
        ft = next(x for x in src if x["score"] == "fork_tv")
        for k, nm in (("spearman", "phase2_fork_tv_spearman"), ("icc", "phase2_fork_tv_icc"),
                      ("kappa", "phase2_fork_tv_kappa_q90")):
            rec[nm] = ft[f"{k}_hi"] - ft[f"{k}_lo"]
        out[tag] = {k: {"width": w, "N_analytic": n0 * (w / TARGET_WIDTH) ** 2}
                    for k, w in rec.items()}
    return out


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--reps", type=int, default=2000)
    args = ap_.parse_args()
    t0 = time.time()
    F = pilot_frames()
    F_no = pilot_frames(drop=DROP)
    B = blocks_from(F)
    B_no = drop_problem(B, DROP)
    # fast-stat checks against the library implementations on the observed data
    D = B["sents"]
    checks = {"spearman_vs_R": abs(spear(D["dp"], D["sd"]) - R.spearman(D["dp"], D["sd"])),
              "ap_vs_sklearn": abs(ap(D["y"], D["ftv"]) - s18.average_precision(D["y"], D["ftv"])),
              "ap_vs_sklearn_position": abs(ap(D["y"], D["b:position_sentence_idx"])
                                            - s18.average_precision(
                                                D["y"], D["b:position_sentence_idx"]))}
    assert all(v < 1e-9 for v in checks.values()), checks
    print("checks ok", checks, f"{time.time() - t0:.0f}s", flush=True)
    imp = imported_widths(F, F_no, args.reps)
    print(f"imported widths done {time.time() - t0:.0f}s", flush=True)
    res = {"stage": "exploratory_post_hoc", "task": "t7_sample_size (item 10)",
           "status": "EXPLORATORY, POST HOC", "grouping": "canonical (arv2.canon)",
           "target_width": TARGET_WIDTH, "ngrid": NGRID, "m_grid": M_GRID,
           "within_dominated_threshold": WITHIN_DOMINATED, "reps": args.reps,
           "contrasts": [{"family": f, "name": n, "block": b, "threshold": t}
                         for f, n, b, t, _ in CONTRASTS],
           "aliases_family_c": ALIASES, "fast_stat_checks": checks,
           "imported_widths_canon": imp, "committed_widths_raw": committed_raw_widths()}
    res["all"] = analyse(B, "all", args.reps)
    res["without_1591"] = analyse(B_no, "no1591", args.reps)

    def bind(r, key):
        ok = {k: v for k, v in r.items() if "skipped" not in v}
        return max(ok, key=lambda k: ok[k][key] if key == "N_analytic"
                   else ok[k]["N_bootstrap"]["N_interp"])
    res["binding"] = {
        tag: {"analytic": bind(res[tag], "N_analytic"),
              "bootstrap": bind(res[tag], "N_bootstrap")} for tag in ("all", "without_1591")}
    res["runtime_s"] = time.time() - t0
    write_json(OUT / "results.json", res)
    print(json.dumps(res["binding"], indent=1))


if __name__ == "__main__":
    main()
