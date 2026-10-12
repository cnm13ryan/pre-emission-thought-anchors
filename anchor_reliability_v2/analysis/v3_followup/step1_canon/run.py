"""v3 follow-up Step 1 (item 6): raw vs canonical answer grouping. EXPLORATORY, POST HOC.

Builds the shared sample tables that every Step 2 task reads (written under
DATA/derived/v3_followup/, not to git), then recomputes every TV-based number of
Phase 2 (s12), the Phase 2 diagnostics (s13) and the plan_v3 pilot (s18) under
  raw   = the grouping those scripts used (answers.category_v2 / normalize_v2), and
  canon = arv2.canon.canonical_answer on the raw answer surface.
The pilot's own files are not touched; this script imports the scoring functions and
feeds them re-grouped inputs. The raw pass must reproduce the committed numbers.

Usage: python analysis/v3_followup/step1_canon/run.py [--reps 2000] [--perms 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import answers as A  # noqa: E402
from arv2 import reliability as R  # noqa: E402
from arv2.canon import canonical_answer, category_canonical, raw_surface  # noqa: E402
from arv2.paths import DATA, DERIVED, INPUTS, RESULTS, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent
CACHE = DATA / "derived/v3_followup"
PHASE2_INPUTS = Path.home() / "preanchor-root/data/processed/redesign_phase2/inputs.parquet"
INDEX = (DATA / "generations/phase2_mirror/20261009T0717Z-phase2-23f5e3e6"
         / "completions_index.parquet")
GROUPINGS = ("raw", "canon")
REBUILD = False


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s12, s13, s18 = load("s12_phase2_score"), load("s13_phase2_diagnostics"), load("s18_forced_answer")


# ----------------------------------------------------------------------------- tables


def released_table(gt_canon: dict) -> pd.DataFrame:
    """All released MVP resamples (replicate 0) with raw surface and both groupings."""
    path = CACHE / "released_canon.parquet"
    if path.exists() and not REBUILD:
        return pd.read_parquet(path)
    cols = ["trajectory_id", "problem_id", "fork_idx", "sample_idx", "v2_status", "v2_answer",
            "boxes", "chunk_resampled", "chunk_removed", "rollout_tokens"]
    parts = [pd.read_parquet(f, columns=cols)
             for f in sorted((DERIVED / "rollouts").glob("*/fork_*.parquet"))]
    r = pd.concat(parts, ignore_index=True)
    raws, src = [], []
    for st, ans, bx in zip(r.v2_status, r.v2_answer, r.boxes, strict=True):
        if st not in ("answered", "unboxed_stated") or not ans:
            raws.append(None), src.append("none")
            continue
        hit = None
        if st == "answered" and bx is not None:
            for b in reversed(list(bx)):
                if str(b).strip() and A.normalize_v2(str(b)) == ans:
                    hit = str(b)
                    break
        raws.append(hit if hit is not None else ans)
        src.append("box" if hit is not None else "normalised_fallback")
    r["raw_surface"], r["raw_source"] = raws, src
    r["cat_raw"] = [a if s in ("answered", "unboxed_stated") and a else
                    A.TRUNCATED if s == "length_truncated" else A.NON_ANSWER
                    for s, a in zip(r.v2_status, r.v2_answer, strict=True)]
    r["cat_canon"] = [category_canonical(s, x) if s in ("answered", "unboxed_stated") and x
                      else c for s, x, c in zip(r.v2_status, r.raw_surface, r.cat_raw,
                                                strict=True)]
    rel = pd.read_parquet(DERIVED / "rollouts_v2.parquet",
                          columns=["trajectory_id", "fork_idx", "sample_idx", "cat",
                                   "correct_v2", "similarity"])
    r = r.merge(rel, on=["trajectory_id", "fork_idx", "sample_idx"], how="left",
                validate="one_to_one")
    r["correct_raw"] = r.correct_v2.astype(bool)
    r["correct_canon"] = [c == gt_canon[t] for c, t in zip(r.cat_canon, r.trajectory_id,
                                                          strict=True)]
    r["replicate"] = 0
    r = r.drop(columns=["boxes", "v2_answer"])
    CACHE.mkdir(parents=True, exist_ok=True)
    r.to_parquet(path)
    return r


def phase2_table(gt: dict, gt_canon: dict, device: str) -> pd.DataFrame:
    """Phase 2 replicates 1 and 2 exactly as s12 prepares them, plus the canonical columns."""
    path = CACHE / "phase2_canon.parquet"
    if path.exists() and not REBUILD:
        return pd.read_parquet(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                device=device)
    idx = pd.read_parquet(INDEX)
    ok, _ = s12.prepare_replicates(idx, pd.read_parquet(PHASE2_INPUTS), gt, model)
    rs = [raw_surface(t, finish_reason=f, token_count=c) for t, f, c in
          zip(ok.raw_answer, ok.finish_reason, ok.generated_token_count, strict=True)]
    ok["raw_surface"] = [x for _, x in rs]
    assert [s for s, _ in rs] == ok.v2_status.tolist()
    ok["cat_raw"], ok["correct_raw"] = ok.cat, ok.correct
    ok["cat_canon"] = [category_canonical(s, x) for s, x in rs]
    ok["correct_canon"] = [c == gt_canon[t] for c, t in zip(ok.cat_canon, ok.trajectory_id,
                                                           strict=True)]
    keep = ["replicate", "trajectory_id", "fork_idx", "sample_idx", "v2_status", "finish_reason",
            "generated_token_count", "first_replacement_sentence", "raw_surface", "similarity",
            "cat_raw", "correct_raw", "cat_canon", "correct_canon"]
    out = ok[keep].reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def view(df: pd.DataFrame, g: str) -> pd.DataFrame:
    return df.assign(cat=df[f"cat_{g}"], correct=df[f"correct_{g}"].astype(bool))


# ----------------------------------------------------------------------------- s12 / s13


def s12_block(p2: pd.DataFrame, rel_win: pd.DataFrame, units, q90, g: str) -> dict:
    reps = {r: s12.label_table(view(p2[p2.replicate == r], g), units) for r in (1, 2)}
    reps[0] = s12.label_table(view(rel_win, g), units)
    keep = ("score", "spearman", "spearman_lo", "spearman_hi", "spearman_decision", "icc",
            "icc_lo", "icc_hi", "icc_decision", "kappa", "kappa_lo", "kappa_hi",
            "kappa_decision", "flag_rate_1", "flag_rate_2", "within_problem_spearman_median",
            "rows", "continuous_form", "flag_form")

    def slim(recs):
        return {x["score"]: {k: x.get(k) for k in keep} for x in recs}

    return {
        "replicate_1_vs_2": slim(s12.agreement(reps[1], reps[2], q90, "20260914:arv2:bootstrap")),
        "without_1591": slim(s12.agreement(reps[1][reps[1].problem_id != 1591],
                                           reps[2][reps[2].problem_id != 1591], q90,
                                           "20260914:arv2:bootstrap:no1591")),
        "vs_released": {f"{r}_vs_0": slim(s12.agreement(reps[r], reps[0], q90,
                                                        f"20260914:arv2:bootstrap:{r}v0"))
                        for r in (1, 2)},
    }, reps


def bh(p, q=0.10):
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.nonzero(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def s13_block(pool_df: pd.DataFrame, units, perms: int) -> tuple[dict, pd.DataFrame]:
    """Noise floor, replicate decomposition and real effects (s13 procedure and seeds)."""
    codes = {c: i for i, c in enumerate(sorted(pool_df.cat.unique()))}
    k = len(codes)
    df = pool_df.assign(code=pool_df.cat.map(codes))
    groups = np.array([p for _, p, _ in units])
    halves = {r: {key: (g.code.to_numpy(), g.correct.to_numpy(float))
                  for key, g in df[df.replicate == r].groupby(["trajectory_id", "fork_idx"])}
              for r in (0, 1, 2)}
    sc = {r: s13.unit_scores(halves[r], units, k) for r in (0, 1, 2)}
    floor = {}
    for r in (0, 1, 2):
        f = np.full(len(units), np.nan)
        for j, (tid, _, i) in enumerate(units):
            if (tid, i) in halves[r] and (tid, i + 1) in halves[r]:
                f[j] = s13.tv_null_analytic(halves[r][(tid, i)][0], halves[r][(tid, i + 1)][0], k)
        floor[r] = f
    out: dict = {"categories": k}
    decomp = {}
    for name in ("fork_tv", "tv_debiased", "acc_level"):
        a, b = sc[1][name], sc[2][name]
        decomp[name] = s13.rel_stats(a, b, groups)
        decomp[name]["interval_spearman"] = R.interval(R.cluster_bootstrap(
            groups, lambda r, a=a, b=b: R.spearman(a[r], b[r]), 2000,
            R.seed_from(f"20260914:arv2:v3f:step1:{name}")))[:2]
    out["replicates_1v2_n100"] = decomp
    f1, f2, t1 = floor[1], floor[2], sc[1]["fork_tv"]
    out["noise_floor"] = {
        "mean_floor_r1": float(np.nanmean(f1)), "mean_floor_r2": float(np.nanmean(f2)),
        "mean_fork_tv_r1": float(np.nanmean(t1)),
        "floor_r1_vs_floor_r2_spearman": R.spearman(f1, f2),
        "fork_tv_r1_vs_floor_r2_spearman": R.spearman(t1, f2),
        "fork_tv_r1_vs_fork_tv_r2_spearman": R.spearman(t1, sc[2]["fork_tv"]),
        "share_fork_tv_below_1p5x_floor_r1": float(np.nanmean(t1 < 1.5 * f1)),
        "interval_floor_r1_vs_r2": R.interval(R.cluster_bootstrap(
            groups, lambda r: R.spearman(f1[r], f2[r]), 2000,
            R.seed_from("20260914:arv2:v3f:step1:floor")))[:2],
    }
    # real effects: s13 permutation procedure and seed, ~300 vs ~300 per sentence
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s13:perm"))
    pool = {key: (g.code.to_numpy(), g.correct.to_numpy(float))
            for key, g in df.groupby(["trajectory_id", "fork_idx"])}
    rows = []
    for tid, prob, i in units:
        (ca, ya), (cb, yb) = pool[(tid, i)], pool[(tid, i + 1)]
        obs_tv, obs_d = s13.tv(ca, cb, k), abs(yb.mean() - ya.mean())
        cc, yy, na = np.concatenate([ca, cb]), np.concatenate([ya, yb]), len(ca)
        ge_tv = ge_d = 0
        for _ in range(perms):
            p = rng.permutation(len(cc))
            ge_tv += s13.tv(cc[p[:na]], cc[p[na:]], k) >= obs_tv
            ge_d += abs(yy[p[na:]].mean() - yy[p[:na]].mean()) >= obs_d
        rows.append({"trajectory_id": tid, "problem_id": prob, "sentence_idx": i,
                     "pooled_fork_tv": obs_tv, "pooled_floor": s13.tv_null_analytic(ca, cb, k),
                     "pooled_abs_dacc": obs_d, "acc_before": ya.mean(), "acc_after": yb.mean(),
                     "p_tv": (ge_tv + 1) / (perms + 1), "p_dacc": (ge_d + 1) / (perms + 1)})
    eff = pd.DataFrame(rows)
    eff["real_tv_bh10"] = bh(eff.p_tv.to_numpy())
    eff["real_dacc_bh10"] = bh(eff.p_dacc.to_numpy())
    out["real_effects"] = {
        "tv_bh10": int(eff.real_tv_bh10.sum()), "dacc_bh10": int(eff.real_dacc_bh10.sum()),
        "either": int((eff.real_tv_bh10 | eff.real_dacc_bh10).sum()),
        "flagged": [f"{t}:{s}" for t, s, a, b in zip(eff.trajectory_id, eff.sentence_idx,
                                                      eff.real_tv_bh10, eff.real_dacc_bh10,
                                                      strict=True) if a or b],
        "pooled_floor_mean": float(eff.pooled_floor.mean()),
        "pooled_tv_debiased_mean": float((eff.pooled_fork_tv - eff.pooled_floor).mean())}
    return out, eff


# ----------------------------------------------------------------------------- pilot


def canon_forced(text):
    if not isinstance(text, str) or not text.lstrip(" ").startswith("{"):
        return A.NON_ANSWER
    boxes, _ = A.balanced_boxes("\\boxed" + text.lstrip(" "))
    if not boxes or not boxes[0].strip():
        return A.NON_ANSWER
    return canonical_answer(boxes[0])


def canon_candidates(cands: dict) -> tuple[dict, dict, dict]:
    """Merge candidate categories with equal canonical keys. Returns (cands, map, stats)."""
    new, cmap, stats = {}, {}, {}
    for tid, c in cands.items():
        gt = canonical_answer(c["gt_raw"])
        m = {}
        for which in ("out", "in"):
            for x in c[which]:
                m[x["category"]] = canonical_answer(x["surfaces"][0]) if x["surfaces"] \
                    else canonical_answer(x["category"])
        cmap[tid] = m
        rec = {"gt_category": gt, "gt_raw": c["gt_raw"]}
        for which in ("out", "in"):
            merged: dict = {}
            for x in c[which]:
                key = m[x["category"]]
                y = merged.setdefault(key, {"category": key, "share": 0.0, "is_gt": key == gt,
                                            "surfaces": []})
                y["share"] += x["share"]
                y["surfaces"] += [s for s in x["surfaces"] if s not in y["surfaces"]]
            if gt not in merged:
                merged[gt] = {"category": gt, "share": 0.0, "is_gt": True,
                              "surfaces": [c["gt_raw"]]}
            rec[which] = list(merged.values())
            stats[f"{tid}|{which}"] = {"before": len(c[which]), "after": len(merged)}
        new[tid] = rec
    return new, cmap, stats


def pilot_block(targets, cands, resp, effects, units, reps, oof, nr_acc) -> dict:
    def analyse(which, labels, drop=None):
        tg = targets[(targets.label_set == labels) & (targets.problem_id != drop)
                     ].drop(columns="label_set")
        fr = s18.fork_readouts(resp, cands, "primary", "main", which).merge(
            tg[["trajectory_id", "fork_idx", "acc_level", "entropy"]],
            on=["trajectory_id", "fork_idx"])
        fr["problem_baseline"] = fr.problem_id.map(nr_acc)
        sents = s18.sentence_table(fr, tg, units, effects)
        sents["sentence_id"] = sents.trajectory_id + ":" + sents.sentence_idx.astype(str)
        sents = sents.merge(oof[["sentence_id"] + [b for b in s18.BASELINES if b in oof]],
                            on="sentence_id", how="left")
        sampled = s18.sample_readouts(resp, cands, which)
        m = s18.compute_metrics(fr, sents, sampled=sampled, baselines=oof, reps=reps)
        m["candidate_mass_median"] = float(fr.mass.median())
        return m

    return {"primary": analyse("out", "r12"), "in_window": analyse("in", "r12"),
            "r012": analyse("out", "r012"), "without_1591": analyse("out", "r12", 1591)}


def build_targets(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label_set, reps in s18.LABEL_SETS.items():
        sub = win[win.replicate.isin(reps)]
        for (tid, fork), g in sub.groupby(["trajectory_id", "fork_idx"]):
            counts = g.cat.value_counts().to_dict()
            rows.append({"label_set": label_set, "trajectory_id": tid,
                         "problem_id": int(tid.split(":")[0]), "fork_idx": int(fork),
                         "n": len(g), "acc_level": float(g.correct.mean()),
                         "entropy": s18.entropy_bits(np.array(list(counts.values())) / len(g)),
                         "counts_json": json.dumps(counts, sort_keys=True)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main


def pick(m: dict, key: str) -> dict:
    x = m.get(key, {})
    return {k: x.get(k) for k in ("estimate", "lo", "hi", "decision")} if isinstance(x, dict) else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rebuild", action="store_true", help="recompute the cached tables")
    args = ap.parse_args()
    global REBUILD
    REBUILD = args.rebuild
    t0 = time.time()
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt_raw = dict(zip(trajs.trajectory_id, trajs.ground_truth_answer, strict=True))
    gt = {t: A.normalize_v2(g) for t, g in gt_raw.items()}
    gt_canon = {t: canonical_answer(g) for t, g in gt_raw.items()}
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    window = {(t["trajectory_id"], f) for t in plan["trajectories"] for f in t["forks"]}
    q90 = json.loads((RESULTS / "s06_power.json").read_text())["q90_thresholds"]

    rel = released_table(gt_canon)
    p2 = phase2_table(gt, gt_canon, args.device)
    rel_win = rel[[(t, int(f)) in window for t, f in zip(rel.trajectory_id, rel.fork_idx,
                                                       strict=True)]]
    print(f"tables ready {time.time() - t0:.0f}s", flush=True)

    res: dict = {"stage": "exploratory_post_hoc", "task": "step1_canon (item 6)",
                 "module": "src/arv2/canon.py", "ground_truth_keys": gt_canon}
    # category counts
    pool_cols = ["replicate", "trajectory_id", "fork_idx", "cat_raw", "cat_canon",
                 "correct_raw", "correct_canon"]
    pool = pd.concat([p2[pool_cols], rel_win[pool_cols]], ignore_index=True)
    merges = (pool.groupby("cat_canon").cat_raw.unique().map(list))
    merges = {k: sorted(v) for k, v in merges.items() if len(v) > 1}
    res["category_counts"] = {
        "phase2_pool_global": {g: int(pool[f"cat_{g}"].nunique()) for g in GROUPINGS},
        "phase2_pool_per_trajectory": {
            t: {g: int(x[f"cat_{g}"].nunique()) for g in GROUPINGS}
            for t, x in pool.groupby("trajectory_id")},
        "phase2_pool_merges": merges,
        "released_mvp_global": {g: int(rel[f"cat_{g}"].nunique()) for g in GROUPINGS},
        "released_mvp_per_trajectory_sum": {
            g: int(rel.groupby("trajectory_id")[f"cat_{g}"].nunique().sum()) for g in GROUPINGS},
        "samples_changing_correctness": {
            "phase2_pool": int((pool.correct_raw != pool.correct_canon).sum()),
            "released_mvp": int((rel.correct_raw != rel.correct_canon).sum())},
        "released_raw_surface_source": rel.raw_source.value_counts().to_dict(),
        "released_cat_raw_equals_rollouts_v2_cat": float((rel.cat_raw == rel.cat).mean()),
    }
    rm = rel.groupby("cat_canon").cat_raw.unique().map(list)
    res["category_counts"]["released_mvp_merges_top"] = dict(sorted(
        ((k, sorted(v)[:8]) for k, v in rm.items() if len(v) > 1),
        key=lambda kv: -len(kv[1]))[:40])
    print(json.dumps(res["category_counts"]["phase2_pool_global"]), flush=True)

    # pilot inputs (shared)
    rows, forced = s18.load_build(s18.BUILD)
    cands = json.loads((s18.BUILD / "candidates.json").read_text())["candidates"]
    resp = s18.load_responses(rows, forced, s18.STORE)
    oof = pd.read_parquet(RESULTS / "s14_oof_predictions.parquet")
    mem = pd.read_csv(RESULTS / "s08_memorisation.csv")
    nr_acc = dict(zip(mem.problem_id, mem.no_reasoning_accuracy, strict=True))
    cands_c, cmap, cstats = canon_candidates(cands)
    resp_c = resp.copy()
    resp_c["category"] = [cmap[t].get(c, c) if isinstance(c, str) else c
                          for t, c in zip(resp.trajectory_id, resp.category, strict=True)]
    res["pilot_candidate_counts"] = cstats

    blocks = {}
    for g in GROUPINGS:
        print(f"[{g}] s12 ...", flush=True)
        s12r, _ = s12_block(p2, rel_win, units, q90, g)
        print(f"[{g}] s13 ... {time.time() - t0:.0f}s", flush=True)
        # s13 pools all OK replicate 1/2 rows and released r0 for the window forks
        s13r, eff = s13_block(view(pool, g), units, args.perms)
        eff.to_csv(CACHE / f"sentence_effects_{g}.csv", index=False)
        print(f"[{g}] pilot ... {time.time() - t0:.0f}s", flush=True)
        win = view(pool, g)
        targets = build_targets(win)
        if g == "raw":
            ref = pd.read_parquet(s18.BUILD / "targets.parquet")
            j = ref.merge(targets, on=["label_set", "trajectory_id", "fork_idx"])
            res["raw_targets_match_build"] = bool(
                len(j) == len(ref) and (j.counts_json_x == j.counts_json_y).all()
                and np.allclose(j.acc_level_x, j.acc_level_y))
            orig = s18.parse_forced_sample
            pr = pilot_block(targets, cands, resp, eff, units, args.reps, oof, nr_acc)
        else:
            orig = s18.parse_forced_sample
            s18.parse_forced_sample = canon_forced
            try:
                pr = pilot_block(targets, cands_c, resp_c, eff, units, args.reps, oof, nr_acc)
            finally:
                s18.parse_forced_sample = orig
        blocks[g] = {"phase2_s12": s12r, "diagnostics_s13": s13r, "pilot_s18": pr}
        print(f"[{g}] done {time.time() - t0:.0f}s", flush=True)
    res["by_grouping"] = blocks

    # compact comparison of the headline numbers
    keys = ["H1a_spearman_level", "H1b_within_level", "S2_sampled_level", "S3_entropy",
            "S4_signed_change", "S5_tv_debiased", "S6_partial_fork_tv_given_floor",
            "S7_ap_real_effects"]
    cmp = {}
    for g in GROUPINGS:
        b = blocks[g]
        ft = b["phase2_s12"]["replicate_1_vs_2"]["fork_tv"]
        tvs = b["phase2_s12"]["replicate_1_vs_2"]["tv"]
        cmp[g] = {
            "categories_phase2_pool": res["category_counts"]["phase2_pool_global"][g],
            "phase2_fork_tv_spearman": [ft["spearman"], ft["spearman_lo"], ft["spearman_hi"],
                                        ft["spearman_decision"]],
            "phase2_fork_tv_icc": [ft["icc"], ft["icc_lo"], ft["icc_hi"], ft["icc_decision"]],
            "phase2_fork_tv_kappa": [ft["kappa"], ft["kappa_lo"], ft["kappa_hi"],
                                     ft["kappa_decision"]],
            "phase2_tv_spearman": [tvs["spearman"], tvs["spearman_lo"], tvs["spearman_hi"]],
            "fork_tv_vs_released_r1": b["phase2_s12"]["vs_released"]["1_vs_0"]["fork_tv"]["icc"],
            "floor": b["diagnostics_s13"]["noise_floor"],
            "tv_debiased_spearman_r1v2": b["diagnostics_s13"]["replicates_1v2_n100"]
            ["tv_debiased"]["spearman"],
            "real_effects": b["diagnostics_s13"]["real_effects"],
            "pilot_primary": {k: pick(b["pilot_s18"]["primary"], k) for k in keys},
            "pilot_candidate_mass_median": b["pilot_s18"]["primary"]["candidate_mass_median"],
            "pilot_other_rate": b["pilot_s18"]["primary"].get("sample_other_rate_mean"),
            "pilot_without_1591": {k: pick(b["pilot_s18"]["without_1591"], k)
                                   for k in ("H1a_spearman_level", "H1b_within_level")},
            "pilot_in_window_H1a": pick(b["pilot_s18"]["in_window"], "H1a_spearman_level"),
        }
    res["comparison"] = cmp
    # reproduction check of the raw pass against the committed results
    p2c = json.loads((RESULTS / "phase2_reliability.json").read_text())
    pc = json.loads((RESULTS / "plan_v3_pilot.json").read_text())
    s13c = json.loads((RESULTS / "s13_phase2_diagnostics.json").read_text())
    c0 = next(x for x in p2c["replicate_1_vs_2"] if x["score"] == "fork_tv")
    res["raw_reproduces_committed"] = {
        "phase2_fork_tv_spearman": [c0["spearman"], cmp["raw"]["phase2_fork_tv_spearman"][0]],
        "phase2_fork_tv_icc_ci": [[c0["icc_lo"], c0["icc_hi"]],
                                  cmp["raw"]["phase2_fork_tv_icc"][1:3]],
        "s13_categories": [s13c["categories"], blocks["raw"]["diagnostics_s13"]["categories"]],
        "s13_real_effects_tv": [s13c["real_effects"]["tv_bh10"],
                                cmp["raw"]["real_effects"]["tv_bh10"]],
        "pilot_S5": [pc["primary_candidates_out_of_window"]["S5_tv_debiased"]["estimate"],
                     cmp["raw"]["pilot_primary"]["S5_tv_debiased"]["estimate"]],
        "pilot_H1a": [pc["primary_candidates_out_of_window"]["H1a_spearman_level"]["estimate"],
                      cmp["raw"]["pilot_primary"]["H1a_spearman_level"]["estimate"]],
        "pilot_S7": [pc["primary_candidates_out_of_window"]["S7_ap_real_effects"]["estimate"],
                     cmp["raw"]["pilot_primary"]["S7_ap_real_effects"]["estimate"]],
    }
    res["runtime_s"] = time.time() - t0
    write_json(OUT / "results.json", res)
    print(json.dumps({"comparison": cmp, "repro": res["raw_reproduces_committed"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
