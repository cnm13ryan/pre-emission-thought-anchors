"""v3 follow-up T3 (item 7): does window selection explain the low Phase 2 anchor rate?
EXPLORATORY, POST HOC.

Everything below (anchor definitions, criteria list, grids, seeds) is fixed in this file
before any rate is computed.

1. Anchor definitions on the released MVP resamples (replicate 0, 100 samples per fork, every
   sentence i whose forks i and i+1 both exist; 6,434 sentences):
   primary    permutation test of fork i vs fork i+1 with TV on canonical categories.
              A label permutation of 100 vs 100 pooled samples gives group-A category counts
              that are exactly multivariate-hypergeometric, so permutations are drawn as
              hypergeometric count vectors (vectorised, identical in law to shuffling labels).
              The integer statistic D = sum_k |cA_k - cB_k| = 200 * TV avoids float ties.
              Sequential Monte Carlo (Besag-Clifford): batches of 20,000, stop once
              >= H=100 exceedances (p = exceedances / draws), otherwise continue up to
              1,000,000 draws (p = (exceedances + 1) / (draws + 1)). BH at 10% over ALL.
   secondary  the same comparison on |delta accuracy| (canonical correctness). With one
              binary variable the permutation null is hypergeometric, so the p-value is
              computed EXACTLY (no Monte Carlo). BH at 10% over all.
   reference  MVP flag: mvp_importance >= 1.3549810893213545 (q90), sentence_scores.parquet.
   sensitivity primary test on raw categories (cat_raw).
2. Anchor rate under each fixed selection criterion with a problem-cluster bootstrap 95% CI
   (2,000 replicates), enrichment vs all sentences (paired bootstrap), coverage and recall.
3. The Phase 2 rule (research/redesign/phase2_select.py) re-run under 1,000 alternative seeds.
4. Cross-check of the 190 Phase 2 units against the 300-vs-300 real effects.

Usage: python analysis/v3_followup/t3_window_selection/run.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

STUDY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STUDY / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DATA, DERIVED, write_json  # noqa: E402

TASK = "t3_window_selection"
OUT = Path(__file__).resolve().parent
SHARED = DATA / "derived/v3_followup"
CACHE = SHARED / TASK
LABELS = Path.home() / "preanchor-root/data/processed/stage4/labels.parquet"
TEST = {6998, 2236, 2870, 3916}
N_PROBLEMS, WINDOW = 10, 20
MVP_Q90 = 1.3549810893213545
BH_Q = 0.10
BATCH, H_STOP, MAX_DRAWS = 20_000, 100, 1_000_000
REPS = 2000
N_RULE_SEEDS = 1000
PILOT_N = 20                       # s17 cheap estimate: first 20 samples of fork i
DEFS = ("primary_tv_canon", "secondary_dacc_canon", "reference_mvp_q90", "sens_tv_raw")


def seed(name: str) -> int:
    return R.seed_from(f"20260914:arv2:v3f:{TASK}:{name}")


def bh(p: np.ndarray, q: float = BH_Q) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.nonzero(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------------- tests


def perm_tv_p(ca: np.ndarray, cb: np.ndarray, rng) -> tuple[float, int, int]:
    """Sequential MC permutation p for D = sum|cA-cB| (equal group sizes)."""
    pooled = ca + cb
    pooled = pooled[pooled > 0]
    na = int(ca.sum())
    assert na == int(cb.sum())
    obs = int(np.abs(ca - cb).sum())
    if len(pooled) <= 1 or obs == 0:
        return 1.0, 0, 0
    ge = draws = 0
    while draws < MAX_DRAWS:
        x = rng.multivariate_hypergeometric(pooled, na, size=BATCH)
        d = np.abs(2 * x - pooled).sum(1)
        ge += int((d >= obs).sum())
        draws += BATCH
        if ge >= H_STOP:
            return ge / draws, ge, draws
    return (ge + 1) / (draws + 1), ge, draws


def exact_dacc_p(ya: int, yb: int, n: int) -> float:
    """Exact permutation p for |acc_A - acc_B| with n vs n samples."""
    c = ya + yb
    obs = abs(ya - yb)
    x = np.arange(max(0, c - n), min(c, n) + 1)
    stat = np.abs(2 * x - c)
    pmf = hypergeom.pmf(x, 2 * n, c, n)
    return float(min(1.0, pmf[stat >= obs].sum()))


# ----------------------------------------------------------------------------- data


def build_sentences(rel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.groupby("trajectory_id").agg(
        problem_id=("problem_id", "first"), trace_kind=("trace_kind", "first"),
        n=("sentence_idx", "size"), conv=("convergence_idx", "first"),
        final=("final_answer_idx", "first"))
    rows = []
    for (tid, fork), g in rel.groupby(["trajectory_id", "fork_idx"], sort=True):
        g = g.sort_values("sample_idx")
        assert len(g) == 100
        rows.append((tid, fork, g.cat_canon.to_numpy(), g.cat_raw.to_numpy(),
                     g.correct_canon.to_numpy(bool)))
    forks = {(t, f): (cc, cr, y) for t, f, cc, cr, y in rows}
    out = []
    for (tid, i), (cc, cr, y) in forks.items():
        if (tid, i + 1) not in forks:
            continue
        cc2, cr2, y2 = forks[(tid, i + 1)]
        L = lab.loc[tid]
        cats = sorted(set(cc) | set(cc2))
        code = {c: k for k, c in enumerate(cats)}
        ca = np.bincount([code[c] for c in cc], minlength=len(cats))
        cb = np.bincount([code[c] for c in cc2], minlength=len(cats))
        rcats = sorted(set(cr) | set(cr2))
        rcode = {c: k for k, c in enumerate(rcats)}
        ra = np.bincount([rcode[c] for c in cr], minlength=len(rcats))
        rb = np.bincount([rcode[c] for c in cr2], minlength=len(rcats))
        c20 = pd.Series(cc[:PILOT_N]).value_counts().to_numpy()
        limit = min(x for x in (L.n - 1, L.conv, L.final) if pd.notna(x))
        out.append({"trajectory_id": tid, "problem_id": int(L.problem_id),
                    "trace_kind": L.trace_kind, "sentence_idx": int(i), "n": int(L.n),
                    "conv": L.conv, "final": L.final, "limit": float(limit),
                    "ca": ca, "cb": cb, "ra": ra, "rb": rb,
                    "ya": int(y.sum()), "yb": int(y2.sum()),
                    "fork_tv": 0.5 * np.abs(ca - cb).sum() / 100,
                    "entropy_i": entropy(ca), "entropy20_i": entropy(c20),
                    "rel_pos": i / (L.n - 1)})
    return pd.DataFrame(out).sort_values(["trajectory_id", "sentence_idx"]).reset_index(drop=True)


def run_tests(s: pd.DataFrame) -> pd.DataFrame:
    path = CACHE / "sentence_tests.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(s):
            return cached
    res = []
    t0 = time.time()
    for j, r in enumerate(s.itertuples()):
        rng = np.random.default_rng(seed(f"perm:canon:{r.trajectory_id}:{r.sentence_idx}"))
        p, ge, dr = perm_tv_p(r.ca, r.cb, rng)
        rng = np.random.default_rng(seed(f"perm:raw:{r.trajectory_id}:{r.sentence_idx}"))
        praw, _, drraw = perm_tv_p(r.ra, r.rb, rng)
        res.append({"trajectory_id": r.trajectory_id, "sentence_idx": r.sentence_idx,
                    "p_tv_canon": p, "exceed_canon": ge, "draws_canon": dr,
                    "p_tv_raw": praw, "draws_raw": drraw,
                    "p_dacc_canon": exact_dacc_p(r.ya, r.yb, 100)})
        if j % 500 == 0:
            print(f"  tests {j}/{len(s)} {time.time() - t0:.0f}s", flush=True)
    t = pd.DataFrame(res)
    CACHE.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path, index=False)
    return t


# ----------------------------------------------------------------------------- rule


def phase2_rule(labels: pd.DataFrame, rng_seed: int, info: dict) -> list[dict]:
    """Exact re-implementation of phase2_select.py's draw for a given generator seed."""
    rng = np.random.default_rng(rng_seed)
    problems = sorted(set(labels.problem_id) - TEST)
    order = rng.permutation(problems).tolist()
    kinds = rng.choice(["correct", "incorrect"], size=len(order)).tolist()
    chosen = []
    for problem, kind in zip(order, kinds, strict=True):
        if len(chosen) == N_PROBLEMS:
            break
        n, conv, final = info[f"{problem}:{kind}"]
        limit = min(x for x in (n - 1, conv, final) if pd.notna(x))
        starts = [s for s in range(0, n) if s + WINDOW - 2 < limit and s + WINDOW - 1 <= n - 1]
        if not starts:
            continue
        st = int(rng.choice(starts))
        chosen.append({"trajectory_id": f"{problem}:{kind}", "start_fork": st,
                       "sentence_labels": list(range(st, st + WINDOW - 1))})
    return chosen


# ----------------------------------------------------------------------------- main


def main() -> None:
    t_start = time.time()
    rel = pd.read_parquet(SHARED / "released_canon.parquet",
                          columns=["trajectory_id", "problem_id", "fork_idx", "sample_idx",
                                   "cat_canon", "cat_raw", "correct_canon"])
    labels = pd.read_parquet(LABELS)
    scores = pd.read_parquet(DERIVED / "sentence_scores.parquet")
    eff = pd.read_csv(SHARED / "sentence_effects_canon.csv")
    cfg = json.loads((STUDY / "configs/phase2_forks.json").read_text())

    s = build_sentences(rel, labels)
    print(f"sentences {len(s)}", flush=True)
    tests = run_tests(s)
    s = s.merge(tests, on=["trajectory_id", "sentence_idx"], how="left", validate="1:1")
    s = s.merge(scores[["trajectory_id", "sentence_idx", "mvp_importance", "mvp_eligible"]],
                on=["trajectory_id", "sentence_idx"], how="left", validate="1:1")

    flags = {
        "primary_tv_canon": bh(s.p_tv_canon.to_numpy()),
        "secondary_dacc_canon": bh(s.p_dacc_canon.to_numpy()),
        "reference_mvp_q90": (s.mvp_importance.fillna(-np.inf) >= MVP_Q90).to_numpy(),
        "sens_tv_raw": bh(s.p_tv_raw.to_numpy()),
    }
    for k, v in flags.items():
        s[k] = v
    pstar = {k: float(s.loc[s[k], col].max()) if s[k].any() else None
             for k, col in (("primary_tv_canon", "p_tv_canon"),
                            ("secondary_dacc_canon", "p_dacc_canon"),
                            ("sens_tv_raw", "p_tv_raw"))}

    # ---- criteria, fixed list
    tie = np.random.default_rng(seed("tiebreak")).random(len(s))
    H = np.round(s.entropy_i.to_numpy(), 10)
    rank_h = np.lexsort((tie, H))                    # ascending entropy, random ties
    ter = np.empty(len(s), int)
    ter[rank_h] = np.arange(len(s)) * 3 // len(s)    # 0 low, 1 mid, 2 high
    H20 = np.round(s.entropy20_i.to_numpy(), 10)
    rank20 = np.lexsort((tie, -H20))                 # descending, random ties
    top50 = np.zeros(len(s), bool)
    top50[rank20[: int(np.ceil(0.5 * len(s)))]] = True
    pre = (s.sentence_idx < s.limit).to_numpy()
    rp = s.rel_pos.to_numpy()
    p2_units = {(t["trajectory_id"], i) for t in cfg["trajectories"] for i in t["sentence_labels"]}
    in_p2 = np.array([(t, i) in p2_units for t, i in zip(s.trajectory_id, s.sentence_idx)])
    criteria = {
        "a_all": np.ones(len(s), bool),
        "b_pre_convergence": pre,
        "b_post_convergence": ~pre,
        "c_phase2_realised_window": in_p2,
        "d_position_early": rp < 1 / 3,
        "d_position_mid": (rp >= 1 / 3) & (rp < 2 / 3),
        "d_position_late": rp >= 2 / 3,
        "e_entropy_low": ter == 0,
        "e_entropy_mid": ter == 1,
        "e_entropy_high_top_tercile": ter == 2,
        "f_trace_correct": (s.trace_kind == "correct").to_numpy(),
        "f_trace_incorrect": (s.trace_kind == "incorrect").to_numpy(),
        "g_excl_after_final_answer": (s.final.isna() | (s.sentence_idx <= s.final)).to_numpy(),
        "h_s17_top50_entropy20": top50,
        "x_non_test_problems": ~s.problem_id.isin(TEST).to_numpy(),
        "x_pre_convergence_and_non_test": pre & ~s.problem_id.isin(TEST).to_numpy(),
        "x_pre_convergence_and_entropy_high": pre & (ter == 2),
        "x_pre_convergence_and_s17_top50": pre & top50,
        "x_entropy_zero_at_fork_i": H == 0,
    }

    groups = s.problem_id.to_numpy()
    res_crit = {}
    for cname, m in criteria.items():
        row = {"n_sentences": int(m.sum()), "coverage": float(m.mean()),
               "n_problems": int(len(np.unique(groups[m])))}
        for d in DEFS:
            f = flags[d].astype(float)
            tot = f.sum()

            def rate(rows, f=f, m=m):
                mm = m[rows]
                return f[rows][mm].mean() if mm.any() else np.nan

            def ratio(rows, f=f, m=m):
                mm = m[rows]
                base = f[rows].mean()
                return f[rows][mm].mean() / base if mm.any() and base > 0 else np.nan

            sd = seed(f"boot:{cname}:{d}")
            ci = R.interval(R.cluster_bootstrap(groups, rate, REPS, sd))
            cr = R.interval(R.cluster_bootstrap(groups, ratio, REPS, sd))
            row[d] = {"anchors": int(f[m].sum()), "rate": float(f[m].mean()) if m.any() else None,
                      "rate_ci95": ci[:2], "boot_undefined": ci[2],
                      "enrichment_vs_all": float(f[m].mean() / f.mean()) if tot else None,
                      "enrichment_ci95": cr[:2],
                      "recall_of_all_anchors": float(f[m].sum() / tot) if tot else None}
        res_crit[cname] = row
        print(cname, row["n_sentences"], row["primary_tv_canon"]["anchors"],
              round(row["primary_tv_canon"]["rate"] or 0, 4), flush=True)

    # ---- (c) Phase 2 rule under alternative seeds
    info = {t: (len(g), g.convergence_idx.iloc[0], g.final_answer_idx.iloc[0])
            for t, g in labels.sort_values("sentence_idx").groupby("trajectory_id")}
    realised_seed = int(hashlib.sha256(b"20260914:phase2_selection").hexdigest()[:8], 16)
    real = phase2_rule(labels, realised_seed, info)
    reproduced = [(c["trajectory_id"], c["start_fork"]) for c in real] == \
        [(c["trajectory_id"], c["start_fork"]) for c in cfg["trajectories"]]
    key = {(t, i): j for j, (t, i) in enumerate(zip(s.trajectory_id, s.sentence_idx))}

    def window_rates(chosen):
        idx = [key[(c["trajectory_id"], i)] for c in chosen for i in c["sentence_labels"]]
        return {d: float(flags[d][idx].mean()) for d in DEFS}, len(idx), len(chosen)

    sims, short = [], 0
    for j in range(N_RULE_SEEDS):
        ch = phase2_rule(labels, seed(f"rule_seed:{j}"), info)
        r, nu, nt = window_rates(ch)
        short += nt < N_PROBLEMS
        sims.append({**r, "units": nu, "trajectories": nt,
                     "mean_entropy_i": float(s.entropy_i.to_numpy()[
                         [key[(c["trajectory_id"], i)] for c in ch for i in c["sentence_labels"]]
                     ].mean())})
    sims = pd.DataFrame(sims)
    sims.to_parquet(CACHE / "rule_seed_rates.parquet", index=False)
    real_rates, _, _ = window_rates(real)
    rule = {"realised_seed_reproduces_config": bool(reproduced), "n_seeds": N_RULE_SEEDS,
            "seeds_with_fewer_than_10_trajectories": int(short),
            "realised_mean_entropy_i": float(s.entropy_i[in_p2].mean()),
            "seed_mean_entropy_i_quantiles": sims.mean_entropy_i.quantile(
                [.025, .5, .975]).round(4).to_dict()}
    for d in DEFS:
        v = sims[d].to_numpy()
        rule[d] = {"realised_rate": real_rates[d],
                   "realised_anchors": int(round(real_rates[d] * 190)),
                   "seed_mean": float(v.mean()), "seed_sd": float(v.std(ddof=1)),
                   "seed_quantiles": {str(q): float(np.quantile(v, q))
                                      for q in (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)},
                   "share_seeds_rate_le_realised": float((v <= real_rates[d] + 1e-12).mean()),
                   "share_seeds_rate_ge_realised": float((v >= real_rates[d] - 1e-12).mean()),
                   "share_seeds_zero_anchors": float((v == 0).mean())}

    # ---- 4. cross-check with Phase 2 300 vs 300
    e = eff.merge(s[["trajectory_id", "sentence_idx", "p_tv_canon", "p_dacc_canon"] + list(DEFS)],
                  on=["trajectory_id", "sentence_idx"], how="left", validate="1:1")
    assert e.p_tv_canon.notna().all() and len(e) == 190
    e["r0_local_bh_tv"] = bh(e.p_tv_canon.to_numpy())
    e["r0_local_bh_dacc"] = bh(e.p_dacc_canon.to_numpy())

    # r1, r2 at 100 vs 100 with the same test, flagged at the global BH cutoff p* of r0
    p2 = pd.read_parquet(SHARED / "phase2_canon.parquet",
                         columns=["replicate", "trajectory_id", "fork_idx", "sample_idx",
                                  "cat_canon", "correct_canon"])
    rep = {}
    for r_ in (1, 2):
        d = p2[p2.replicate == r_]
        fk = {k: g.sort_values("sample_idx") for k, g in d.groupby(["trajectory_id", "fork_idx"])}
        ps, pd_ = [], []
        for t, i in zip(e.trajectory_id, e.sentence_idx):
            a, b = fk[(t, i)], fk[(t, i + 1)]
            cats = sorted(set(a.cat_canon) | set(b.cat_canon))
            code = {c: k for k, c in enumerate(cats)}
            ca = np.bincount(a.cat_canon.map(code), minlength=len(cats))
            cb = np.bincount(b.cat_canon.map(code), minlength=len(cats))
            rng = np.random.default_rng(seed(f"perm:r{r_}:{t}:{i}"))
            if ca.sum() != cb.sum():
                ps.append(np.nan)
            else:
                ps.append(perm_tv_p(ca, cb, rng)[0])
            pd_.append(exact_dacc_p(int(a.correct_canon.sum()), int(b.correct_canon.sum()), 100)
                       if len(a) == len(b) == 100 else np.nan)
        e[f"p_tv_r{r_}"], e[f"p_dacc_r{r_}"] = ps, pd_
        rep[f"r{r_}"] = {
            "units_tested": int(np.isfinite(e[f"p_tv_r{r_}"]).sum()),
            "tv_at_global_pstar": int((e[f"p_tv_r{r_}"] <= pstar["primary_tv_canon"]).sum())
            if pstar["primary_tv_canon"] else 0,
            "dacc_at_global_pstar": int((e[f"p_dacc_r{r_}"] <= pstar["secondary_dacc_canon"]).sum())
            if pstar["secondary_dacc_canon"] else 0,
            "tv_local_bh10": int(bh(e[f"p_tv_r{r_}"].fillna(1).to_numpy()).sum()),
        }
    e.to_csv(CACHE / "phase2_units_crosscheck.csv", index=False)
    real_set = e.real_tv_bh10.to_numpy(bool)
    cross = {
        "phase2_300v300_real_tv_bh10": int(real_set.sum()),
        "phase2_300v300_real_dacc_bh10": int(e.real_dacc_bh10.sum()),
        "r0_global_flags_in_window": {d: int(e[d].sum()) for d in DEFS},
        "r0_local_bh10_tv": int(e.r0_local_bh_tv.sum()),
        "r0_local_bh10_dacc": int(e.r0_local_bh_dacc.sum()),
        "overlap_r0_global_primary_with_300v300": int((e.primary_tv_canon & real_set).sum()),
        "overlap_r0_local_tv_with_300v300": int((e.r0_local_bh_tv & real_set).sum()),
        "kappa_r0_global_primary_vs_300v300": R.cohen_kappa(e.primary_tv_canon, real_set),
        "replicates_100v100_same_test": rep,
        "real_units": e.loc[real_set, ["trajectory_id", "sentence_idx", "p_tv",
                                       "p_tv_canon", "p_tv_r1", "p_tv_r2",
                                       "primary_tv_canon"]].to_dict("records"),
    }

    # reference flag vs noise floor
    ref = flags["reference_mvp_q90"]
    refinfo = {
        "spearman_mvp_importance_vs_entropy_i": R.spearman(
            np.round(s.mvp_importance.fillna(0).to_numpy(), 10), H),
        "spearman_mvp_importance_vs_entropy_i_eligible_only": R.spearman(
            np.round(s.mvp_importance[s.mvp_importance.notna()].to_numpy(), 10),
            H[s.mvp_importance.notna().to_numpy()]),
        "share_reference_anchors_in_entropy_top_tercile": float((ter[ref] == 2).mean()),
        "share_reference_anchors_that_are_primary_anchors": float(flags["primary_tv_canon"][ref].mean()),
        "kappa_reference_vs_primary": R.cohen_kappa(ref, flags["primary_tv_canon"]),
        "mvp_importance_missing": int(s.mvp_importance.isna().sum()),
    }

    test_info = {
        "sentences": int(len(s)), "trajectories": int(s.trajectory_id.nunique()),
        "problems": int(s.problem_id.nunique()),
        "bh_pstar": pstar,
        "anchors_total": {d: int(flags[d].sum()) for d in DEFS},
        "draws_canon": {"median": float(tests.draws_canon.median()),
                        "max": int(tests.draws_canon.max()),
                        "n_at_cap": int((tests.draws_canon >= MAX_DRAWS).sum()),
                        "n_trivial_p1": int((tests.draws_canon == 0).sum())},
        "min_p_canon": float(tests.p_tv_canon.min()),
        "n_p_canon_le_1e-4": int((tests.p_tv_canon <= 1e-4).sum()),
        "bh_rank1_threshold": BH_Q / len(s),
        "kappa_primary_vs_secondary": R.cohen_kappa(flags["primary_tv_canon"],
                                                    flags["secondary_dacc_canon"]),
        "kappa_primary_canon_vs_raw": R.cohen_kappa(flags["primary_tv_canon"],
                                                    flags["sens_tv_raw"]),
        "entropy_tercile_cuts": [float(H[rank_h[len(s) // 3]]), float(H[rank_h[2 * len(s) // 3]])],
        "entropy20_top50_cut": float(H20[rank20[int(np.ceil(0.5 * len(s))) - 1]]),
        "share_entropy_i_zero": float((H == 0).mean()),
        "share_entropy20_zero": float((H20 == 0).mean()),
        "g_drops_sentences": int((~criteria["g_excl_after_final_answer"]).sum()),
    }
    s.drop(columns=["ca", "cb", "ra", "rb"]).to_parquet(CACHE / "sentence_table.parquet",
                                                        index=False)
    write_json(OUT / "results.json", {
        "status": "EXPLORATORY, POST HOC",
        "task": "T3 (item 7): window selection and anchor rate",
        "settings": {"batch": BATCH, "h_stop": H_STOP, "max_draws": MAX_DRAWS, "bh_q": BH_Q,
                     "reps": REPS, "rule_seeds": N_RULE_SEEDS, "pilot_n": PILOT_N,
                     "mvp_q90": MVP_Q90, "seed_material": f"20260914:arv2:v3f:{TASK}:<name>"},
        "tests": test_info, "criteria": res_crit, "phase2_rule_simulation": rule,
        "phase2_crosscheck": cross, "reference_flag": refinfo,
        "runtime_s": round(time.time() - t_start, 1),
    })
    print(json.dumps({"tests": test_info, "rule": rule, "cross": {k: v for k, v in cross.items()
                                                                   if k != "real_units"}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
