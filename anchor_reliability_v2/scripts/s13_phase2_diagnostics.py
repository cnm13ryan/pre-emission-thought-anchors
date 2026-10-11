"""EXPLORATORY diagnostics of the Phase 2 data (after the confirmatory s12 result; not
preregistered). Questions:

1. How much of each score is signal vs sampling noise at n = 100 (replicate 1 vs 2)?
2. Measured (not forecast) reliability as a function of resamples per fork: pool replicate
   0 (released), 1 and 2 (about 300 samples per fork), draw two disjoint halves of size n
   per fork, score both, repeat. Extrapolate with R(n) = n / (n + c).
3. Do coarser targets (accuracy level, signed or smoothed accuracy change) reproduce
   better than fork_tv?
4. How many sentences carry a real effect? Pooled permutation test per sentence.

Input: the Phase 2 completions index, inputs.parquet, derived/rollouts_v2.parquet.
Output: results/s13_phase2_diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, STUDY, write_json  # noqa: E402

N_GRID = [20, 40, 60, 80, 100, 120, 150]
DRAWS = 40
PERMS = 2000
SCORES = ["fork_tv", "abs_dacc", "signed_dacc", "smooth_dacc", "acc_level", "tv_debiased"]


def load_samples(index: Path) -> pd.DataFrame:
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt = {t: A.normalize_v2(g) for t, g in zip(trajs.trajectory_id, trajs.ground_truth_answer,
                                                strict=True)}
    idx = pd.read_parquet(index)
    idx = idx[idx.status == "OK"]
    parsed = [A.parse_v2(t, finish_reason=f, token_count=c) for t, f, c in
              zip(idx.raw_answer, idx.finish_reason, idx.generated_token_count, strict=True)]
    new = pd.DataFrame({
        "replicate": idx.replicate.to_numpy(), "trajectory_id": idx.trajectory_id.to_numpy(),
        "fork_idx": idx.fork_idx.to_numpy(),
        "cat": [A.category_v2(p) for p in parsed],
        "correct": [p.answer is not None and p.answer == gt[t]
                    for p, t in zip(parsed, idx.trajectory_id, strict=True)]})
    rel = pd.read_parquet(DERIVED / "rollouts_v2.parquet",
                          columns=["trajectory_id", "fork_idx", "cat", "correct_v2"])
    keys = set(zip(new.trajectory_id, new.fork_idx))
    rel = rel[[k in keys for k in zip(rel.trajectory_id, rel.fork_idx)]]
    rel = rel.rename(columns={"correct_v2": "correct"}).assign(replicate=0)
    return pd.concat([new, rel[new.columns]], ignore_index=True)


def tv(a: np.ndarray, b: np.ndarray, k: int) -> float:
    return 0.5 * np.abs(np.bincount(a, minlength=k) / len(a)
                        - np.bincount(b, minlength=k) / len(b)).sum()


def tv_null_analytic(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Normal approximation to E[TV] under no effect, from the pooled distribution."""
    p = np.bincount(np.concatenate([a, b]), minlength=k) / (len(a) + len(b))
    sd = np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    return 0.5 * float((sd * np.sqrt(2 / np.pi)).sum())


def unit_scores(f: dict, units, k: int) -> dict[str, np.ndarray]:
    """f[(tid, fork)] = (cat codes, correct) for one half. Returns score arrays over units."""
    out = {s: np.full(len(units), np.nan) for s in SCORES}
    for j, (tid, _, i) in enumerate(units):
        if (tid, i) not in f or (tid, i + 1) not in f:
            continue
        (ca, ka), (cb, kb) = f[(tid, i)], f[(tid, i + 1)]
        out["fork_tv"][j] = tv(ca, cb, k)
        out["tv_debiased"][j] = out["fork_tv"][j] - tv_null_analytic(ca, cb, k)
        out["abs_dacc"][j] = abs(kb.mean() - ka.mean())
        out["signed_dacc"][j] = kb.mean() - ka.mean()
        out["acc_level"][j] = ka.mean()
        if (tid, i - 1) in f and (tid, i + 2) in f:
            before = np.concatenate([f[(tid, i - 1)][1], ka]).mean()
            after = np.concatenate([kb, f[(tid, i + 2)][1]]).mean()
            out["smooth_dacc"][j] = after - before
    return out


def rel_stats(a: np.ndarray, b: np.ndarray, g: np.ndarray) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, g = a[ok], b[ok], g[ok]
    within = [R.spearman(a[g == p], b[g == p]) for p in np.unique(g)]
    within = [w for w in within if np.isfinite(w)]
    return {"icc": R.icc_2_1(a, b), "spearman": R.spearman(a, b),
            "within_spearman": float(np.median(within)) if within else np.nan, "rows": int(ok.sum())}


def fit_c(ns, vals):
    """Least-squares fit of R(n) = n / (n + c) to measured reliabilities; returns c."""
    ns, vals = np.asarray(ns, float), np.asarray(vals, float)
    keep = (vals > 0) & (vals < 1)
    if keep.sum() < 2:
        return np.nan
    cs = np.linspace(1, 5000, 50000)
    err = ((ns[keep, None] / (ns[keep, None] + cs[None, :]) - vals[keep, None]) ** 2).sum(0)
    return float(cs[err.argmin()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    groups = np.array([p for _, p, _ in units])
    s = load_samples(args.index)
    codes = {c: i for i, c in enumerate(sorted(s.cat.unique()))}
    s["code"] = s.cat.map(codes)
    k = len(codes)
    out: dict = {"stage": "exploratory_post_confirmatory", "categories": k,
                 "pool_size_per_fork": s.groupby(["trajectory_id", "fork_idx"]).size()
                 .describe()[["min", "50%", "max"]].to_dict()}

    # 1. Actual replicates 1 vs 2 at n = 100: signal / noise decomposition.
    halves = {r: {key: (g.code.to_numpy(), g.correct.to_numpy(float))
                  for key, g in s[s.replicate == r].groupby(["trajectory_id", "fork_idx"])}
              for r in (0, 1, 2)}
    sc = {r: unit_scores(halves[r], units, k) for r in (0, 1, 2)}
    decomp = {}
    for name in SCORES:
        a, b = sc[1][name], sc[2][name]
        ok = np.isfinite(a) & np.isfinite(b)
        sig = float(np.cov(a[ok], b[ok])[0, 1])
        noise = float(np.var(a[ok] - b[ok], ddof=1) / 2)
        means = pd.Series((a[ok] + b[ok]) / 2).groupby(groups[ok]).mean()
        decomp[name] = {**rel_stats(a, b, groups), "signal_var": sig, "noise_var": noise,
                        "between_problem_share_of_signal": float(np.var(means, ddof=1) / sig)
                        if sig > 0 else None}
    out["replicates_1v2_n100"] = decomp

    # 2. Measured reliability vs n from the pooled ~300 samples per fork.
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s13:curve"))
    pool = {key: (g.code.to_numpy(), g.correct.to_numpy(float))
            for key, g in s.groupby(["trajectory_id", "fork_idx"])}
    curve = {name: {} for name in SCORES}
    for n in N_GRID:
        acc = {name: [] for name in SCORES}
        for _ in range(DRAWS):
            ha, hb = {}, {}
            for key, (c, y) in pool.items():
                perm = rng.permutation(len(c))
                ha[key] = (c[perm[:n]], y[perm[:n]])
                hb[key] = (c[perm[n:2 * n]], y[perm[n:2 * n]])
            sa, sb = unit_scores(ha, units, k), unit_scores(hb, units, k)
            for name in SCORES:
                acc[name].append(rel_stats(sa[name], sb[name], groups))
        for name in SCORES:
            d = pd.DataFrame(acc[name])
            curve[name][n] = {m: float(d[m].mean()) for m in ("icc", "spearman", "within_spearman")}
            curve[name][n].update({f"{m}_sd": float(d[m].std()) for m in ("icc", "spearman")})
    proj = {}
    for name in SCORES:
        p = {}
        for m, target in (("icc", 0.75), ("spearman", 0.80), ("within_spearman", 0.80)):
            c = fit_c(N_GRID, [curve[name][n][m] for n in N_GRID])
            p[m] = {"c": c, "n_needed": float(target / (1 - target) * c) if np.isfinite(c) else None,
                    "at_n300": float(300 / (300 + c)) if np.isfinite(c) else None}
        proj[name] = p
    out["curve"] = curve
    out["projection"] = proj

    # 3/4. Real effects: pooled permutation test per sentence (all ~300 per fork).
    rng = np.random.default_rng(R.seed_from("20260914:arv2:s13:perm"))
    rows = []
    for tid, prob, i in units:
        (ca, ya), (cb, yb) = pool[(tid, i)], pool[(tid, i + 1)]
        obs_tv, obs_d = tv(ca, cb, k), abs(yb.mean() - ya.mean())
        cc, yy, na = np.concatenate([ca, cb]), np.concatenate([ya, yb]), len(ca)
        ge_tv = ge_d = 0
        for _ in range(PERMS):
            p = rng.permutation(len(cc))
            ge_tv += tv(cc[p[:na]], cc[p[na:]], k) >= obs_tv
            ge_d += abs(yy[p[na:]].mean() - yy[p[:na]].mean()) >= obs_d
        rows.append({"trajectory_id": tid, "problem_id": prob, "sentence_idx": i,
                     "pooled_fork_tv": obs_tv, "pooled_abs_dacc": obs_d,
                     "acc_before": ya.mean(), "acc_after": yb.mean(),
                     "p_tv": (ge_tv + 1) / (PERMS + 1), "p_dacc": (ge_d + 1) / (PERMS + 1)})
    eff = pd.DataFrame(rows)

    def bh(p, q=0.10):
        order = np.argsort(p)
        ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
        passed = np.zeros(len(p), bool)
        below = np.nonzero(ranked <= q)[0]
        if len(below):
            passed[order[: below.max() + 1]] = True
        return passed

    eff["real_tv_bh10"] = bh(eff.p_tv.to_numpy())
    eff["real_dacc_bh10"] = bh(eff.p_dacc.to_numpy())
    eff.to_csv(RESULTS / "s13_sentence_effects.csv", index=False)
    out["real_effects"] = {
        "sentences": len(eff),
        "tv_bh10": int(eff.real_tv_bh10.sum()), "dacc_bh10": int(eff.real_dacc_bh10.sum()),
        "per_problem_tv_bh10": eff.groupby("problem_id").real_tv_bh10.sum().astype(int).to_dict(),
        "per_problem_dacc_bh10": eff.groupby("problem_id").real_dacc_bh10.sum().astype(int).to_dict(),
        "pooled_fork_tv_quantiles": eff.pooled_fork_tv.quantile([.5, .9, .99]).to_dict(),
    }
    acc = s.groupby(["trajectory_id", "fork_idx"]).correct.mean()
    out["fork_accuracy_by_trajectory"] = {
        t: [round(float(v), 3) for v in g.sort_index(level=1).to_numpy()]
        for t, g in acc.groupby(level=0)}
    write_json(RESULTS / "s13_phase2_diagnostics.json", out)
    print(json.dumps({"replicates_1v2_n100": decomp, "projection": proj,
                      "real_effects": out["real_effects"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
