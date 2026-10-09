"""Phase 2 power / minimum-resample simulation (plan §9). Exploratory, CPU only.

Truth for each planned sentence is its released rollouts: fork i rows (category,
correctness, dissimilar flag) and fork i+1 rows. Each simulated replicate resamples n rows
with replacement from each fork, independently for replicates 1 and 2. Each score is
computed per replicate; agreement over the planned sentences is computed with a
problem-cluster bootstrap interval and classified by the three-way rule.

Caveat (stated in the report): treating 100 empirical rows as truth adds their own sampling
noise to the true between-sentence variance, so reliability is optimistic, mildly at large
n and more at small n.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, RESULTS, STUDY, write_json  # noqa: E402

NS = [25, 50, 100, 200, 400]
SIMS, BOOT = 200, 300
SCORES = ["tv", "jsd", "dacc", "fork_tv"]
THRESH = {"icc": 0.75, "spearman": 0.80, "kappa": 0.60}


def tv_counts(a: np.ndarray, b: np.ndarray, k: int) -> float:
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return 0.5 * np.abs(pa - pb).sum()


def jsd_counts(a, b, k) -> float:
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    m = 0.5 * (pa + pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(pa > 0, pa * np.log2(pa / m), 0).sum()
        t2 = np.where(pb > 0, pb * np.log2(pb / m), 0).sum()
    return float(min(max(0.5 * (t1 + t2), 0), 1))


def main() -> None:
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    roll = pd.read_parquet(DERIVED / "rollouts_v2.parquet")
    scores = pd.read_parquet(DERIVED / "sentence_scores.parquet")
    elig_train = scores[scores.eligible_v2 & (scores.split == "train")]
    q90 = {s: float(np.nanquantile(elig_train[s].dropna(), 0.90)) for s in SCORES}
    units = []
    for t in plan["trajectories"]:
        tid = t["trajectory_id"]
        g = roll[roll.trajectory_id == tid]
        for i in t["sentence_labels"]:
            F = g[g.fork_idx == i].sort_values("sample_idx")
            G = g[g.fork_idx == i + 1].sort_values("sample_idx")
            cats = sorted(set(F.cat) | set(G.cat))
            code = {c: j for j, c in enumerate(cats)}
            units.append({"problem": t["problem_id"], "k": len(cats),
                          "f_cat": F.cat.map(code).to_numpy(), "f_ok": F.correct_v2.to_numpy(),
                          "f_dis": F.dissimilar.to_numpy(),
                          "g_cat": G.cat.map(code).to_numpy(), "g_ok": G.correct_v2.to_numpy()})
    groups = np.asarray([u["problem"] for u in units])
    rng = np.random.default_rng(R.seed_from("20260914:arv2:simulate"))
    results = []
    for n in NS:
        outcome = {s: {"icc": [], "spearman": [], "kappa": [], "three_way": [],
                       "no_support": []} for s in SCORES}
        for sim in range(SIMS):
            vals = {s: np.full((len(units), 2), np.nan) for s in SCORES}
            for ui, u in enumerate(units):
                for r in range(2):
                    fi = rng.integers(0, len(u["f_cat"]), n)
                    gi = rng.integers(0, len(u["g_cat"]), n)
                    fc, gc = u["f_cat"][fi], u["g_cat"][gi]
                    vals["fork_tv"][ui, r] = tv_counts(fc, gc, u["k"])
                    dis = u["f_dis"][fi]
                    if dis.any():
                        wc = fc[dis]
                        vals["tv"][ui, r] = tv_counts(wc, gc, u["k"])
                        vals["jsd"][ui, r] = jsd_counts(wc, gc, u["k"])
                        vals["dacc"][ui, r] = abs(u["f_ok"][fi][dis].mean() - u["g_ok"][gi].mean())
            for s in SCORES:
                v = vals[s]
                ok = np.isfinite(v).all(1)
                x, y, gr = v[ok, 0], v[ok, 1], groups[ok]
                icc, sp = R.icc_2_1(x, y), R.spearman(x, y)
                kap = R.cohen_kappa(x >= q90[s], y >= q90[s])
                seed = int(rng.integers(0, 2**31))
                d_icc = R.cluster_bootstrap(gr, lambda rr: R.icc_2_1(x[rr], y[rr]), BOOT, seed)
                d_sp = R.cluster_bootstrap(gr, lambda rr: R.spearman(x[rr], y[rr]), BOOT, seed + 1)
                v_icc = R.three_way(*R.interval(d_icc)[:2], THRESH["icc"])
                v_sp = R.three_way(*R.interval(d_sp)[:2], THRESH["spearman"])
                combined = "FAIL" if "FAIL" in (v_icc, v_sp) else (
                    "PASS" if v_icc == v_sp == "PASS" else "INCONCLUSIVE")
                o = outcome[s]
                o["icc"].append(icc); o["spearman"].append(sp); o["kappa"].append(kap)
                o["three_way"].append(combined); o["no_support"].append(int((~ok).sum()))
        for s in SCORES:
            o = outcome[s]
            tw = pd.Series(o["three_way"])
            results.append({"n": n, "score": s, "sims": SIMS,
                            "icc_mean": float(np.nanmean(o["icc"])),
                            "icc_p05": float(np.nanquantile(o["icc"], .05)),
                            "spearman_mean": float(np.nanmean(o["spearman"])),
                            "kappa_q90_mean": float(np.nanmean(o["kappa"])),
                            "p_pass": float((tw == "PASS").mean()),
                            "p_fail": float((tw == "FAIL").mean()),
                            "p_inconclusive": float((tw == "INCONCLUSIVE").mean()),
                            "no_support_mean": float(np.mean(o["no_support"]))})
        print(n, "done", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(RESULTS / "s06_power.csv", index=False)
    write_json(RESULTS / "s06_power.json", {"stage": "exploratory", "units": len(units),
                                            "problems": int(len(set(groups))),
                                            "q90_thresholds": q90, "table": results})
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
