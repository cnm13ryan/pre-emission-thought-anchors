"""Split-half reliability of every candidate score on the released rollouts (plan §3, §7).

For each score: Spearman, ICC(2,1) and the Spearman–Brown projection to full N; Cohen's κ
of q80/q90/q95 flags (each half uses its own training-split quantile); prevalence;
within-problem Spearman; and 95% problem-cluster bootstrap intervals with the three-way
rule. Applies the preregistered primary-score selection rule (highest split-half ICC
among tv, jsd, dacc, fork_tv; ties within 0.02 broken in that order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DERIVED, RESULTS, write_json  # noqa: E402

SCORES = ["tv", "jsd", "dacc", "fork_tv", "kl_1e-9"]
CANDIDATES = ["tv", "jsd", "dacc", "fork_tv"]
THRESH = {"icc": 0.75, "spearman": 0.80, "kappa": 0.60}
REPS = 2000


def main() -> None:
    d = pd.read_parquet(DERIVED / "sentence_scores.parquet")
    elig = d[d.eligible_v2]
    rows, perproblem = [], []
    for s in SCORES:
        a, b = f"hA_{s}", f"hB_{s}"
        e = elig[elig[a].notna() & elig[b].notna()].reset_index(drop=True)
        x, y = e[a].to_numpy(float), e[b].to_numpy(float)
        if s == "kl_1e-9":
            x, y = np.log1p(x), np.log1p(y)
        groups = e.problem_id.to_numpy()
        tr = (e.split == "train").to_numpy()
        rec = {"score": s, "rows": len(e),
               "missing_in_a_half": int(len(elig) - len(e)),
               "spearman": R.spearman(x, y), "icc": R.icc_2_1(x, y)}
        rec["icc_projected_full_n"] = R.spearman_brown(rec["icc"], 2)
        rec["spearman_projected_full_n"] = R.spearman_brown(rec["spearman"], 2)
        seed = R.seed_from(f"20260914:arv2:bootstrap:splithalf:{s}")
        for stat, fn in (("spearman", R.spearman), ("icc", R.icc_2_1)):
            draws = R.cluster_bootstrap(groups, lambda r, fn=fn: fn(x[r], y[r]), REPS, seed)
            lo, hi, bad = R.interval(draws)
            rec[f"{stat}_lo"], rec[f"{stat}_hi"], rec[f"{stat}_nonfinite"] = lo, hi, bad
            rec[f"{stat}_three_way"] = R.three_way(lo, hi, THRESH[stat])
        for q in (0.80, 0.90, 0.95):
            ta, tb = np.quantile(x[tr], q), np.quantile(y[tr], q)
            fa, fb = x >= ta, y >= tb
            k = R.cohen_kappa(fa, fb)
            draws = R.cluster_bootstrap(groups, lambda r: R.cohen_kappa(fa[r], fb[r]), REPS,
                                        seed + int(q * 100))
            lo, hi, bad = R.interval(draws)
            tag = f"q{int(q * 100)}"
            rec.update({f"kappa_{tag}": k, f"kappa_{tag}_lo": lo, f"kappa_{tag}_hi": hi,
                        f"kappa_{tag}_three_way": R.three_way(lo, hi, THRESH["kappa"]),
                        f"prevalence_{tag}_halfA": float(fa.mean()),
                        f"prevalence_{tag}_halfB": float(fb.mean())})
        within = []
        for p, g in e.groupby("problem_id"):
            idx = g.index.to_numpy()
            r = R.spearman(x[idx], y[idx])
            within.append(r)
            perproblem.append({"score": s, "problem_id": int(p), "rows": len(idx),
                               "spearman": r})
        w = np.asarray([v for v in within if np.isfinite(v)])
        rec.update({"within_problem_spearman_median": float(np.median(w)) if len(w) else None,
                    "within_problem_spearman_min": float(w.min()) if len(w) else None,
                    "within_problem_spearman_max": float(w.max()) if len(w) else None,
                    "within_problem_nonfinite": int(len(within) - len(w)),
                    "noise_ceiling_projected": float(np.sqrt(max(rec["icc_projected_full_n"],
                                                                 0)))})
        rows.append(rec)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "s05_split_half_reliability.csv", index=False)
    pd.DataFrame(perproblem).to_csv(RESULTS / "s05_within_problem.csv", index=False)
    cand = table[table.score.isin(CANDIDATES)].set_index("score").loc[CANDIDATES]
    best = cand.icc.max()
    primary = next(s for s in CANDIDATES if cand.loc[s, "icc"] >= best - 0.02)
    out = {"stage": "exploratory", "eligible_v2": int(len(elig)),
           "primary_selection_rule": "max split-half ICC among tv,jsd,dacc,fork_tv; "
                                     "ties within 0.02 broken in that order",
           "candidate_icc": cand.icc.to_dict(), "primary_score": primary,
           "table": rows}
    write_json(RESULTS / "s05_reliability.json", out)
    print(table[["score", "rows", "spearman", "spearman_lo", "spearman_hi", "icc", "icc_lo",
                 "icc_hi", "icc_projected_full_n", "kappa_q90", "kappa_q90_lo",
                 "kappa_q90_hi", "within_problem_spearman_median"]].round(3).to_string())
    print("primary:", primary)


if __name__ == "__main__":
    main()
