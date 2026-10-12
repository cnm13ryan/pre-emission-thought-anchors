"""Phase 2 confirmatory scoring (plan §7, v2.1). Written and tested BEFORE any Phase 2 data
were read.

Input: ``completions_index.parquet`` (from the generation driver's ``index`` command:
replicate, trajectory_id, fork_idx, sample_idx, status, raw_answer, finish_reason,
generated_token_count, first_replacement_sentence), and ``configs/phase2_forks.json``.

Procedure:
1. Common N per fork across both replicates (plan §3, early-stop rule).
2. Parse with v2, using the exact ``finish_reason`` for truncation, and use the primary
   handling for categories.
3. MiniLM similarity of each replacement sentence to S_i; "without" = cosine < 0.8.
4. Scores per replicate: fork_tv (primary), tv, jsd, dacc and MVP KL (reference).
   Replicate 0 = the released samples for the same forks.
5. Replicate 1 vs 2: Spearman and ICC(2,1) on the continuous score; κ of the flag at the
   FIXED q90 threshold from results/s06_power.json. 95% problem-cluster bootstrap
   (2,000 replicates, seed material 20260914:arv2:bootstrap). Three-way rule against
   thresholds 0.80 / 0.75 / 0.60.
6. Sensitivities: excluding problem 1591; each replicate vs replicate 0.
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
from arv2 import scores as S  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, STUDY, write_json  # noqa: E402

PRIMARY = "fork_tv"
SCORES = ["fork_tv", "tv", "jsd", "dacc", "kl_1e-9"]
THRESH = {"spearman": 0.80, "icc": 0.75, "kappa": 0.60}
REPS = 2000


def label_table(samples: pd.DataFrame, units: list[tuple[str, int, int]]) -> pd.DataFrame:
    """samples: trajectory_id, fork_idx, cat, correct, similarity (to S_fork)."""
    by = {k: g for k, g in samples.groupby(["trajectory_id", "fork_idx"])}
    rows = []
    for tid, problem, i in units:
        F, G = by.get((tid, i)), by.get((tid, i + 1))
        rec = {"trajectory_id": tid, "problem_id": problem, "sentence_idx": i}
        if F is None or G is None or not len(F) or not len(G):
            rec["status"] = "MISSING_FORK"
            rows.append(rec)
            continue
        rec["fork_tv"] = S.total_variation(F.cat.tolist(), G.cat.tolist())
        w = F[F.similarity < 0.8]
        rec["n_without"] = len(w)
        rec["status"] = "OK" if len(w) else "NO_SUPPORT"
        if len(w):
            rec["tv"] = S.total_variation(w.cat.tolist(), G.cat.tolist())
            rec["jsd"] = S.jensen_shannon(w.cat.tolist(), G.cat.tolist())
            rec["dacc"] = S.delta_accuracy(w.correct.tolist(), G.correct.tolist())
            rec["kl_1e-9"] = S.kl_smoothed(w.cat.tolist(), G.cat.tolist(), 1e-9,
                                           extra_categories=())
        rows.append(rec)
    return pd.DataFrame(rows)


def agreement(x: pd.DataFrame, y: pd.DataFrame, q90: dict, seed_material: str) -> list[dict]:
    m = x.merge(y, on=["trajectory_id", "problem_id", "sentence_idx"], suffixes=("_1", "_2"))
    out = []
    for s in SCORES:
        d = m[m[f"{s}_1"].notna() & m[f"{s}_2"].notna()].reset_index(drop=True)
        a, b = d[f"{s}_1"].to_numpy(float), d[f"{s}_2"].to_numpy(float)
        if s == "kl_1e-9":
            a, b = np.log1p(a), np.log1p(b)
        g = d.problem_id.to_numpy()
        rec = {"score": s, "rows": len(d), "excluded": int(len(m) - len(d)),
               "problems": int(len(set(g)))}
        seed = R.seed_from(f"{seed_material}:{s}")
        stats = {"spearman": lambda r: R.spearman(a[r], b[r]),
                 "icc": lambda r: R.icc_2_1(a[r], b[r])}
        if s in q90:
            t = q90[s]
            stats["kappa"] = lambda r: R.cohen_kappa(a[r] >= t, b[r] >= t)
            rec["flag_rate_1"], rec["flag_rate_2"] = float((a >= t).mean()), float((b >= t).mean())
        verdicts = []
        for k, fn in stats.items():
            rec[k] = fn(np.arange(len(d)))
            lo, hi, bad = R.interval(R.cluster_bootstrap(g, fn, REPS, seed))
            rec[f"{k}_lo"], rec[f"{k}_hi"], rec[f"{k}_nonfinite"] = lo, hi, bad
            rec[f"{k}_decision"] = R.three_way(lo, hi, THRESH[k])
            verdicts.append(rec[f"{k}_decision"])
        cont = [rec["spearman_decision"], rec["icc_decision"]]
        rec["continuous_form"] = ("FAIL" if "FAIL" in cont else
                                  "PASS" if all(v == "PASS" for v in cont) else "INCONCLUSIVE")
        if "kappa" in stats:
            rec["flag_form"] = rec["kappa_decision"]
        within = [R.spearman(a[g == p], b[g == p]) for p in sorted(set(g))]
        w = [v for v in within if np.isfinite(v)]
        rec["within_problem_spearman_median"] = float(np.median(w)) if w else None
        out.append(rec)
    return out


def common_counts(ok: pd.DataFrame, forks: pd.DataFrame | None = None) -> pd.Series:
    """OK samples per fork usable by both replicates: min over replicates 1 and 2.

    A replicate with no OK samples for a fork counts as 0, not as missing, so such a fork
    gets common_n 0 instead of the other replicate's count. Forks listed in ``forks``
    (trajectory_id, fork_idx) with no OK rows at all are included with 0 as well.
    """
    n = (ok.groupby(["trajectory_id", "fork_idx", "replicate"]).size().unstack()
         .reindex(columns=[1, 2]).fillna(0).astype(int))
    if forks is not None:
        n = n.reindex(pd.MultiIndex.from_frame(forks[["trajectory_id", "fork_idx"]]
                                               .drop_duplicates()), fill_value=0)
    return n.min(axis=1).rename("common_n")


def prepare_replicates(idx: pd.DataFrame, inputs: pd.DataFrame, gt: dict, model) -> tuple:
    ok = idx[idx.status == "OK"].copy()
    common = common_counts(ok, inputs)
    ok = ok.merge(common, left_on=["trajectory_id", "fork_idx"], right_index=True)
    ok = ok[ok.sample_idx < ok.common_n].copy()
    parsed = [A.parse_v2(t, finish_reason=f, token_count=c) for t, f, c in
              zip(ok.raw_answer, ok.finish_reason, ok.generated_token_count, strict=True)]
    ok["v2_status"] = [p.status for p in parsed]
    ok["cat"] = [A.category_v2(p) for p in parsed]
    ok["correct"] = [p.answer is not None and p.answer == gt[t]
                     for p, t in zip(parsed, ok.trajectory_id, strict=True)]
    target = inputs.set_index(["trajectory_id", "fork_idx"]).target_text
    ok["target"] = [target[(t, f)] for t, f in zip(ok.trajectory_id, ok.fork_idx, strict=True)]
    has = ok.first_replacement_sentence.notna()
    texts = sorted(set(ok.loc[has, "first_replacement_sentence"]) | set(ok.target))
    vec = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    pos = {t: i for i, t in enumerate(texts)}
    sim = np.full(len(ok), np.nan)
    ia = ok.loc[has, "first_replacement_sentence"].map(pos).to_numpy()
    ib = ok.loc[has, "target"].map(pos).to_numpy()
    sim[has.to_numpy()] = np.einsum("ij,ij->i", vec[ia].astype(np.float64),
                                    vec[ib].astype(np.float64))
    ok["similarity"] = sim
    return ok, common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True,
                    help="Phase 2 inputs.parquet (prompts + target sentences)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-tag", default="phase2")
    args = ap.parse_args()
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    q90 = json.loads((RESULTS / "s06_power.json").read_text())["q90_thresholds"]
    inputs = pd.read_parquet(args.inputs)
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt = {t: A.normalize_v2(g) for t, g in zip(trajs.trajectory_id, trajs.ground_truth_answer,
                                                strict=True)}
    idx = pd.read_parquet(args.index)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                device=args.device)
    ok, common = prepare_replicates(idx, inputs, gt, model)
    reps = {r: label_table(ok[ok.replicate == r], units) for r in (1, 2)}
    released = pd.read_parquet(DERIVED / "rollouts_v2.parquet")
    keys = {(t, f) for t, f in zip(inputs.trajectory_id, inputs.fork_idx)}
    rel = released[[k in keys for k in zip(released.trajectory_id, released.fork_idx)]].rename(
        columns={"correct_v2": "correct"})
    reps[0] = label_table(rel, units)
    primary = agreement(reps[1], reps[2], q90, "20260914:arv2:bootstrap")
    no1591 = agreement(reps[1][reps[1].problem_id != 1591], reps[2][reps[2].problem_id != 1591],
                       q90, "20260914:arv2:bootstrap:no1591")
    vs0 = {f"{r}_vs_0": agreement(reps[r], reps[0], q90, f"20260914:arv2:bootstrap:{r}v0")
           for r in (1, 2)}
    for r, f in reps.items():
        f.to_csv(RESULTS / f"{args.out_tag}_scores_replicate{r}.csv", index=False)
    prim = next(x for x in primary if x["score"] == PRIMARY)
    summary = {
        "stage": "confirmatory_reliability" if args.out_tag == "phase2" else "test",
        "preregistration": "phase2_plan.md v2.1", "primary_score": PRIMARY,
        "index_status": {f"{a}:{b}": int(v) for (a, b), v in
                         idx.groupby(["replicate", "status"]).size().items()},
        "common_n_min": int(common.min()), "common_n_median": float(common.median()),
        "zero_common_forks": [f"{t}:{f}" for (t, f), v in common.items() if v == 0],
        "v2_status_counts": {f"{a}:{b}": int(v) for (a, b), v in
                             ok.groupby(["replicate", "v2_status"]).size().items()},
        "similarity_missing": int(ok.similarity.isna().sum()),
        "label_status": {str(r): f.status.value_counts().to_dict() for r, f in reps.items()},
        "primary_decision": {"continuous": prim["continuous_form"], "flag": prim["flag_form"]},
        "replicate_1_vs_2": primary, "sensitivity_without_1591": no1591,
        "vs_released": vs0,
        "no_form_passes": not any(x["continuous_form"] == "PASS" or x.get("flag_form") == "PASS"
                                  for x in primary)}
    write_json(RESULTS / f"{args.out_tag}_reliability.json", summary)
    print(json.dumps({k: summary[k] for k in ("primary_decision", "common_n_min",
                                              "no_form_passes")}, indent=1))


if __name__ == "__main__":
    main()
