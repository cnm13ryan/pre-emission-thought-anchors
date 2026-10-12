"""Per-sentence scores, eligibility and split-half scores on the released rollouts.

Outputs ``$ARV2_DATA/derived/sentence_scores.parquet`` (one row per MVP sentence) and
``results/s04_summary.json``. Reproduction gate: the v1-category KL at α = 1e-9 must equal
the MVP ``importance_canonical`` on every MVP-eligible row.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2 import eligibility as E  # noqa: E402
from arv2 import scores as S  # noqa: E402
from arv2.paths import DERIVED, INPUTS, RESULTS, SEED_ROOT, SPLIT, verify_inputs, write_json  # noqa: E402

SPLIT_OF = {p: s for s, ps in SPLIT.items() for p in ps}
COLS = ["trajectory_id", "problem_id", "fork_idx", "sample_idx", "has_error",
        "is_correct_stored", "category_v1", "v2_status", "v2_answer", "boxes", "tail",
        "rollout_tokens", "dup_sentence_frac", "max_repeat_run", "has_think_close"]


def reparse(r) -> A.Parsed:
    """Recompute the v2 answer from the stored last box / tail with the current normaliser
    (the fetch ran before the thousands-separator fix). Exact for boxed answers because the
    last box of the full text is the last box of the answer region whenever one exists."""
    if r.v2_status == "answered":
        boxes = [b for b in (list(r.boxes) if r.boxes is not None else []) if b.strip()]
        value = A.normalize_v2(boxes[-1]) if boxes else ""
        return A.Parsed("answered", value) if value else A.Parsed("no_answer", None)
    if r.v2_status == "unboxed_stated":
        again = A.parse_v2(r.tail or "")
        return again if again.status == "unboxed_stated" else A.Parsed("no_answer", None)
    return A.Parsed(r.v2_status, None)


def half_mask(tid: str, fork: int, n: int) -> np.ndarray:
    seed = int(hashlib.sha256(f"{SEED_ROOT}:arv2:split_half:{tid}:{fork}".encode())
               .hexdigest()[:8], 16)
    m = np.zeros(n, bool)
    m[np.random.default_rng(seed).permutation(n)[: n // 2]] = True
    return m


def score_pair(w_cat, w_ok, g_cat, g_ok, f_cat) -> dict:
    out = {}
    if g_cat and f_cat:
        out["fork_tv"] = S.total_variation(f_cat, g_cat)
    if w_cat and g_cat:
        out["tv"] = S.total_variation(w_cat, g_cat)
        out["jsd"] = S.jensen_shannon(w_cat, g_cat)
        out["dacc"] = S.delta_accuracy(w_ok, g_ok)
        out["kl_1e-9"] = S.kl_smoothed(w_cat, g_cat, 1e-9, extra_categories=())
    return out


def main() -> None:
    verify_inputs()
    files = sorted(glob.glob(str(DERIVED / "rollouts/*/fork_*.parquet")))
    if len(files) != 6474:
        raise SystemExit(f"expected 6474 fork files, found {len(files)}")
    roll = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    roll = roll.merge(pd.read_parquet(DERIVED / "similarity.parquet"),
                      on=["trajectory_id", "fork_idx", "sample_idx"], validate="one_to_one")
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt = {t: A.normalize_v2(g) for t, g in zip(trajs.trajectory_id, trajs.ground_truth_answer,
                                                strict=True)}
    parsed = [reparse(r) for r in roll.itertuples()]
    roll["status"] = [p.status for p in parsed]
    roll["cat"] = [A.category_v2(p) for p in parsed]
    roll["cat_nounboxed"] = [A.category_v2(p, include_unboxed=False) for p in parsed]
    roll["correct_v2"] = [p.answer is not None and p.answer == gt[t]
                          for p, t in zip(parsed, roll.trajectory_id, strict=True)]
    answered = roll.status.isin(["answered"])
    agree = float((roll.loc[answered, "correct_v2"] == roll.loc[answered,
                                                               "is_correct_stored"]).mean())
    roll["dissimilar"] = roll.similarity < 0.8
    # rollout-level table kept for later stages
    roll[["trajectory_id", "problem_id", "fork_idx", "sample_idx", "status", "cat",
          "cat_nounboxed", "category_v1", "correct_v2", "is_correct_stored", "similarity",
          "dissimilar", "rollout_tokens", "dup_sentence_frac", "max_repeat_run",
          "has_think_close"]].to_parquet(DERIVED / "rollouts_v2.parquet", index=False)

    forks = {}
    for (tid, fk), g in roll.groupby(["trajectory_id", "fork_idx"], sort=False):
        g = g.sort_values("sample_idx")
        forks[(tid, int(fk))] = g
    sents = pd.read_parquet(INPUTS / "sentences.parquet")
    mvp = pd.read_parquet(INPUTS / "mvp_labels.parquet").set_index("sentence_id")
    rows = []
    for tid, ss in sents.groupby("trajectory_id"):
        ss = ss.sort_values("sentence_idx")
        n = len(ss)
        acc = [float(forks[(tid, i)].correct_v2.mean()) if (tid, i) in forks else None
               for i in range(n)]
        acc_mvp = [mvp.loc[s, "fork_accuracy"] for s in ss.sentence_id]
        acc_mvp = [None if pd.isna(a) else float(a) for a in acc_mvp]
        conv_pre = E.converged_pre(acc)
        conv_mvp = E.converged_mvp(acc_mvp)
        boxed = E.boxed_before(ss.text.tolist())
        fin_mvp = E.final_answer_mvp(ss.text.tolist())
        for k, s in enumerate(ss.itertuples()):
            i = s.sentence_idx
            F, G = forks.get((tid, i)), forks.get((tid, i + 1))
            rec = {"sentence_id": s.sentence_id, "trajectory_id": tid,
                   "problem_id": s.problem_id, "split": SPLIT_OF[s.problem_id],
                   "sentence_idx": i, "fork_accuracy_v2": acc[k],
                   "rule_boxed_before": boxed[k], "rule_fork_i_missing": F is None,
                   "rule_converged_pre": conv_pre[k], "rule_converged_mvp": conv_mvp[k],
                   "rule_final_answer_mvp": fin_mvp[k],
                   "mvp_eligible": bool(mvp.loc[s.sentence_id, "eligible"]),
                   "mvp_importance": mvp.loc[s.sentence_id, "importance_canonical"]}
            rec["eligible_v2"] = bool(s.alignment_ok and not boxed[k] and F is not None
                                      and not conv_pre[k])
            if F is None or G is None:
                rec["label_status"] = "MISSING_FORK"
                rows.append(rec)
                continue
            w = F[F.dissimilar]
            rec["n_fork"], rec["n_without"], rec["n_with"] = len(F), len(w), len(G)
            rec["label_status"] = "OK" if len(w) else "NO_SUPPORT"
            rec.update(score_pair(w.cat.tolist(), w.correct_v2.tolist(), G.cat.tolist(),
                                  G.correct_v2.tolist(), F.cat.tolist()))
            if len(w):
                rec["kl_v1_1e-9"] = S.kl_smoothed(w.category_v1.tolist(),
                                                  G.category_v1.tolist(), 1e-9)
                rec["tv_v1"] = S.total_variation(w.category_v1.tolist(), G.category_v1.tolist())
                rec["tv_nounboxed"] = S.total_variation(w.cat_nounboxed.tolist(),
                                                        G.cat_nounboxed.tolist())
                wa = [c for c in w.cat if not c.startswith("<")]
                ga = [c for c in G.cat if not c.startswith("<")]
                rec["tv_answers_only"] = S.total_variation(wa, ga) if wa and ga else None
                pooled = Counter(w.cat.tolist() + G.cat.tolist())
                tot = sum(pooled.values())
                rec["tv_null_floor"] = S.tv_null_expectation(
                    {c: v / tot for c, v in pooled.items()}, len(w), len(G), draws=300,
                    seed=int(hashlib.sha256(s.sentence_id.encode()).hexdigest()[:8], 16))
            # split halves
            hf, hg = half_mask(tid, i, len(F)), half_mask(tid, i + 1, len(G))
            for tag, fm, gm in (("A", hf, hg), ("B", ~hf, ~hg)):
                Fh, Gh = F[fm], G[gm]
                wh = Fh[Fh.dissimilar]
                for name, v in score_pair(wh.cat.tolist(), wh.correct_v2.tolist(),
                                          Gh.cat.tolist(), Gh.correct_v2.tolist(),
                                          Fh.cat.tolist()).items():
                    rec[f"h{tag}_{name}"] = v
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_parquet(DERIVED / "sentence_scores.parquet", index=False)

    # reproduction gate against the MVP
    m = out[out.mvp_eligible & out["kl_v1_1e-9"].notna()]
    diff = (m["kl_v1_1e-9"] - m.mvp_importance).abs()
    status_counts = roll.status.value_counts().to_dict()
    summary = {
        "stage": "exploratory", "rollouts": int(len(roll)),
        "v2_status_counts": status_counts,
        "v1_missing": int((roll.category_v1 == "<NO_ANSWER>").sum()),
        "v1_vs_v2_category_differs_on_answered": int(
            (answered & (roll.category_v1 != roll.cat)).sum()),
        "correctness_v2_vs_stored_agreement_on_answered": agree,
        "reproduction": {"rows": int(len(m)), "max_abs_diff": float(diff.max()),
                         "rows_diff_gt_1e-9": int((diff > 1e-9).sum()),
                         "passed": bool((diff <= 1e-9).all())},
        "eligible_v2": int(out.eligible_v2.sum()), "mvp_eligible": int(out.mvp_eligible.sum()),
        "label_status_eligible_v2": out[out.eligible_v2].label_status.value_counts().to_dict(),
        "rules": {c: int(out[c].sum()) for c in out.columns if c.startswith("rule_")}}
    write_json(RESULTS / "s04_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
