"""Data-validity audit (plan §5, §10): parser v1 vs v2, missing answers and truncation,
looping, eligibility changes and the recomputed MVP label statistics. Exploratory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.paths import DERIVED, MVP_THRESHOLD, RESULTS, write_json  # noqa: E402


def main() -> None:
    roll = pd.read_parquet(DERIVED / "rollouts_v2.parquet")
    sc = pd.read_parquet(DERIVED / "sentence_scores.parquet")
    out = {"stage": "exploratory", "rollouts": int(len(roll))}

    # ---- parsing: v1 vs v2
    v1_missing = roll.category_v1 == "<NO_ANSWER>"
    answered = roll.status == "answered"
    changed = answered & ~v1_missing & (roll.cat != roll.category_v1)
    out["parsing"] = {
        "v2_status_counts": roll.status.value_counts().to_dict(),
        "v1_no_answer": int(v1_missing.sum()),
        "v1_no_answer_recovered_by_v2": int((v1_missing & roll.status.isin(
            ["answered", "unboxed_stated"])).sum()),
        "answer_changed_v1_to_v2": int(changed.sum()),
        "answer_changed_share_of_answered": float(changed.sum() / answered.sum()),
        "v1_answer_v2_nonanswer": int((~v1_missing & ~roll.status.isin(
            ["answered", "unboxed_stated"])).sum()),
        "correct_v2_minus_stored": int(roll.correct_v2.sum() - roll.is_correct_stored.sum())}
    examples = roll[changed].groupby(["category_v1", "cat"]).size().sort_values(
        ascending=False).head(15)
    out["parsing"]["top_changes"] = [{"v1": a, "v2": b, "n": int(n)}
                                     for (a, b), n in examples.items()]

    # ---- truncation and looping per fork
    roll["trunc"] = roll.status == "length_truncated"
    roll["loop"] = (roll.dup_sentence_frac >= 0.5) | (roll.max_repeat_run >= 5)
    fk = roll.groupby(["trajectory_id", "fork_idx"]).agg(
        n=("trunc", "size"), trunc=("trunc", "mean"), loop=("loop", "mean"),
        tokens=("rollout_tokens", "mean")).reset_index()
    fk.to_csv(RESULTS / "s07_fork_pathology.csv", index=False)
    tt = roll[roll.trunc]
    out["pathology"] = {
        "truncated": int(roll.trunc.sum()),
        "truncated_with_loop_signature": int(tt.loop.sum()),
        "truncated_loop_share": float(tt.loop.mean()) if len(tt) else None,
        "nontruncated_loop_share": float(roll[~roll.trunc].loop.mean()),
        "loop_rule": "dup_sentence_frac >= 0.5 or max_repeat_run >= 5 (last 8,000 chars)",
        "forks_trunc_ge_10pct": int((fk.trunc >= 0.10).sum()),
        "forks_trunc_ge_50pct": int((fk.trunc >= 0.50).sum()),
        "by_trajectory": roll.groupby("trajectory_id").agg(
            trunc=("trunc", "mean"), loop=("loop", "mean")).sort_values(
            "trunc", ascending=False).head(8).round(4).to_dict("index")}

    # ---- eligibility: MVP vs v2
    out["eligibility"] = {
        "mvp_eligible": int(sc.mvp_eligible.sum()), "v2_eligible": int(sc.eligible_v2.sum()),
        "both": int((sc.mvp_eligible & sc.eligible_v2).sum()),
        "only_mvp": int((sc.mvp_eligible & ~sc.eligible_v2).sum()),
        "only_v2": int((~sc.mvp_eligible & sc.eligible_v2).sum()),
        "rule_counts": {c: int(sc[c].sum()) for c in sc.columns if c.startswith("rule_")},
        "v2_label_status": sc[sc.eligible_v2].label_status.value_counts().to_dict()}

    # ---- recomputed MVP statistics under v2 eligibility / parsing
    def stats(frame, col, thr=None):
        v = frame[col].dropna()
        r = {"rows": int(len(v)), "median": float(v.median()), "p90": float(v.quantile(.9))}
        if thr is not None:
            for s in ("train", "validation", "test"):
                sub = frame[(frame.split == s)][col].dropna()
                r[f"prevalence_{s}"] = float((sub >= thr).mean()) if len(sub) else None
        return r
    mvp_rows = sc[sc.mvp_eligible]
    v2_rows = sc[sc.eligible_v2]
    out["recomputed"] = {
        "kl_v1_on_mvp_eligible": stats(mvp_rows, "kl_v1_1e-9", MVP_THRESHOLD),
        "kl_v1_on_v2_eligible": stats(v2_rows, "kl_v1_1e-9", MVP_THRESHOLD),
        "kl_v2cats_on_v2_eligible": stats(v2_rows, "kl_1e-9", MVP_THRESHOLD),
        "anchors_mvp_definition": int((mvp_rows["kl_v1_1e-9"] >= MVP_THRESHOLD).sum()),
        "anchors_v2_eligibility_v1_parse": int((v2_rows["kl_v1_1e-9"] >= MVP_THRESHOLD).sum()),
        "anchors_v2_eligibility_v2_parse": int((v2_rows["kl_1e-9"] >= MVP_THRESHOLD).sum()),
        "anchor_flag_changed_by_v2_parse_on_v2_rows": int(
            ((v2_rows["kl_v1_1e-9"] >= MVP_THRESHOLD) != (v2_rows["kl_1e-9"] >= MVP_THRESHOLD))
            [v2_rows["kl_1e-9"].notna() & v2_rows["kl_v1_1e-9"].notna()].sum()),
        "tv_on_v2_eligible": stats(v2_rows, "tv"),
        "tv_null_floor_median": float(v2_rows.tv_null_floor.median()),
        "tv_above_floor_share": float((v2_rows.tv > v2_rows.tv_null_floor)[
            v2_rows.tv.notna()].mean()),
        "tv_sensitivity_spearman": {
            c: float(v2_rows[["tv", c]].dropna().corr("spearman").iloc[0, 1])
            for c in ["tv_v1", "tv_nounboxed", "tv_answers_only"]}}
    write_json(RESULTS / "s07_validity.json", out)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
