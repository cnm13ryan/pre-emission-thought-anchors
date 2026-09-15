"""Stage 3 normalized trajectory and sentence records."""

from __future__ import annotations

import hashlib


def trajectory_record(base: dict, problem: dict, source_path: str, revision: str,
                      problem_id: int, trace_kind: str) -> dict:
    trace = base["full_cot"]
    return {"schema_version": 1, "trajectory_id": f"{problem_id}:{trace_kind}",
            "problem_id": problem_id, "trace_kind": trace_kind,
            "problem_text": problem["problem"], "ground_truth_answer": problem["gt_answer"],
            "prompt": base["prompt"], "solution": base["solution"], "full_cot": trace,
            "full_cot_sha256": hashlib.sha256(trace.encode()).hexdigest(),
            "normalized_final_answer": base.get("answer", ""),
            "is_correct": bool(base.get("is_correct", False)), "source_path": source_path,
            "source_revision": revision, "n_chars": len(trace)}


def sentence_record(trajectory: dict, idx: int, span, labeled: dict, *, eligible=True,
                    exclusion_reason=None) -> dict:
    text = trajectory["full_cot"][span.start:span.end]
    return {"schema_version": 1, "sentence_id": f"{trajectory['trajectory_id']}:{idx}",
            "trajectory_id": trajectory["trajectory_id"], "problem_id": trajectory["problem_id"],
            "trace_kind": trajectory["trace_kind"], "sentence_idx": idx, "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "char_start": span.start,
            "char_end": span.end, "alignment_method": span.method, "alignment_ok": True,
            "eligible": eligible, "exclusion_reason": exclusion_reason,
            "importance_released": labeled.get("counterfactual_importance_kl"),
            "importance_legacy": None, "importance_canonical": None,
            "importance_primary": labeled.get("counterfactual_importance_kl"),
            "label_source": "released"}
