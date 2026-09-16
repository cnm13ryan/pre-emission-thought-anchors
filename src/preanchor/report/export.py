"""Compatibility exporter for the vendored thought-anchors.com interface (SPEC §Stage 11).

The interface reads, per problem, ``step_importance.json``: a list of
``{source_chunk_idx, source_chunk_text, target_impacts: [{target_chunk_idx,
importance_score}]}`` (see ``vendor/thought-anchors.com/src/components/visualization/
ProblemVisualizer.js`` and the shipped data files). This MVP measures per-sentence anchor
importance, not pairwise source-to-target attribution, so ``target_impacts`` is exported
empty by design and the real per-sentence quantities travel in extension fields the
interface ignores. Also exports ``chunks_labeled.json``-shaped records whose ``importance``
is the canonical counterfactual importance, which is what the interface colors nodes by.
"""

from __future__ import annotations

from pathlib import Path

from preanchor.io import write_json
from preanchor.report.loaders import Sources, load_tables

SENTENCE_COLUMNS = ["sentence_id", "trajectory_id", "problem_id", "trace_kind",
                    "sentence_idx", "text", "eligible", "exclusion_reason",
                    "importance_released", "importance_legacy", "importance_canonical",
                    "anchor", "anchor_threshold"]
PREDICTION_COLUMNS = ["sentence_id", "probability", "probability_calibrated", "score_raw",
                      "y_true", "split"]


def export_interface(root: Path, out: Path, *, stage9_run_id: str) -> dict:
    sources = Sources(Path(root))
    labels = load_tables(sources, ["stage4_labels"],
                         {"stage4_labels": SENTENCE_COLUMNS})["stage4_labels"].to_pylist()
    predictions = {}
    for split in ("validation", "test"):
        table = sources.read_table(
            sources.run_relative(stage9_run_id, f"predictions/{split}.parquet"))
        for r in table.to_pylist():
            if r["probe_name"] == "activation_plus_observables" and r["task"] == "binary":
                predictions[r["sentence_id"]] = r

    by_trajectory: dict[tuple, list[dict]] = {}
    for row in labels:
        by_trajectory.setdefault(
            (int(row["problem_id"]), row["trace_kind"], row["trajectory_id"]),
            []).append(row)

    out = Path(out)
    written: dict[str, str] = {}
    for (problem_id, trace_kind, trajectory_id), rows in sorted(by_trajectory.items()):
        rows.sort(key=lambda r: int(r["sentence_idx"]))
        steps, chunks = [], []
        for row in rows:
            pred = predictions.get(row["sentence_id"])
            extra = {
                "sentence_id": row["sentence_id"],
                "eligible": bool(row["eligible"]),
                "exclusion_reason": row["exclusion_reason"],
                "importance_released": row["importance_released"],
                "importance_legacy": row["importance_legacy"],
                "importance_canonical": row["importance_canonical"],
                "anchor": None if row["anchor"] is None else bool(row["anchor"]),
                "predicted_split": pred["split"] if pred else None,
                "predicted_anchor_probability": (
                    None if pred is None else
                    pred.get("probability_calibrated") or pred["probability"]),
            }
            steps.append({
                "source_chunk_idx": int(row["sentence_idx"]),
                "source_chunk_text": row["text"],
                # Pairwise source->target attribution is not measured by this MVP.
                "target_impacts": [],
                **extra,
            })
            chunks.append({
                "chunk_idx": int(row["sentence_idx"]),
                "chunk": row["text"],
                "importance": row["importance_canonical"],
                **extra,
            })
        directory = out / f"problem_{problem_id}" / trace_kind / trajectory_id
        written[f"{directory.relative_to(out)}/step_importance.json"] = write_json(
            directory / "step_importance.json", steps)
        written[f"{directory.relative_to(out)}/chunks_labeled.json"] = write_json(
            directory / "chunks_labeled.json", chunks)
    return {"out": str(out), "trajectories": len(by_trajectory),
            "input_sha256": sources.ledger(), "outputs": written}
