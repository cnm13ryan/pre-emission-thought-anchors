"""Import exact Stage 2 source units and align released chunks."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from preanchor.data.schemas import sentence_record, trajectory_record
from preanchor.data.segment import align_sequential
from preanchor.io import read_json, write_json, write_parquet


def import_trajectory(
    source_manifest: dict, problem_id: int, trace_kind: str
) -> tuple[dict, list[dict]]:
    rows = [r for r in source_manifest["files"] if r["problem_id"] == problem_id
            and r["trace_kind"] == trace_kind]
    def load(path: str):
        p = Path(path)
        if p.is_symlink() or not p.is_file():
            raise FileNotFoundError(path)
        if Path(str(p) + ".sha256").exists():
            # Checksummed Stage 2 artifact: a mismatch is an error, never a silent fallback.
            return read_json(p)
        return json.loads(p.read_bytes())  # Unit-test fixtures without sidecars only.

    by_role = {r["role"]: load(r["local_path"]) for r in rows}
    base = by_role["base_solution"]
    chunks = by_role["chunks"]["chunks"]
    labels = by_role["chunks_labeled"]
    problem = by_role["problem"]
    trajectory = trajectory_record(
        base, problem, rows[0]["local_path"], source_manifest["revision"], problem_id, trace_kind
    )
    think = trajectory["full_cot"].find("<think>")
    start = think + len("<think>") if think >= 0 else 0
    spans = align_sequential(trajectory["full_cot"], chunks, start)
    if len(spans) != len(labels):
        raise ValueError("Released chunks and labels have different lengths")
    sentences = [sentence_record(trajectory, i, span, labels[i])
                 for i, span in enumerate(spans)]
    return trajectory, sentences


def import_all(source_manifest: dict, problem_ids: list[int], output: Path) -> dict:
    trajectories, sentences = [], []
    for problem_id in problem_ids:
        for trace_kind in ("correct", "incorrect"):
            trajectory, rows = import_trajectory(source_manifest, problem_id, trace_kind)
            trajectories.append(trajectory)
            sentences.extend(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "import_summary.json", {"schema_version": 1,
               "trajectories": len(trajectories), "sentences": len(sentences),
               "unresolved": 0})
    write_parquet(output / "trajectories.parquet", pa.Table.from_pylist(trajectories))
    write_parquet(output / "sentences.parquet", pa.Table.from_pylist(sentences))
    return {"trajectories": trajectories, "sentences": sentences}
