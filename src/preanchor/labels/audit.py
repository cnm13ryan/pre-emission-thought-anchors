"""Audit released labels against the canonical implementation on Problem 4682."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
from scipy.stats import spearmanr

from preanchor.data.importer import import_trajectory
from preanchor.io import write_json, write_parquet
from preanchor.labels.importance import canonical_from_forks


def audit_4682(source_manifest: dict, output: Path) -> dict:
    trajectory, sentences = import_trajectory(source_manifest, 4682, "correct")
    rows = [r for r in source_manifest["files"] if r.get("problem_id") == 4682
            and r.get("trace_kind") == "correct" and r.get("role") == "audit_resamples"]
    forks = {int(Path(r["path"]).parent.name.removeprefix("chunk_")): r["local_path"] for r in rows}
    audit = []
    released, canonical = [], []
    for row in sentences:
        idx = row["sentence_idx"]
        if idx not in forks or idx + 1 not in forks:
            continue
        # Load only adjacent fork files; retaining all 145 multi-megabyte lists is unnecessary.
        adjacent = {k: json.loads(Path(forks[k]).read_bytes()) for k in (idx, idx + 1)
                    if k in forks}
        value = canonical_from_forks(adjacent, idx)
        released_value = row["importance_released"]
        released.append(released_value)
        canonical.append(value)
        audit.append({"schema_version": 1, "trajectory_id": trajectory["trajectory_id"],
                      "sentence_idx": idx, "importance_released": released_value,
                      "importance_legacy": value, "importance_canonical": value,
                      "finite_nonnegative_released": isinstance(released_value, (int, float))
                      and math.isfinite(released_value) and released_value >= 0})
    finite = [x for x in released if isinstance(x, (int, float)) and math.isfinite(x) and x >= 0]
    correlation = spearmanr(released, canonical).statistic
    output.mkdir(parents=True, exist_ok=True)
    write_parquet(output / "upstream_audit.parquet", pa.Table.from_pylist(audit))
    report = {"schema_version": 1, "trajectory_id": trajectory["trajectory_id"],
              "eligible_rows": len(audit), "finite_nonnegative_fraction": len(finite) / len(audit),
              "spearman_released_vs_canonical": float(correlation),
              "gate_passed": bool(len(finite) / len(audit) >= 0.99 and correlation >= 0.95),
              "primary_field": "importance_released"}
    write_json(output / "upstream_audit.json", report)
    return report
