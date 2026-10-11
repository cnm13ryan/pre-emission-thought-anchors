"""Sidecar-verified, read-only loaders for the Stage 3-10 artifacts Stage 11 consumes.

``Sources`` records the SHA-256 of every file it opens; the report embeds that ledger so
each figure and table is traceable to exact input bytes (SPEC §10.7). Loading never touches
model weights, ``.joblib`` payloads or activation shards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from preanchor.ids import file_hash
from preanchor.io import validate_artifact

STAGE9_FILES = {
    "predictions_validation": "predictions/validation.parquet",
    "predictions_test": "predictions/test.parquet",
    "metrics_validation": "metrics/validation.json",
    "metrics_test": "metrics/test.json",
    "test_receipt": "metrics/test_receipt.json",
    "calibrator": "probe_models/calibrator.json",
    "frozen_selection": "frozen_selection.json",
}
STAGE10_FILES = {
    "sensitivities": "metrics/sensitivities.parquet",
    "permutations": "metrics/permutations.parquet",
    "lead_time": "predictions/lead_time.parquet",
}
STAGE8_FILES = {
    "cv_results": "probe_models/cv_results.parquet",
    "pending_selection": "probe_models/frozen_selection.pending.json",
}
TABLE_FILES = {
    "stage3_sentences": "data/processed/stage3/sentences.parquet",
    "stage3_trajectories": "data/processed/stage3/trajectories.parquet",
    "stage4_labels": "data/processed/stage4/labels.parquet",
    "stage7_features": "data/processed/stage7/features.parquet",
}


class Sources:
    """Read-only artifact access with a hash ledger of everything opened."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.hashes: dict[str, str] = {}

    def _verified(self, path: Path) -> Path:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing Stage 11 input: {path}")
        validate_artifact(path)  # read-only: hash against the .sha256 sidecar
        self.hashes[str(path.relative_to(self.root))] = file_hash(path)
        return path

    def read_json(self, relative: str) -> dict:
        return json.loads(self._verified(self.root / relative).read_bytes())

    def read_table(self, relative: str, columns: list[str] | None = None):
        path = self._verified(self.root / relative)
        return pq.read_table(path, columns=columns)

    def run_relative(self, run_id: str, name: str) -> str:
        return f"artifacts/runs/{run_id}/{name}"

    def ledger(self) -> dict[str, str]:
        return dict(sorted(self.hashes.items()))


def load_stage9(sources: Sources, run_id: str) -> dict:
    out = {}
    for key, name in STAGE9_FILES.items():
        rel = sources.run_relative(run_id, name)
        out[key] = (sources.read_table(rel) if name.endswith(".parquet")
                    else sources.read_json(rel))
    return out


def load_stage10(sources: Sources, run_id: str) -> dict:
    return {key: sources.read_table(sources.run_relative(run_id, name))
            for key, name in STAGE10_FILES.items()}


def load_stage8(sources: Sources, run_id: str) -> dict:
    out = {}
    for key, name in STAGE8_FILES.items():
        rel = sources.run_relative(run_id, name)
        out[key] = (sources.read_table(rel) if name.endswith(".parquet")
                    else sources.read_json(rel))
    return out


def check_lineage(sources: Sources, s9: dict, *, stage8_run_id: str, stage9_run_id: str,
                  stage10_run_id: str) -> None:
    """Refuse Stage 8, 9 and 10 runs that are not one experiment lineage.

    Each run is sidecar-verified on its own, but independently supplied run ids can still
    come from different experiments. Stage 9 records the Stage 8 run and the hashes of the
    Stage 8 files it froze from. Stage 9's metrics and test receipt carry the freeze hash,
    which is the hash of Stage 9's frozen_selection.json. Stage 10's manifest records the
    Stage 9 run and freeze hash it consumed. Call after the Stage 8 and 9 files are loaded
    through ``sources``.
    """
    problems = []
    seen = sources.hashes
    frozen = s9["frozen_selection"]
    if frozen.get("probe_run_id") != stage8_run_id:
        problems.append(f"Stage 9 was frozen from Stage 8 run {frozen.get('probe_run_id')!r}, "
                        f"not {stage8_run_id!r}")
    for field, key in (("cv_results_sha256", "cv_results"),
                       ("pending_selection_sha256", "pending_selection")):
        actual = seen.get(sources.run_relative(stage8_run_id, STAGE8_FILES[key]))
        if frozen.get(field) != actual:
            problems.append(f"Stage 9 {field} does not match Stage 8 {STAGE8_FILES[key]}")
    freeze = seen.get(sources.run_relative(stage9_run_id, STAGE9_FILES["frozen_selection"]))
    for key in ("metrics_test", "test_receipt"):
        record = s9[key]
        if record.get("run_id") != stage9_run_id:
            problems.append(f"Stage 9 {STAGE9_FILES[key]} belongs to run {record.get('run_id')!r}")
        if record.get("freeze_sha256") != freeze:
            problems.append(f"Stage 9 {STAGE9_FILES[key]} freeze hash does not match "
                            "frozen_selection.json")
    manifest = sources.read_json(sources.run_relative(stage10_run_id, "manifest.json"))
    consumed = manifest.get("input_hashes", {})
    if manifest.get("run_id") != stage10_run_id:
        problems.append(f"Stage 10 manifest belongs to run {manifest.get('run_id')!r}")
    if consumed.get("stage9_run_id") != stage9_run_id:
        problems.append(f"Stage 10 consumed Stage 9 run {consumed.get('stage9_run_id')!r}, "
                        f"not {stage9_run_id!r}")
    if consumed.get("probe_run_id") != stage8_run_id:
        problems.append(f"Stage 10 consumed Stage 8 run {consumed.get('probe_run_id')!r}, "
                        f"not {stage8_run_id!r}")
    if consumed.get("stage9/freeze_sha256") != freeze:
        problems.append("Stage 10 consumed a different Stage 9 freeze")
    if problems:
        raise ValueError("Stage 8-10 runs are not one lineage: " + "; ".join(problems))


def find_stage5_pilot(root: Path) -> str | None:
    """Latest run holding a Stage 5 production summary, or None. The caller verifies it."""
    runs = Path(root) / "artifacts/runs"
    if not runs.is_dir():
        return None
    found = [run.name for run in sorted(runs.iterdir())
             if (run / "stages/stage5/summary.json").is_file()
             and (run / "stages/stage5/fresh_labels.parquet").is_file()]
    return found[-1] if found else None


def load_stage5_pilot(sources: Sources, run_id: str) -> dict:
    """Sidecar-verified Stage 5 summary, plus a verified read of its fresh labels."""
    summary = sources.read_json(sources.run_relative(run_id, "stages/stage5/summary.json"))
    labels = sources.read_table(sources.run_relative(run_id, "stages/stage5/fresh_labels.parquet"))
    return {"summary": summary, "fresh_label_rows": labels.num_rows}


def load_tables(sources: Sources, keys: list[str] | None = None,
                columns: dict[str, list[str]] | None = None) -> dict:
    out = {}
    for key in (keys or list(TABLE_FILES)):
        out[key] = sources.read_table(TABLE_FILES[key], (columns or {}).get(key))
    return out
