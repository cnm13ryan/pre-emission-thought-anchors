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


def load_tables(sources: Sources, keys: list[str] | None = None,
                columns: dict[str, list[str]] | None = None) -> dict:
    out = {}
    for key in (keys or list(TABLE_FILES)):
        out[key] = sources.read_table(TABLE_FILES[key], (columns or {}).get(key))
    return out
