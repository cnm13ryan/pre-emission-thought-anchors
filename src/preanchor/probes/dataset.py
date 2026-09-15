"""Stage 8 inputs: the Stage 7 feature table joined to its activation run.

Only ``PRIMARY_FEATURES`` and one activation site at a time are model inputs. Rows are
selected by ``eligible`` and ``split``; the fixed problem split is the only partition and
no row-wise splitter exists here (SPEC §3.4, §10.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preanchor.config import ExperimentConfig
from preanchor.features.dataset import split_of
from preanchor.features.observables import PRIMARY_FEATURES, check_feature_names, feature_path
from preanchor.features.store import read_shard
from preanchor.ids import file_hash
from preanchor.io import UnitStore, validate_parquet
from preanchor.manifest import load_manifest

# SPEC §3.10 baseline inputs, expressed with the frozen §3.9 column names (docs/decisions.md).
OBSERVABLE_BLOCKS = {
    "position": ["sentence_idx", "generated_token_count", "prefix_context_fraction"],
    "context": ["prefix_token_count", "log1p_prefix_token_count", "previous_sentence_tokens"],
    "uncertainty": ["previous_emitted_token_logprob", "next_token_entropy",
                    "next_token_max_probability", "next_token_probability_margin",
                    "next_token_top5_mass", "next_token_top10_mass"],
}
OBSERVABLE_BLOCKS["observable_combined"] = (OBSERVABLE_BLOCKS["position"]
                                            + OBSERVABLE_BLOCKS["context"]
                                            + OBSERVABLE_BLOCKS["uncertainty"])
if sorted(OBSERVABLE_BLOCKS["observable_combined"]) != sorted(PRIMARY_FEATURES):
    raise ImportError("Observable blocks must partition the frozen allow-list")
FEATURES_FILE = "stage7/features.parquet"
LABELS_FILE = "stage4/labels.parquet"
INDEX_FILE = "activations/index.parquet"


def load_features(root: Path, config: ExperimentConfig, activation_run: Path) -> tuple:
    """The published Stage 7 table, checked against the activation run it points at.

    Returns ``(table, input_hashes)``; the hashes are the Stage 8 run identity inputs.
    """
    manifest = load_manifest(activation_run)
    complete = [s for s in ("stage7", "stage6")
                if manifest["stages"].get(s, {}).get("state") == "complete"]
    if not complete:
        raise ValueError(f"Activation run {activation_run.name} has no complete extraction")
    stage = complete[0]
    if config.execution.mode == "mvp" and stage != "stage7":
        raise ValueError("The mvp configuration requires a complete Stage 7 run")
    # The canonical table is the mvp publication; a smoke run keeps its own copy.
    path = feature_path(root) if config.execution.mode == "mvp" else (
        activation_run / "features" / "features.parquet")
    table = validate_parquet(path)
    check_feature_names(config, table.column_names)
    if manifest["stages"].get(f"{stage}_features", {}).get("state") != "complete":
        raise ValueError("Activation run has no complete feature build")
    index_path = activation_run / INDEX_FILE
    index_sha = file_hash(index_path)
    if manifest["output_hashes"].get(INDEX_FILE) != index_sha:
        raise ValueError("Activation index hash differs from the run manifest")
    features_sha = file_hash(path)
    if manifest["output_hashes"].get("features/features.parquet") != features_sha:
        raise ValueError("Feature table differs from the activation run's recorded table")
    runs = set(table.column("activation_run_id").to_pylist())
    if runs != {activation_run.name}:
        raise ValueError(f"Feature rows point at other activation runs: {sorted(runs)}")
    labels_sha = file_hash(root / "data/processed" / LABELS_FILE)
    if manifest["input_hashes"].get(LABELS_FILE) != labels_sha:
        raise ValueError("Stage 4 labels changed since the activation run")
    if config.execution.mode == "mvp":
        trajectories = set(table.column("trajectory_id").to_pylist())
        if len(trajectories) != 40:
            raise ValueError("Stage 8 requires the full 40-trajectory Stage 7 table")
    hashes = {FEATURES_FILE: features_sha, INDEX_FILE: index_sha, LABELS_FILE: labels_sha,
              "activation_run_id": activation_run.name}
    return table, hashes


@dataclass
class Partition:
    """Rows of one split in deterministic order; observables as float64 columns."""

    split: str
    sentence_id: list[str]
    trajectory_id: list[str]
    problem_id: np.ndarray
    y: np.ndarray                 # anchor, int
    importance_log1p: np.ndarray  # Ridge target
    importance_primary: np.ndarray
    observables: np.ndarray       # [n, 12] in PRIMARY_FEATURES order
    shard_path: list[str]
    row_idx: np.ndarray
    anchor_threshold: float

    def __len__(self) -> int:
        return len(self.sentence_id)

    def observable_block(self, name: str) -> np.ndarray:
        cols = [PRIMARY_FEATURES.index(c) for c in OBSERVABLE_BLOCKS[name]]
        return self.observables[:, cols]


def partition_rows(table, config: ExperimentConfig, split: str) -> Partition:
    """Eligible rows of ``split`` sorted by ``(trajectory_id, sentence_idx)``."""
    splits = split_of(config)
    rows = [r for r in table.to_pylist() if r["eligible"] and r["split"] == split]
    if any(splits[int(r["problem_id"])] != split for r in rows):
        raise ValueError("A row's split column disagrees with the frozen problem split")
    rows.sort(key=lambda r: (r["trajectory_id"], r["sentence_idx"]))
    if not rows:
        raise ValueError(f"No eligible rows in split {split}")
    thresholds = {r["anchor_threshold"] for r in rows}
    if len(thresholds) != 1:
        raise ValueError("Anchor threshold is not constant across rows")
    obs = np.array([[float(r[c]) for c in PRIMARY_FEATURES] for r in rows], dtype=np.float64)
    if not np.isfinite(obs).all():
        raise ValueError("Non-finite observable value")
    y = np.array([int(r["anchor"]) for r in rows])
    log1p = np.array([float(r["importance_log1p"]) for r in rows])
    primary = np.array([float(r["importance_primary"]) for r in rows])
    if not (np.isfinite(log1p).all() and np.isfinite(primary).all()):
        raise ValueError("Non-finite importance in eligible rows")
    return Partition(split, [r["sentence_id"] for r in rows],
                     [r["trajectory_id"] for r in rows],
                     np.array([int(r["problem_id"]) for r in rows]), y, log1p, primary, obs,
                     [r["shard_path"] for r in rows], np.array([int(r["row_idx"]) for r in rows]),
                     float(thresholds.pop()))


class ActivationLoader:
    """Selected rows of every validated shard, held once in float16; one site in float32."""

    def __init__(self, activation_run: Path, partition: Partition, config: ExperimentConfig):
        store = UnitStore(activation_run / "activations" / "units")
        by_unit: dict[str, list[int]] = {}
        for i, path in enumerate(partition.shard_path):
            by_unit.setdefault(path.split("/")[1], []).append(i)
        self.n_sites = self.d_model = None
        self.resid = None
        for key in sorted(by_unit):
            store.validate(key)
            meta, tensors = read_shard(store.root / key)
            resid = tensors["resid"]
            if self.resid is None:
                self.n_sites, self.d_model = resid.shape[1], resid.shape[2]
                self.resid = np.empty((len(partition), self.n_sites, self.d_model),
                                      dtype=np.float16)
            positions = by_unit[key]
            rows = partition.row_idx[positions]
            for pos, row in zip(positions, rows, strict=True):
                if meta["boundaries"][row]["sentence_id"] != partition.sentence_id[pos]:
                    raise ValueError("Shard row does not match the feature row")
            self.resid[positions] = resid[rows]
        expected = len(config.features["layers"])
        if config.execution.mode == "mvp" and self.n_sites != expected:
            raise ValueError(f"Expected {expected} activation sites, found {self.n_sites}")

    def site(self, layer: int) -> np.ndarray:
        if not 0 <= layer < self.n_sites:
            raise ValueError(f"Site {layer} outside 0..{self.n_sites - 1}")
        return self.resid[:, layer, :].astype(np.float32)
