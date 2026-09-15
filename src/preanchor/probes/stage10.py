"""SPEC Stage 10: controls, sensitivities and lead time on a finalized Stage 9 run.

A Stage 10 run is its own ``run_id`` whose ``input_hashes`` name the frozen, test-evaluated
Stage 9 run (freeze hash, registry, calibrator, validation/test predictions and metrics) and
the Stage 7/8 inputs behind it. Nothing here writes into the Stage 9 run and nothing changes
the frozen selection: every model refitted here uses the frozen family, site and
hyperparameter of the Stage 8 selection (``payload["models"]``), and every alternate anchor
threshold is a quantile of eligible *training* importance only.

Analyses (SPEC §3.12 items 5-11, config ``evaluation``), each an immutable, checksummed
work unit under ``stages/stage10/units`` so ``--resume`` skips validated work:

``threshold:q80|q90|q95``      train quantile thresholds; q90 must equal the frozen one
``convergence:k5|k4``          the primary K=5 eligible set versus the K=4 sensitivity
``replacement_quality:*``      dissimilar fraction >= 0.10 or >= 10 dissimilar rollouts
``trajectory_subgroup:*``      correct versus incorrect trajectories (frozen scores)
``position_quintile:1..5``     train-derived quintiles of ``sentence_idx`` (frozen scores)
``entropy_quintile:1..5``      train-derived quintiles of ``next_token_entropy``
``lead_time``                  lead one: features of the boundary before S_{i-1}, label of S_i
``permutations``               100 label permutations within (problem, trace_kind, quintile)
``paired_baseline_differences``  recomputed from the frozen Stage 9 predictions

Two model sources are distinguished in every table: ``frozen_stage9`` (the Stage 9
predictions re-scored on a subset or against relabelled anchors) and
``refit_frozen_hyperparameters`` (a sensitivity model refitted on training rows at the
frozen site and hyperparameter, never re-selected).
"""

from __future__ import annotations

import json
import math
import os
import resource
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed

from preanchor.config import ExperimentConfig, config_hash
from preanchor.features.extract import write_current_json
from preanchor.features.observables import PRIMARY_FEATURES
from preanchor.ids import cache_key, derived_seed, file_hash
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_artifact,
    validate_parquet,
    write_json,
    write_parquet,
)
from preanchor.labels.threshold import train_threshold
from preanchor.logging import redact
from preanchor.manifest import check_resume, create_run, load_manifest, save_manifest, utc_now
from preanchor.probes import models as M
from preanchor.probes.bootstrap import clustered_bootstrap
from preanchor.probes.dataset import (
    LABELS_FILE,
    ActivationLoader,
    Partition,
    partition_rows,
)
from preanchor.probes.evaluation import binary_metrics
from preanchor.probes.fit import THRESHOLD_QUANTILE
from preanchor.probes.folds import assert_disjoint
from preanchor.probes.metrics import auroc, average_precision
from preanchor.probes.select import PRIMARY_MODEL
from preanchor.probes.stage9 import (
    BASELINE,
    CALIBRATOR_FILE,
    FREEZE_FILE,
    RECEIPT_FILE,
    REGISTRY_FILE,
    TEST,
    VALIDATION,
    test_state,
    verify_freeze,
)
from preanchor.runtime import Runtime

STAGE = "stage10"
RUN_STAGE = "stage10_sensitivities"
UNITS_DIR = "stages/stage10/units"
SENSITIVITIES_FILE = "metrics/sensitivities.parquet"
PERMUTATIONS_FILE = "metrics/permutations.parquet"
LEAD_FILE = "predictions/lead_time.parquet"
SUMMARY_FILE = "stages/stage10/summary.json"
FROZEN, REFIT = "frozen_stage9", "refit_frozen_hyperparameters"
REFIT_FAMILIES = (PRIMARY_MODEL, "activation_only", BASELINE)
STAGE9_OUTPUTS = (FREEZE_FILE, REGISTRY_FILE, CALIBRATOR_FILE, RECEIPT_FILE,
                  "predictions/validation.parquet", "predictions/test.parquet",
                  "metrics/validation.json", "metrics/test.json")
STAGE11_MARKERS = ("SUCCESS.json", "SAFE_TO_TERMINATE", "reports")
PERMUTATION_PURPOSE = "permutation"
POINT_METRICS = ("average_precision", "auroc", "brier", "log_loss", "ece",
                 "precision_top10_percent", "ndcg_top10_percent")
PAIRED_METRICS = ("average_precision", "auroc")
EVAL_SPLITS = (VALIDATION, TEST)
QUANTILE_METHOD = "linear interpolation on the sorted values (numpy default, labels/threshold.py)"

SENSITIVITY_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("analysis", pa.string(), nullable=False),
    pa.field("variant", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("model_source", pa.string(), nullable=False),
    pa.field("probe_name", pa.string(), nullable=False),
    pa.field("task", pa.string(), nullable=False),
    pa.field("layer", pa.int64(), nullable=True),
    pa.field("hyperparameter", pa.float64(), nullable=True),
    pa.field("lead_sentences", pa.int64(), nullable=False),
    pa.field("threshold_quantile", pa.float64(), nullable=False),
    pa.field("anchor_threshold", pa.float64(), nullable=False),
    pa.field("n_rows", pa.int64(), nullable=False),
    pa.field("n_positive", pa.int64(), nullable=False),
    pa.field("n_problems", pa.int64(), nullable=False),
    pa.field("n_trajectories", pa.int64(), nullable=False),
    pa.field("metric", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=True),
    pa.field("boot_low", pa.float64(), nullable=True),
    pa.field("boot_high", pa.float64(), nullable=True),
    pa.field("boot_mean", pa.float64(), nullable=True),
    pa.field("boot_n_valid", pa.int64(), nullable=True),
    pa.field("comparison", pa.string(), nullable=True),
    pa.field("detail", pa.string(), nullable=True),
])

PERMUTATION_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),          # observed | permutation
    pa.field("permutation_index", pa.int64(), nullable=False),  # -1 for observed
    pa.field("seed", pa.int64(), nullable=True),
    pa.field("seed_purpose", pa.string(), nullable=True),
    pa.field("split", pa.string(), nullable=False),
    pa.field("probe_name", pa.string(), nullable=False),
    pa.field("layer", pa.int64(), nullable=False),
    pa.field("hyperparameter", pa.float64(), nullable=False),
    pa.field("strata", pa.string(), nullable=False),
    pa.field("n_strata", pa.int64(), nullable=False),
    pa.field("n_permutable_strata", pa.int64(), nullable=False),
    pa.field("n_single_class_strata", pa.int64(), nullable=False),
    pa.field("n_rows_permuted", pa.int64(), nullable=False),
    pa.field("n_rows_fixed", pa.int64(), nullable=False),
    pa.field("labels_changed", pa.int64(), nullable=True),
    pa.field("n_rows", pa.int64(), nullable=False),
    pa.field("n_positive", pa.int64(), nullable=False),
    pa.field("average_precision", pa.float64(), nullable=True),
    pa.field("auroc", pa.float64(), nullable=True),
    pa.field("converged", pa.bool_(), nullable=True),
])

LEAD_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("sentence_id", pa.string(), nullable=False),
    pa.field("problem_id", pa.int64(), nullable=False),
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("lead_sentences", pa.int64(), nullable=False),
    pa.field("threshold_quantile", pa.float64(), nullable=False),
    pa.field("layer", pa.int64(), nullable=True),
    pa.field("probe_name", pa.string(), nullable=False),
    pa.field("task", pa.string(), nullable=False),
    pa.field("hyperparameter", pa.float64(), nullable=True),
    pa.field("model_source", pa.string(), nullable=False),
    pa.field("target_sentence_idx", pa.int64(), nullable=False),
    pa.field("feature_sentence_id", pa.string(), nullable=True),
    pa.field("feature_sentence_idx", pa.int64(), nullable=True),
    pa.field("feature_boundary_token_idx", pa.int64(), nullable=True),
    pa.field("available", pa.bool_(), nullable=False),
    pa.field("unavailable_reason", pa.string(), nullable=True),
    pa.field("y_true", pa.int64(), nullable=False),
    pa.field("importance_true", pa.float64(), nullable=False),
    pa.field("score_raw", pa.float64(), nullable=True),
    pa.field("probability", pa.float64(), nullable=True),
])


# --------------------------------------------------------------------------- Stage 9 handoff


def load_stage9_run(root: Path, repo: Path, config: ExperimentConfig, stage9_run: Path) -> dict:
    """The finalized Stage 9 run, verified read-only: freeze intact, test evaluated exactly
    once with a receipt, every output hash equal to the manifest. Nothing is written."""
    manifest = load_manifest(stage9_run)
    frozen, freeze_sha, handoff = verify_freeze(stage9_run, repo, config, manifest, None)
    state = test_state(stage9_run, manifest)
    if not state["receipt"] or state["state"] != "complete":
        raise ValueError("Stage 9 held-out test is not complete; Stage 10 needs the receipt")
    receipt = read_json(stage9_run / RECEIPT_FILE)
    if receipt["freeze_sha256"] != freeze_sha:
        raise ValueError("Test receipt names a different freeze")
    hashes = {}
    for name in STAGE9_OUTPUTS:
        digest = validate_artifact(stage9_run / name)
        if manifest["output_hashes"].get(name) != digest:
            raise ValueError(f"Stage 9 {name} differs from its manifest")
        hashes[name] = digest
    if hashes[FREEZE_FILE] != freeze_sha:
        raise ValueError("Freeze hash mismatch")
    if (receipt["predictions_sha256"], receipt["metrics_sha256"]) != (
            hashes["predictions/test.parquet"], hashes["metrics/test.json"]):
        raise ValueError("Stage 9 test outputs differ from the test receipt")
    if frozen["features"]["lead_sentences"] != 0 or frozen["threshold_quantile"] != \
            THRESHOLD_QUANTILE:
        raise ValueError("Frozen primary analysis is not lead zero at q90")
    payload = handoff["payload"]
    for family in REFIT_FAMILIES:
        if f"{family}:{M.BINARY}" not in payload["models"]:
            raise ValueError(f"Frozen selection has no {family} binary model")
    if payload["models"][f"{PRIMARY_MODEL}:{M.BINARY}"]["layer"] != frozen["binary"]["layer"] or \
            payload["models"][f"{PRIMARY_MODEL}:{M.BINARY}"]["hyperparameter"] != \
            frozen["binary"]["C"]:
        raise ValueError("Serialized primary model differs from the freeze")
    input_hashes = {**manifest["input_hashes"], "stage9_run_id": stage9_run.name,
                    "stage9/freeze_sha256": freeze_sha,
                    **{f"stage9/{n}": h for n, h in hashes.items()}}
    return {"manifest": manifest, "frozen": frozen, "freeze_sha256": freeze_sha,
            "handoff": handoff, "receipt": receipt, "hashes": hashes,
            "input_hashes": input_hashes, "run": stage9_run}


def check_spec(config: ExperimentConfig) -> dict:
    """The prespecified Stage 10 design, read from the frozen configuration."""
    e = config.evaluation
    spec = {"threshold_quantiles": [float(q) for q in e["threshold_sensitivities"]],
            "convergence": [int(k) for k in e["convergence_sensitivities"]],
            "replacement_minimum_fraction": float(e["replacement_minimum_fraction"]),
            "replacement_minimum_count": int(e["replacement_minimum_count"]),
            "trajectory_subgroups": list(e["trajectory_subgroups"]),
            "position_quintiles": int(e["position_quintiles"]),
            "entropy_quintiles": int(e["entropy_quintiles"]),
            "permutation_count": int(e["permutation_count"]),
            "permutation_strata": list(e["permutation_strata"]),
            "leads": [int(x) for x in config.features["leads"]]}
    if THRESHOLD_QUANTILE not in spec["threshold_quantiles"]:
        raise ValueError("Threshold sensitivities must include the frozen q90")
    if sorted(spec["convergence"]) != [4, 5]:
        raise ValueError("Convergence sensitivities must be K=4 and K=5")
    if spec["permutation_strata"] != ["problem_id", "trace_kind", "position_quintile"]:
        raise ValueError("Permutation strata differ from the prespecified design")
    if spec["leads"] != [0, 1]:
        raise ValueError("Lead times must be exactly 0 and 1")
    if spec["trajectory_subgroups"] != ["correct", "incorrect"]:
        raise ValueError("Trajectory subgroups differ from the prespecified design")
    if spec["position_quintiles"] != 5 or spec["entropy_quintiles"] != 5:
        raise ValueError("Position and entropy stratification must use five strata")
    if spec["permutation_count"] < 1:
        raise ValueError("At least one permutation is required")
    return spec


def prepare_run(root: Path, repo: Path, config: ExperimentConfig, stage9_run: Path) -> Path:
    assert_disjoint(config)
    M.check_frozen(config)
    check_spec(config)
    stage9 = load_stage9_run(root, repo, config, stage9_run)
    run = create_run(root, repo, config, ["run-sensitivities"])
    manifest = load_manifest(run)
    manifest["input_hashes"] = dict(stage9["input_hashes"])
    manifest["identity"]["input_hashes"] = dict(stage9["input_hashes"])
    manifest["gpu_gate"] = "cpu_only_stage"
    manifest["stages"][RUN_STAGE] = {"state": "prepared", "stage9_run_id": stage9_run.name,
                                     "freeze_sha256": stage9["freeze_sha256"]}
    save_manifest(run, manifest)
    return run


def discover_stage9(root: Path) -> Path:
    """The single run with a complete, receipted Stage 9 test; ambiguity fails closed."""
    found = []
    for directory in sorted((root / "artifacts/runs").iterdir()):
        if not (directory / "manifest.json").is_file():
            continue
        try:
            manifest = load_manifest(directory)
        except (ValueError, OSError, KeyError):
            continue
        stage = manifest.get("stages", {}).get("stage9_test", {})
        if stage.get("state") == "complete" and (directory / RECEIPT_FILE).is_file():
            found.append(directory)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one test-evaluated Stage 9 run, found "
                         f"{[d.name for d in found]}; pass --stage9-run-id")
    return found[0]


# --------------------------------------------------------------------------- rows


@dataclass
class Rows:
    """Eligible rows of one split in ``Partition`` order plus the Stage 10 stratifiers."""

    partition: Partition
    trace_kind: np.ndarray
    sentence_idx: np.ndarray
    boundary_token_idx: np.ndarray
    target_token_start: np.ndarray
    entropy: np.ndarray
    eligible_k4: np.ndarray
    dissimilar_count: np.ndarray
    sample_count: np.ndarray

    def __len__(self) -> int:
        return len(self.partition)


def take(partition: Partition, idx: np.ndarray) -> Partition:
    idx = np.asarray(idx)
    return Partition(partition.split, [partition.sentence_id[i] for i in idx],
                     [partition.trajectory_id[i] for i in idx], partition.problem_id[idx],
                     partition.y[idx], partition.importance_log1p[idx],
                     partition.importance_primary[idx], partition.observables[idx],
                     [partition.shard_path[i] for i in idx], partition.row_idx[idx],
                     partition.anchor_threshold)


def relabel(partition: Partition, threshold: float) -> Partition:
    """Anchor = importance_primary >= threshold (the frozen comparison), nothing else changes."""
    y = (partition.importance_primary >= threshold).astype(int)
    return replace(partition, y=y, anchor_threshold=float(threshold))


def load_rows(root: Path, config: ExperimentConfig, table, labels: dict, split: str) -> Rows:
    partition = partition_rows(table, config, split)
    by_id = {r["sentence_id"]: r for r in table.to_pylist()}
    rows = [by_id[s] for s in partition.sentence_id]
    for r, s in zip(rows, partition.sentence_id, strict=True):
        if r["sentence_id"] != s or s not in labels:
            raise ValueError("Feature row does not align with the Stage 4 label row")
    return Rows(partition, np.array([r["trace_kind"] for r in rows]),
                np.array([int(r["sentence_idx"]) for r in rows]),
                np.array([int(r["boundary_token_idx"]) for r in rows]),
                np.array([int(r["target_token_start"]) for r in rows]),
                np.array([float(r["next_token_entropy"]) for r in rows]),
                np.array([bool(r["eligible_k4"]) for r in rows]),
                np.array([int(labels[s]["dissimilar_count"]) for s in partition.sentence_id]),
                np.array([int(labels[s]["sample_count"]) for s in partition.sentence_id]))


def load_labels(root: Path, expected_sha: str) -> dict:
    path = root / "data/processed" / LABELS_FILE
    if file_hash(path) != expected_sha:
        raise ValueError("Stage 4 labels differ from the frozen inputs")
    table = validate_parquet(path)
    return {r["sentence_id"]: r for r in table.to_pylist()}


def train_thresholds(train: Partition, config: ExperimentConfig, quantiles: list[float]) -> dict:
    """Every alternate threshold from eligible training importance only (SPEC §3.7)."""
    rows = [{"importance_primary": float(v), "problem_id": int(p), "eligible": True}
            for v, p in zip(train.importance_primary, train.problem_id, strict=True)]
    problems = set(int(p) for p in config.splits.train)
    if set(int(p) for p in train.problem_id) - problems:
        raise ValueError("Threshold rows contain a non-training problem")
    return {f"q{int(round(q * 100))}": train_threshold(rows, problems, q) for q in quantiles}


def quintile_cuts(values: np.ndarray, bins: int) -> list[float]:
    """Interior cut points from training values (numpy linear quantiles)."""
    probs = [k / bins for k in range(1, bins)]
    return [float(v) for v in np.quantile(np.asarray(values, dtype=np.float64), probs)]


def assign_quintile(values: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Deterministic bin 1..len(cuts)+1: bin k holds values in (cut_{k-1}, cut_k]."""
    return np.searchsorted(np.asarray(cuts, dtype=np.float64), np.asarray(values, dtype=np.float64),
                           side="left") + 1


def replacement_support(rows: Rows, spec: dict) -> tuple[np.ndarray, dict]:
    """SPEC §3.12 item 8: dissimilar fraction >= 0.10 OR >= 10 dissimilar released rollouts."""
    if (rows.sample_count <= 0).any():
        raise ValueError("Replacement support needs a positive sample_count for every row")
    fraction = rows.dissimilar_count / rows.sample_count
    by_fraction = fraction >= spec["replacement_minimum_fraction"]
    by_count = rows.dissimilar_count >= spec["replacement_minimum_count"]
    supported = by_fraction | by_count
    reasons = {"supported": int(supported.sum()),
               "excluded": int((~supported).sum()),
               "excluded_fraction_below_minimum": int((~by_fraction).sum()),
               "excluded_count_below_minimum": int((~by_count).sum()),
               "excluded_both_below_minimum": int((~by_fraction & ~by_count).sum()),
               "rule": f"dissimilar_count / sample_count >= {spec['replacement_minimum_fraction']}"
                       f" OR dissimilar_count >= {spec['replacement_minimum_count']}"}
    return supported, reasons


# --------------------------------------------------------------------------- lead one


def lead_one_source(rows: Rows, table_by_id: dict) -> tuple[np.ndarray, Partition, list[dict]]:
    """For every target S_i, the stored row of S_{i-1} in the same trajectory.

    Returns ``(available, source_partition_over_available_rows, mapping)``. The source
    boundary is ``target_token_start(S_{i-1}) - 1``: strictly before every token of S_{i-1}
    and of S_i. A target with ``sentence_idx == 0`` has no S_{i-1} and is unavailable; it is
    never shifted to another boundary.
    """
    p = rows.partition
    available = np.zeros(len(p), dtype=bool)
    mapping, source_rows = [], []
    for k in range(len(p)):
        traj, idx = p.trajectory_id[k], int(rows.sentence_idx[k])
        record = {"sentence_id": p.sentence_id[k], "trajectory_id": traj,
                  "target_sentence_idx": idx, "feature_sentence_id": None,
                  "feature_sentence_idx": None, "feature_boundary_token_idx": None,
                  "available": False, "unavailable_reason": None}
        if idx == 0:
            record["unavailable_reason"] = "first_sentence_has_no_predecessor"
            mapping.append(record)
            continue
        source_id = f"{traj}:{idx - 1}"
        src = table_by_id.get(source_id)
        if src is None or src["trajectory_id"] != traj or int(src["sentence_idx"]) != idx - 1:
            record["unavailable_reason"] = "predecessor_row_missing"
            mapping.append(record)
            continue
        if not (int(src["boundary_token_idx"]) < int(src["target_token_start"])
                <= int(rows.target_token_start[k])):
            raise ValueError(f"Lead-one boundary for {p.sentence_id[k]} does not precede "
                             "S_(i-1)")
        if int(src["boundary_token_idx"]) >= int(rows.boundary_token_idx[k]):
            raise ValueError(f"Lead-one boundary for {p.sentence_id[k]} is not earlier "
                             "than lead zero")
        record.update({"feature_sentence_id": source_id, "feature_sentence_idx": idx - 1,
                       "feature_boundary_token_idx": int(src["boundary_token_idx"]),
                       "available": True})
        available[k] = True
        mapping.append(record)
        source_rows.append((k, src))
    idx = np.array([k for k, _ in source_rows], dtype=int)
    obs = np.array([[float(src[c]) for c in PRIMARY_FEATURES] for _, src in source_rows],
                   dtype=np.float64).reshape(len(source_rows), len(PRIMARY_FEATURES))
    if len(source_rows) and not np.isfinite(obs).all():
        raise ValueError("Non-finite observable at a lead-one source boundary")
    source = Partition(p.split, [src["sentence_id"] for _, src in source_rows],
                       [src["trajectory_id"] for _, src in source_rows],
                       p.problem_id[idx] if len(idx) else np.array([], dtype=int),
                       p.y[idx] if len(idx) else np.array([], dtype=int),
                       p.importance_log1p[idx] if len(idx) else np.array([]),
                       p.importance_primary[idx] if len(idx) else np.array([]),
                       obs, [src["shard_path"] for _, src in source_rows],
                       np.array([int(src["row_idx"]) for _, src in source_rows], dtype=int),
                       p.anchor_threshold)
    return available, source, mapping


# --------------------------------------------------------------------------- scoring


def frozen_scores(stage9: dict, split: str, partition: Partition) -> dict[str, dict]:
    """Binary Stage 9 predictions of every family aligned to ``partition`` order."""
    path = stage9["run"] / "predictions" / f"{split}.parquet"
    if file_hash(path) != stage9["hashes"][f"predictions/{split}.parquet"]:
        raise ValueError(f"Stage 9 {split} predictions changed")
    table = validate_parquet(path)
    by_model: dict[str, dict[str, tuple]] = {}
    for r in table.to_pylist():
        if r["task"] != M.BINARY:
            continue
        by_model.setdefault(r["probe_name"], {})[r["sentence_id"]] = (
            r["score_raw"], r["probability"], r["y_true"], r["layer"], r["hyperparameter"])
    out = {}
    for family, rows in by_model.items():
        score = np.array([rows[s][0] for s in partition.sentence_id], dtype=np.float64)
        prob = np.array([rows[s][1] for s in partition.sentence_id], dtype=np.float64)
        layers = {rows[s][3] for s in partition.sentence_id}
        hps = {rows[s][4] for s in partition.sentence_id}
        if len(layers) != 1 or len(hps) != 1:
            raise ValueError("Frozen predictions are not one configuration per family")
        out[family] = {"score": score, "probability": prob, "layer": layers.pop(),
                       "hyperparameter": hps.pop()}
    if PRIMARY_MODEL not in out or BASELINE not in out:
        raise ValueError("Frozen predictions lack the primary model or the combined baseline")
    return out


def refit_family(family: str, meta: dict, train: Partition, site: np.ndarray | None,
                 config: ExperimentConfig, y: np.ndarray | None = None):
    """One sensitivity model at the frozen site and hyperparameter (no selection)."""
    X = M.design(family, train, site)
    target = train.y if y is None else np.asarray(y, dtype=int)
    if len(np.unique(target)) < 2:
        raise ValueError(f"Cannot refit {family}: training labels have a single class")
    pipeline = M.make_pipeline(family, M.BINARY, meta["hyperparameter"], config)
    pipeline.fit(X, target)
    return pipeline


def family_meta(stage9: dict, family: str) -> dict:
    record = stage9["handoff"]["payload"]["models"][f"{family}:{M.BINARY}"]
    return {"layer": record["layer"], "hyperparameter": record["hyperparameter"]}


# --------------------------------------------------------------------------- metric rows


def _num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def evaluate_group(*, analysis: str, variant: str, split: str, source: str,
                   partition: Partition, binary: dict[str, dict], meta: dict[str, dict],
                   config: ExperimentConfig, lead: int = 0, detail: dict | None = None,
                   bootstrap: bool = True) -> list[dict]:
    """Long-format rows for one (analysis, variant, split, source): every binary metric per
    model, primary-minus-other paired differences, clustered bootstrap intervals. An empty
    or single-class group stays explicit (``value`` null, reason in ``detail``)."""
    base = {"schema_version": 1, "analysis": analysis, "variant": variant, "split": split,
            "model_source": source, "lead_sentences": lead,
            "threshold_quantile": (detail or {}).get("threshold_quantile", THRESHOLD_QUANTILE),
            "anchor_threshold": float(partition.anchor_threshold),
            "n_rows": int(len(partition)), "n_positive": int(partition.y.sum()) if len(partition)
            else 0, "n_problems": int(len(set(int(p) for p in partition.problem_id))),
            "n_trajectories": int(len(set(partition.trajectory_id)))}
    note = json.dumps(detail or {}, sort_keys=True)
    if len(partition) == 0:
        return [{**base, "probe_name": "none", "task": M.BINARY, "layer": None,
                 "hyperparameter": None, "metric": "empty_group", "value": None,
                 "boot_low": None, "boot_high": None, "boot_mean": None, "boot_n_valid": None,
                 "comparison": None,
                 "detail": json.dumps({**(detail or {}), "undefined": "empty_group"},
                                      sort_keys=True)}]
    two = len(np.unique(partition.y)) == 2
    bins = int(config.evaluation["ece_bins"])
    boot = None
    if two and bootstrap:
        boot = clustered_bootstrap(
            problem_id=partition.problem_id, trajectory_id=partition.trajectory_id,
            y=partition.y, importance=partition.importance_log1p,
            binary={n: {"score": m["score"], "probability": m["probability"]}
                    for n, m in binary.items()},
            continuous={}, primary=PRIMARY_MODEL, baseline=BASELINE, config=config, split=split)
    rows = []
    point = {}
    for family in sorted(binary):
        m = binary[family]
        metrics = binary_metrics(partition.y, m["score"], m["probability"],
                                 partition.trajectory_id, bins)
        point[family] = metrics
        for metric in POINT_METRICS:
            value = metrics.get(metric)
            interval = (boot or {}).get("binary", {}).get(family, {}).get(metric)
            undefined = None if (value is not None or metric in ("brier", "log_loss", "ece")) \
                else "single_class"
            rows.append({**base, "probe_name": family, "task": M.BINARY,
                         "layer": meta[family]["layer"],
                         "hyperparameter": meta[family]["hyperparameter"], "metric": metric,
                         "value": _num(value),
                         "boot_low": interval["low"] if interval else None,
                         "boot_high": interval["high"] if interval else None,
                         "boot_mean": interval["mean"] if interval else None,
                         "boot_n_valid": interval["n_valid"] if interval else None,
                         "comparison": None,
                         "detail": note if undefined is None else json.dumps(
                             {**(detail or {}), "undefined": undefined}, sort_keys=True)})
        rows.append({**base, "probe_name": family, "task": M.BINARY,
                     "layer": meta[family]["layer"],
                     "hyperparameter": meta[family]["hyperparameter"], "metric": "prevalence",
                     "value": float(partition.y.mean()), "boot_low": None, "boot_high": None,
                     "boot_mean": None, "boot_n_valid": None, "comparison": None,
                     "detail": note})
    for family in sorted(binary):
        if family == PRIMARY_MODEL:
            continue
        for metric in PAIRED_METRICS:
            a, b = point[PRIMARY_MODEL].get(metric), point[family].get(metric)
            interval = ((boot or {}).get("paired_differences", {})
                        .get(f"{PRIMARY_MODEL}-minus-{family}", {}).get(metric))
            rows.append({**base, "probe_name": PRIMARY_MODEL, "task": M.BINARY,
                         "layer": meta[PRIMARY_MODEL]["layer"],
                         "hyperparameter": meta[PRIMARY_MODEL]["hyperparameter"],
                         "metric": metric,
                         "value": None if a is None or b is None else float(a - b),
                         "boot_low": interval["low"] if interval else None,
                         "boot_high": interval["high"] if interval else None,
                         "boot_mean": interval["mean"] if interval else None,
                         "boot_n_valid": interval["n_valid"] if interval else None,
                         "comparison": f"minus:{family}",
                         "detail": note if two else json.dumps(
                             {**(detail or {}), "undefined": "single_class"}, sort_keys=True)})
    return rows


def rows_table(rows: list[dict]) -> pa.Table:
    rows = sorted(rows, key=lambda r: (r["analysis"], r["variant"], r["split"],
                                       r["model_source"], r["probe_name"], r["metric"],
                                       r["comparison"] or ""))
    return pa.Table.from_pylist(rows, schema=SENSITIVITY_SCHEMA)


# --------------------------------------------------------------------------- context


class Context:
    """Everything the analyses share: partitions, strata, frozen scores, one site."""

    def __init__(self, root: Path, config: ExperimentConfig, stage9: dict, spec: dict):
        self.root, self.config, self.stage9, self.spec = root, config, stage9, spec
        handoff = stage9["handoff"]
        self.activation_run = handoff["activation_run"]
        self.table = handoff["table"]
        self.table_by_id = {r["sentence_id"]: r for r in self.table.to_pylist()}
        labels = load_labels(root, stage9["manifest"]["input_hashes"][LABELS_FILE])
        self.rows = {s: load_rows(root, config, self.table, labels, s)
                     for s in ("train", VALIDATION, TEST)}
        train = self.rows["train"].partition
        frozen = stage9["frozen"]
        if train.anchor_threshold != frozen["anchor_threshold"]:
            raise ValueError("Training anchor threshold differs from the freeze")
        self.thresholds = train_thresholds(train, config, spec["threshold_quantiles"])
        q90 = self.thresholds[f"q{int(round(THRESHOLD_QUANTILE * 100))}"]
        if q90 != frozen["anchor_threshold"]:
            raise ValueError(f"Recomputed q90 {q90!r} differs from the frozen threshold "
                             f"{frozen['anchor_threshold']!r}")
        self.layer = int(frozen["binary"]["layer"])
        self.meta = {f: family_meta(stage9, f) for f in REFIT_FAMILIES}
        self.position_cuts = quintile_cuts(self.rows["train"].sentence_idx,
                                           spec["position_quintiles"])
        self.entropy_cuts = quintile_cuts(self.rows["train"].entropy, spec["entropy_quintiles"])
        self.position_quintile = {s: assign_quintile(r.sentence_idx, self.position_cuts)
                                  for s, r in self.rows.items()}
        self.entropy_quintile = {s: assign_quintile(r.entropy, self.entropy_cuts)
                                 for s, r in self.rows.items()}
        self.frozen = {s: frozen_scores(stage9, s, self.rows[s].partition) for s in EVAL_SPLITS}
        self.frozen_meta = {s: {f: {"layer": m["layer"], "hyperparameter": m["hyperparameter"]}
                                for f, m in self.frozen[s].items()} for s in EVAL_SPLITS}
        for s in EVAL_SPLITS:
            if self.frozen_meta[s][PRIMARY_MODEL] != self.meta[PRIMARY_MODEL]:
                raise ValueError("Frozen predictions were not produced by the frozen primary")
        self._sites: dict[tuple[str, int], np.ndarray] = {}
        self._lead: dict[str, tuple] = {}
        self._lead_sites: dict[tuple[str, int], np.ndarray] = {}

    def site(self, split: str, layer: int | None = None) -> np.ndarray:
        """The frozen site of ``layer`` (default: the primary model's frozen layer) for the
        eligible rows of ``split`` (float32). Cached per (split, layer): each activation
        family must be fitted and scored at its own frozen layer, not the primary's."""
        key = (split, self.layer if layer is None else int(layer))
        if key not in self._sites:
            loader = ActivationLoader(self.activation_run, self.rows[split].partition, self.config)
            self._sites[key] = loader.site(key[1])
            del loader
        return self._sites[key]

    def lead(self, split: str, layer: int | None = None) \
            -> tuple[np.ndarray, Partition, list[dict], np.ndarray]:
        """``(available, source_partition, mapping, source_site)`` for lead one, with the
        source site at ``layer`` (default: the primary model's frozen layer)."""
        if split not in self._lead:
            self._lead[split] = lead_one_source(self.rows[split], self.table_by_id)
        available, source, mapping = self._lead[split]
        key = (split, self.layer if layer is None else int(layer))
        if key not in self._lead_sites:
            loader = ActivationLoader(self.activation_run, source, self.config)
            self._lead_sites[key] = loader.site(key[1])
            del loader
        return available, source, mapping, self._lead_sites[key]

    def scores(self, family: str, pipeline, split: str, partition: Partition,
               site: np.ndarray | None) -> dict:
        out = M.scores(pipeline, M.BINARY, M.design(family, partition, site))
        return {"score": out["score_raw"], "probability": out["probability"]}


# --------------------------------------------------------------------------- analyses


def _refit_all(ctx: Context, train: Partition, site_for, y=None) -> dict:
    """``site_for(layer)`` returns the training activation matrix at that layer, so each
    activation family is refitted at its own frozen layer."""
    fitted = {}
    for family in REFIT_FAMILIES:
        site = site_for(ctx.meta[family]["layer"]) if M.uses_activation(family) else None
        fitted[family] = refit_family(family, ctx.meta[family], train, site, ctx.config, y)
    return fitted


def _score_all(ctx: Context, fitted: dict, split: str, partition: Partition,
               site_for) -> dict:
    return {family: ctx.scores(family, pipeline, split, partition,
                               site_for(ctx.meta[family]["layer"])
                               if M.uses_activation(family) else None)
            for family, pipeline in fitted.items()}


def analysis_threshold(ctx: Context, variant: str) -> list[dict]:
    q = float(variant[1:]) / 100.0
    threshold = ctx.thresholds[variant]
    train = relabel(ctx.rows["train"].partition, threshold)
    detail = {"threshold_quantile": q, "threshold": threshold,
              "threshold_source": "train_eligible_importance_quantile",
              "quantile_method": QUANTILE_METHOD,
              "train_n_rows": int(len(train)), "train_n_positive": int(train.y.sum()),
              "train_prevalence": float(train.y.mean()),
              "equals_frozen_primary": bool(threshold == ctx.stage9["frozen"]["anchor_threshold"])}
    fitted = _refit_all(ctx, train, lambda ly: ctx.site("train", ly))
    rows = []
    for split in EVAL_SPLITS:
        part = relabel(ctx.rows[split].partition, threshold)
        rows += evaluate_group(analysis="threshold", variant=variant, split=split, source=FROZEN,
                               partition=part, binary=ctx.frozen[split],
                               meta=ctx.frozen_meta[split], config=ctx.config, detail=detail)
        rows += evaluate_group(analysis="threshold", variant=variant, split=split, source=REFIT,
                               partition=part,
                               binary=_score_all(ctx, fitted, split, part,
                                                 lambda ly, s=split: ctx.site(s, ly)),
                               meta=ctx.meta, config=ctx.config, detail=detail)
    return rows


def analysis_convergence(ctx: Context, variant: str) -> list[dict]:
    k = int(variant[1:])
    masks = {s: (r.eligible_k4 if k == 4 else np.ones(len(r), dtype=bool))
             for s, r in ctx.rows.items()}
    detail = {"convergence_consecutive": k,
              "primary_rule": int(ctx.config.labels["convergence_consecutive"]),
              "eligible_rows": {s: int(m.sum()) for s, m in masks.items()},
              "rows_lost_vs_k5": {s: int(len(m) - m.sum()) for s, m in masks.items()},
              "rows_gained_vs_k5": {s: 0 for s in masks},
              "anchor_threshold": "frozen_q90_unchanged"}
    train = take(ctx.rows["train"].partition, np.flatnonzero(masks["train"]))
    fitted = _refit_all(ctx, train,
                        lambda ly: ctx.site("train", ly)[masks["train"]])
    rows = []
    for split in EVAL_SPLITS:
        idx = np.flatnonzero(masks[split])
        part = take(ctx.rows[split].partition, idx)
        rows += evaluate_group(analysis="convergence", variant=variant, split=split,
                               source=FROZEN, partition=part,
                               binary={f: {**m, "score": m["score"][idx],
                                           "probability": m["probability"][idx]}
                                       for f, m in ctx.frozen[split].items()},
                               meta=ctx.frozen_meta[split], config=ctx.config, detail=detail)
        rows += evaluate_group(analysis="convergence", variant=variant, split=split,
                               source=REFIT, partition=part,
                               binary=_score_all(ctx, fitted, split, part,
                                                 lambda ly, s=split, i=idx: ctx.site(s, ly)[i]),
                               meta=ctx.meta, config=ctx.config, detail=detail)
    return rows


def _subset_frozen(ctx: Context, split: str, idx: np.ndarray) -> dict:
    return {f: {**m, "score": m["score"][idx], "probability": m["probability"][idx]}
            for f, m in ctx.frozen[split].items()}


def analysis_replacement(ctx: Context, variant: str) -> list[dict]:
    supported = {s: replacement_support(r, ctx.spec) for s, r in ctx.rows.items()}
    keep = variant == "supported"
    detail = {"filter": supported["train"][1]["rule"], "variant": variant,
              "counts": {s: v[1] for s, v in supported.items()}}
    rows = []
    fitted = None
    if keep:
        mask = supported["train"][0]
        train = take(ctx.rows["train"].partition, np.flatnonzero(mask))
        fitted = _refit_all(ctx, train, lambda ly, m=mask: ctx.site("train", ly)[m])
    for split in EVAL_SPLITS:
        mask = supported[split][0] if keep else ~supported[split][0]
        idx = np.flatnonzero(mask)
        part = take(ctx.rows[split].partition, idx)
        rows += evaluate_group(analysis="replacement_quality", variant=variant, split=split,
                               source=FROZEN, partition=part,
                               binary=_subset_frozen(ctx, split, idx),
                               meta=ctx.frozen_meta[split], config=ctx.config, detail=detail)
        if fitted is not None:
            rows += evaluate_group(analysis="replacement_quality", variant=variant, split=split,
                                   source=REFIT, partition=part,
                                   binary=_score_all(ctx, fitted, split, part,
                                                     lambda ly, s=split, i=idx:
                                                     ctx.site(s, ly)[i]),
                                   meta=ctx.meta, config=ctx.config, detail=detail)
    return rows


def analysis_subgroup(ctx: Context, variant: str) -> list[dict]:
    rows = []
    for split in EVAL_SPLITS:
        r = ctx.rows[split]
        idx = np.flatnonzero(r.trace_kind == variant)
        part = take(r.partition, idx)
        detail = {"trace_kind": variant, "n_trajectories": int(len(set(part.trajectory_id))),
                  "n_problems": int(len(set(int(p) for p in part.problem_id))),
                  "note": "eventual correctness is a diagnostic subgroup variable only"}
        rows += evaluate_group(analysis="trajectory_subgroup", variant=variant, split=split,
                               source=FROZEN, partition=part,
                               binary=_subset_frozen(ctx, split, idx),
                               meta=ctx.frozen_meta[split], config=ctx.config, detail=detail)
    return rows


def analysis_quintile(ctx: Context, which: str, variant: str) -> list[dict]:
    q = int(variant)
    cuts = ctx.position_cuts if which == "position" else ctx.entropy_cuts
    quintiles = ctx.position_quintile if which == "position" else ctx.entropy_quintile
    column = "sentence_idx" if which == "position" else "next_token_entropy"
    rows = []
    for split in EVAL_SPLITS:
        idx = np.flatnonzero(quintiles[split] == q)
        part = take(ctx.rows[split].partition, idx)
        detail = {"quintile": q, "column": column, "cut_points": cuts,
                  "cut_source": "eligible training rows, numpy linear quantiles 0.2/0.4/0.6/0.8",
                  "bin_rule": "bin k holds values in (cut_{k-1}, cut_k]; bin 1 is <= cut_1",
                  "train_rows_in_bin": int((quintiles["train"] == q).sum())}
        rows += evaluate_group(analysis=f"{which}_quintile", variant=variant, split=split,
                               source=FROZEN, partition=part,
                               binary=_subset_frozen(ctx, split, idx),
                               meta=ctx.frozen_meta[split], config=ctx.config, detail=detail)
    return rows


def analysis_lead_time(ctx: Context) -> tuple[list[dict], pa.Table, dict]:
    """Lead one at the frozen site: refit on lead-one training rows, evaluate on the mappable
    validation/test rows, paired with the frozen lead-zero scores on the same rows."""
    avail_tr, source_tr, mapping_tr, _ = ctx.lead("train")
    target_tr = take(ctx.rows["train"].partition, np.flatnonzero(avail_tr))
    fitted = _refit_all(ctx, source_tr, lambda ly: ctx.lead("train", ly)[3], y=target_tr.y)
    rows, preds = [], []
    summary = {"layer": ctx.layer, "families": list(REFIT_FAMILIES), "unavailable": {},
               "available": {}, "train_rows": int(avail_tr.sum()),
               "train_unavailable": int((~avail_tr).sum())}
    for split in EVAL_SPLITS:
        available, source, mapping, _ = ctx.lead(split)
        idx = np.flatnonzero(available)
        target = take(ctx.rows[split].partition, idx)
        if source.sentence_id != [m["feature_sentence_id"] for m in mapping if m["available"]]:
            raise ValueError("Lead-one source rows do not follow the mapping")
        summary["unavailable"][split] = int((~available).sum())
        summary["available"][split] = int(available.sum())
        detail = {"lead_sentences": 1, "rule": "features from the boundary immediately before "
                  "S_(i-1); label of S_i", "n_unavailable": int((~available).sum()),
                  "unavailable_reasons": sorted({m["unavailable_reason"] for m in mapping
                                                 if not m["available"]})}
        scored = _score_all(ctx, fitted, split, source, lambda ly, s=split: ctx.lead(s, ly)[3])
        rows += evaluate_group(analysis="lead_time", variant="lead1_refit", split=split,
                               source=REFIT, partition=target, binary=scored, meta=ctx.meta,
                               config=ctx.config, lead=1, detail=detail)
        rows += evaluate_group(analysis="lead_time", variant="lead0_frozen_same_rows",
                               split=split, source=FROZEN, partition=target,
                               binary=_subset_frozen(ctx, split, idx), meta=ctx.frozen_meta[split],
                               config=ctx.config, lead=0,
                               detail={**detail, "lead_sentences": 0,
                                       "rule": "frozen Stage 9 lead-zero scores restricted to "
                                               "the lead-one-mappable rows"})
        full = ctx.rows[split].partition
        frozen = ctx.frozen[split][PRIMARY_MODEL]
        for k, m in enumerate(mapping):
            common = {"schema_version": 1, "sentence_id": full.sentence_id[k],
                      "problem_id": int(full.problem_id[k]),
                      "trajectory_id": full.trajectory_id[k], "split": split,
                      "threshold_quantile": THRESHOLD_QUANTILE, "task": M.BINARY,
                      "target_sentence_idx": m["target_sentence_idx"],
                      "y_true": int(full.y[k]), "importance_true": float(full.importance_log1p[k])}
            preds.append({**common, "lead_sentences": 0, "layer": ctx.meta[PRIMARY_MODEL]["layer"],
                          "probe_name": PRIMARY_MODEL,
                          "hyperparameter": ctx.meta[PRIMARY_MODEL]["hyperparameter"],
                          "model_source": FROZEN, "feature_sentence_id": full.sentence_id[k],
                          "feature_sentence_idx": m["target_sentence_idx"],
                          "feature_boundary_token_idx": int(ctx.rows[split].boundary_token_idx[k]),
                          "available": True, "unavailable_reason": None,
                          "score_raw": float(frozen["score"][k]),
                          "probability": float(frozen["probability"][k])})
            for family in REFIT_FAMILIES:
                pos = int(np.searchsorted(idx, k)) if m["available"] else None
                preds.append({**common, "lead_sentences": 1, "layer": ctx.meta[family]["layer"],
                              "probe_name": family,
                              "hyperparameter": ctx.meta[family]["hyperparameter"],
                              "model_source": REFIT,
                              "feature_sentence_id": m["feature_sentence_id"],
                              "feature_sentence_idx": m["feature_sentence_idx"],
                              "feature_boundary_token_idx": m["feature_boundary_token_idx"],
                              "available": m["available"],
                              "unavailable_reason": m["unavailable_reason"],
                              "score_raw": (float(scored[family]["score"][pos])
                                            if m["available"] else None),
                              "probability": (float(scored[family]["probability"][pos])
                                              if m["available"] else None)})
    preds.sort(key=lambda r: (r["split"], r["lead_sentences"], r["probe_name"],
                              r["trajectory_id"], r["target_sentence_idx"]))
    return rows, pa.Table.from_pylist(preds, schema=LEAD_SCHEMA), summary


def strata_of(problem_id: np.ndarray, trace_kind: np.ndarray, quintile: np.ndarray) -> np.ndarray:
    keys = [f"{int(p)}:{k}:{int(q)}" for p, k, q in zip(problem_id, trace_kind, quintile,
                                                       strict=True)]
    return np.array(keys)


def permute_within_strata(y: np.ndarray, strata: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Shuffle labels inside every stratum that holds both classes; single-class strata are
    left exactly as they are (nothing crosses a stratum)."""
    y = np.asarray(y, dtype=int)
    out = y.copy()
    counts = {"n_strata": 0, "n_permutable_strata": 0, "n_single_class_strata": 0,
              "n_rows_permuted": 0, "n_rows_fixed": 0}
    for key in sorted(set(strata.tolist())):
        idx = np.flatnonzero(strata == key)
        counts["n_strata"] += 1
        if len(np.unique(y[idx])) == 2:
            counts["n_permutable_strata"] += 1
            counts["n_rows_permuted"] += int(len(idx))
            out[idx] = y[idx][rng.permutation(len(idx))]
        else:
            counts["n_single_class_strata"] += 1
            counts["n_rows_fixed"] += int(len(idx))
    return out, counts


def permutation_seed(config: ExperimentConfig, index: int) -> int:
    return derived_seed(config.seed_root, PERMUTATION_PURPOSE, index)


def _permutation_job(index: int, seed: int, X: np.ndarray, y: np.ndarray, strata: np.ndarray,
                     evals: dict, hyperparameter: float, config: ExperimentConfig) -> list[dict]:
    rng = np.random.default_rng(seed)
    y_perm, counts = permute_within_strata(y, strata, rng)
    pipeline = M.make_pipeline(PRIMARY_MODEL, M.BINARY, hyperparameter, config)
    pipeline.fit(X, y_perm)
    rows = []
    for split, (X_eval, y_eval) in evals.items():
        score = np.asarray(pipeline.decision_function(X_eval), dtype=np.float64)
        two = len(np.unique(y_eval)) == 2
        rows.append({"kind": "permutation", "permutation_index": index, "seed": seed,
                     "split": split, **counts, "labels_changed": int((y_perm != y).sum()),
                     "n_rows": int(len(y_eval)), "n_positive": int(y_eval.sum()),
                     "average_precision": _num(average_precision(y_eval, score)) if two else None,
                     "auroc": _num(auroc(y_eval, score)) if two else None,
                     "converged": M.converged(pipeline)})
    return rows


def analysis_permutations(ctx: Context, workers: int) -> tuple[pa.Table, dict]:
    spec = ctx.spec
    train = ctx.rows["train"]
    strata = strata_of(train.partition.problem_id, train.trace_kind,
                       ctx.position_quintile["train"])
    X = M.design(PRIMARY_MODEL, train.partition, ctx.site("train"))
    evals = {s: (M.design(PRIMARY_MODEL, ctx.rows[s].partition, ctx.site(s)),
                 ctx.rows[s].partition.y) for s in EVAL_SPLITS}
    meta = ctx.meta[PRIMARY_MODEL]
    common = {"schema_version": 1, "probe_name": PRIMARY_MODEL, "layer": int(meta["layer"]),
              "hyperparameter": float(meta["hyperparameter"]),
              "strata": ",".join(spec["permutation_strata"])}
    _, counts = permute_within_strata(train.partition.y, strata, np.random.default_rng(0))
    rows = []
    for split in EVAL_SPLITS:
        y_eval = ctx.rows[split].partition.y
        frozen = ctx.frozen[split][PRIMARY_MODEL]
        rows.append({**common, "kind": "observed", "permutation_index": -1, "seed": None,
                     "seed_purpose": None, "split": split, **counts, "labels_changed": 0,
                     "n_rows": int(len(y_eval)), "n_positive": int(y_eval.sum()),
                     "average_precision": _num(average_precision(y_eval, frozen["score"])),
                     "auroc": _num(auroc(y_eval, frozen["score"])), "converged": None})
    seeds = [permutation_seed(ctx.config, k) for k in range(spec["permutation_count"])]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Permutation seeds collide")
    results = Parallel(n_jobs=max(1, workers), backend="loky")(
        delayed(_permutation_job)(k, seeds[k], X, train.partition.y, strata, evals,
                                  float(meta["hyperparameter"]), ctx.config)
        for k in range(spec["permutation_count"]))
    for result in results:
        for r in result:
            rows.append({**common, "seed_purpose": f"derived_seed(seed_root, "
                         f"'{PERMUTATION_PURPOSE}', index)", **r})
    rows.sort(key=lambda r: (r["split"], r["permutation_index"]))
    table = pa.Table.from_pylist(rows, schema=PERMUTATION_SCHEMA)
    summary = {"requested": spec["permutation_count"], "strata": spec["permutation_strata"],
               "position_cut_points": ctx.position_cuts, **counts,
               "seed_root": ctx.config.seed_root,
               "seeds_sha256": config_hash(seeds), "splits": {}}
    for split in EVAL_SPLITS:
        obs = next(r for r in rows if r["kind"] == "observed" and r["split"] == split)
        null = [r for r in rows if r["kind"] == "permutation" and r["split"] == split]
        effective = [r for r in null if r["labels_changed"] > 0]
        values = np.array([r["average_precision"] for r in effective
                           if r["average_precision"] is not None], dtype=np.float64)
        summary["splits"][split] = {
            "observed_average_precision": obs["average_precision"],
            "observed_auroc": obs["auroc"],
            "effective_permutations": int(len(effective)),
            "null_defined": int(len(values)),
            "null_mean": _num(values.mean()) if len(values) else None,
            "null_sd": _num(values.std(ddof=1)) if len(values) > 1 else None,
            "null_p95": _num(np.percentile(values, 95)) if len(values) else None,
            "null_max": _num(values.max()) if len(values) else None,
            "fraction_null_at_or_above_observed": (
                _num(np.mean(values >= obs["average_precision"]))
                if len(values) and obs["average_precision"] is not None else None),
            "empirical_p_value_plus_one": (
                (1 + int((values >= obs["average_precision"]).sum())) / (len(values) + 1)
                if len(values) and obs["average_precision"] is not None else None),
            "descriptive_only": True}
    return table, summary


def analysis_paired(ctx: Context) -> list[dict]:
    """The registry's paired baseline differences, recomputed from the frozen predictions on
    the full validation/test partitions and checked against the Stage 9 metrics."""
    rows = []
    for split in EVAL_SPLITS:
        part = ctx.rows[split].partition
        out = evaluate_group(analysis="paired_baseline_differences", variant="all_eligible_rows",
                             split=split, source=FROZEN, partition=part, binary=ctx.frozen[split],
                             meta=ctx.frozen_meta[split], config=ctx.config,
                             detail={"registry": "paired_baseline_differences"})
        recorded = read_json(ctx.stage9["run"] / "metrics" / f"{split}.json")
        for r in out:
            if r["comparison"] and r["metric"] in PAIRED_METRICS:
                family = r["comparison"].split(":", 1)[1]
                expected = recorded["paired_differences_point"][
                    f"{PRIMARY_MODEL}-minus-{family}"][r["metric"]]
                if r["value"] is None or expected is None or abs(r["value"] - expected) > 1e-9:
                    raise ValueError(f"Paired difference for {family} differs from Stage 9")
        rows += out
    return rows


# --------------------------------------------------------------------------- execution


def unit_specs(spec: dict) -> list[str]:
    units = [f"threshold:q{int(round(q * 100))}" for q in spec["threshold_quantiles"]]
    units += [f"convergence:k{k}" for k in sorted(spec["convergence"], reverse=True)]
    units += ["replacement_quality:supported", "replacement_quality:unsupported"]
    units += [f"trajectory_subgroup:{k}" for k in spec["trajectory_subgroups"]]
    units += [f"position_quintile:{q}" for q in range(1, spec["position_quintiles"] + 1)]
    units += [f"entropy_quintile:{q}" for q in range(1, spec["entropy_quintiles"] + 1)]
    units += ["lead_time", "permutations", "paired_baseline_differences"]
    return units


def unit_key(config: ExperimentConfig, manifest: dict, name: str) -> str:
    return cache_key(stage=RUN_STAGE, scientific_config_hash=config.scientific_hash(),
                     model_revision=config.model.revision,
                     tokenizer_revision=config.model.tokenizer_revision, trace_sha256=None,
                     segmentation_version="released_chunks_v1", boundary_mapping_sha256=None,
                     generation_parameters={}, fork_idx=None, sample_seed=None,
                     code_version=manifest["identity"]["git_sha"],
                     input_hashes=dict(manifest["input_hashes"]), unit_id=name)


def run_unit(ctx: Context, name: str, workers: int) -> dict:
    """Returns ``{"rows": [...], "tables": {filename: pa.Table}, "record": {...}}``."""
    analysis, _, variant = name.partition(":")
    extra: dict = {}
    if analysis == "threshold":
        rows = analysis_threshold(ctx, variant)
    elif analysis == "convergence":
        rows = analysis_convergence(ctx, variant)
    elif analysis == "replacement_quality":
        rows = analysis_replacement(ctx, variant)
    elif analysis == "trajectory_subgroup":
        rows = analysis_subgroup(ctx, variant)
    elif analysis in ("position_quintile", "entropy_quintile"):
        rows = analysis_quintile(ctx, analysis.split("_")[0], variant)
    elif analysis == "lead_time":
        rows, table, summary = analysis_lead_time(ctx)
        extra = {"tables": {"lead_time.parquet": table}, "record": summary}
    elif analysis == "permutations":
        table, summary = analysis_permutations(ctx, workers)
        rows, extra = [], {"tables": {"permutations.parquet": table}, "record": summary}
    elif analysis == "paired_baseline_differences":
        rows = analysis_paired(ctx)
    else:
        raise ValueError(f"Unknown Stage 10 unit {name}")
    return {"rows": rows, "tables": extra.get("tables", {}), "record": extra.get("record", {})}


def execute(run: Path, repo: Path, config: ExperimentConfig, *, resume: bool = False,
            workers: int | None = None, interrupt_after: int | None = None) -> dict:
    root = run.parents[2]
    assert_disjoint(config)
    M.check_frozen(config)
    spec = check_spec(config)
    workers = workers or int(config.probes["max_cpu_workers"])
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        state = manifest["stages"].get(RUN_STAGE, {}).get("state")
        if state not in ("prepared", "running", "failed", "complete") or (
                state != "prepared" and not resume):
            raise ValueError(f"{RUN_STAGE} is {state}; pass --resume to continue")
        stage9_run = root / "artifacts/runs" / manifest["input_hashes"]["stage9_run_id"]
        stage9 = load_stage9_run(root, repo, config, stage9_run)
        if stage9["input_hashes"] != manifest["input_hashes"]:
            raise ValueError("Stage 9 inputs changed since this Stage 10 run was prepared")
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        stage_dir = run / "stages" / STAGE
        stage_dir.mkdir(parents=True, exist_ok=True)
        specs = unit_specs(spec)
        manifest["commands"].append({"command": "run-sensitivities", "attempt": attempt,
                                     "resume": resume, "workers": workers})
        manifest["stages"][RUN_STAGE] = {**manifest["stages"][RUN_STAGE], "state": "running",
                                         "attempt": attempt, "n_units": len(specs)}
        save_manifest(run, manifest)
        runtime = Runtime(run, RUN_STAGE, len(specs), None)
        store = UnitStore(run / UNITS_DIR)
        units: list[dict] = []
        ctx: Context | None = None
        result: dict = {}
        try:
            with runtime:
                for n, name in enumerate(specs):
                    if interrupt_after is not None and n >= interrupt_after:
                        raise KeyboardInterrupt("test interruption")
                    key = unit_key(config, manifest, name)
                    timing: dict = {}

                    def writer(directory: Path, name=name, timing=timing) -> None:
                        nonlocal ctx
                        if ctx is None:
                            ctx = Context(root, config, stage9, spec)
                        unit_started = time.monotonic()
                        out = run_unit(ctx, name, workers)
                        timing["compute_seconds"] = time.monotonic() - unit_started
                        write_parquet(directory / "rows.parquet", rows_table(out["rows"]))
                        for filename, table in out["tables"].items():
                            write_parquet(directory / filename, table)
                        write_json(directory / "unit.json", {
                            "schema_version": 1, "unit": name, "stage": RUN_STAGE,
                            "stage9_run_id": stage9_run.name,
                            "freeze_sha256": stage9["freeze_sha256"],
                            "selected_layer": stage9["frozen"]["binary"]["layer"],
                            "record": out["record"]})

                    directory, created = store.produce(key, writer)
                    units.append({"key": key, "unit": name, "created": created,
                                  "attempt": attempt if created else None, **timing})
                    runtime.completed += 1
                    runtime.last_output = str(directory.relative_to(run))
                    runtime.pulse(force=True)
                result = finalize(run, config, manifest, stage9, spec, units, attempt, started,
                                  ctx)
                manifest["stages"][RUN_STAGE].update({"state": "complete", "attempt": attempt})
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": manifest["run_id"],
                              "stage": RUN_STAGE, "attempt": attempt, "at": utc_now(),
                              "completed": runtime.completed, "exception": type(exc).__name__,
                              "traceback": traceback.format_exc(),
                              "recovery": "run-sensitivities --resume --run-id <id>"})
            write_json(stage_dir / "failures" / f"{attempt}.json", failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"][RUN_STAGE]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            manifest["stage_timings"].setdefault(RUN_STAGE, []).append(
                {"schema_version": 1, "attempt": attempt, "workers": workers,
                 "elapsed_seconds": time.monotonic() - started,
                 "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "cpu_count": os.cpu_count()})
            save_manifest(run, manifest)
        return result


def aggregate_units(run: Path, units: list[dict]) -> tuple[pa.Table, pa.Table, pa.Table, dict]:
    store = UnitStore(run / UNITS_DIR)
    rows, perms, leads, records = [], None, None, {}
    for u in units:
        store.validate(u["key"])
        directory = store.root / u["key"]
        rows.append(validate_parquet(directory / "rows.parquet"))
        record = read_json(directory / "unit.json")
        records[u["unit"]] = record["record"]
        if u["unit"] == "permutations":
            perms = validate_parquet(directory / "permutations.parquet")
        if u["unit"] == "lead_time":
            leads = validate_parquet(directory / "lead_time.parquet")
    if perms is None or leads is None:
        raise ValueError("Permutation or lead-time unit missing")
    return pa.concat_tables(rows).combine_chunks(), perms, leads, records


def finalize(run: Path, config: ExperimentConfig, manifest: dict, stage9: dict, spec: dict,
             units: list[dict], attempt: str, started: float, ctx: Context | None) -> dict:
    table, perms, leads, records = aggregate_units(run, units)
    for name in (SENSITIVITIES_FILE, PERMUTATIONS_FILE, LEAD_FILE):
        for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
            (run / (name + suffix)).unlink(missing_ok=True)
    hashes = {SENSITIVITIES_FILE: write_parquet(run / SENSITIVITIES_FILE, table),
              PERMUTATIONS_FILE: write_parquet(run / PERMUTATIONS_FILE, perms),
              LEAD_FILE: write_parquet(run / LEAD_FILE, leads)}
    from preanchor.viz.stage10 import write_figures

    figures = write_figures(run, table, perms, leads)
    thresholds = {r["variant"]: json.loads(r["detail"])["threshold"] for r in table.to_pylist()
                  if r["analysis"] == "threshold" and r["metric"] == "prevalence"}
    summary = {"schema_version": 1, "run_id": run.name, "stage": RUN_STAGE, "attempt": attempt,
               "stage_tag": "sensitivity_control", "preregistered": True,
               "confirmatory": False, "stage9_run_id": stage9["run"].name,
               "freeze_sha256": stage9["freeze_sha256"],
               "selected": {"binary": stage9["frozen"]["binary"],
                            "continuous": stage9["frozen"]["continuous"]},
               "primary_anchor_threshold": stage9["frozen"]["anchor_threshold"],
               "train_thresholds": thresholds, "quantile_method": QUANTILE_METHOD,
               "spec": spec, "units": units, "records": records,
               "position_cut_points": ctx.position_cuts if ctx else records.get(
                   "permutations", {}).get("position_cut_points"),
               "entropy_cut_points": ctx.entropy_cuts if ctx else None,
               "output_hashes": hashes, "figures": figures,
               "sensitivity_rows": table.num_rows, "permutation_rows": perms.num_rows,
               "lead_rows": leads.num_rows, "elapsed_seconds": time.monotonic() - started}
    if ctx is None:
        # Resumed with every unit already valid: recover the cut points from a unit record.
        for r in table.to_pylist():
            if r["analysis"] == "entropy_quintile":
                summary["entropy_cut_points"] = json.loads(r["detail"])["cut_points"]
                break
        for r in table.to_pylist():
            if r["analysis"] == "position_quintile":
                summary["position_cut_points"] = json.loads(r["detail"])["cut_points"]
                break
    write_current_json(run / SUMMARY_FILE, summary, attempt)
    manifest["output_hashes"].update(hashes)
    manifest["output_hashes"].update({f"figures/{k}": v for k, v in figures.items()})
    return summary


def read_summary(run: Path) -> dict:
    return read_json(run / SUMMARY_FILE)


# --------------------------------------------------------------------------- validation


def validate_stage10(run: Path, repo: Path, config: ExperimentConfig, *,
                     write_record: bool = True) -> dict:
    """Read-only re-check of a Stage 10 run against its Stage 9 inputs."""
    manifest = load_manifest(run)
    root = run.parents[2]
    report = {"schema_version": 1, "run_id": run.name, "ok": True, "problems": [],
              "validated_at": utc_now()}

    def fail(reason: str) -> None:
        report["ok"] = False
        report["problems"].append(reason)

    if manifest["stages"].get(RUN_STAGE, {}).get("state") != "complete":
        fail("run-sensitivities is not complete")
        if write_record:
            write_current_json(run / "stages" / STAGE / "validation.json", report)
        return report
    stage9_run = root / "artifacts/runs" / manifest["input_hashes"]["stage9_run_id"]
    try:
        stage9 = load_stage9_run(root, repo, config, stage9_run)
    except (ValueError, OSError, KeyError) as exc:
        fail(f"Stage 9 inputs no longer verify: {exc}")
        if write_record:
            write_current_json(run / "stages" / STAGE / "validation.json", report)
        return report
    if stage9["input_hashes"] != manifest["input_hashes"]:
        fail("Stage 9 artifacts differ from the hashes this run was prepared with")
    frozen = stage9["frozen"]
    report["stage9"] = {"run_id": stage9_run.name, "freeze_sha256": stage9["freeze_sha256"],
                        "hashes": stage9["hashes"], "unchanged": stage9["input_hashes"]
                        == manifest["input_hashes"]}
    for name in (FREEZE_FILE, "predictions/test.parquet", "metrics/test.json",
                 "predictions/validation.parquet", "metrics/validation.json"):
        if (run / name).exists():
            fail(f"Stage 10 run contains a Stage 9 primary artifact {name}")
    for marker in STAGE11_MARKERS:
        if (run / marker).exists():
            fail(f"Stage 11 artifact {marker} present")
    for name in (SENSITIVITIES_FILE, PERMUTATIONS_FILE, LEAD_FILE):
        try:
            if validate_artifact(run / name) != manifest["output_hashes"].get(name):
                fail(f"{name} differs from the manifest")
        except (ValueError, OSError) as exc:
            fail(f"{name}: {exc}")
    if not report["ok"]:
        if write_record:
            write_current_json(run / "stages" / STAGE / "validation.json", report)
        return report
    summary = read_summary(run)
    table = validate_parquet(run / SENSITIVITIES_FILE).to_pylist()
    perms = validate_parquet(run / PERMUTATIONS_FILE).to_pylist()
    leads = validate_parquet(run / LEAD_FILE).to_pylist()
    layer = int(frozen["binary"]["layer"])
    # Thresholds: train-only, q90 equals the frozen threshold.
    thresholds = {}
    for r in table:
        if r["analysis"] == "threshold":
            d = json.loads(r["detail"])
            thresholds[r["variant"]] = d
            if d["threshold_source"] != "train_eligible_importance_quantile":
                fail(f"{r['variant']} threshold is not from training")
            if abs(r["anchor_threshold"] - d["threshold"]) > 0:
                fail(f"{r['variant']} rows use a different numeric threshold")
    q90 = thresholds.get(f"q{int(round(THRESHOLD_QUANTILE * 100))}")
    if q90 is None or q90["threshold"] != frozen["anchor_threshold"]:
        fail("q90 threshold does not reproduce the frozen primary threshold")
    expected = {f"q{int(round(q * 100))}" for q in config.evaluation["threshold_sensitivities"]}
    if set(thresholds) != expected:
        fail(f"threshold variants {sorted(thresholds)} differ from {sorted(expected)}")
    # Layer identity: every activation family only at its Stage 8-frozen site (the primary
    # at the selected layer); refits, lead one and permutations at the selected layer.
    frozen_sites = {name.split(":")[0]: rec["layer"]
                    for name, rec in stage9["handoff"]["payload"]["models"].items()
                    if rec["task"] == M.BINARY and rec["layer"] is not None}
    for r in table:
        if r["probe_name"] in frozen_sites and r["layer"] != frozen_sites[r["probe_name"]]:
            fail(f"{r['probe_name']} evaluated at site {r['layer']} instead of its frozen "
                 f"site {frozen_sites[r['probe_name']]}")
    if {r["layer"] for r in table if r["probe_name"] == PRIMARY_MODEL} != {layer}:
        fail("primary model rows are not all at the frozen selected layer")
    if {r["layer"] for r in perms} != {layer}:
        fail("permutations did not use the frozen selected layer")
    if {r["layer"] for r in leads if r["probe_name"] == PRIMARY_MODEL} != {layer}:
        fail("lead-time predictions did not use the frozen selected layer")
    primary_hp = frozen["binary"]["C"]
    if any(r["hyperparameter"] != primary_hp for r in table if r["probe_name"] == PRIMARY_MODEL):
        fail("primary model rows use a hyperparameter other than the frozen C")
    # Lead one mapping.
    for r in leads:
        if r["lead_sentences"] == 1:
            if r["available"]:
                expected_id = f"{r['trajectory_id']}:{r['target_sentence_idx'] - 1}"
                if r["feature_sentence_idx"] != r["target_sentence_idx"] - 1 or \
                        r["feature_sentence_id"] != expected_id:
                    fail(f"lead-one feature row is not S_(i-1) for {r['sentence_id']}")
                if r["score_raw"] is None:
                    fail(f"available lead-one row without a score: {r['sentence_id']}")
            else:
                if r["score_raw"] is not None or r["feature_sentence_id"] is not None:
                    fail(f"unavailable lead-one row carries a score or mapping: {r['sentence_id']}")
                if r["target_sentence_idx"] != 0 and r["unavailable_reason"] == \
                        "first_sentence_has_no_predecessor":
                    fail("non-first sentence marked as first")
        elif r["lead_sentences"] == 0:
            if r["feature_sentence_id"] != r["sentence_id"]:
                fail("lead-zero row features do not belong to the target sentence")
    zero_targets = [r for r in leads if r["lead_sentences"] == 1 and r["target_sentence_idx"] == 0]
    if any(r["available"] for r in zero_targets):
        fail("a sentence-zero target was mapped at lead one")
    # Permutations.
    requested = int(config.evaluation["permutation_count"])
    indices = {r["permutation_index"] for r in perms if r["kind"] == "permutation"}
    if indices != set(range(requested)):
        fail(f"permutation indices {len(indices)} differ from the requested {requested}")
    if any(r["seed"] != permutation_seed(config, r["permutation_index"]) for r in perms
           if r["kind"] == "permutation"):
        fail("permutation seeds are not the derived seeds")
    if {r["strata"] for r in perms} != {",".join(config.evaluation["permutation_strata"])}:
        fail("permutation strata differ from the prespecified strata")
    if not all(r["kind"] == "observed" for r in perms if r["permutation_index"] == -1) or \
            {r["split"] for r in perms if r["kind"] == "observed"} != set(EVAL_SPLITS):
        fail("observed statistic missing for a split")
    # Empty / undefined groups stay explicit.
    report["undefined_groups"] = sorted({f"{r['analysis']}:{r['variant']}:{r['split']}"
                                         for r in table if r["detail"] and
                                         "undefined" in json.loads(r["detail"])})
    report.update({"selected_layer": layer, "thresholds": {k: v["threshold"]
                                                            for k, v in thresholds.items()},
                   "sensitivity_rows": len(table), "permutation_rows": len(perms),
                   "lead_rows": len(leads),
                   "lead_unavailable": summary["records"]["lead_time"]["unavailable"],
                   "permutations": summary["records"]["permutations"]["splits"],
                   "figures": summary["figures"]})
    if write_record:
        write_current_json(run / "stages" / STAGE / "validation.json", report)
    return report
