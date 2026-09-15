"""SPEC Stage 9: validation scoring, validation-only Platt calibration, the immutable
selection freeze and the exactly-once, hash-locked held-out test evaluation.

A Stage 9 run is its own ``run_id`` whose ``input_hashes`` name the finalized Stage 8 run
(pending selection, serialized models, CV tables) and the Stage 7 inputs behind it. The
three commands run at one code identity (``check_resume``):

``calibrate``        validation rows only -> predictions/validation.parquet,
                     metrics/validation.json, probe_models/calibrator.json
``freeze-selection`` -> metrics/analysis_registry.json, frozen_selection.json; the file's
                     SHA-256 is the freeze hash recorded in the manifest
``evaluate-test``    requires the exact freeze hash, re-verifies every frozen identity on
                     disk, claims the evaluation in an immutable manifest revision before
                     reading a test row, then publishes predictions/test.parquet,
                     metrics/test.json and metrics/test_receipt.json exactly once

Exactly-once semantics: outputs are byte-deterministic for a given freeze, so a retry of
an interrupted (claimed, unpublished) attempt under the same freeze hash either reproduces
identical bytes or fails on the first differing artifact; a published receipt refuses any
further evaluation; a selection change after the freeze fails the hash checks and the run
is invalid (new ``run_id``). Test metrics are never read before the freeze exists.
"""

from __future__ import annotations

import io
import re
import time
import traceback
import uuid
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa

from preanchor.config import ExperimentConfig, config_hash
from preanchor.features.extract import write_current_json
from preanchor.features.observables import check_feature_names
from preanchor.ids import file_hash
from preanchor.io import (
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_artifact,
    validate_parquet,
    write_json,
    write_parquet,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, create_run, load_manifest, save_manifest, utc_now
from preanchor.probes import models as M
from preanchor.probes.bootstrap import bootstrap_seed, check_spec, clustered_bootstrap
from preanchor.probes.calibrate import METHOD, apply_platt, calibration_decision, fit_platt
from preanchor.probes.dataset import FEATURES_FILE, LABELS_FILE, load_features
from preanchor.probes.evaluation import binary_metrics, continuous_metrics
from preanchor.probes.fit import FIT_STAGE, THRESHOLD_QUANTILE, load_partition
from preanchor.probes.folds import assert_disjoint
from preanchor.probes.select import PRIMARY_MODEL, SELECT_STAGE

STAGE = "stage9"
CALIBRATE_STAGE = "stage9_calibrate"
FREEZE_STAGE = "stage9_freeze"
TEST_STAGE = "stage9_test"
VALIDATION, TEST = "validation", "test"
BASELINE = "observable_combined"
FREEZE_FILE = "frozen_selection.json"
REGISTRY_FILE = "metrics/analysis_registry.json"
CALIBRATOR_FILE = "probe_models/calibrator.json"
RECEIPT_FILE = "metrics/test_receipt.json"
PENDING_FILE = "probe_models/frozen_selection.pending.json"
MODEL_FILE = "probe_models/selected_model.joblib"
PROBE_TABLES = ("probe_models/cv_results.parquet", "probe_models/fold_predictions.parquet")
STAGE10_OUTPUTS = ("metrics/sensitivities.parquet", "metrics/permutations.parquet",
                   "predictions/lead_time.parquet")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

EVAL_SCHEMA = pa.schema([
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
    pa.field("y_true", pa.int64(), nullable=False),
    pa.field("importance_true", pa.float64(), nullable=False),
    pa.field("score_raw", pa.float64(), nullable=False),
    pa.field("probability", pa.float64(), nullable=True),
    pa.field("probability_calibrated", pa.float64(), nullable=True),
    pa.field("importance_predicted", pa.float64(), nullable=True),
])


class TestLockError(ValueError):
    """Any refusal of the held-out test evaluation."""


# --------------------------------------------------------------------------- Stage 8 handoff


def load_probe_run(root: Path, config: ExperimentConfig, probe_run: Path) -> dict:
    """The finalized Stage 8 run, verified: fit and select complete, validate-probes ok,
    every recorded hash resolves on disk, selection basis is train CV, inputs unchanged."""
    manifest = load_manifest(probe_run)
    for stage in (FIT_STAGE, SELECT_STAGE):
        if manifest["stages"].get(stage, {}).get("state") != "complete":
            raise ValueError(f"Stage 8 run {probe_run.name}: {stage} is not complete")
    report = read_json(probe_run / "probe_models" / "probe_validation.json")
    if not (report.get("ok") and report.get("selected")):
        raise ValueError("Stage 8 run has no successful validate-probes record after selection")
    hashes = {}
    for name in (PENDING_FILE, MODEL_FILE, *PROBE_TABLES):
        digest = validate_artifact(probe_run / name)
        if manifest["output_hashes"].get(name) != digest:
            raise ValueError(f"{name} differs from the Stage 8 manifest")
        hashes[name] = digest
    pending = read_json(probe_run / PENDING_FILE)
    if pending["status"] != "pending" or pending["run_id"] != probe_run.name:
        raise ValueError("Pending selection does not belong to this Stage 8 run")
    if pending["selection_basis"] != "train_cv_only":
        raise ValueError("Selection basis is not training CV")
    if pending["input_hashes"] != manifest["input_hashes"]:
        raise ValueError("Pending selection inputs differ from the Stage 8 manifest")
    if pending["scientific_config_hash"] != config.scientific_hash():
        raise ValueError("Scientific configuration differs from the Stage 8 selection")
    if pending["selected_model_sha256"] != hashes[MODEL_FILE] or \
            pending["cv_results_sha256"] != hashes[PROBE_TABLES[0]] or \
            pending["fold_predictions_sha256"] != hashes[PROBE_TABLES[1]]:
        raise ValueError("Pending selection hashes differ from the Stage 8 artifacts")
    if pending["threshold_quantile"] != THRESHOLD_QUANTILE or \
            float(config.labels["binary_train_quantile"]) != THRESHOLD_QUANTILE:
        raise ValueError("Anchor quantile differs from the frozen 0.90")
    if (pending["binary"]["probe_name"], pending["continuous"]["probe_name"]) != (
            PRIMARY_MODEL, PRIMARY_MODEL):
        raise ValueError("Pending selection is not the primary model family")
    activation_run = root / "artifacts/runs" / manifest["input_hashes"]["activation_run_id"]
    table, current = load_features(root, config, activation_run)
    if current != manifest["input_hashes"]:
        raise ValueError("Stage 7 inputs changed since the Stage 8 run")
    payload = joblib.load(io.BytesIO((probe_run / MODEL_FILE).read_bytes()))
    if payload["run_id"] != probe_run.name or payload["train_rows_sha256"] != \
            pending["train_rows_sha256"]:
        raise ValueError("Serialized models do not belong to the pending selection")
    binary_key, ridge_key = payload["selected"]["binary"], payload["selected"]["continuous"]
    for key, record in ((binary_key, pending["binary"]), (ridge_key, pending["continuous"])):
        model = payload["models"][key]
        hp = record.get("C", record.get("alpha"))
        if (model["layer"], model["hyperparameter"]) != (record["layer"], hp):
            raise ValueError("Serialized selected model differs from the pending selection")
    return {"manifest": manifest, "pending": pending, "payload": payload, "table": table,
            "activation_run": activation_run,
            "input_hashes": {**manifest["input_hashes"], "probe_run_id": probe_run.name,
                             **hashes}}



def prepare_run(root: Path, repo: Path, config: ExperimentConfig, probe_run: Path) -> Path:
    assert_disjoint(config)
    M.check_frozen(config)
    check_spec(config)
    handoff = load_probe_run(root, config, probe_run)
    run = create_run(root, repo, config, ["calibrate"])
    manifest = load_manifest(run)
    manifest["input_hashes"] = dict(handoff["input_hashes"])
    manifest["identity"]["input_hashes"] = dict(handoff["input_hashes"])
    manifest["gpu_gate"] = "cpu_only_stage"
    manifest["stages"][CALIBRATE_STAGE] = {"state": "prepared", "partition": VALIDATION}
    save_manifest(run, manifest)
    return run


def _probe_run(root: Path, manifest: dict) -> Path:
    return root / "artifacts/runs" / manifest["input_hashes"]["probe_run_id"]


def _handoff(root: Path, config: ExperimentConfig, manifest: dict) -> dict:
    handoff = load_probe_run(root, config, _probe_run(root, manifest))
    if handoff["input_hashes"] != manifest["input_hashes"]:
        raise ValueError("Stage 8 inputs changed since this Stage 9 run was prepared")
    return handoff


# --------------------------------------------------------------------------- scoring


def score_split(root: Path, config: ExperimentConfig, handoff: dict, split: str,
                calibrator: dict | None) -> tuple:
    """Every refitted family at its selected configuration on the eligible rows of
    ``split``; the anchor label is the Stage 4 train-derived threshold, never recomputed."""
    pending, payload = handoff["pending"], handoff["payload"]
    partition, loader, hashes = load_partition(root, config, handoff["activation_run"], split)
    if hashes != handoff["manifest"]["input_hashes"]:
        raise ValueError("Stage 7 inputs changed during scoring")
    if partition.anchor_threshold != pending["anchor_threshold"]:
        raise ValueError("Partition anchor threshold differs from the training-derived q90")
    expected = (partition.importance_primary >= partition.anchor_threshold).astype(int)
    if not np.array_equal(expected, partition.y):
        raise ValueError("Anchor labels are not importance >= the frozen threshold")
    rows, arrays = [], {}
    binary_key = payload["selected"]["binary"]
    for key in sorted(payload["pipelines"]):
        record = payload["models"][key]
        family, task = record["probe_name"], record["task"]
        site = loader.site(int(record["layer"])) if M.uses_activation(family) else None
        X = M.design(family, partition, site)
        out = M.scores(payload["pipelines"][key], task, X)
        calibrated = None
        if task == M.BINARY and key == binary_key and calibrator and calibrator["fitted"]:
            calibrated = apply_platt(calibrator["parameters"], out["score_raw"])
        arrays[key] = {**out, "probability_calibrated": calibrated}
        for i in range(len(partition)):
            rows.append({
                "schema_version": 1, "sentence_id": partition.sentence_id[i],
                "problem_id": int(partition.problem_id[i]),
                "trajectory_id": partition.trajectory_id[i], "split": split,
                "lead_sentences": 0, "threshold_quantile": THRESHOLD_QUANTILE,
                "layer": record["layer"], "probe_name": family, "task": task,
                "hyperparameter": record["hyperparameter"],
                "y_true": int(partition.y[i]),
                "importance_true": float(partition.importance_log1p[i]),
                "score_raw": float(out["score_raw"][i]),
                "probability": float(out["probability"][i]) if "probability" in out else None,
                "probability_calibrated": (None if calibrated is None
                                           else float(calibrated[i])),
                "importance_predicted": (float(out["importance_predicted"][i])
                                         if "importance_predicted" in out else None)})
    rows.sort(key=lambda r: (r["probe_name"], r["task"], r["trajectory_id"], r["sentence_id"]))
    return partition, pa.Table.from_pylist(rows, schema=EVAL_SCHEMA), arrays


def split_metrics(partition, arrays: dict, handoff: dict, config: ExperimentConfig,
                  split: str, calibrator: dict | None) -> dict:
    payload, pending = handoff["payload"], handoff["pending"]
    binary_key = payload["selected"]["binary"]
    bins = int(config.evaluation["ece_bins"])
    models, binary, continuous = {}, {}, {}
    for key, out in arrays.items():
        family, task = key.split(":")
        if task == M.BINARY:
            models[key] = binary_metrics(partition.y, out["score_raw"], out["probability"],
                                         partition.trajectory_id, bins,
                                         out["probability_calibrated"])
            binary[family] = {"score": out["score_raw"], "probability": out["probability"]}
        else:
            models[key] = continuous_metrics(partition.importance_log1p,
                                             out["importance_predicted"])
            continuous[family] = out["importance_predicted"]
    primary_ap = models[binary_key]["average_precision"]
    baseline_ap = models[f"{BASELINE}:{M.BINARY}"]["average_precision"]
    paired = {}
    for key, m in models.items():
        family, task = key.split(":")
        if task == M.BINARY and key != binary_key:
            paired[f"{PRIMARY_MODEL}-minus-{family}"] = {
                s: (None if models[binary_key][s] is None or m[s] is None
                    else models[binary_key][s] - m[s]) for s in ("average_precision", "auroc")}
    boot = clustered_bootstrap(
        problem_id=partition.problem_id, trajectory_id=partition.trajectory_id, y=partition.y,
        importance=partition.importance_log1p, binary=binary, continuous=continuous,
        primary=PRIMARY_MODEL, baseline=BASELINE, config=config, split=split)
    e = config.evaluation
    problems = sorted({int(p) for p in partition.problem_id})
    return {"schema_version": 1, "split": split, "stage": e["stage"],
            "preregistered": bool(e["preregistered"]),
            "held_out_description": e["held_out_description"] if split == TEST else None,
            "partition": {"n_rows": len(partition), "n_positive": int(partition.y.sum()),
                          "prevalence": float(partition.y.mean()),
                          "n_problems": len(problems), "problems": problems,
                          "n_trajectories": len(set(partition.trajectory_id)),
                          "anchor_threshold": partition.anchor_threshold,
                          "threshold_quantile": THRESHOLD_QUANTILE,
                          "threshold_source": "stage4_train_eligible_quantile",
                          "meets_minimum_sentences": len(partition) >= int(
                              e[f"minimum_{split}_sentences"]),
                          "meets_minimum_anchors": int(partition.y.sum()) >= int(
                              e[f"minimum_{split}_anchors"])},
            "selected": {"binary": pending["binary"], "continuous": pending["continuous"]},
            "calibration": calibrator, "primary": {
                "model": binary_key, "metric": e["primary_metric"], "value": primary_ap},
            "aucpr_gain_over_observable_combined": (
                None if primary_ap is None or baseline_ap is None else primary_ap - baseline_ap),
            "paired_differences_point": paired, "models": models, "bootstrap": boot}


# --------------------------------------------------------------------------- calibrate


def calibrate(run: Path, repo: Path, config: ExperimentConfig) -> dict:
    root = run.parents[2]
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        state = manifest["stages"].get(CALIBRATE_STAGE, {}).get("state")
        if state != "prepared":
            raise ValueError(f"{CALIBRATE_STAGE} is {state}; calibration runs once per run")
        handoff = _handoff(root, config, manifest)
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        manifest["commands"].append({"command": "calibrate", "attempt": attempt})
        manifest["stages"][CALIBRATE_STAGE] = {"state": "running", "attempt": attempt,
                                               "partition": VALIDATION}
        save_manifest(run, manifest)
        try:
            partition, _, arrays = score_split(root, config, handoff, VALIDATION, None)
            binary_key = handoff["payload"]["selected"]["binary"]
            y, score = partition.y, arrays[binary_key]["score_raw"]
            decision = calibration_decision(int(y.sum()), len(np.unique(y)), config)
            calibrator = {"schema_version": 1, "run_id": run.name, "method": METHOD,
                          "policy": config.probes["calibration"], "fitted_on": VALIDATION,
                          "fitted": bool(decision["calibrate"]), "skip_reason": decision["reason"],
                          "parameters": None, "model": binary_key,
                          "n_rows": int(len(y)), "n_positive": int(y.sum()),
                          "n_classes": int(len(np.unique(y))),
                          "validation_rows_sha256": config_hash(partition.sentence_id),
                          "minimum_positives": int(config.probes["calibration_minimum_positives"]),
                          "minimum_classes": int(config.probes["calibration_minimum_classes"]),
                          "in_sample_on_validation": True}
            if calibrator["fitted"]:
                calibrator["parameters"] = fit_platt(score, y)
            # Second pass applies the calibrator to the selected model's validation scores.
            partition, table, arrays = score_split(root, config, handoff, VALIDATION, calibrator)
            metrics = split_metrics(partition, arrays, handoff, config, VALIDATION, calibrator)
            metrics["run_id"] = run.name
            pred_sha = write_parquet(run / "predictions" / "validation.parquet", table)
            metrics_sha = write_json(run / "metrics" / "validation.json", metrics)
            cal_sha = write_json(run / CALIBRATOR_FILE, calibrator)
            manifest["output_hashes"].update({"predictions/validation.parquet": pred_sha,
                                              "metrics/validation.json": metrics_sha,
                                              CALIBRATOR_FILE: cal_sha})
            manifest["stages"][CALIBRATE_STAGE] = {
                "state": "complete", "attempt": attempt, "partition": VALIDATION,
                "n_rows": int(len(partition)), "n_positive": int(partition.y.sum()),
                "calibration_fitted": calibrator["fitted"],
                "skip_reason": calibrator["skip_reason"]}
            return {"calibrator": calibrator, "metrics": metrics, "predictions_sha256": pred_sha,
                    "metrics_sha256": metrics_sha}
        except BaseException:
            manifest["stages"][CALIBRATE_STAGE]["state"] = "failed"
            raise
        finally:
            manifest["stage_timings"].setdefault(CALIBRATE_STAGE, []).append(
                {"schema_version": 1, "attempt": attempt,
                 "elapsed_seconds": time.monotonic() - started})
            save_manifest(run, manifest)


# --------------------------------------------------------------------------- registry, freeze


def analysis_registry(run: Path, config: ExperimentConfig, manifest: dict, pending: dict,
                      calibrator: dict) -> dict:
    """Machine-readable record of the prespecified analyses; contains no realized test
    result. Stage 9 test analyses are the confirmatory set; Stage 10 items are
    sensitivity/control analyses (SPEC §3.12, §3.11)."""
    e = config.evaluation
    binary_key = f"{PRIMARY_MODEL}:{M.BINARY}"
    comparisons = [f for f in M.BASELINES + M.ACTIVATION_MODELS + [M.RESIDUALIZED]
                   if f != PRIMARY_MODEL]
    confirmatory = [
        {"id": "primary_test_aucpr", "stage": STAGE, "confirmatory": True,
         "endpoint": "primary", "metric": e["primary_metric"], "model": binary_key,
         "partition": TEST, "description": "test AUCPR of the selected activation + "
         "observable probe (SPEC §3.11)"},
        *[{"id": f"secondary_test_{m}", "stage": STAGE, "confirmatory": True,
           "endpoint": "secondary", "metric": m, "model": binary_key
           if m not in ("spearman", "rmse") else f"{PRIMARY_MODEL}:{M.CONTINUOUS}",
           "partition": TEST} for m in e["secondary_metrics"]],
        {"id": "paired_baseline_differences", "stage": STAGE, "confirmatory": True,
         "endpoint": "secondary", "metric": "paired_differences",
         "model": binary_key, "against": [f"{f}:{M.BINARY}" for f in comparisons],
         "partition": TEST, "statistics": ["average_precision", "auroc"]},
    ]
    later = [{"id": name, "stage": "stage10", "confirmatory": False,
              "status": "prespecified_not_started"} for name in e["minimum_experiments"]
             if name not in ("released_label_replay", "observable_baselines",
                             "all_layer_probes", "residualized_control", "fresh_4682_pilot")]
    completed_before = [{"id": name, "stage": "stage8", "confirmatory": False,
                         "status": "training_cv_diagnostic"} for name in
                        ("released_label_replay", "observable_baselines", "all_layer_probes",
                         "residualized_control")]
    return {
        "schema_version": 1, "run_id": run.name, "probe_run_id": pending["run_id"],
        "stage_tag": e["stage"], "preregistered": bool(e["preregistered"]),
        "held_out_description": e["held_out_description"],
        "primary_endpoint": {"metric": e["primary_metric"], "model": binary_key,
                             "partition": TEST, "config_primary_model": e["primary_model"]},
        "secondary_metrics": list(e["secondary_metrics"]),
        "selected_model": {"binary": pending["binary"], "continuous": pending["continuous"],
                           "selected_model_sha256": pending["selected_model_sha256"],
                           "selection_basis": pending["selection_basis"]},
        "calibration": {"method": calibrator["method"], "fitted": calibrator["fitted"],
                        "skip_reason": calibrator["skip_reason"], "fitted_on": VALIDATION,
                        "parameters": calibrator["parameters"]},
        "anchor_threshold": {"value": pending["anchor_threshold"],
                             "quantile": THRESHOLD_QUANTILE,
                             "source": "stage4_train_eligible_quantile"},
        "bootstrap": {"replicates": int(e["bootstrap_replicates"]), "unit": e["bootstrap_unit"],
                      "paired": bool(e["bootstrap_paired"]), "with_replacement": True,
                      "interval": "percentile_2.5_97.5",
                      "seeds": {s: bootstrap_seed(config, s) for s in (VALIDATION, TEST)},
                      "descriptive_only": True},
        "metric_definitions": {
            "average_precision": "sklearn average_precision_score on score_raw",
            "auroc": "sklearn roc_auc_score on score_raw",
            "brier": "mean squared error of probability", "log_loss": "clipped at 1e-15",
            "ece": f"{int(e['ece_bins'])}-bin {e['ece_binning']} on probability",
            "precision_top10_percent": "per trajectory, k = ceil(0.10 n), mean over trajectories",
            "ndcg_top10_percent": "per trajectory, binary gains, log2 discount, trajectories "
                                  "without a positive excluded and counted",
            "spearman": "log1p importance vs Ridge prediction",
            "rmse": "log1p importance vs Ridge prediction"},
        "test_partition": {"problems": list(config.splits.test),
                           "features_sha256": manifest["input_hashes"][FEATURES_FILE],
                           "labels_sha256": manifest["input_hashes"][LABELS_FILE]},
        "analyses": completed_before + confirmatory + later,
        "code_commit": manifest["identity"]["git_sha"],
        "scientific_config_hash": manifest["identity"]["scientific_config_hash"],
        "resolved_config_hash": manifest["identity"]["resolved_config_hash"],
    }


def freeze(run: Path, repo: Path, config: ExperimentConfig) -> dict:
    root = run.parents[2]
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        if manifest["stages"].get(CALIBRATE_STAGE, {}).get("state") != "complete":
            raise ValueError("calibrate must complete before the freeze")
        if manifest["stages"].get(FREEZE_STAGE, {}).get("state") == "complete" or \
                (run / FREEZE_FILE).exists():
            raise ValueError("The selection is already frozen for this run")
        handoff = _handoff(root, config, manifest)
        pending = handoff["pending"]
        for name in ("predictions/validation.parquet", "metrics/validation.json",
                     CALIBRATOR_FILE):
            if validate_artifact(run / name) != manifest["output_hashes"][name]:
                raise ValueError(f"{name} changed after calibration")
        calibrator = read_json(run / CALIBRATOR_FILE)
        attempt = uuid.uuid4().hex
        manifest["commands"].append({"command": "freeze-selection", "attempt": attempt})
        registry = analysis_registry(run, config, manifest, pending, calibrator)
        registry_sha = write_json(run / REGISTRY_FILE, registry)
        table = handoff["table"]
        feature_check = check_feature_names(config, table.column_names)
        split_path = repo / config.dataset.split_manifest
        frozen = {
            "schema_version": 1, "status": "frozen", "run_id": run.name,
            "probe_run_id": pending["run_id"], "stage": FREEZE_STAGE, "attempt": attempt,
            "frozen_at": utc_now(), "stage_tag": pending["stage_tag"],
            "preregistered": pending["preregistered"],
            "code_commit": manifest["identity"]["git_sha"],
            "code_dirty_diff_hash": manifest["identity"]["dirty_diff_hash"],
            "lock_hash": manifest["identity"]["lock_hash"],
            "resolved_config_hash": manifest["identity"]["resolved_config_hash"],
            "scientific_config_hash": manifest["identity"]["scientific_config_hash"],
            "selection_code_commit": pending["code_commit"],
            "input_hashes": dict(manifest["input_hashes"]),
            "activation_run_id": manifest["input_hashes"]["activation_run_id"],
            "split": {"manifest": config.dataset.split_manifest,
                      "sha256": file_hash(split_path), "train": list(config.splits.train),
                      "validation": list(config.splits.validation),
                      "test": list(config.splits.test)},
            "features": {"primary": list(config.features["observables"]),
                         "excluded": list(config.features["excluded"]),
                         "forbidden_feature_check": feature_check,
                         "layers": list(config.features["layers"]),
                         "activation_site": config.features["activation_site"],
                         "boundary": config.features["boundary"], "lead_sentences": 0},
            "anchor_threshold": pending["anchor_threshold"],
            "threshold_quantile": THRESHOLD_QUANTILE,
            "threshold_source": "stage4_train_eligible_quantile",
            "anchor_comparison": config.labels["anchor_comparison"],
            "binary": pending["binary"], "continuous": pending["continuous"],
            "residualized_control": {"layer": pending["residualized_control"]["layer"],
                                     "unit_key": pending["residualized_control"]["unit_key"]},
            "probe_families": {k: config.probes[k] for k in (
                "scaler", "binary", "class_weight", "max_iter", "logistic_C", "continuous",
                "ridge_alpha", "cv_splitter", "cv_folds", "cv_shuffle", "group_column",
                "selection_metric", "ridge_selection_metric", "ridge_tiebreak_metric",
                "fit_scope", "residualization", "baselines", "activation_models")},
            "fold_seed": pending["fold_seed"], "train_rows_sha256": pending["train_rows_sha256"],
            "n_train_rows": pending["n_train_rows"],
            "n_train_positive": pending["n_train_positive"],
            "selection_basis": pending["selection_basis"],
            "pending_selection_sha256": manifest["input_hashes"][PENDING_FILE],
            "selected_model_sha256": pending["selected_model_sha256"],
            "cv_results_sha256": pending["cv_results_sha256"],
            "fold_predictions_sha256": pending["fold_predictions_sha256"],
            "calibration": {"method": calibrator["method"], "policy": calibrator["policy"],
                            "fitted": calibrator["fitted"],
                            "skip_reason": calibrator["skip_reason"],
                            "parameters": calibrator["parameters"], "fitted_on": VALIDATION,
                            "n_validation_rows": calibrator["n_rows"],
                            "n_validation_positive": calibrator["n_positive"],
                            "calibrator_sha256": manifest["output_hashes"][CALIBRATOR_FILE]},
            "validation_predictions_sha256": manifest["output_hashes"][
                "predictions/validation.parquet"],
            "validation_metrics_sha256": manifest["output_hashes"]["metrics/validation.json"],
            "metrics": {"primary": config.evaluation["primary_metric"],
                        "primary_model": f"{PRIMARY_MODEL}:{M.BINARY}",
                        "secondary": list(config.evaluation["secondary_metrics"]),
                        "ece_bins": int(config.evaluation["ece_bins"]),
                        "ece_binning": config.evaluation["ece_binning"],
                        "definitions": registry["metric_definitions"]},
            "bootstrap": registry["bootstrap"],
            "evaluation_config": dict(config.evaluation),
            "analysis_registry_sha256": registry_sha,
            "test_evaluation": {"partition": TEST, "problems": list(config.splits.test),
                                "runs_once": True, "requires_confirm_hash": True},
        }
        payload = json_bytes(frozen)
        freeze_sha = atomic_bytes(run / FREEZE_FILE, payload)
        atomic_bytes(run / (FREEZE_FILE + ".sha256"), (freeze_sha + "\n").encode())
        manifest["output_hashes"].update({REGISTRY_FILE: registry_sha,
                                          FREEZE_FILE: freeze_sha})
        manifest["stages"][FREEZE_STAGE] = {"state": "complete", "attempt": attempt,
                                            "freeze_sha256": freeze_sha,
                                            "frozen_at": frozen["frozen_at"]}
        save_manifest(run, manifest)
        return {"frozen": frozen, "freeze_sha256": freeze_sha,
                "analysis_registry_sha256": registry_sha}


# --------------------------------------------------------------------------- test lock


def verify_freeze(run: Path, repo: Path, config: ExperimentConfig, manifest: dict,
                  confirm: str | None) -> tuple[dict, str, dict]:
    """Every precondition of the held-out evaluation; raises ``TestLockError``."""
    if confirm is not None and not SHA256.match(confirm):
        raise TestLockError("--confirm-frozen must be the exact lowercase SHA-256 hex digest")
    path = run / FREEZE_FILE
    if not path.is_file() or path.is_symlink():
        raise TestLockError("frozen_selection.json does not exist; freeze-selection first")
    try:
        actual = validate_artifact(path)
        frozen = read_json(path)
    except (ValueError, OSError, KeyError) as exc:
        raise TestLockError(f"frozen_selection.json fails integrity checks: {exc}") from None
    if confirm is not None and confirm != actual:
        raise TestLockError("Supplied freeze hash does not equal the frozen_selection.json hash")
    stage = manifest["stages"].get(FREEZE_STAGE, {})
    if stage.get("state") != "complete" or stage.get("freeze_sha256") != actual or \
            manifest["output_hashes"].get(FREEZE_FILE) != actual:
        raise TestLockError("Freeze hash is not the one recorded in the run manifest")
    if frozen.get("status") != "frozen" or frozen.get("schema_version") != 1:
        raise TestLockError("Malformed freeze record")
    if frozen["run_id"] != run.name or frozen["probe_run_id"] != \
            manifest["input_hashes"]["probe_run_id"]:
        raise TestLockError("Freeze belongs to another run")
    identity = manifest["identity"]
    if (frozen["code_commit"], frozen["code_dirty_diff_hash"], frozen["lock_hash"],
            frozen["resolved_config_hash"], frozen["scientific_config_hash"]) != (
            identity["git_sha"], identity["dirty_diff_hash"], identity["lock_hash"],
            identity["resolved_config_hash"], identity["scientific_config_hash"]):
        raise TestLockError("Code, lock or configuration identity differs from the freeze")
    if frozen["input_hashes"] != manifest["input_hashes"]:
        raise TestLockError("Run inputs differ from the freeze")
    if frozen["split"]["sha256"] != file_hash(repo / config.dataset.split_manifest) or \
            frozen["split"]["test"] != list(config.splits.test):
        raise TestLockError("The fixed problem split differs from the freeze")
    root = run.parents[2]
    try:
        handoff = load_probe_run(root, config, _probe_run(root, manifest))
    except (ValueError, OSError, KeyError) as exc:
        raise TestLockError(f"Stage 8 inputs no longer verify: {exc}") from None
    if handoff["input_hashes"] != frozen["input_hashes"]:
        raise TestLockError("Selection artifacts changed after the freeze")
    pending = handoff["pending"]
    if (pending["binary"], pending["continuous"], pending["anchor_threshold"]) != (
            frozen["binary"], frozen["continuous"], frozen["anchor_threshold"]):
        raise TestLockError("Selected layer, regularisation or threshold differs from the freeze")
    for name, expected in ((CALIBRATOR_FILE, frozen["calibration"]["calibrator_sha256"]),
                           (REGISTRY_FILE, frozen["analysis_registry_sha256"]),
                           ("predictions/validation.parquet",
                            frozen["validation_predictions_sha256"]),
                           ("metrics/validation.json", frozen["validation_metrics_sha256"])):
        try:
            digest = validate_artifact(run / name)
        except (ValueError, OSError) as exc:
            raise TestLockError(f"{name}: {exc}") from None
        if digest != expected:
            raise TestLockError(f"{name} changed after the freeze")
    calibrator = read_json(run / CALIBRATOR_FILE)
    if calibrator["fitted_on"] != VALIDATION or calibrator["parameters"] != \
            frozen["calibration"]["parameters"]:
        raise TestLockError("Calibrator differs from the freeze or was not fitted on validation")
    if frozen["bootstrap"]["replicates"] != int(config.evaluation["bootstrap_replicates"]):
        raise TestLockError("Bootstrap specification differs from the freeze")
    return frozen, actual, handoff


def test_state(run: Path, manifest: dict) -> dict:
    """What has happened to the held-out evaluation of this run (metadata only)."""
    stage = manifest["stages"].get(TEST_STAGE)
    receipt = run / RECEIPT_FILE
    outputs = {n: (run / n).exists() for n in ("predictions/test.parquet", "metrics/test.json")}
    return {"claimed": stage is not None, "state": None if stage is None else stage["state"],
            "receipt": receipt.is_file(), "outputs": outputs,
            "attempts": [] if stage is None else stage.get("attempts", []),
            "freeze_sha256": None if stage is None else stage.get("freeze_sha256")}


def evaluate_test(run: Path, repo: Path, config: ExperimentConfig, confirm: str) -> dict:
    root = run.parents[2]
    with lock(run / ".run.lock"):
        try:
            manifest = check_resume(run, repo, config)
        except ValueError as exc:
            raise TestLockError(f"Run identity check failed: {exc}") from None
        if confirm is None:
            raise TestLockError("--confirm-frozen is required")
        frozen, freeze_sha, handoff = verify_freeze(run, repo, config, manifest, confirm)
        state = test_state(run, manifest)
        if state["receipt"] or state["state"] == "complete":
            raise TestLockError("The held-out test was already evaluated for this run; a new "
                                "selection needs a new run_id")
        if any(state["outputs"].values()) and not state["claimed"]:
            raise TestLockError("Test outputs exist without a recorded claim; refusing")
        if state["claimed"] and state["freeze_sha256"] != freeze_sha:
            raise TestLockError("A previous attempt was claimed under a different freeze")
        if state["claimed"] and state["state"] not in ("claimed", "failed"):
            raise TestLockError(f"Unexpected test state {state['state']}")
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        attempts = state["attempts"] + [{"attempt": attempt, "claimed_at": utc_now(),
                                         "resumed": state["claimed"]}]
        # The claim is durable (immutable manifest revision) before any test row is read.
        manifest["commands"].append({"command": "evaluate-test", "attempt": attempt,
                                     "confirm_frozen": confirm})
        manifest["stages"][TEST_STAGE] = {"state": "claimed", "attempt": attempt,
                                          "freeze_sha256": freeze_sha, "attempts": attempts,
                                          "partition": TEST}
        save_manifest(run, manifest)
        stage_dir = run / "stages" / STAGE
        try:
            calibrator = read_json(run / CALIBRATOR_FILE)
            partition, table, arrays = score_split(root, config, handoff, TEST, calibrator)
            problems = sorted({int(p) for p in partition.problem_id})
            if config.execution.mode == "mvp" and problems != sorted(config.splits.test):
                raise TestLockError(f"Test partition problems {problems} are not the four "
                                    "frozen test problems")
            if any(p not in set(config.splits.test) for p in problems):
                raise TestLockError("A non-test problem is present in the test partition")
            metrics = split_metrics(partition, arrays, handoff, config, TEST, calibrator)
            metrics["run_id"] = run.name
            metrics["freeze_sha256"] = freeze_sha
            metrics["mvp_incomplete_insufficient_test_anchors"] = not metrics["partition"][
                "meets_minimum_anchors"]
            pred_sha = write_parquet(run / "predictions" / "test.parquet", table)
            metrics_sha = write_json(run / "metrics" / "test.json", metrics)
            receipt = {"schema_version": 1, "run_id": run.name, "stage": TEST_STAGE,
                       "attempt": attempt, "attempts": attempts, "freeze_sha256": freeze_sha,
                       "confirm_frozen": confirm, "evaluated_at": utc_now(),
                       "code_commit": manifest["identity"]["git_sha"],
                       "predictions_sha256": pred_sha, "metrics_sha256": metrics_sha,
                       "n_rows": int(len(partition)), "n_positive": int(partition.y.sum()),
                       "n_problems": len(problems), "problems": problems,
                       "elapsed_seconds": time.monotonic() - started}
            receipt_sha = write_json(run / RECEIPT_FILE, receipt)
            manifest["output_hashes"].update({"predictions/test.parquet": pred_sha,
                                              "metrics/test.json": metrics_sha,
                                              RECEIPT_FILE: receipt_sha})
            manifest["stages"][TEST_STAGE].update({
                "state": "complete", "n_rows": int(len(partition)),
                "n_positive": int(partition.y.sum()), "n_problems": len(problems),
                "predictions_sha256": pred_sha, "metrics_sha256": metrics_sha,
                "receipt_sha256": receipt_sha})
            return {"metrics": metrics, "receipt": receipt}
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": run.name, "stage": TEST_STAGE,
                              "attempt": attempt, "at": utc_now(), "freeze_sha256": freeze_sha,
                              "exception": type(exc).__name__,
                              "traceback": traceback.format_exc(),
                              "recovery": "evaluate-test again with the same freeze hash "
                                          "reproduces identical bytes or fails closed"})
            write_json(stage_dir / "failures" / f"{attempt}.json", failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"][TEST_STAGE]["state"] = "failed"
            raise
        finally:
            manifest["stage_timings"].setdefault(TEST_STAGE, []).append(
                {"schema_version": 1, "attempt": attempt,
                 "elapsed_seconds": time.monotonic() - started})
            save_manifest(run, manifest)


# --------------------------------------------------------------------------- validation


def validate_stage9(run: Path, repo: Path, config: ExperimentConfig) -> dict:
    """Read-only re-check of a Stage 9 run. ``ok`` is false, with reasons, when any frozen
    identity no longer verifies (the run is then invalid: SPEC Stage 9 failure/recovery)."""
    manifest = load_manifest(run)
    report = {"schema_version": 1, "run_id": run.name, "ok": True, "problems": [],
              "validated_at": utc_now()}

    def fail(reason: str) -> None:
        report["ok"] = False
        report["problems"].append(reason)

    try:
        frozen, freeze_sha, handoff = verify_freeze(run, repo, config, manifest, None)
    except TestLockError as exc:
        fail(str(exc))
        report["frozen"] = False
        write_current_json(run / "stages" / STAGE / "validation.json", report)
        return report
    report.update({"frozen": True, "freeze_sha256": freeze_sha,
                   "calibration_fitted": frozen["calibration"]["fitted"],
                   "calibration_skip_reason": frozen["calibration"]["skip_reason"]})
    val = validate_parquet(run / "predictions" / "validation.parquet")
    if set(val.column("split").to_pylist()) != {VALIDATION}:
        fail("validation predictions contain non-validation rows")
    if set(val.column("problem_id").to_pylist()) - set(config.splits.validation):
        fail("validation predictions contain a non-validation problem")
    calibrator = read_json(run / CALIBRATOR_FILE)
    n_val = len({s for s in val.column("sentence_id").to_pylist()})
    if calibrator["n_rows"] != n_val or calibrator["fitted_on"] != VALIDATION:
        fail("calibrator row count does not equal the validation partition")
    for name in STAGE10_OUTPUTS:
        if (run / name).exists():
            fail(f"Stage 10 output {name} present in a Stage 9 run")
    state = test_state(run, manifest)
    report["test"] = state
    if state["receipt"]:
        receipt = read_json(run / RECEIPT_FILE)
        if receipt["freeze_sha256"] != freeze_sha:
            fail("test receipt names a different freeze")
        for name, key in (("predictions/test.parquet", "predictions_sha256"),
                          ("metrics/test.json", "metrics_sha256")):
            if validate_artifact(run / name) != receipt[key] or \
                    manifest["output_hashes"].get(name) != receipt[key]:
                fail(f"{name} differs from the test receipt")
        test = validate_parquet(run / "predictions" / "test.parquet")
        if not test.schema.equals(EVAL_SCHEMA):
            fail("test predictions schema differs from the Stage 9 schema")
        if set(test.column("split").to_pylist()) != {TEST}:
            fail("test predictions contain non-test rows")
        problems = set(test.column("problem_id").to_pylist())
        if problems - set(config.splits.test):
            fail("test predictions contain a non-test problem")
        if config.execution.mode == "mvp" and problems != set(config.splits.test):
            fail("test predictions do not cover exactly the four test problems")
        if set(val.column("sentence_id").to_pylist()) & set(test.column("sentence_id").to_pylist()):
            fail("a sentence appears in both validation and test predictions")
        probs = [p for p in test.column("probability").to_pylist() if p is not None]
        cal = [p for p in test.column("probability_calibrated").to_pylist() if p is not None]
        if any(p < 0 or p > 1 for p in probs + cal):
            fail("test probability outside [0, 1]")
        if bool(cal) != bool(frozen["calibration"]["fitted"]):
            fail("calibrated probabilities disagree with the frozen calibration state")
        metrics = read_json(run / "metrics" / "test.json")
        boot = metrics["bootstrap"]
        if boot["replicates_requested"] != frozen["bootstrap"]["replicates"] or \
                boot["sampling_unit"] != "problem_id" or not boot["paired"]:
            fail("test bootstrap does not match the frozen specification")
        if metrics["partition"]["anchor_threshold"] != frozen["anchor_threshold"]:
            fail("test anchor threshold differs from the frozen training threshold")
        if len(state["attempts"]) != len(receipt["attempts"]):
            fail("attempt history differs between manifest and receipt")
        report["test_summary"] = {"n_rows": receipt["n_rows"], "n_positive": receipt["n_positive"],
                                  "n_problems": receipt["n_problems"],
                                  "attempts": len(receipt["attempts"]),
                                  "predictions_sha256": receipt["predictions_sha256"],
                                  "metrics_sha256": receipt["metrics_sha256"]}
    elif state["claimed"]:
        report["test_summary"] = {"interrupted_claim": True, "attempts": len(state["attempts"])}
    write_current_json(run / "stages" / STAGE / "validation.json", report)
    return report
