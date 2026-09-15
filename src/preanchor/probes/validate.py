"""``preanchor validate-probes``: re-check a finished Stage 8 run without refitting."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np

from preanchor.config import ExperimentConfig
from preanchor.features.extract import write_current_json
from preanchor.ids import file_hash
from preanchor.io import read_json, validate_artifact, validate_parquet
from preanchor.manifest import load_manifest, utc_now
from preanchor.probes import models as M
from preanchor.probes.dataset import load_features
from preanchor.probes.fit import FIT_STAGE, aggregate_units, read_summary
from preanchor.probes.metrics import average_precision
from preanchor.probes.select import SELECT_STAGE, mean_cv, selection_table


def validate_probes(run: Path, config: ExperimentConfig, root: Path) -> dict:
    manifest = load_manifest(run)
    if manifest["stages"].get(FIT_STAGE, {}).get("state") != "complete":
        raise ValueError("fit has not completed")
    selected = manifest["stages"].get(SELECT_STAGE, {}).get("state") == "complete"
    out = run / "probe_models"
    summary = read_summary(run)
    activation_run = root / "artifacts/runs" / manifest["input_hashes"]["activation_run_id"]
    _, hashes = load_features(root, config, activation_run)
    if hashes != manifest["input_hashes"]:
        raise ValueError("Stage 7 inputs differ from the run's recorded inputs")
    cv = validate_parquet(out / "cv_results.parquet")
    pred = validate_parquet(out / "fold_predictions.parquet")
    for name in ("cv_results.parquet", "fold_predictions.parquet"):
        if file_hash(out / name) != manifest["output_hashes"][f"probe_models/{name}"]:
            raise ValueError(f"{name} differs from the manifest")
    keys = [u["key"] for u in summary["units"]]
    pending = None
    if selected:
        pending = read_json(out / "frozen_selection.pending.json")
        keys.append(pending["residualized_control"]["unit_key"])
    cv_units, pred_units = aggregate_units(run, keys)
    if not (cv_units.equals(cv) and pred_units.equals(pred)):
        raise ValueError("Run-level tables differ from the validated units")
    rows = pred.to_pylist()
    if {r["split"] for r in rows} != {"train"}:
        raise ValueError("A non-training row is present in fold predictions")
    # Every training row predicted exactly once per (probe, layer, task, hp).
    counts: dict[tuple, set] = {}
    for r in rows:
        counts.setdefault((r["probe_name"], r["layer"], r["task"], r["hyperparameter"]),
                          set()).add(r["sentence_id"])
    n = summary["n_rows"]
    for key, ids in counts.items():
        if len(ids) != n:
            raise ValueError(f"Incomplete coverage for {key}: {len(ids)} of {n}")
    total_rows = sum(1 for _ in rows)
    if total_rows != n * len(counts):
        raise ValueError("Duplicate fold predictions")
    for c in ("score_raw", "importance_true"):
        if not np.isfinite(np.array(pred.column(c).to_pylist(), dtype=float)).all():
            raise ValueError(f"Non-finite {c}")
    probs = [p for p in pred.column("probability").to_pylist() if p is not None]
    if probs and (min(probs) < 0 or max(probs) > 1):
        raise ValueError("Probability outside [0, 1]")
    # Independent recomputation of the selection metric from stored predictions.
    by_key: dict[tuple, dict] = {}
    for r in rows:
        if r["task"] != M.BINARY:
            continue
        by_key.setdefault((r["probe_name"], r["layer"], r["hyperparameter"], r["fold"]),
                          {"y": [], "s": []})
        by_key[(r["probe_name"], r["layer"], r["hyperparameter"], r["fold"])]["y"].append(
            r["y_true"])
        by_key[(r["probe_name"], r["layer"], r["hyperparameter"], r["fold"])]["s"].append(
            r["score_raw"])
    max_diff = 0.0
    for r in cv.to_pylist():
        if r["task"] != M.BINARY:
            continue
        d = by_key[(r["probe_name"], r["layer"], r["hyperparameter"], r["fold"])]
        recomputed = average_precision(np.array(d["y"]), np.array(d["s"]))
        max_diff = max(max_diff, abs(recomputed - r["average_precision"]))
        if abs(recomputed - r["average_precision"]) > 1e-9:
            raise ValueError("Stored average precision differs from recomputation")
    prevalence_rows = [r for r in cv.to_pylist() if r["probe_name"] == "prevalence"]
    for r in prevalence_rows:
        if abs(r["average_precision"] - r["val_pos"] / r["n_val"]) > 1e-9:
            raise ValueError("Prevalence AUCPR does not equal fold prevalence")
    sites = sorted({r["layer"] for r in cv.to_pylist() if r["layer"] is not None})
    report = {"schema_version": 1, "ok": True, "run_id": run.name, "rows": n,
              "units": len(keys), "sites": len(sites), "cv_rows": cv.num_rows,
              "prediction_rows": pred.num_rows, "max_ap_recompute_diff": max_diff,
              "selected": selected, "validated_at": utc_now()}
    if selected:
        means = mean_cv(cv)
        best = selection_table(means)
        primary = best["activation_plus_observables"]
        if (primary[M.BINARY]["layer"], primary[M.BINARY]["hyperparameter"]) != (
                pending["binary"]["layer"], pending["binary"]["C"]):
            raise ValueError("Recomputed binary selection differs from the pending freeze")
        if (primary[M.CONTINUOUS]["layer"], primary[M.CONTINUOUS]["hyperparameter"]) != (
                pending["continuous"]["layer"], pending["continuous"]["alpha"]):
            raise ValueError("Recomputed Ridge selection differs from the pending freeze")
        if validate_artifact(out / "selected_model.joblib") != pending["selected_model_sha256"]:
            raise ValueError("selected_model.joblib hash differs from the pending freeze")
        payload = joblib.load(io.BytesIO((out / "selected_model.joblib").read_bytes()))
        if set(payload["pipelines"]) != set(payload["models"]):
            raise ValueError("Serialized pipelines do not match the model records")
        for key, record in payload["models"].items():
            model = payload["pipelines"][key].named_steps["model"]
            if record["task"] == M.BINARY and key.split(":")[0] != "prevalence" \
                    and abs(model.C - record["hyperparameter"]) > 0:
                raise ValueError(f"Serialized C differs for {key}")
        report["selection"] = {"binary": pending["binary"], "continuous": pending["continuous"]}
        if config.execution.mode == "mvp" and len(sites) != len(config.features["layers"]):
            raise ValueError("Not every activation site has CV results")
    write_current_json(run / "probe_models" / "probe_validation.json", report)
    return report
