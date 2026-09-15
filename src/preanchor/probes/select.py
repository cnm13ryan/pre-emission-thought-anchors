"""``preanchor select``: choose layer and regularisation from training CV alone.

Binary selection maximises mean train-CV average precision; Ridge selection maximises mean
train-CV Spearman with RMSE as tie-breaker (SPEC §3.10). The residualised control is then
fitted at the selected site, every family is refitted on all training rows, and a pending
freeze record is written for Stage 9. No validation or test row is read here.
"""

from __future__ import annotations

import io
import math
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa

from preanchor.config import ExperimentConfig
from preanchor.features.extract import write_current_json
from preanchor.ids import file_hash
from preanchor.io import UnitStore, lock, validate_parquet, write_artifact, write_parquet
from preanchor.manifest import check_resume, save_manifest
from preanchor.probes import models as M
from preanchor.probes.fit import (
    FIT_STAGE,
    THRESHOLD_QUANTILE,
    aggregate_units,
    load_partition,
    read_summary,
    run_unit,
    unit_key,
)
from preanchor.probes.folds import fold_seed, grouped_folds

SELECT_STAGE = "stage8_select"
PRIMARY_MODEL = "activation_plus_observables"


def mean_cv(cv: pa.Table) -> list[dict]:
    """Mean metrics per (probe, layer, task, hyperparameter) over folds, in a fixed order."""
    groups: dict[tuple, list[dict]] = {}
    for r in cv.to_pylist():
        groups.setdefault((r["probe_name"], r["layer"], r["task"], r["hyperparameter"]),
                          []).append(r)
    out = []
    for (name, layer, task, hp), rows in groups.items():
        folds = sorted(r["fold"] for r in rows)
        if folds != list(range(len(folds))):
            raise ValueError(f"Incomplete folds for {name} layer {layer} hp {hp}")
        metric = {}
        for key in ("average_precision", "auroc", "spearman", "rmse"):
            values = [r[key] for r in rows]
            metric[key] = (None if any(v is None for v in values)
                           else float(np.mean(values)))
        out.append({"probe_name": name, "layer": layer, "task": task, "hyperparameter": hp,
                    "n_folds": len(folds), "converged": all(r["converged"] for r in rows),
                    **metric})
    out.sort(key=lambda r: (r["probe_name"], -1 if r["layer"] is None else r["layer"],
                            r["task"], r["hyperparameter"] or 0.0))
    return out


def _best(rows: list[dict], task: str) -> dict | None:
    """Deterministic argmax: metric, then tie-breaks (lower RMSE, smaller hp, lower layer)."""
    candidates = [r for r in rows if r["task"] == task]
    if task == M.BINARY:
        keyed = [(-(r["average_precision"] if r["average_precision"] is not None else -math.inf),
                  r["hyperparameter"] or 0.0, r["layer"] or -1, r) for r in candidates]
    else:
        keyed = [(-(r["spearman"] if r["spearman"] is not None else -math.inf),
                  r["rmse"] if r["rmse"] is not None else math.inf,
                  r["hyperparameter"] or 0.0, r["layer"] or -1, r) for r in candidates]
    if not keyed:
        return None
    keyed.sort(key=lambda t: t[:-1])
    return keyed[0][-1]


def selection_table(means: list[dict]) -> dict:
    """Best configuration per (family, task); the primary choice is the SPEC one."""
    families = sorted({r["probe_name"] for r in means})
    best = {}
    for family in families:
        rows = [r for r in means if r["probe_name"] == family]
        best[family] = {task: _best(rows, task) for task in (M.BINARY, M.CONTINUOUS)}
    return best


def layer_curve(means: list[dict], family: str, task: str) -> list[dict]:
    """Per-site best hyperparameter row, for the required per-layer curve."""
    curve = []
    layers = sorted({r["layer"] for r in means if r["probe_name"] == family
                     and r["layer"] is not None})
    for layer in layers:
        curve.append(_best([r for r in means if r["probe_name"] == family
                            and r["layer"] == layer], task))
    return curve


def execute(run: Path, repo: Path, config: ExperimentConfig, *, workers: int | None = None,
            tracking: bool = False) -> dict:
    root = run.parents[2]
    workers = workers or int(config.probes["max_cpu_workers"])
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        if manifest["stages"].get(FIT_STAGE, {}).get("state") != "complete":
            raise ValueError("fit must be complete before selection")
        if manifest["stages"].get(SELECT_STAGE, {}).get("state") == "complete":
            raise ValueError("Selection is complete; a new selection needs a new run")
        summary = read_summary(run)
        out = run / "probe_models"
        for suffix in ("", ".sha256"):
            (out / ("selected_model.joblib" + suffix)).unlink(missing_ok=True)
        cv = validate_parquet(out / "cv_results.parquet")
        if file_hash(out / "cv_results.parquet") != summary["cv_results_sha256"]:
            raise ValueError("cv_results.parquet differs from the fit summary")
        attempt = uuid.uuid4().hex
        started = time.monotonic()
        manifest["commands"].append({"command": "select", "attempt": attempt})
        manifest["stages"][SELECT_STAGE] = {"state": "running", "attempt": attempt}
        save_manifest(run, manifest)
        try:
            means = mean_cv(cv)
            fit_units = {u["probe_name"] for u in summary["units"]}
            if any(f not in fit_units for f in M.BASELINES + M.ACTIVATION_MODELS):
                raise ValueError("Not every SPEC §3.10 family has CV results")
            best = selection_table(means)
            binary = best[PRIMARY_MODEL][M.BINARY]
            ridge = best[PRIMARY_MODEL][M.CONTINUOUS]
            if binary is None or ridge is None:
                raise ValueError("Primary model has no CV results")
            layer = int(binary["layer"])
            # Residualised control at the selected site, same grid, same folds.
            activation_run = root / "artifacts/runs" / manifest["input_hashes"][
                "activation_run_id"]
            partition, loader, hashes = load_partition(root, config, activation_run)
            if hashes != manifest["input_hashes"]:
                raise ValueError("Stage 7 inputs changed since fit")
            folds = grouped_folds(partition.y, partition.problem_id, config)
            store = UnitStore(out / "units")
            key = unit_key(config, manifest, M.RESIDUALIZED, layer)
            site = loader.site(layer)

            def writer(directory: Path) -> None:
                cv_r, pred_r, _ = run_unit(M.RESIDUALIZED, layer, partition, site, folds,
                                           config, workers)
                write_parquet(directory / "cv_results.parquet", cv_r)
                write_parquet(directory / "fold_predictions.parquet", pred_r)

            store.produce(key, writer)
            keys = [u["key"] for u in summary["units"]] + [key]
            cv_all, pred_all = aggregate_units(run, keys)
            for name in ("cv_results.parquet", "fold_predictions.parquet"):
                for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
                    (out / (name + suffix)).unlink(missing_ok=True)
            cv_sha = write_parquet(out / "cv_results.parquet", cv_all)
            pred_sha = write_parquet(out / "fold_predictions.parquet", pred_all)
            means = mean_cv(cv_all)
            best = selection_table(means)
            # Refit every family at its selected configuration on all training rows.
            refits, fitted = {}, {}
            for family, tasks in best.items():
                for task, row in tasks.items():
                    if row is None:
                        continue
                    site_f = loader.site(int(row["layer"])) if M.uses_activation(family) \
                        else None
                    X = M.design(family, partition, site_f)
                    target = partition.y if task == M.BINARY else partition.importance_log1p
                    pipeline = M.make_pipeline(family, task, row["hyperparameter"], config)
                    pipeline.fit(X, target)
                    fitted[f"{family}:{task}"] = pipeline
                    refits[f"{family}:{task}"] = {
                        "probe_name": family, "task": task, "layer": row["layer"],
                        "hyperparameter": row["hyperparameter"],
                        "mean_cv": {k: row[k] for k in ("average_precision", "auroc",
                                                         "spearman", "rmse")},
                        "converged": M.converged(pipeline), "n_features": int(X.shape[1])}
            payload = {"schema_version": 1, "run_id": run.name, "fold_seed": fold_seed(config),
                       "train_rows_sha256": summary["train_rows_sha256"],
                       "selected": {"binary": f"{PRIMARY_MODEL}:{M.BINARY}",
                                    "continuous": f"{PRIMARY_MODEL}:{M.CONTINUOUS}"},
                       "models": refits, "pipelines": fitted}
            buffer = io.BytesIO()
            joblib.dump(payload, buffer)
            model_sha = write_artifact(out / "selected_model.joblib", buffer.getvalue())
            pending = {
                "schema_version": 1, "run_id": run.name, "stage": SELECT_STAGE,
                "attempt": attempt, "status": "pending", "stage_tag": "exploratory",
                "preregistered": True,
                "selection_basis": "train_cv_only", "selection_metric":
                    config.probes["selection_metric"],
                "ridge_selection_metric": config.probes["ridge_selection_metric"],
                "ridge_tiebreak_metric": config.probes["ridge_tiebreak_metric"],
                "binary": {"probe_name": PRIMARY_MODEL, "layer": layer,
                           "C": binary["hyperparameter"],
                           "mean_cv_average_precision": binary["average_precision"]},
                "continuous": {"probe_name": PRIMARY_MODEL, "layer": ridge["layer"],
                               "alpha": ridge["hyperparameter"],
                               "mean_cv_spearman": ridge["spearman"],
                               "mean_cv_rmse": ridge["rmse"]},
                "residualized_control": {"layer": layer, "unit_key": key,
                                         "best": best[M.RESIDUALIZED]},
                "per_family_best": best,
                "layer_curve": {f: {t: layer_curve(means, f, t) for t in (M.BINARY,
                                                                          M.CONTINUOUS)}
                                for f in M.ACTIVATION_MODELS},
                "anchor_threshold": summary["anchor_threshold"],
                "threshold_quantile": THRESHOLD_QUANTILE,
                "primary_features": list(config.features["observables"]),
                "fold_seed": fold_seed(config), "n_train_rows": summary["n_rows"],
                "n_train_positive": summary["n_positive"],
                "train_rows_sha256": summary["train_rows_sha256"],
                "input_hashes": manifest["input_hashes"],
                "code_commit": manifest["identity"]["git_sha"],
                "scientific_config_hash": manifest["identity"]["scientific_config_hash"],
                "cv_results_sha256": cv_sha, "fold_predictions_sha256": pred_sha,
                "selected_model_sha256": model_sha,
                "elapsed_seconds": time.monotonic() - started}
            pending_sha = write_current_json(out / "frozen_selection.pending.json", pending,
                                             attempt)
            manifest["output_hashes"].update({
                "probe_models/cv_results.parquet": cv_sha,
                "probe_models/fold_predictions.parquet": pred_sha,
                "probe_models/selected_model.joblib": model_sha,
                "probe_models/frozen_selection.pending.json": pending_sha})
            manifest["stages"][SELECT_STAGE] = {"state": "complete", "attempt": attempt,
                                                "binary_layer": layer,
                                                "binary_C": binary["hyperparameter"],
                                                "ridge_layer": ridge["layer"],
                                                "ridge_alpha": ridge["hyperparameter"]}
            return pending
        except BaseException:
            manifest["stages"][SELECT_STAGE]["state"] = "failed"
            raise
        finally:
            manifest["stage_timings"].setdefault(SELECT_STAGE, []).append(
                {"schema_version": 1, "attempt": attempt,
                 "elapsed_seconds": time.monotonic() - started})
            save_manifest(run, manifest)
