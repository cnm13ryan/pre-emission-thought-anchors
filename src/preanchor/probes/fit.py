"""``preanchor fit``: every SPEC §3.10 model under four-fold grouped CV on training problems.

Work unit = one (model, site) pair. Each unit is an immutable ``UnitStore`` entry holding
its per-fold CV metrics and fold predictions; the run-level tables are rebuilt from
validated units at the end, so a partial run never publishes aggregate results.
"""

from __future__ import annotations

import math
import os
import resource
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed

from preanchor.config import ExperimentConfig, config_hash
from preanchor.features.extract import write_current_json
from preanchor.ids import cache_key
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_parquet,
    write_json,
    write_parquet,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, create_run, load_manifest, save_manifest, utc_now
from preanchor.probes import models as M
from preanchor.probes.dataset import (
    ActivationLoader,
    Partition,
    load_features,
    partition_rows,
)
from preanchor.probes.folds import assert_disjoint, fold_seed, grouped_folds
from preanchor.probes.metrics import auroc, average_precision, rmse, spearman
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking

STAGE = "stage8"
FIT_STAGE = "stage8_fit"
PARTITION = "train"
THRESHOLD_QUANTILE = 0.90

CV_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("probe_name", pa.string(), nullable=False),
    pa.field("layer", pa.int64(), nullable=True),
    pa.field("task", pa.string(), nullable=False),
    pa.field("hyperparameter", pa.float64(), nullable=True),
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("fold_seed", pa.int64(), nullable=False),
    pa.field("n_train", pa.int64(), nullable=False),
    pa.field("n_val", pa.int64(), nullable=False),
    pa.field("train_pos", pa.int64(), nullable=False),
    pa.field("val_pos", pa.int64(), nullable=False),
    pa.field("val_problems", pa.string(), nullable=False),
    pa.field("average_precision", pa.float64(), nullable=True),
    pa.field("auroc", pa.float64(), nullable=True),
    pa.field("spearman", pa.float64(), nullable=True),
    pa.field("rmse", pa.float64(), nullable=True),
    pa.field("converged", pa.bool_(), nullable=False),
])
PREDICTIONS_SCHEMA = pa.schema([
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
    pa.field("fold", pa.int64(), nullable=False),
    pa.field("y_true", pa.int64(), nullable=False),
    pa.field("importance_true", pa.float64(), nullable=False),
    pa.field("score_raw", pa.float64(), nullable=False),
    pa.field("probability", pa.float64(), nullable=True),
    pa.field("probability_calibrated", pa.float64(), nullable=True),
    pa.field("importance_predicted", pa.float64(), nullable=True),
])


def unit_specs(config: ExperimentConfig, n_sites: int) -> list[tuple[str, int | None]]:
    specs: list[tuple[str, int | None]] = [(b, None) for b in M.BASELINES]
    for name in M.ACTIVATION_MODELS:
        specs.extend((name, layer) for layer in range(n_sites))
    return specs


def unit_key(config: ExperimentConfig, manifest: dict, name: str, layer: int | None) -> str:
    return cache_key(stage=FIT_STAGE, scientific_config_hash=config.scientific_hash(),
                     model_revision=config.model.revision,
                     tokenizer_revision=config.model.tokenizer_revision, trace_sha256=None,
                     segmentation_version="released_chunks_v1", boundary_mapping_sha256=None,
                     generation_parameters={}, fork_idx=None, sample_seed=None,
                     code_version=manifest["identity"]["git_sha"],
                     input_hashes={**manifest["input_hashes"], "partition": PARTITION,
                                   "fold_seed": str(fold_seed(config))},
                     unit_id=f"{name}:{'' if layer is None else layer}")


def _fold_job(name: str, task: str, hps: list[float], X: np.ndarray, partition: Partition,
              fold: dict, config: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    tr, va = fold["train_idx"], fold["val_idx"]
    target = partition.y if task == M.BINARY else partition.importance_log1p
    cv_rows, pred_rows, seconds = [], [], []
    for hp in hps:
        started = time.perf_counter()
        pipeline = M.make_pipeline(name, task, hp, config).fit(X[tr], target[tr])
        out = M.scores(pipeline, task, X[va])
        elapsed = time.perf_counter() - started
        metric = {"average_precision": None, "auroc": None, "spearman": None, "rmse": None}
        if task == M.BINARY:
            metric["average_precision"] = average_precision(partition.y[va], out["score_raw"])
            metric["auroc"] = auroc(partition.y[va], out["score_raw"])
        else:
            metric["spearman"] = spearman(partition.importance_log1p[va], out["score_raw"])
            metric["rmse"] = rmse(partition.importance_log1p[va], out["score_raw"])
        for key, value in metric.items():
            if value is not None and not math.isfinite(value):
                metric[key] = None
        cv_rows.append({"schema_version": 1, "probe_name": name, "task": task,
                        "hyperparameter": None if name == "prevalence" else hp,
                        "fold": fold["fold"], "fold_seed": fold["seed"],
                        "n_train": fold["n_train"], "n_val": fold["n_val"],
                        "train_pos": fold["train_pos"], "val_pos": fold["val_pos"],
                        "val_problems": ",".join(map(str, fold["val_problems"])),
                        **metric, "converged": M.converged(pipeline)})
        seconds.append(elapsed)
        for j, i in enumerate(va):
            pred_rows.append({
                "schema_version": 1, "sentence_id": partition.sentence_id[i],
                "problem_id": int(partition.problem_id[i]),
                "trajectory_id": partition.trajectory_id[i], "split": partition.split,
                "lead_sentences": 0, "threshold_quantile": THRESHOLD_QUANTILE,
                "probe_name": name, "task": task,
                "hyperparameter": None if name == "prevalence" else hp, "fold": fold["fold"],
                "y_true": int(partition.y[i]),
                "importance_true": float(partition.importance_log1p[i]),
                "score_raw": float(out["score_raw"][j]),
                "probability": float(out["probability"][j]) if "probability" in out else None,
                "probability_calibrated": None,
                "importance_predicted": (float(out["importance_predicted"][j])
                                         if "importance_predicted" in out else None)})
        if name == "prevalence":
            break
    return cv_rows, pred_rows, seconds


def run_unit(name: str, layer: int | None, partition: Partition, site: np.ndarray | None,
             folds: list[dict], config: ExperimentConfig,
             workers: int) -> tuple[pa.Table, pa.Table, dict]:
    """CV tables for one unit (byte-deterministic) plus non-deterministic timing."""
    X = M.design(name, partition, site)
    tasks = [M.BINARY] if name == "prevalence" else [M.BINARY, M.CONTINUOUS]
    jobs = [(task, fold) for task in tasks for fold in folds]
    results = Parallel(n_jobs=max(1, min(workers, len(jobs))), backend="loky")(
        delayed(_fold_job)(name, task, [None] if name == "prevalence"
                           else M.hyperparameters(task, config), X, partition, fold, config)
        for task, fold in jobs)
    cv_rows = [dict(r, layer=layer) for rows, _, _ in results for r in rows]
    pred_rows = [dict(r, layer=layer) for _, rows, _ in results for r in rows]
    timing = {"fit_seconds_total": float(sum(s for _, _, sec in results for s in sec)),
              "fits": int(sum(len(sec) for _, _, sec in results)), "n_features": int(X.shape[1])}
    cv_rows.sort(key=lambda r: (r["task"], r["hyperparameter"] or 0.0, r["fold"]))
    pred_rows.sort(key=lambda r: (r["task"], r["hyperparameter"] or 0.0, r["fold"],
                                  r["trajectory_id"], r["sentence_id"]))
    return (pa.Table.from_pylist(cv_rows, schema=CV_SCHEMA),
            pa.Table.from_pylist(pred_rows, schema=PREDICTIONS_SCHEMA), timing)


def prepare_run(root: Path, repo: Path, config: ExperimentConfig, activation_run: Path,
                command: str) -> Path:
    _, hashes = load_features(root, config, activation_run)
    run = create_run(root, repo, config, [command])
    manifest = load_manifest(run)
    manifest["input_hashes"] = dict(hashes)
    manifest["identity"]["input_hashes"] = dict(hashes)
    manifest["gpu_gate"] = "cpu_only_stage"
    manifest["stages"][FIT_STAGE] = {"state": "prepared", "partition": PARTITION}
    save_manifest(run, manifest)
    return run


def load_partition(root: Path, config: ExperimentConfig, activation_run: Path,
                   split: str = PARTITION) -> tuple[Partition, ActivationLoader, dict]:
    table, hashes = load_features(root, config, activation_run)
    partition = partition_rows(table, config, split)
    return partition, ActivationLoader(activation_run, partition, config), hashes


def execute(run: Path, repo: Path, config: ExperimentConfig, *, resume: bool = False,
            workers: int | None = None, limit_sites: int | None = None,
            tracking: bool = True, interrupt_after: int | None = None) -> dict:
    """Fit every unit (resumable), then rebuild the run-level tables from validated units."""
    root = run.parents[2]
    assert_disjoint(config)
    M.check_frozen(config)
    if limit_sites is not None and config.execution.mode != "smoke":
        raise ValueError("Site limits are for smoke configurations only")
    workers = workers or int(config.probes["max_cpu_workers"])
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        state = manifest["stages"].get(FIT_STAGE, {}).get("state")
        if state not in ("prepared", "running", "failed", "complete") or (
                state != "prepared" and not resume):
            raise ValueError(f"{FIT_STAGE} is {state}; pass --resume to continue")
        activation_run = root / "artifacts/runs" / manifest["input_hashes"]["activation_run_id"]
        partition, loader, hashes = load_partition(root, config, activation_run)
        if hashes != manifest["input_hashes"]:
            raise ValueError("Stage 7 inputs changed since this run was prepared")
        folds = grouped_folds(partition.y, partition.problem_id, config)
        n_sites = loader.n_sites if limit_sites is None else min(limit_sites, loader.n_sites)
        specs = unit_specs(config, n_sites)
        attempt = uuid.uuid4().hex
        stage_dir = run / "stages" / STAGE
        stage_dir.mkdir(exist_ok=True)
        session, record = (start_tracking(run, manifest, config, stage=FIT_STAGE) if tracking
                           else (None, {"mode": "disabled_for_test", "url": None}))
        manifest["wandb"].append(record)
        manifest["commands"].append({"command": "fit", "attempt": attempt, "resume": resume,
                                     "workers": workers, "limit_sites": limit_sites})
        manifest["fold_seeds"] = [{"purpose": "cv_folds", "seed": f["seed"], "fold": f["fold"],
                                   "val_problems": f["val_problems"]} for f in folds]
        manifest["stages"][FIT_STAGE] = {"state": "running", "attempt": attempt,
                                         "partition": PARTITION, "n_units": len(specs),
                                         "n_sites": n_sites, "n_rows": len(partition)}
        save_manifest(run, manifest)
        runtime = Runtime(run, FIT_STAGE, len(specs), session)
        store = UnitStore(run / "probe_models" / "units")
        started = time.monotonic()
        units: list[dict] = []
        result: dict = {}
        try:
            with runtime:
                for n, (name, layer) in enumerate(specs):
                    if interrupt_after is not None and n >= interrupt_after:
                        raise KeyboardInterrupt("test interruption")
                    key = unit_key(config, manifest, name, layer)
                    site = loader.site(layer) if M.uses_activation(name) else None

                    timing: dict = {}

                    def writer(directory: Path, name=name, layer=layer, site=site,
                               timing=timing) -> None:
                        cv, pred, took = run_unit(name, layer, partition, site, folds, config,
                                                  workers)
                        timing.update(took)
                        write_parquet(directory / "cv_results.parquet", cv)
                        write_parquet(directory / "fold_predictions.parquet", pred)
                        write_json(directory / "unit.json", {
                            "schema_version": 1, "probe_name": name, "layer": layer,
                            "partition": PARTITION, "n_rows": len(partition),
                            "fold_seed": fold_seed(config)})

                    unit_started = time.monotonic()
                    directory, created = store.produce(key, writer)
                    units.append({"key": key, "probe_name": name, "layer": layer,
                                  "created": created, "attempt": attempt if created else None,
                                  "wall_seconds": time.monotonic() - unit_started, **timing})
                    runtime.completed += 1
                    runtime.last_output = str(directory.relative_to(run))
                    runtime.pulse(force=True)
                result = finalize(run, config, manifest, partition, folds, units, stage_dir,
                                  attempt, started)
                manifest["stages"][FIT_STAGE].update({"state": "complete", "attempt": attempt})
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": manifest["run_id"],
                              "stage": FIT_STAGE, "attempt": attempt, "at": utc_now(),
                              "completed": runtime.completed, "identity": manifest["identity"],
                              "exception": type(exc).__name__,
                              "traceback": traceback.format_exc(),
                              "recovery": "fit --resume --run-id <id>"})
            write_json(stage_dir / "failures" / f"{attempt}.json", failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"][FIT_STAGE]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            manifest["stage_timings"].setdefault(FIT_STAGE, []).append(
                {"schema_version": 1, "attempt": attempt, "workers": workers,
                 "elapsed_seconds": time.monotonic() - started,
                 "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "cpu_count": os.cpu_count()})
            if session:
                try:
                    session.finish(exit_code=0 if manifest["stages"][FIT_STAGE]["state"]
                                   == "complete" else 1)
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "offline records require backup"
            save_manifest(run, manifest)
        return result


def aggregate_units(run: Path, keys: list[str]) -> tuple[pa.Table, pa.Table]:
    store = UnitStore(run / "probe_models" / "units")
    cv, pred = [], []
    for key in keys:
        store.validate(key)
        cv.append(validate_parquet(store.root / key / "cv_results.parquet"))
        pred.append(validate_parquet(store.root / key / "fold_predictions.parquet"))
    return (pa.concat_tables(cv).combine_chunks(), pa.concat_tables(pred).combine_chunks())


def finalize(run: Path, config: ExperimentConfig, manifest: dict, partition: Partition,
             folds: list[dict], units: list[dict], stage_dir: Path, attempt: str,
             started: float) -> dict:
    cv, pred = aggregate_units(run, [u["key"] for u in units])
    out = run / "probe_models"
    for name in ("cv_results.parquet", "fold_predictions.parquet"):
        for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
            (out / (name + suffix)).unlink(missing_ok=True)
    cv_sha = write_parquet(out / "cv_results.parquet", cv)
    pred_sha = write_parquet(out / "fold_predictions.parquet", pred)
    rows_sha = config_hash(partition.sentence_id)
    summary = {"schema_version": 1, "run_id": run.name, "stage": FIT_STAGE, "attempt": attempt,
               "stage_tag": "exploratory", "preregistered": True,
               "partition": PARTITION, "n_rows": len(partition),
               "n_positive": int(partition.y.sum()), "prevalence": float(partition.y.mean()),
               "anchor_threshold": partition.anchor_threshold,
               "train_rows_sha256": rows_sha, "units": units,
               "folds": [{k: v for k, v in f.items() if k not in ("train_idx", "val_idx")}
                         for f in folds],
               "cv_results_sha256": cv_sha, "fold_predictions_sha256": pred_sha,
               "cv_rows": cv.num_rows, "prediction_rows": pred.num_rows,
               "elapsed_seconds": time.monotonic() - started}
    write_current_json(stage_dir / "summary.json", summary, attempt)
    manifest["output_hashes"]["probe_models/cv_results.parquet"] = cv_sha
    manifest["output_hashes"]["probe_models/fold_predictions.parquet"] = pred_sha
    return summary


def read_summary(run: Path) -> dict:
    return read_json(run / "stages" / STAGE / "summary.json")
