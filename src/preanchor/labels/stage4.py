"""Stage 4: raw resample acquisition, Problem 4682 audit and canonical label build.

All outputs live under a run directory with manifest provenance; nothing is written to
hard-coded paths. Every expensive unit (one raw fork file) is an immutable checksum
unit; the audit and build are deterministic functions of the run's inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from preanchor.config import ExperimentConfig, config_hash
from preanchor.data.fetch import PREFIX, metadata, validate_source
from preanchor.ids import file_hash
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_artifact,
    validate_parquet,
    write_artifact,
    write_json,
    write_parquet,
)
from preanchor.labels.convergence import convergence_idx
from preanchor.labels.importance import (
    canonical_from_forks,
    legacy_from_forks,
    normalize_answer,
    smoothed_kl,
)
from preanchor.labels.similarity import ReplacementSimilarity, similarity_fingerprint
from preanchor.labels.threshold import train_threshold
from preanchor.labels.validation import validate_label_rows
from preanchor.logging import redact
from preanchor.manifest import check_resume, save_manifest, utc_now
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking

RAW_ROLE = "raw_resamples"
LOCK_FILES = "configs/locks/raw_resample_files.txt"
LOCK_METADATA = "configs/locks/raw_resample_metadata.json"


# ---------------------------------------------------------------------------
# Discovery of every raw fork file for the 40 selected trajectories
# ---------------------------------------------------------------------------
def fork_index(path: str) -> int:
    name = PurePosixPath(path).parent.name
    if not name.startswith("chunk_") or not name[6:].isdigit():
        raise ValueError(f"Unexpected fork directory: {path}")
    return int(name[6:])


def discover_raw(config: ExperimentConfig, *, api=None, progress=lambda: None) -> dict:
    api = api or HfApi()
    kwargs = dict(repo_id=config.dataset.id, repo_type="dataset", revision=config.dataset.revision)
    records, forks = [], {}
    for kind in ("correct", "incorrect"):
        for problem in sorted(config.dataset.problem_ids):
            progress()
            directory = f"{PREFIX}/{kind}_base_solution/problem_{problem}"
            entries = api.list_repo_tree(**kwargs, path_in_repo=directory, recursive=True)
            files = [e for e in entries if isinstance(e, RepoFile)
                     and e.path.startswith(f"{directory}/chunk_")
                     and PurePosixPath(e.path).name == "solutions.json"
                     and PurePosixPath(e.path).parent.parent.as_posix() == directory]
            indices = sorted(fork_index(e.path) for e in files)
            if not indices or indices != list(range(len(indices))):
                raise ValueError(f"Fork directories must be contiguous from zero: {directory}")
            forks[f"{problem}:{kind}"] = len(indices)
            records.extend(metadata(e, RAW_ROLE, problem, kind) for e in files)
    result = {"schema_version": 1, "dataset_id": config.dataset.id,
              "revision": config.dataset.revision, "forks_per_trajectory": forks,
              "files": sorted(records, key=lambda r: r["path"])}
    validate_raw_discovery(result, config)
    return result


def validate_raw_discovery(discovery: dict, config: ExperimentConfig) -> None:
    if (discovery.get("schema_version") != 1 or discovery.get("dataset_id") != config.dataset.id
            or discovery.get("revision") != config.dataset.revision):
        raise ValueError("Dataset identity mismatch")
    pairs = {(p, k) for p in config.dataset.problem_ids for k in ("correct", "incorrect")}
    if set(discovery["forks_per_trajectory"]) != {f"{p}:{k}" for p, k in pairs}:
        raise ValueError("Raw discovery must cover exactly the 40 selected trajectories")
    paths = [r["path"] for r in discovery["files"]]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Empty or duplicate raw allowlist")
    seen: dict[str, set[int]] = {}
    for row in discovery["files"]:
        path = PurePosixPath(row["path"])
        if path.is_absolute() or ".." in path.parts or any(c in row["path"] for c in "*?[]"):
            raise ValueError("Allowlist entries must be exact safe paths")
        if row["role"] != RAW_ROLE or (row["problem_id"], row["trace_kind"]) not in pairs:
            raise ValueError("Raw record outside the selected scope")
        base = f"{PREFIX}/{row['trace_kind']}_base_solution/problem_{row['problem_id']}"
        index = fork_index(row["path"])
        if row["path"] != f"{base}/chunk_{index}/solutions.json":
            raise ValueError("Path is outside the exact fork layout")
        for field, length in (("git_blob_sha1", 40), ("lfs_sha256", 64)):
            value = row[field]
            if field == "lfs_sha256" and value is None:
                continue
            if not isinstance(value, str) or len(value) != length \
                    or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid upstream hash")
        if not isinstance(row["size"], int) or row["size"] < 0:
            raise ValueError("Invalid source size")
        seen.setdefault(f"{row['problem_id']}:{row['trace_kind']}", set()).add(index)
    for key, count in discovery["forks_per_trajectory"].items():
        if seen.get(key) != set(range(count)):
            raise ValueError(f"Fork coverage mismatch for {key}")


def save_raw_discovery(discovery: dict, repo: Path) -> None:
    payload = "".join(r["path"] + "\n" for r in discovery["files"]).encode()
    write_artifact(repo / LOCK_FILES, payload)
    write_json(repo / LOCK_METADATA, discovery)


def load_raw_discovery(repo: Path, config: ExperimentConfig) -> dict:
    value = read_json(repo / LOCK_METADATA)
    validate_raw_discovery(value, config)
    path = repo / LOCK_FILES
    validate_artifact(path)
    if path.read_text() != "".join(r["path"] + "\n" for r in value["files"]):
        raise ValueError("Raw allowlist differs from pinned metadata")
    return value


# ---------------------------------------------------------------------------
# Resumable parallel fetch of raw fork files into immutable units
# ---------------------------------------------------------------------------
def raw_unit_key(record: dict, discovery: dict, identity: dict) -> str:
    return config_hash({"stage": "stage4_raw", "dataset_revision": discovery["revision"],
                        "record": record, "identity": identity})


def default_download(*, repo_id, repo_type, revision, filename, cache_dir):
    return Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type, revision=revision,
                                filename=filename, cache_dir=str(cache_dir)))


def fetch_raw_file(store: UnitStore, record: dict, discovery: dict, identity: dict,
                   cache_dir: Path, *, download=default_download, stop_event=None) -> dict:
    key = raw_unit_key(record, discovery, identity)
    started = time.monotonic()

    def writer(directory: Path) -> None:
        write_json(directory / "request.json", record)
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Download interrupted before request")
        source = download(repo_id=discovery["dataset_id"], repo_type="dataset",
                          revision=discovery["revision"], filename=record["path"],
                          cache_dir=cache_dir)
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Download interrupted before publication")
        validate_source(Path(source), record)
        write_artifact(directory / "solutions.json", Path(source).read_bytes())

    directory, created = store.produce(key, writer)
    path = directory / "solutions.json"
    digest = validate_source(path, record)
    return {**record, "unit_key": key, "local_path": str(path), "sha256": digest,
            "created": created, "elapsed_seconds": time.monotonic() - started}


def _stage_failure(directory: Path, stage_dir: Path, stage: str, manifest: dict,
                   attempt: str, exc: BaseException, extra: dict) -> None:
    failure = redact({"schema_version": 1, "stage": stage, "attempt": attempt, "at": utc_now(),
                      "exception": type(exc).__name__, "traceback": traceback.format_exc(),
                      "identity": manifest["identity"], **extra})
    write_json(stage_dir / "failures" / f"{attempt}.json", failure)
    atomic_bytes(directory / "failure.json", json_bytes(failure), replace=True)


def run_raw_fetch(directory: Path, repo: Path, config: ExperimentConfig, cache_dir: Path, *,
                  resume: bool = False, workers: int = 4, tracking: bool = True,
                  download=None, limit: int | None = None) -> dict:
    """Fetch every raw fork file for the 40 trajectories as independent checksum units."""
    discovery = load_raw_discovery(repo, config)
    records = discovery["files"][:limit] if limit else discovery["files"]
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    stage_name = "stage4_raw"
    with lock(directory / ".run.lock"):
        manifest = check_resume(directory, repo, config)
        inputs = {"raw_discovery": config_hash(discovery)}
        previous = manifest["stages"].get(stage_name)
        if previous and not resume:
            raise ValueError("Existing stage requires --resume")
        if previous and previous.get("limit") != limit:
            raise ValueError("Resume scope (limit) differs from the existing stage")
        if manifest["input_hashes"].get("raw_discovery", inputs["raw_discovery"]) \
                != inputs["raw_discovery"]:
            raise ValueError("Changed raw discovery")
        manifest["input_hashes"].update(inputs)
        manifest["identity"]["input_hashes"] = manifest["input_hashes"]
        stage = directory / "stages" / stage_name
        stage.mkdir(parents=True, exist_ok=True)
        attempt = uuid.uuid4().hex
        if previous and previous["state"] == "running":
            write_json(stage / "failures" / f"{previous['attempt']}.json",
                       {"schema_version": 1, "exception": "UnfinishedAttempt", "at": utc_now(),
                        "attempt": previous["attempt"], "signal": "unknown",
                        "recovery": "validate completed units and quarantine incomplete units"})
        session, tracking_record = (start_tracking(directory, manifest, config) if tracking
                                    else (None, {"mode": "disabled_for_test"}))
        manifest["wandb"].append(tracking_record)
        manifest["commands"].append({"command": "fetch-raw-resamples", "resume": resume,
                                     "workers": workers, "limit": limit, "attempt": attempt})
        manifest["stages"][stage_name] = {"state": "running", "attempt": attempt, "limit": limit}
        save_manifest(directory, manifest)
        runtime = Runtime(directory, stage_name, len(records), session)
        store = UnitStore(stage / "units")
        rows: dict[str, dict] = {}
        timings = []
        created = 0
        started = time.monotonic()
        stop_event = threading.Event()
        kwargs = {"download": download} if download is not None else {}
        try:
            with runtime, ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                queue = list(records)
                while queue or pending:
                    while queue and len(pending) < workers * 2:
                        record = queue.pop(0)
                        future = pool.submit(fetch_raw_file, store, record, discovery,
                                             manifest["identity"], cache_dir,
                                             stop_event=stop_event, **kwargs)
                        pending[future] = record["path"]
                    done, _ = wait(list(pending), timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        path = pending.pop(future)
                        row = future.result()  # Propagates worker failures.
                        created += row.pop("created")
                        timings.append({"path": path, "seconds": row.pop("elapsed_seconds")})
                        rows[path] = row
                        runtime.completed += 1
                        runtime.last_output = row["local_path"]
                    try:
                        runtime.pulse(force=bool(done))
                    except BaseException:
                        stop_event.set()
                        for future in pending:
                            future.cancel()
                        raise
                ordered = [rows[r["path"]] for r in records]
                result = {"schema_version": 1, "dataset_id": discovery["dataset_id"],
                          "revision": discovery["revision"],
                          "discovery_sha256": config_hash(discovery), "limit": limit,
                          "files": ordered}
                digest = write_json(stage / "source_manifest.json", result)
                write_json(stage / "SUCCESS.json", {"schema_version": 1,
                                                   "source_manifest_sha256": digest,
                                                   "files": len(ordered),
                                                   "bytes": sum(r["size"] for r in records)})
                manifest["output_hashes"]["stage4_raw_source_manifest"] = digest
                manifest["stages"][stage_name]["state"] = "complete"
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            _stage_failure(directory, stage, stage_name, manifest, attempt, exc,
                           {"completed": runtime.completed,
                            "resume": "fetch-raw-resamples --resume with unchanged identity"})
            manifest["stages"][stage_name]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            timing = {"schema_version": 1, "attempt": attempt, "created": created,
                      "completed": runtime.completed, "workers": workers,
                      "file_timings": timings, "elapsed_seconds": time.monotonic() - started,
                      "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "peak_gpu_memory_bytes": None, "gpu_used": False}
            write_json(stage / "attempts" / f"{attempt}.json", timing)
            manifest["stage_timings"].setdefault(stage_name, []).append(timing)
            if session:
                try:
                    session.finish(exit_code=0 if manifest["stages"][stage_name]["state"]
                                   == "complete" else 1)
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "offline records require backup"
            save_manifest(directory, manifest)
        return {**validate_raw_download(directory, repo, config), "created": created,
                "skipped": len(records) - created}


def validate_raw_download(directory: Path, repo: Path, config: ExperimentConfig) -> dict:
    discovery = load_raw_discovery(repo, config)
    manifest = check_resume(directory, repo, config)
    stage = directory / "stages/stage4_raw"
    result = read_json(stage / "source_manifest.json")
    limit = manifest["stages"]["stage4_raw"].get("limit")
    records = discovery["files"][:limit] if limit else discovery["files"]
    if manifest["input_hashes"].get("raw_discovery") != config_hash(discovery):
        raise ValueError("Changed raw discovery")
    if result["discovery_sha256"] != config_hash(discovery) or len(result["files"]) != len(records):
        raise ValueError("Raw manifest coverage mismatch")
    store = UnitStore(stage / "units")
    for record, row in zip(records, result["files"], strict=True):
        key = raw_unit_key(record, discovery, manifest["identity"])
        store.validate(key)
        path = stage / "units" / key / "solutions.json"
        if (row["unit_key"] != key or row["local_path"] != str(path)
                or any(row[k] != v for k, v in record.items())
                or validate_source(path, record) != row["sha256"]):
            raise ValueError("Raw record identity/hash mismatch")
    completion = read_json(stage / "SUCCESS.json")
    if completion["source_manifest_sha256"] != file_hash(stage / "source_manifest.json"):
        raise ValueError("Stage completion mismatch")
    return {"state": "passed", "run_id": directory.name, "files": len(records),
            "bytes": sum(r["size"] for r in records),
            "source_manifest_sha256": file_hash(stage / "source_manifest.json")}


# ---------------------------------------------------------------------------
# Shared loading helpers
# ---------------------------------------------------------------------------
def raw_fork_paths(raw_run: Path, source_manifest: dict) -> dict[str, dict[int, Path]]:
    """trajectory_id -> fork index -> raw file path, derived from unit keys (not stored paths)."""
    units = raw_run / "stages/stage4_raw/units"
    out: dict[str, dict[int, Path]] = {}
    for row in source_manifest["files"]:
        if row["role"] != RAW_ROLE:
            continue
        out.setdefault(f"{row['problem_id']}:{row['trace_kind']}", {})[fork_index(row["path"])] \
            = units / row["unit_key"] / "solutions.json"
    return out


def load_raw_run(raw_run: Path) -> tuple[dict, dict]:
    """The completed raw-acquisition run is an input: verify completion and hashes only."""
    from preanchor.manifest import load_manifest

    manifest = load_manifest(raw_run)
    if manifest["stages"].get("stage4_raw", {}).get("state") != "complete":
        raise ValueError("Raw resample run is not complete")
    if manifest["stages"]["stage4_raw"].get("limit"):
        raise ValueError("Raw resample run was limited; a full run is required")
    stage = raw_run / "stages/stage4_raw"
    source_manifest = read_json(stage / "source_manifest.json")
    completion = read_json(stage / "SUCCESS.json")
    digest = file_hash(stage / "source_manifest.json")
    if completion["source_manifest_sha256"] != digest \
            or manifest["output_hashes"].get("stage4_raw_source_manifest") != digest:
        raise ValueError("Raw run completion record does not match its source manifest")
    inputs = {"raw_run_id": raw_run.name, "raw_source_manifest": digest,
              "raw_run_manifest": file_hash(raw_run / "manifest.json")}
    return source_manifest, inputs


def load_fork(path: Path) -> list[dict]:
    """Checksum-validated raw fork file as a list of sample dicts."""
    if Path(str(path) + ".sha256").exists():
        validate_artifact(path)
    value = json.loads(path.read_bytes())
    if isinstance(value, dict):
        value = value.get("solutions", value.get("samples"))
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"Raw fork file is not a sample list: {path}")
    return value


def load_stage3(root: Path) -> tuple[list[dict], list[dict], dict[str, str]]:
    stage3 = root / "data/processed/stage3"
    trajectories = validate_parquet(stage3 / "trajectories.parquet").to_pylist()
    sentences = validate_parquet(stage3 / "sentences.parquet").to_pylist()
    hashes = {"stage3_trajectories": file_hash(stage3 / "trajectories.parquet"),
              "stage3_sentences": file_hash(stage3 / "sentences.parquet")}
    return trajectories, sentences, hashes


def final_answer_index(sentences: list[dict]) -> int | None:
    """Zero-based index of the final-answer-emission sentence (last ``\\boxed{``), if any."""
    boxed = [row["sentence_idx"] for row in sentences if "\\boxed{" in row["text"]]
    return max(boxed) if boxed else None


def fork_accuracy(samples: list[dict]) -> float | None:
    if not samples:
        return None
    return sum(bool(s.get("is_correct")) for s in samples) / len(samples)


# ---------------------------------------------------------------------------
# Per-trajectory scoring shared by the audit and the build
# ---------------------------------------------------------------------------
def score_trajectory(sentences: list[dict], forks: dict[int, Path],
                     similarity: ReplacementSimilarity, labels_config: dict) -> list[dict]:
    """Canonical/legacy importance, support counts and eligibility inputs per sentence."""
    alpha = float(labels_config.get("smoothing_alpha", 1e-9))
    sensitivities = [float(a) for a in labels_config.get("smoothing_sensitivities", [0.5, 1.0])]
    threshold = float(labels_config.get("similarity_threshold", 0.8))
    cache: dict[int, list[dict]] = {}

    def fork(index: int) -> list[dict] | None:
        if index not in forks:
            return None
        if index not in cache:
            cache[index] = load_fork(forks[index])
        return cache[index]

    accuracy = {index: fork_accuracy(fork(index)) for index in sorted(forks)}
    rows = []
    for sentence in sorted(sentences, key=lambda r: r["sentence_idx"]):
        idx = sentence["sentence_idx"]
        current, following = fork(idx), fork(idx + 1)
        record = {"sentence_id": sentence["sentence_id"], "sentence_idx": idx,
                  "importance_canonical": None, "importance_legacy": None,
                  "canonical_sensitivities": {}, "sample_count": 0, "dissimilar_count": 0,
                  "similar_count": 0, "no_answer_dissimilar_count": 0, "with_count": 0,
                  "support": "missing_adjacent_fork", "similarity_sha256": None,
                  "fork_accuracy": accuracy.get(idx)}
        if current and following:
            scores = similarity.scores(current)
            canonical = canonical_from_forks({idx: current, idx + 1: following}, idx, scores,
                                             alpha=alpha, threshold=threshold)
            record.update({k: canonical[k] for k in ("sample_count", "dissimilar_count",
                                                     "similar_count",
                                                     "no_answer_dissimilar_count",
                                                     "with_count", "support",
                                                     "importance_canonical")})
            record["similarity_sha256"] = similarity_fingerprint(scores)
            record["importance_legacy"] = legacy_from_forks(
                {idx: current, idx + 1: following}, idx, scores, alpha=1.0, threshold=threshold)
            if canonical["importance_canonical"] is not None:
                keep = [i for i, s in enumerate(scores) if s < threshold]
                without = [normalize_answer(current[i].get("answer")) for i in keep]
                with_side = [normalize_answer(r.get("answer")) for r in following]
                record["canonical_sensitivities"] = {
                    f"alpha_{a}": smoothed_kl(without, with_side, a) for a in sensitivities}
        rows.append(record)
        # Fork i is never needed again once sentence i is scored; keep memory flat.
        cache.pop(idx, None)
    return rows


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Problem 4682 audit (SPEC §3.6 gate)
# ---------------------------------------------------------------------------
def audit_4682(directory: Path, repo: Path, config: ExperimentConfig, root: Path,
               raw_run: Path, *, device: str = "cpu") -> dict:
    from scipy.stats import spearmanr

    stage_name = "stage4_audit"
    with lock(directory / ".run.lock"):
        manifest = check_resume(directory, repo, config)
        raw, raw_inputs = load_raw_run(raw_run)
        trajectories, sentences, hashes = load_stage3(root)
        target = config.labels["audit"]["trajectory"]
        problem_id, trace_kind = target.split(":")
        rows = [r for r in sentences if r["trajectory_id"] == target]
        forks = raw_fork_paths(raw_run, raw).get(target)
        if not rows or not forks:
            raise ValueError(f"Audit trajectory {target} is missing from Stage 3 or raw data")
        inputs = {**hashes, **raw_inputs}
        for key, value in inputs.items():
            if manifest["input_hashes"].get(key, value) != value:
                raise ValueError(f"Changed input: {key}")
        manifest["input_hashes"].update(inputs)
        manifest["identity"]["input_hashes"] = manifest["input_hashes"]
        similarity = ReplacementSimilarity(device=device)
        stage = directory / "stages/stage4"
        stage.mkdir(parents=True, exist_ok=True)
        attempt = uuid.uuid4().hex
        manifest["commands"].append({"command": "audit-upstream", "attempt": attempt,
                                     "device": device, "raw_run_id": raw_run.name})
        manifest["stages"][stage_name] = {"state": "running", "attempt": attempt}
        save_manifest(directory, manifest)
        started = time.monotonic()
        try:
            scored = {r["sentence_idx"]: r for r in score_trajectory(rows, forks, similarity,
                                                                    config.labels)}
            audit = []
            for sentence in sorted(rows, key=lambda r: r["sentence_idx"]):
                score = scored[sentence["sentence_idx"]]
                released = sentence["importance_released"]
                audit.append({"schema_version": 1, "trajectory_id": target,
                              "sentence_idx": sentence["sentence_idx"],
                              "importance_released": released,
                              "importance_legacy": score["importance_legacy"],
                              "importance_canonical": score["importance_canonical"],
                              "sample_count": score["sample_count"],
                              "dissimilar_count": score["dissimilar_count"],
                              "similar_count": score["similar_count"],
                              "support": score["support"],
                              "finite_nonnegative_released": _finite(released) and released >= 0})
            compared = [a for a in audit if a["support"] == "ok"
                        and _finite(a["importance_released"])
                        and _finite(a["importance_canonical"])
                        and _finite(a["importance_legacy"])]
            if len(compared) < 3:
                raise ValueError("Too few comparable rows for the audit")
            released = [a["importance_released"] for a in compared]
            canonical = [a["importance_canonical"] for a in compared]
            legacy = [a["importance_legacy"] for a in compared]
            adjacent = [a for a in audit if a["support"] != "missing_adjacent_fork"]
            finite_fraction = (sum(a["finite_nonnegative_released"] for a in adjacent)
                               / len(adjacent))
            rho_canonical = float(spearmanr(released, canonical).statistic)
            rho_legacy = float(spearmanr(released, legacy).statistic)
            minimum_rho = float(config.labels["audit"][
                "minimum_spearman_released_vs_legacy_or_canonical"])
            minimum_finite = float(config.labels["audit"]["minimum_finite_nonnegative_fraction"])
            gate = bool(finite_fraction >= minimum_finite
                        and max(rho_canonical, rho_legacy) >= minimum_rho)
            report = {"schema_version": 2, "trajectory_id": target, "problem_id": int(problem_id),
                      "trace_kind": trace_kind, "rows": len(audit),
                      "rows_with_adjacent_fork": len(adjacent), "rows_compared": len(compared),
                      "finite_nonnegative_fraction": finite_fraction,
                      "spearman_released_vs_canonical": rho_canonical,
                      "spearman_released_vs_legacy": rho_legacy,
                      "spearman_canonical_vs_legacy": float(spearmanr(canonical, legacy).statistic),
                      "gate": {"minimum_finite_nonnegative_fraction": minimum_finite,
                               "minimum_spearman": minimum_rho, "passed": gate},
                      "primary_field": "importance_released" if gate else "importance_canonical",
                      "label_source": "released" if gate else "canonical_fallback",
                      "dissimilar_summary": {
                          "mean_dissimilar_count": float(np.mean([a["dissimilar_count"]
                                                                  for a in adjacent])),
                          "rows_without_dissimilar": sum(a["support"] == "no_dissimilar_samples"
                                                         for a in audit)},
                      "similarity": similarity.identity(),
                      "legacy_defaults": {"alpha": 1.0, "use_similar_chunks": True},
                      "canonical_alpha": float(config.labels.get("smoothing_alpha", 1e-9)),
                      "inputs": inputs, "elapsed_seconds": time.monotonic() - started}
            digest = write_parquet(stage / "upstream_audit.parquet", pa.Table.from_pylist(audit))
            report["audit_parquet_sha256"] = digest
            report_digest = write_json(stage / "upstream_audit.json", report)
            manifest["output_hashes"]["stage4_upstream_audit"] = report_digest
            manifest["stages"][stage_name] = {"state": "complete", "attempt": attempt,
                                              "gate_passed": gate}
        except BaseException as exc:
            _stage_failure(directory, stage, stage_name, manifest, attempt, exc, {})
            manifest["stages"][stage_name]["state"] = "failed"
            raise
        finally:
            manifest["stage_timings"].setdefault(stage_name, []).append(
                {"schema_version": 1, "attempt": attempt,
                 "elapsed_seconds": time.monotonic() - started, "device": device})
            save_manifest(directory, manifest)
        return report


# ---------------------------------------------------------------------------
# Label build for all 40 trajectories
# ---------------------------------------------------------------------------
def build_rows(sentences: list[dict], scored: list[dict], labels_config: dict,
               label_source: str) -> tuple[list[dict], dict]:
    """Assemble one trajectory's label rows with eligibility masks applied."""
    consecutive = int(labels_config.get("convergence_consecutive", 5))
    by_idx = {s["sentence_idx"]: s for s in scored}
    ordered = sorted(sentences, key=lambda r: r["sentence_idx"])
    accuracies = [by_idx[r["sentence_idx"]]["fork_accuracy"] for r in ordered]
    known = [a for a in accuracies if a is not None]
    convergence = convergence_idx(known, consecutive) if known else None
    convergence_k4 = convergence_idx(known, 4) if known else None
    final_idx = final_answer_index(ordered) if labels_config.get("exclude_final_answer", True) \
        else None
    rows = []
    for sentence in ordered:
        idx = sentence["sentence_idx"]
        score = by_idx[idx]
        reason = None
        if not sentence.get("alignment_ok", True) or not sentence.get("eligible", True):
            reason = sentence.get("exclusion_reason") or "alignment"
        elif idx == 0 and sentence["char_start"] == 0:
            reason = "first_token"
        elif final_idx is not None and idx >= final_idx:
            reason = "final_answer"
        elif score["support"] == "missing_adjacent_fork":
            reason = "missing_adjacent_fork"
        elif convergence is not None and idx >= convergence:
            reason = "convergence"
        elif score["support"] == "no_dissimilar_samples":
            reason = "no_dissimilar_samples"
        elif label_source == "released" and not (_finite(sentence["importance_released"])
                                                 and sentence["importance_released"] >= 0):
            reason = "nonfinite_importance"
        elif not (_finite(score["importance_canonical"]) and score["importance_canonical"] >= 0):
            reason = "nonfinite_importance"
        eligible = reason is None
        primary = None
        if eligible:
            primary = (sentence["importance_released"] if label_source == "released"
                       else score["importance_canonical"])
        row = {**{k: sentence[k] for k in ("schema_version", "sentence_id", "trajectory_id",
                                            "problem_id", "trace_kind", "sentence_idx", "text",
                                            "text_sha256", "char_start", "char_end",
                                            "alignment_method", "alignment_ok")},
               "eligible": eligible, "exclusion_reason": reason,
               "importance_released": sentence["importance_released"],
               "importance_legacy": score["importance_legacy"],
               "importance_canonical": score["importance_canonical"],
               "importance_primary": primary, "label_source": label_source,
               "importance_log1p": math.log1p(primary) if primary is not None else None,
               "sample_count": score["sample_count"], "dissimilar_count": score["dissimilar_count"],
               "similar_count": score["similar_count"],
               "no_answer_dissimilar_count": score["no_answer_dissimilar_count"],
               "with_count": score["with_count"], "fork_accuracy": score["fork_accuracy"],
               "convergence_idx": convergence, "convergence_k4_idx": convergence_k4,
               "final_answer_idx": final_idx,
               "eligible_k4": eligible and (convergence_k4 is None or idx < convergence_k4),
               "similarity_sha256": score["similarity_sha256"]}
        for name, value in score["canonical_sensitivities"].items():
            row[f"importance_canonical_{name.replace('.', '_')}"] = value
        rows.append(row)
    summary = {"convergence_idx": convergence, "convergence_k4_idx": convergence_k4,
               "final_answer_idx": final_idx, "sentences": len(rows),
               "eligible": sum(r["eligible"] for r in rows)}
    return rows, summary


def build_labels(directory: Path, repo: Path, config: ExperimentConfig, root: Path,
                 raw_run: Path, *, device: str = "cpu", publish: bool = True) -> dict:
    stage_name = "stage4_labels"
    with lock(directory / ".run.lock"):
        manifest = check_resume(directory, repo, config)
        if manifest["stages"].get("stage4_audit", {}).get("state") != "complete":
            raise ValueError("The Problem 4682 audit must complete before building labels")
        stage = directory / "stages/stage4"
        audit = read_json(stage / "upstream_audit.json")
        label_source = audit["label_source"]
        raw, raw_inputs = load_raw_run(raw_run)
        trajectories, sentences, hashes = load_stage3(root)
        inputs = {**hashes, **raw_inputs,
                  "upstream_audit": file_hash(stage / "upstream_audit.json")}
        for key, value in inputs.items():
            if manifest["input_hashes"].get(key, value) != value:
                raise ValueError(f"Changed input: {key}")
        manifest["input_hashes"].update(inputs)
        manifest["identity"]["input_hashes"] = manifest["input_hashes"]
        forks_by_trajectory = raw_fork_paths(raw_run, raw)
        expected = {t["trajectory_id"] for t in trajectories}
        if set(forks_by_trajectory) != expected:
            raise ValueError("Raw resamples do not cover exactly the Stage 3 trajectories")
        attempt = uuid.uuid4().hex
        manifest["commands"].append({"command": "build-labels", "attempt": attempt,
                                     "device": device, "raw_run_id": raw_run.name})
        manifest["stages"][stage_name] = {"state": "running", "attempt": attempt}
        save_manifest(directory, manifest)
        similarity = ReplacementSimilarity(device=device)
        runtime = Runtime(directory, stage_name, len(expected))
        started = time.monotonic()
        try:
            rows, per_trajectory = [], {}
            with runtime:
                for trajectory_id in sorted(expected):
                    subset = [s for s in sentences if s["trajectory_id"] == trajectory_id]
                    scored = score_trajectory(subset, forks_by_trajectory[trajectory_id],
                                              similarity, config.labels)
                    built, summary = build_rows(subset, scored, config.labels, label_source)
                    rows.extend(built)
                    per_trajectory[trajectory_id] = summary
                    similarity.clear()
                    runtime.completed += 1
                    runtime.pulse(force=True)
                validation = validate_label_rows(rows, primary_source=label_source)
                train = set(config.splits.train)
                eligible = [r for r in rows if r["eligible"]]
                quantiles = sorted({float(config.labels.get("binary_train_quantile", 0.9)),
                                    *map(float, config.evaluation.get(
                                        "threshold_sensitivities", [0.8, 0.9, 0.95]))})
                thresholds = {f"q{int(round(q * 100))}": train_threshold(eligible, train, q)
                              for q in quantiles}
                primary_quantile = float(config.labels.get("binary_train_quantile", 0.9))
                primary_q = f"q{int(round(primary_quantile * 100))}"
                for row in rows:
                    row["anchor_threshold"] = thresholds[primary_q]
                    row["anchor"] = bool(row["eligible"]
                                         and row["importance_primary"] >= thresholds[primary_q])
                table = pa.Table.from_pylist(rows)
                labels_digest = write_parquet(stage / "labels.parquet", table)
                summary = {"schema_version": 2, "run_id": directory.name,
                           "label_source": label_source,
                           "primary_field": audit["primary_field"], "rows": len(rows),
                           "eligible": validation["eligible"],
                           "eligible_k4": sum(r["eligible_k4"] for r in rows),
                           "anchors": sum(r["anchor"] for r in rows),
                           "thresholds_train_only": thresholds,
                           "primary_threshold": primary_q,
                           "threshold_problem_ids": sorted(train),
                           "exclusions": dict(sorted(
                               {reason: sum(r["exclusion_reason"] == reason for r in rows)
                                for reason in {r["exclusion_reason"] for r in rows
                                               if r["exclusion_reason"]}}.items())),
                           "support": {"mean_dissimilar_count": float(np.mean(
                               [r["dissimilar_count"] for r in rows if r["sample_count"]])),
                               "rows_without_dissimilar": sum(
                                   r["exclusion_reason"] == "no_dissimilar_samples" for r in rows),
                               "rows_with_at_least_10_dissimilar": sum(
                                   r["dissimilar_count"] >= 10 for r in rows)},
                           "per_trajectory": per_trajectory, "similarity": similarity.identity(),
                           "audit": {k: audit[k] for k in ("spearman_released_vs_canonical",
                                                           "spearman_released_vs_legacy",
                                                           "finite_nonnegative_fraction",
                                                           "gate")},
                           "inputs": inputs, "labels_parquet_sha256": labels_digest,
                           "validation": validation, "device": device,
                           "elapsed_seconds": time.monotonic() - started}
                summary_digest = write_json(stage / "labels_summary.json", summary)
                if publish:
                    processed = root / "data/processed/stage4"
                    processed.mkdir(parents=True, exist_ok=True)
                    write_parquet(processed / "labels.parquet", table)
                    write_json(processed / "labels_summary.json", summary)
                manifest["output_hashes"]["stage4_labels"] = labels_digest
                manifest["output_hashes"]["stage4_labels_summary"] = summary_digest
                manifest["stages"][stage_name] = {"state": "complete", "attempt": attempt,
                                                  "label_source": label_source}
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            _stage_failure(directory, stage, stage_name, manifest, attempt, exc, {})
            manifest["stages"][stage_name]["state"] = "failed"
            runtime.stop_signal = None
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            manifest["stage_timings"].setdefault(stage_name, []).append(
                {"schema_version": 1, "attempt": attempt,
                 "elapsed_seconds": time.monotonic() - started, "device": device,
                 "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
            save_manifest(directory, manifest)
        return summary


def labels_fingerprint(path: Path) -> str:
    """SHA-256 over the sorted (sentence_id, primary) pairs for cross-run comparison."""
    table = validate_parquet(path).to_pylist()
    payload = json.dumps(sorted((r["sentence_id"], r["importance_primary"]) for r in table),
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
