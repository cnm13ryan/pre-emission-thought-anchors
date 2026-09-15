"""CPU-only selective fetch orchestration; each source file is one durable unit."""

import resource
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from preanchor.config import config_hash
from preanchor.data.fetch import fetch_file, load_discovery, source_key, validate_source
from preanchor.ids import file_hash
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    same_logical_path,
    write_json,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, save_manifest, utc_now
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking


def selected_records(discovery, smoke):
    if not smoke:
        return discovery["files"]
    # Exercise every JSON layout and a representative raw fork, without sampling labels.
    return [
        r
        for r in discovery["files"]
        if r["problem_id"] == 4682
        and r["trace_kind"] == "correct"
        and (r["role"] != "audit_resamples" or "/chunk_0/" in r["path"])
    ]


def validate_download(directory, repo, config):
    discovery = load_discovery(repo, config)
    manifest = check_resume(directory, repo, config)
    stage = directory / "stages/stage2"
    result = read_json(stage / "source_manifest.json")
    scope = manifest["stages"]["stage2"]["smoke"]
    records = selected_records(discovery, scope)
    if manifest["input_hashes"] != {"dataset_discovery": config_hash(discovery)}:
        raise ValueError("Changed input discovery")
    if result["discovery_sha256"] != config_hash(discovery) or len(result["files"]) != len(records):
        raise ValueError("Source manifest coverage mismatch")
    store = UnitStore(stage / "units")
    for record, row in zip(records, result["files"], strict=True):
        key = source_key(record, discovery, manifest["identity"])
        store.validate(key)
        path = stage / "units" / key / Path(record["path"]).name
        if (
            row["unit_key"] != key
            or not same_logical_path(row["local_path"], path, directory)
            or any(row[k] != v for k, v in record.items())
            or validate_source(path, record) != row["sha256"]
        ):
            raise ValueError("Source record identity/hash mismatch")
    completion = read_json(stage / "SUCCESS.json")
    if completion["source_manifest_sha256"] != file_hash(stage / "source_manifest.json"):
        raise ValueError("Stage completion mismatch")
    return {
        "state": "passed",
        "files": len(records),
        "bytes": sum(r["size"] for r in records),
        "smoke": scope,
        "source_manifest_sha256": file_hash(stage / "source_manifest.json"),
    }


def run_fetch(
    directory,
    repo,
    config,
    cache_dir,
    *,
    resume=False,
    smoke=False,
    delay_seconds=0,
    tracking=True,
    download=None,
):
    discovery = load_discovery(repo, config)
    records = selected_records(discovery, smoke)
    if delay_seconds and (not smoke or not 0 <= delay_seconds <= 60):
        raise ValueError("Interruption delay is restricted to small smoke runs")
    with lock(directory / ".run.lock"):
        manifest = check_resume(directory, repo, config)
        inputs = {"dataset_discovery": config_hash(discovery)}
        previous = manifest["stages"].get("stage2")
        if previous and (not resume or previous["smoke"] != smoke):
            raise ValueError("Existing stage requires --resume with unchanged scope")
        if manifest["input_hashes"] and manifest["input_hashes"] != inputs:
            raise ValueError("Changed input discovery")
        manifest["input_hashes"] = inputs
        manifest["identity"]["input_hashes"] = inputs
        stage = directory / "stages/stage2"
        stage.mkdir(parents=True, exist_ok=True)
        attempt = uuid.uuid4().hex
        if previous and previous["state"] == "running":
            write_json(
                stage / "failures" / f"{previous['attempt']}.json",
                {
                    "schema_version": 1,
                    "exception": "UnfinishedAttempt",
                    "at": utc_now(),
                    "attempt": previous["attempt"],
                    "signal": "unknown",
                    "recovery": "validate completed units and quarantine incomplete units",
                },
            )
        session, tracking_record = (
            start_tracking(directory, manifest, config)
            if tracking
            else (None, {"mode": "disabled_for_test"})
        )
        manifest["wandb"].append(tracking_record)
        manifest["commands"].append(
            {
                "command": "fetch",
                "resume": resume,
                "smoke": smoke,
                "delay_seconds": delay_seconds,
                "attempt": attempt,
            }
        )
        manifest["stages"]["stage2"] = {"state": "running", "smoke": smoke, "attempt": attempt}
        save_manifest(directory, manifest)
        runtime = Runtime(directory, "stage2", len(records), session)
        store = UnitStore(stage / "units")
        rows, timings = [], []
        started = time.monotonic()
        current = None
        created = 0
        try:
            with runtime:
                for record in records:
                    current = record["path"]
                    # Network I/O is off-thread so the owning main loop continues its heartbeat.
                    # On graceful interruption, retain incomplete units for quarantine on resume.
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        kwargs = {"download": download} if download is not None else {}
                        stop_event = threading.Event()
                        kwargs.update(delay_seconds=delay_seconds, stop_event=stop_event)
                        task = pool.submit(
                            fetch_file,
                            store,
                            record,
                            discovery,
                            manifest["identity"],
                            cache_dir,
                            **kwargs,
                        )
                        while True:
                            try:
                                row = task.result(timeout=0.1)
                                break
                            except TimeoutError:
                                if task.done():
                                    # Propagate an actual worker timeout.
                                    task.result()
                                try:
                                    runtime.pulse()
                                except BaseException:
                                    stop_event.set()
                                    raise
                    created += row.pop("created")
                    timings.append({"path": current, "seconds": row.pop("elapsed_seconds")})
                    rows.append(row)
                    runtime.completed += 1
                    runtime.last_output = row["local_path"]
                    runtime.pulse(force=True)
                result = {
                    "schema_version": 1,
                    "dataset_id": discovery["dataset_id"],
                    "revision": discovery["revision"],
                    "discovery_sha256": config_hash(discovery),
                    "smoke": smoke,
                    "files": rows,
                }
                digest = write_json(stage / "source_manifest.json", result)
                write_json(
                    stage / "SUCCESS.json",
                    {"schema_version": 1, "source_manifest_sha256": digest, "files": len(rows)},
                )
                manifest["output_hashes"]["stage2_source_manifest"] = digest
                manifest["stages"]["stage2"]["state"] = "complete"
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            failure = redact(
                {
                    "schema_version": 1,
                    "stage": "stage2",
                    "attempt": attempt,
                    "at": utc_now(),
                    "path": current,
                    "exception": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "identity": manifest["identity"],
                    "resume": "fetch --resume with unchanged code/config/input/scope",
                }
            )
            write_json(stage / "failures" / f"{attempt}.json", failure)
            atomic_bytes(directory / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"]["stage2"]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            timing = {
                "schema_version": 1,
                "attempt": attempt,
                "created": created,
                "completed": runtime.completed,
                "file_timings": timings,
                "elapsed_seconds": time.monotonic() - started,
                "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_gpu_memory_bytes": None,
                "gpu_used": False,
            }
            write_json(stage / "attempts" / f"{attempt}.json", timing)
            manifest["stage_timings"].setdefault("stage2", []).append(timing)
            if session:
                try:
                    session.finish(
                        exit_code=0 if manifest["stages"]["stage2"]["state"] == "complete" else 1
                    )
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "offline records require backup"
            save_manifest(directory, manifest)
        return {
            **validate_download(directory, repo, config),
            "run_id": directory.name,
            "created": created,
            "skipped": len(records) - created,
        }
