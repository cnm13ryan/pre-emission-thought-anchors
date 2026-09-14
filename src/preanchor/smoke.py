"""Small CPU-only Stage 1 exercise of every artifact format and resume boundary."""

import resource
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa

from preanchor.config import config_hash
from preanchor.ids import cache_key, derived_seed
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    write_json,
    write_jsonl,
    write_parquet,
    write_safetensors,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, save_manifest, utc_now
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking


def smoke_key(config, manifest: dict, index: int) -> str:
    code_fields = ["git_sha", "dirty_diff_hash", "lock_hash"]
    return cache_key(stage="stage1", scientific_config_hash=config.scientific_hash(),
                     model_revision=config.model.revision,
                     tokenizer_revision=config.model.tokenizer_revision, trace_sha256=None,
                     segmentation_version=config.segmentation["primary"],
                     boundary_mapping_sha256=None, generation_parameters={}, fork_idx=None,
                     sample_seed=derived_seed(config.seed_root, "io_smoke", index),
                     code_version=config_hash({k: manifest["identity"][k]
                                               for k in code_fields}),
                     input_hashes={}, unit_id=f"synthetic-{index}")


def run_smoke(directory: Path, repo: Path, config, *, resume: bool = False,
              units: int = 3, delay_seconds: float = 0, tracking: bool = True) -> dict:
    if config.execution.mode != "smoke" or not 1 <= units <= 8 or not 0 <= delay_seconds <= 60:
        raise ValueError("Stage 1 smoke requires smoke config, 1–8 units and delay of 0–60s")
    with lock(directory / ".run.lock"):
        manifest = check_resume(directory, repo, config)
        previous = manifest["stages"].get("stage1")
        if previous and (not resume or previous["units"] != units):
            raise ValueError("Existing stage requires --resume with the same unit count")
        attempt = uuid.uuid4().hex
        stage = directory / "stages/stage1"
        stage.mkdir(parents=True, exist_ok=True)
        if previous and previous["state"] == "running":
            recovered = {"schema_version": 1, "run_id": manifest["run_id"], "stage": "stage1",
                         "attempt": previous["attempt"], "at": utc_now(),
                         "exception": "UnfinishedAttempt", "identity": manifest["identity"],
                         "message": "Prior writer exited without a terminal record",
                         "signal": "unknown", "recovery": "validate units and rerun missing work"}
            write_json(stage / "failures" / f"{previous['attempt']}.json", recovered)
            atomic_bytes(directory / "failure.json", json_bytes(recovered), replace=True)
        session, tracking_record = start_tracking(directory, manifest, config) if tracking else (
            None, {"mode": "disabled_for_test", "url": None})
        manifest["wandb"].append(tracking_record)
        manifest["commands"].append({"command": "io-smoke", "resume": resume, "units": units,
                                     "delay_seconds": delay_seconds, "attempt": attempt})
        manifest["stages"]["stage1"] = {"state": "running", "units": units, "attempt": attempt}
        save_manifest(directory, manifest)
        runtime = Runtime(directory, "stage1", units, session)
        store = UnitStore(stage / "units")
        started = time.monotonic()
        current_key = None
        created = 0
        keys = []
        try:
            with runtime:
                for index in range(units):
                    current_key = smoke_key(config, manifest, index)
                    keys.append(current_key)

                    def writer(part: Path, index=index):
                        row = {"schema_version": 1, "unit_id": f"synthetic-{index}",
                               "seed": derived_seed(config.seed_root, "io_smoke", index)}
                        write_json(part / "sample.json", row)
                        # Test kill at a real incomplete-unit boundary; pause remains in main loop.
                        runtime.pause(delay_seconds)
                        write_jsonl(part / "events.jsonl", [row])
                        table = pa.Table.from_pylist([row], schema=pa.schema([
                            pa.field("schema_version", pa.int32(), nullable=False),
                            pa.field("unit_id", pa.string(), nullable=False),
                            pa.field("seed", pa.uint32(), nullable=False)]))
                        write_parquet(part / "sample.parquet", table)
                        write_safetensors(part / "sample.safetensors",
                                          {"synthetic": np.full((2, 3), index, dtype=np.float16)})

                    part, was_created = store.produce(current_key, writer)
                    created += was_created
                    runtime.completed += 1
                    runtime.last_output = str(part / "SUCCESS.json")
                    runtime.pulse(force=True)
                result = {"schema_version": 1, "stage": "stage1", "run_id": manifest["run_id"],
                          "unit_keys": keys, "units": units, "state": "complete"}
                write_json(stage / "SUCCESS.json", result)
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": manifest["run_id"],
                              "stage": "stage1", "attempt": attempt, "at": utc_now(),
                              "unit_key": current_key, "completed": runtime.completed,
                              "identity": manifest["identity"], "exception": type(exc).__name__,
                              "message": str(exc), "traceback": traceback.format_exc(),
                              "resume": "io-smoke --resume with unchanged code/config"})
            write_json(stage / "failures" / f"{attempt}.json", failure)
            atomic_bytes(directory / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"]["stage1"]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"exception": type(exc).__name__, "attempt": attempt}
            runtime.pulse(force=True, state="failed")
            raise
        else:
            manifest["stages"]["stage1"]["state"] = "complete"
            manifest["output_hashes"] = {key: store.validate(key)["files"] for key in keys}
        finally:
            timing = {"attempt": attempt, "elapsed_seconds": time.monotonic() - started,
                      "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "peak_gpu_memory_bytes": None, "gpu_used": False,
                      "completed": runtime.completed, "created": created}
            write_json(stage / "attempts" / f"{attempt}.json", timing)
            write_jsonl(directory / "logs/events" / f"{attempt}.jsonl", [
                {"at": utc_now(), **manifest["stages"]["stage1"], **timing}])
            manifest["stage_timings"].setdefault("stage1", []).append(timing)
            if session:
                try:
                    session.finish(exit_code=0 if manifest["stages"]["stage1"]["state"]
                                   == "complete" else 1)
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "local records require backup"
            save_manifest(directory, manifest)
        return {**result, "created": created, "skipped": units - created}
