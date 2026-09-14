"""Detached CPU-only persistent smoke: SIGTERM, SIGKILL, and unchanged completed artifacts."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from preanchor.config import Settings, load_config
from preanchor.ids import file_hash
from preanchor.io import UnitStore, write_json
from preanchor.manifest import create_run, load_manifest
from preanchor.smoke import smoke_key
from preanchor.storage import persistent_mount


def main():
    repo = Path(__file__).resolve().parents[1]
    root = Settings().persistent_root()
    persistent_mount(root)
    config = load_config(repo / "configs/smoke.yaml")
    directory = create_run(root, repo, config, ["python scripts/stage1_smoke.py"])
    print(json.dumps({"run_id": directory.name, "directory": str(directory)}), flush=True)
    manifest = load_manifest(directory)
    keys = [smoke_key(config, manifest, i) for i in range(3)]
    units = directory / "stages/stage1/units"

    def launch(name, *, resume=False, delay=0):
        args = [sys.executable, "-m", "preanchor.cli", "io-smoke", "--run-id", directory.name,
                "--delay-seconds", str(delay)]
        if resume:
            args.append("--resume")
        with (directory / "logs" / f"{name}.txt").open("x") as stream:
            return subprocess.Popen(args, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                         "WANDB_MODE": "offline"})

    def interrupt(process, signum, *, quarantine_count=0):
        deadline = time.monotonic() + 80
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Worker exited early: {process.returncode}")
                completed = units / keys[0] / "SUCCESS.json"
                partial = units / keys[1] / "sample.json"
                quarantines = list((units / "quarantine").glob("*"))
                if completed.exists() and partial.exists() and len(quarantines) >= quarantine_count:
                    process.send_signal(signum)
                    return process.wait(timeout=30)
                time.sleep(0.1)
            raise TimeoutError("Worker did not reach the expected interruption boundary")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)

    first = launch("01-sigterm", delay=31)
    term_code = interrupt(first, signal.SIGTERM)
    assert term_code != 0
    store = UnitStore(units)
    store.validate(keys[0])
    before = {str(p.relative_to(units / keys[0])): [file_hash(p), p.stat().st_mtime_ns]
              for p in (units / keys[0]).rglob("*") if p.is_file()}
    assert not (units / keys[1] / "SUCCESS.json").exists()
    second = launch("02-sigkill", resume=True, delay=31)
    kill_code = interrupt(second, signal.SIGKILL, quarantine_count=1)
    assert kill_code == -signal.SIGKILL
    third = launch("03-resume", resume=True)
    assert third.wait(timeout=60) == 0
    after = {str(p.relative_to(units / keys[0])): [file_hash(p), p.stat().st_mtime_ns]
             for p in (units / keys[0]).rglob("*") if p.is_file()}
    assert before == after
    for key in keys:
        store.validate(key)
    fourth = launch("04-noop-resume", resume=True)
    assert fourth.wait(timeout=60) == 0
    assert before == {str(p.relative_to(units / keys[0])): [file_hash(p), p.stat().st_mtime_ns]
                      for p in (units / keys[0]).rglob("*") if p.is_file()}
    result = {"schema_version": 1, "kind": "implementation_validation", "run_id": directory.name,
              "sigterm_exit": term_code, "sigkill_exit": kill_code,
              "completed_artifacts_unchanged": True, "completed_file_evidence": before,
              "quarantined_partial_units": len(list((units / "quarantine").iterdir())),
              "units_verified": 3, "gpu_used": False, "state": "passed"}
    write_json(directory / "stages/stage1/resume_validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
