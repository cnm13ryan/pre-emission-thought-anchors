"""Base-only Stage 5 execution using the existing run and unit machinery."""
from __future__ import annotations

import concurrent.futures
import json
import math
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from preanchor.config import config_hash, load_config, read_yaml
from preanchor.ids import file_hash
from preanchor.io import (
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_parquet,
    write_json,
)
from preanchor.labels.generate import (
    CompletionStore,
    MiniLMSimilarityScorer,
    VLLMBackend,
    production_specs,
    prompt_hash,
    support_counts,
)
from preanchor.logging import redact
from preanchor.manifest import (
    check_resume,
    create_run,
    load_manifest,
    provenance,
    save_manifest,
    utc_now,
)
from preanchor.runtime import Runtime, StageInterrupted
from preanchor.tracking import start_tracking

FORKS = [6, 7, 12, 13, 14, 15, 46, 47]
DIGEST = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
BENCHMARK = "20260914T223408Z-53ff3383c9-1f73ff50"


def inputs(root, repo):
    base = root / "data/processed/stage3"
    trajectories = validate_parquet(base / "trajectories.parquet").to_pylist()
    sentences = validate_parquet(base / "sentences.parquet").to_pylist()
    trajectory = next(r for r in trajectories if r["trajectory_id"] == "4682:correct")
    rows = [r for r in sentences if r["trajectory_id"] == "4682:correct"]
    trajectory["sentence_boundaries"] = {
        str(r["sentence_idx"]): r["char_start"] for r in rows}
    if prompt_hash(trajectory["full_cot"]) != trajectory["full_cot_sha256"]:
        raise ValueError("Trace hash mismatch")
    for row in rows:
        if trajectory["full_cot"][row["char_start"]:row["char_end"]] != row["text"]:
            raise ValueError("Target alignment mismatch")
    mapping = repo / "configs/pilot_targets_4682.yaml"
    targets = read_yaml(mapping)["targets"]
    hashes = {str(p): file_hash(p) for p in (
        base / "trajectories.parquet", base / "sentences.parquet", mapping)}
    return trajectory, targets, {r["sentence_idx"]: r["text"] for r in rows}, hashes


def assert_plan(specs):
    pairs = {(s.fork_idx, s.sample_idx) for s in specs}
    expected = {(fork, sample) for fork in FORKS for sample in range(32)}
    if len(specs) != 256 or pairs != expected or len({s.rollout_id for s in specs}) != 256:
        raise ValueError("Base plan identity/count violation")
    for fork, associations in ((13, {(12, "with"), (13, "without")}),
                               (14, {(13, "with"), (14, "without")})):
        if any(set(s.associations) != associations for s in specs if s.fork_idx == fork):
            raise ValueError("Shared fork associations changed")


def tokenizer_check(host_ids, backend_ids):
    if host_ids != backend_ids:
        raise ValueError("Host/vLLM token IDs mismatch")
    return {"equal": True, "count": len(host_ids), "token_ids_sha256": config_hash(host_ids)}


def support_report(rows):
    return {str(fork): support_counts(
        [r for r in rows if r["fork_idx"] == fork],
        target_sentence_idx=fork, condition="without") for fork in [6, 12, 13, 14, 46]}


def guard_reason(rows):
    if not rows:
        return None
    if sum(r["finish_reason"] == "length" for r in rows) / len(rows) > .05:
        return "LENGTH_TRUNCATION_ABOVE_5_PERCENT"
    if len(rows) >= 32:
        seconds = sum(r["runtime_seconds"] for r in rows) / len(rows) * 256
        if seconds > 36000 or seconds / 3600 * 1.99 > 20:
            return "BASE_PROJECTION_EXCEEDS_AUTHORIZATION"
    return None


def checkpoint(run, rows):
    """Incremental immutable payload copy; manifest only names complete units.

    Each checkpoint references all previous copied units, without duplicating
    those payloads on every 15-minute tick. No model caches are visited.
    """
    destination = run.parents[1] / "backups" / ("stage5-live-" + run.name)
    stage = run / "stages/stage5"
    entries = {}
    sources = [run / "resolved_config.yaml", run / "git.patch"]
    sources += list((run / "manifests").glob("*.json*"))
    sources += [p for p in stage.glob("*.json*") if p.name not in (
        "progress.json", "progress.json.sha256", "telemetry.json")]
    for row in rows:
        for namespace in ("units", "units_raw"):
            sources += list((stage / namespace / row["rollout_id"]).glob("*"))
    # A generated response may already be durable while derived scoring failed.
    raw_store = CompletionStore(stage / "units").raw_units
    for success in raw_store.root.glob("*/SUCCESS.json"):
        raw_store.validate(success.parent.name)
        sources += list(success.parent.glob("*"))
    for source in sources:
        if not source.is_file() or source.is_symlink():
            continue
        relative = str(source.relative_to(run))
        target = destination / "run" / relative
        digest = file_hash(source)
        if not target.exists():
            atomic_bytes(target, source.read_bytes())
        if file_hash(target) != digest:
            raise ValueError(f"Backup verification failed: {relative}")
        entries["run/" + relative] = digest
    # Snapshot active offline telemetry bytes; source may still be growing.
    stamp = uuid.uuid4().hex
    for source in (run / "logs").rglob("*.wandb"):
        if source.is_symlink():
            continue
        target = destination / "wandb" / stamp / source.name
        atomic_bytes(target, source.read_bytes())
        entries[str(target.relative_to(destination))] = file_hash(target)
    record = {"schema_version": 1, "created_at": utc_now(), "completed": len(rows),
              "files": entries, "secondary_verified": False}
    path = destination / "checkpoints" / (stamp + ".json")
    write_json(path, record)
    for name, digest in entries.items():
        if file_hash(destination / name) != digest:
            raise ValueError("Independent primary checkpoint verification failed")
    return {"path": str(path), "sha256": file_hash(path), "files": len(entries)}


def prepare(root: Path, repo: Path, config_path: Path) -> Path:
    config = load_config(config_path)
    source, _ = provenance(repo)
    if source["dirty"]:
        raise ValueError("Commit validated code before preparing production")
    trajectory, targets, references, hashes = inputs(root, repo)
    run = create_run(root, repo, config, [
        "preanchor pilot-labels --config configs/pilot.yaml --resume --base-only"])
    manifest = load_manifest(run)
    manifest["input_hashes"] = hashes
    manifest["identity"]["input_hashes"] = hashes
    manifest["stages"]["stage5"] = {"state": "prepared", "base_only": True}
    save_manifest(run, manifest)
    specs = production_specs(trajectory, targets, config.seed_root, config.resolved_hash(),
                             config_hash(hashes))
    assert_plan(specs)
    plan = {"schema_version": 1, "run_id": run.name, "code_commit": source["git_sha"],
            "config_hash": config.resolved_hash(), "input_hashes": hashes,
            "namespace": "production", "physical_jobs": len(specs),
            "duplicate_jobs": 0, "benchmark_contamination": 0,
            "adaptive_jobs": 0, "references": references,
            "jobs": [{**asdict(s), "rollout_id": s.rollout_id,
                      "artifact_path": str(run / "stages/stage5/units" / s.rollout_id
                                           / "completion.json")} for s in specs]}
    stage = run / "stages/stage5"
    write_json(stage / "production_plan.json", plan)
    benchmark_stage = root / "artifacts/runs" / BENCHMARK / "stages/stage5"
    old_preflight = benchmark_stage / "production_preflight_v7.json"
    expected_preflight = "0f74f591fe3074cdb11a6b17e5dd6992f989b1180876cc53058e3adeacc2a335"
    if file_hash(old_preflight) != expected_preflight:
        raise ValueError("Accepted preflight hash mismatch")
    benchmark_rows = [read_json(p) for p in
                      benchmark_stage.glob("benchmark_units/*/completion.json")]
    if len(benchmark_rows) != 8 or sum(r["generated_token_count"] for r in benchmark_rows) != 36619:
        raise ValueError("Benchmark evidence mismatch")
    write_json(stage / "resource_projection.json", {
        "schema_version": 1, "code_commit": source["git_sha"],
        "config_hash": config.resolved_hash(), "input_hashes": hashes,
        "empirical_run": BENCHMARK, "benchmark_units": 8, "unit_count": 256,
        "measured_seconds_per_unit": 884.48 / 8,
        "projected_seconds": 884.48 / 8 * 256,
        "projected_cost_usd": 884.48 / 8 * 256 / 3600 * 1.99,
        "price_usd_per_hour": 1.99, "peak_vram_mib": 34875,
        "disk_growth_bytes_upper_estimate": 5 * 10**9,
        "peak_ram": "measured during production; prior artifact retained",
        "basis": "Existing unchanged prompt/sampling benchmark; operational controls updated",
        "base_authorized": True, "adaptive_authorized": False,
        "stop_after_minimum_completions": 32, "stop_projected_seconds": 36000,
        "stop_projected_cost_usd": 20})
    return run


def execute(root: Path, repo: Path, config_path: Path, run: Path):
    import httpx
    from transformers import AutoTokenizer

    config = load_config(config_path)
    manifest = check_resume(run, repo, config)
    trajectory, targets, references, hashes = inputs(root, repo)
    if hashes != manifest["input_hashes"]:
        raise ValueError("Production input mismatch")
    specs = production_specs(trajectory, targets, config.seed_root, config.resolved_hash(),
                             config_hash(hashes))
    assert_plan(specs)
    stage = run / "stages/stage5"
    plan = read_json(stage / "production_plan.json")
    if [s.rollout_id for s in specs] != [j["rollout_id"] for j in plan["jobs"]]:
        raise ValueError("Production plan mismatch")
    projection = read_json(stage / "resource_projection.json")
    if projection["code_commit"] != manifest["source"]["git_sha"]:
        raise ValueError("Stale projection")
    backend = VLLMBackend(image_digest=DIGEST)
    response = httpx.get(backend.endpoint + "/health", timeout=30)
    response.raise_for_status()
    inspection = json.loads(subprocess.check_output(
        ["sudo", "-n", "docker", "inspect", "preanchor-vllm"], text=True))[0]
    image = json.loads(subprocess.check_output(
        ["sudo", "-n", "docker", "image", "inspect", inspection["Image"]], text=True))[0]
    if not any(d.endswith("@" + DIGEST) for d in image["RepoDigests"]):
        raise ValueError("vLLM digest mismatch")
    args = inspection["Config"]["Cmd"]
    for key, expected in {"--revision": config.model.revision, "--dtype": "bfloat16",
                          "--max-num-seqs": "1", "--max-model-len": "20480",
                          "--gpu-memory-utilization": "0.92"}.items():
        if args[args.index(key) + 1] != expected:
            raise ValueError("vLLM runtime profile mismatch")
    if "--enable-prefix-caching" not in args:
        raise ValueError("Prefix cache not enabled")
    ports = inspection["HostConfig"]["PortBindings"]["8000/tcp"]
    if any(p["HostIp"] != "127.0.0.1" for p in ports):
        raise ValueError("Backend is not localhost only")
    tokenizer = AutoTokenizer.from_pretrained(config.model.id,
        revision=config.model.tokenizer_revision, local_files_only=True)
    records = []
    for fork in FORKS:
        prompt = next(s.prompt for s in specs if s.fork_idx == fork)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        record = tokenizer_check(ids, backend.tokenize_ids(prompt))
        if len(ids) + config.generation["max_new_tokens"] > 20480:
            raise ValueError("Context capacity exceeded")
        records.append({**record, "fork_idx": fork, "prompt_sha256": prompt_hash(prompt)})
    write_json(stage / "tokenizer_validation.json", {
        "model_id": config.model.id, "revision": config.model.tokenizer_revision,
        "add_special_tokens": False, "chat_template": None, "forks": records})
    scorer = MiniLMSimilarityScorer(device="cpu")
    store = CompletionStore(stage / "units")
    rows = []
    session_started = time.monotonic()
    last_backup = -math.inf
    session_new = 0
    session_reused = 0
    peak_hbm_mib = 0
    last_telemetry = -math.inf
    with lock(run / ".run.lock"):
        tracking, tracking_record = start_tracking(run, manifest, config, stage="stage5")
        manifest["wandb"].append(tracking_record)
        manifest["vllm"] = {"image": config.generation["vllm_image"], "digest": DIGEST}
        manifest["gpu_gate"] = "measured_generation_admitted_activation_fit_pending"
        manifest["stages"]["stage5"] = {"state": "running", "started_at": utc_now()}
        save_manifest(run, manifest)
        try:
            with Runtime(run, "stage5", 256, tracking) as runtime:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
                    for spec in specs:
                        runtime.check_stop()
                        task = worker.submit(store.run_one, spec, backend,
                            max_tokens=config.generation["max_new_tokens"],
                            temperature=config.generation["temperature"],
                            top_p=config.generation["top_p"],
                            similarity_scorer=scorer, reference_texts=references)
                        while not task.done():
                            # SIGTERM requests a stop after this atomic completion.
                            if not runtime.stop_signal:
                                runtime.pulse()
                            if time.monotonic() - last_telemetry >= 30:
                                smi = subprocess.check_output([
                                    "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).strip()
                                used, utilization = map(int, smi.split(","))
                                peak_hbm_mib = max(peak_hbm_mib, used)
                                atomic_bytes(stage / "telemetry.json", json_bytes({
                                    "updated_at": utc_now(), "memory_used_mib": used,
                                    "gpu_utilization_percent": utilization,
                                    "peak_hbm_mib": peak_hbm_mib}), replace=True)
                                last_telemetry = time.monotonic()
                            if time.monotonic() - last_backup >= 900:
                                backup = checkpoint(run, rows)
                                manifest["backup"]["latest_primary_checkpoint"] = backup
                                save_manifest(run, manifest)
                                last_backup = time.monotonic()
                            time.sleep(1)
                        row, created = task.result()
                        if row["rollout_id"] != spec.rollout_id:
                            raise ValueError("Completed identity mismatch")
                        rows.append(row)
                        session_new += int(created)
                        session_reused += int(not created)
                        runtime.completed = len(rows)
                        runtime.last_output = str(stage / "units" / spec.rollout_id)
                        seconds = sum(r["runtime_seconds"] for r in rows)
                        progress = {"updated_at": utc_now(), "completed": len(rows), "total": 256,
                            "by_fork": dict(Counter(r["fork_idx"] for r in rows)),
                            "generated_tokens": sum(r["generated_token_count"] for r in rows),
                            "generation_seconds": seconds,
                            "session_seconds": time.monotonic() - session_started,
                            "session_new": session_new, "session_reused": session_reused,
                            "recent_runtime_seconds": row["runtime_seconds"],
                            "recent_decode_seconds": row.get("decode_seconds"),
                            "peak_hbm_mib": peak_hbm_mib,
                            "eta_seconds": seconds / len(rows) * (256 - len(rows)),
                            "estimated_compute_cost_usd": seconds / 3600 * 1.99,
                            "truncations": sum(r["finish_reason"] == "length" for r in rows),
                            "support": support_report(rows)}
                        atomic_bytes(stage / "progress.json", json_bytes(progress), replace=True)
                        print(json.dumps(progress, sort_keys=True), flush=True)
                        if runtime.stop_signal:
                            raise StageInterrupted(runtime.stop_signal)
                        runtime.pulse(force=True)
                        reason = guard_reason(rows)
                        if reason:
                            raise RuntimeError(reason)
                for spec in specs:
                    store.units.validate(spec.rollout_id)
                    store.raw_units.validate(spec.rollout_id)
                report = support_report(rows)
                deficient = [int(f) for f, counts in report.items()
                             if counts["dissimilar_valid"] < 10]
                write_json(stage / "base_validation.json", {
                    "schema_version": 1, "status": "BASE_VERIFIED", "completed": len(rows),
                    "by_fork": dict(Counter(r["fork_idx"] for r in rows)),
                    "support": report, "deficient_forks": deficient,
                    "adaptive_authorized": False, "next_gate": (
                        "ADAPTIVE_AUTHORIZATION_REQUIRED" if deficient else "LABEL_CONSTRUCTION"),
                    "generated_tokens": sum(r["generated_token_count"] for r in rows),
                    "generation_seconds": sum(r["runtime_seconds"] for r in rows),
                    "completed_at": utc_now()})
                runtime.pulse(force=True, state="base_complete")
        except BaseException as exc:
            failure = redact({
                "time": utc_now(), "error_type": type(exc).__name__,
                "reason": str(exc), "completed": len(rows),
                "next_unit": specs[len(rows)].rollout_id if len(rows) < 256 else None})
            write_json(run / "stages/stage5/failures" / (uuid.uuid4().hex + ".json"), failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"]["stage5"]["state"] = "stopped"
            save_manifest(run, manifest)
            raise
        finally:
            if tracking:
                tracking.finish()
            manifest["backup"]["latest_primary_checkpoint"] = checkpoint(run, rows)
            save_manifest(run, manifest)
    return run


def main():
    import argparse

    from preanchor.config import Settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = Settings().persistent_root()
    repo = Path(__file__).resolve().parents[3]
    if args.prepare:
        print(prepare(root, repo, args.config))
    else:
        if not args.run_id or Path(args.run_id).name != args.run_id:
            parser.error("--run-id required")
        execute(root, repo, args.config, root / "artifacts/runs" / args.run_id)


if __name__ == "__main__":
    main()
