"""Measured A6000 Stage 5 admission, isolated benchmark and base-only execution.

Runtime choices live beside the frozen scientific config, never in rollout IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import httpx

from preanchor.config import Settings, config_hash, load_config
from preanchor.doctor import run_doctor
from preanchor.ids import file_hash
from preanchor.io import atomic_bytes, json_bytes, lock, read_json, write_json, write_parquet
from preanchor.labels.batching import (
    KV_BYTES_PER_TOKEN,
    MAX_CONTEXT,
    StreamingVLLMBackend,
    admitted_concurrency,
    continuous,
    required_capacity,
    roofline,
)
from preanchor.labels.fresh import answer_category, canonical_kl, legacy_kl
from preanchor.labels.generate import CompletionStore, MiniLMSimilarityScorer, production_specs
from preanchor.labels.pilot import DIGEST, FORKS, assert_plan, checkpoint, inputs, support_report
from preanchor.manifest import (
    check_resume,
    create_run,
    load_manifest,
    save_manifest,
    utc_now,
)
from preanchor.runtime import Runtime, StageInterrupted
from preanchor.tracking import start_tracking

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/pilot.yaml"
RATE = 1.09  # User-supplied planning rate; no billing API credentials available.


def gpu():
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total, used, free, driver, util = [s.strip() for s in value.split(",")]
    if name != "NVIDIA RTX A6000":
        raise ValueError("This admission applies only to the measured RTX A6000")
    return dict(
        name=name,
        total_mib=int(total),
        used_mib=int(used),
        free_mib=int(free),
        driver=driver,
        utilization_percent=int(util),
    )


def metrics():
    response = httpx.get("http://127.0.0.1:8000/metrics", timeout=10)
    response.raise_for_status()
    result = {}
    for line in response.text.splitlines():
        if line.startswith("vllm:"):
            label, value = line.rsplit(" ", 1)
            name = label.split("{")[0]
            if "_bucket" not in name:
                result[name] = result.get(name, 0) + float(value)
    return result


def server_admission(n, startup):
    inspection = json.loads(
        subprocess.check_output(["sudo", "-n", "docker", "inspect", "preanchor-vllm"], text=True)
    )[0]
    image = json.loads(
        subprocess.check_output(
            ["sudo", "-n", "docker", "image", "inspect", inspection["Image"]], text=True
        )
    )[0]
    if not any(d.endswith("@" + DIGEST) for d in image["RepoDigests"]):
        raise ValueError("vLLM image digest mismatch")
    args = inspection["Config"]["Cmd"]
    config = load_config(CONFIG)
    expected = {
        "--model": config.model.id,
        "--revision": config.model.revision,
        "--dtype": "bfloat16",
        "--max-num-seqs": str(n),
        "--max-model-len": "20480",
        "--gpu-memory-utilization": "0.92",
        "--seed": str(config.seed_root),
    }
    for key, value in expected.items():
        if key not in args or args[args.index(key) + 1] != value:
            raise ValueError(f"Runtime mismatch: {key}")
    if "--enable-prefix-caching" not in args or any(
        "quantization" in a or "speculative" in a for a in args
    ):
        raise ValueError("Unapproved runtime semantics")
    ports = inspection["HostConfig"]["PortBindings"]["8000/tcp"]
    if any(p["HostIp"] != "127.0.0.1" for p in ports):
        raise ValueError("vLLM must be localhost-only")
    log = startup.read_text()
    capacities = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    if not capacities:
        raise ValueError("No measured GPU KV capacity in startup log")
    capacity = int(capacities[-1].replace(",", ""))
    if n > admitted_concurrency(capacity):
        raise ValueError(f"Batch {n} needs {required_capacity(n)} KV tokens, found {capacity}")
    memory_lines = [
        line
        for line in log.splitlines()
        if any(s in line.lower() for s in ("memory", "kv cache size", "maximum concurrency"))
    ]
    return dict(
        schema_version=1,
        concurrency=n,
        max_num_seqs=n,
        kv_capacity_tokens=capacity,
        required_kv_tokens=required_capacity(n),
        live_max_context_tokens=n * MAX_CONTEXT,
        kv_headroom_tokens=capacity - n * MAX_CONTEXT,
        kv_safety_fraction=capacity / (n * MAX_CONTEXT) - 1,
        gpu=gpu(),
        vllm_digest=DIGEST,
        args=args,
        startup_log_sha256=file_hash(startup),
        memory_report=memory_lines,
        resolved_runtime_hash=config_hash(expected),
        kv_bytes_per_token=KV_BYTES_PER_TOKEN,
        bandwidth_bytes_per_second=768e9,
        dense_bf16_flops_per_second=154.85e12,
        critical_intensity_flops_per_byte=154.85e12 / 768e9,
    )


def plan(root, config):
    trajectory, targets, refs, hashes = inputs(root, REPO)
    specs = production_specs(
        trajectory, targets, config.seed_root, config.resolved_hash(), config_hash(hashes)
    )
    assert_plan(specs)
    frozen = (
        config.model.revision,
        config.model.dtype,
        config.seed_root,
        config.generation["temperature"],
        config.generation["top_p"],
        config.generation["max_new_tokens"],
    )
    if frozen != (
        "1df8507178afcc1bef68cd8c393f61a886323761",
        "bfloat16",
        20260914,
        0.6,
        0.95,
        16384,
    ):
        raise ValueError("Frozen scientific settings changed")
    return specs, refs, hashes


def tokenizer_validation(specs, config):
    from transformers import AutoConfig, AutoTokenizer

    from preanchor.labels.pilot import tokenizer_check

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.tokenizer_revision,
        local_files_only=True,
        use_fast=True,
    )
    architecture = AutoConfig.from_pretrained(
        config.model.id, revision=config.model.revision, local_files_only=True
    )
    dimensions = (
        architecture.num_hidden_layers,
        architecture.num_key_value_heads,
        architecture.hidden_size // architecture.num_attention_heads,
    )
    if dimensions != (48, 8, 128) or 2 * 48 * 8 * 128 * 2 != KV_BYTES_PER_TOKEN:
        raise ValueError("KV architecture assumption mismatch")
    if not tokenizer.is_fast:
        raise ValueError("Pinned fast tokenizer required")
    backend = StreamingVLLMBackend(image_digest=DIGEST)
    records = []
    for fork in FORKS:
        spec = next(s for s in specs if s.fork_idx == fork)
        ids = tokenizer(spec.prompt, add_special_tokens=False)["input_ids"]
        record = tokenizer_check(ids, backend.tokenize_ids(spec.prompt))
        if len(ids) + 16384 > MAX_CONTEXT:
            raise ValueError("Actual scientific context exceeds admission assumption")
        records.append({**record, "fork_idx": fork})
    return records


def initialize(root, namespace, n, startup):
    config = load_config(CONFIG)
    doctor = run_doctor(config, Settings(), REPO)
    if not doctor["ok"]:
        raise ValueError("Production doctor failed")
    admission = server_admission(n, startup)
    specs, refs, hashes = plan(root, config)
    tokens = tokenizer_validation(specs, config)
    run = create_run(root, REPO, config, ["python -m preanchor.labels.a6000", namespace])
    manifest = load_manifest(run)
    if manifest["source"]["dirty"]:
        raise ValueError("Commit tested code before benchmarking or production")
    manifest["input_hashes"] = hashes
    manifest["identity"]["input_hashes"] = hashes
    manifest["source"]["hardware"] = admission["gpu"]
    manifest["vllm"] = {"digest": DIGEST}
    manifest["gpu_gate"] = "generation_doctor_and_measured_kv_admission"
    manifest["stages"]["stage5"] = {"state": "prepared", "namespace": namespace}
    manifest["sample_seeds"] = [s.seed for s in specs]
    save_manifest(run, manifest)
    stage = run / "stages/stage5"
    write_json(stage / "doctor.json", doctor)
    write_json(stage / "runtime_admission.json", admission)
    write_json(stage / "tokenizer_validation.json", tokens)
    atomic_bytes(stage / "startup.log", startup.read_bytes())
    collected = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "torch.utils.collect_env"],
        text=True,
        capture_output=True,
        check=False,
    )
    atomic_bytes(
        run / "environment/torch_collect_env_native.txt",
        (collected.stdout + collected.stderr).encode(),
    )
    environment = collected.stdout
    if collected.returncode:
        if "get_pip_packages" not in collected.stderr or "splitlines" not in collected.stderr:
            raise RuntimeError("Unexpected torch environment collector failure")
        # uv's locked environment intentionally lacks pip. Only package enumeration
        # changes; every hardware/driver/CUDA probe stays in torch's own collector.
        environment = subprocess.check_output(
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import torch.utils.collect_env as e; import importlib.metadata as m; "
                "e.get_pip_packages=lambda *a, **k: ('importlib.metadata', "
                "'\\n'.join(sorted(d.metadata['Name']+'=='+d.version "
                "for d in m.distributions()))); "
                "print(e.get_pretty_env_info())",
            ],
            text=True,
        )
    atomic_bytes(run / "environment/torch_collect_env.txt", environment.encode())
    selected = [s for s in specs if s.sample_idx == 0] if namespace == "benchmark" else specs
    write_json(
        stage / "plan.json",
        dict(
            schema_version=1,
            namespace=namespace,
            physical_jobs=len(selected),
            references=refs,
            input_hashes=hashes,
            jobs=[{**asdict(s), "rollout_id": s.rollout_id} for s in selected],
        ),
    )
    print(json.dumps({"run_id": run.name, "namespace": namespace}), flush=True)
    return run


def warning_lines(startup):
    return [
        line
        for line in startup.read_text().splitlines()
        if re.search(
            r"out of memory|CUDA error|preempt|not enough KV|allocation.*(fail|warn)|"
            r"(warn|error).*KV cache|KV cache.*(warn|error)",
            line,
            re.I,
        )
    ]


def summarize(rows, wall, telemetry, before, after):
    tokens = sum(r["generated_token_count"] for r in rows)
    lengths = [r["generated_token_count"] for r in rows]
    rates = [
        r["decode_tokens_per_second"] for r in rows if r.get("decode_tokens_per_second") is not None
    ]

    def delta(name):
        return after.get(name, 0) - before.get(name, 0)

    hits = delta("vllm:prefix_cache_hits_total")
    queries = delta("vllm:prefix_cache_queries_total")
    return dict(
        schema_version=1,
        stage="exploratory",
        preregistered=True,
        wall_seconds=wall,
        completed=len(rows),
        generated_tokens=tokens,
        aggregate_tokens_per_second=tokens / wall,
        mean_per_request_decode_tokens_per_second=statistics.mean(rates) if rates else None,
        mean_output_tokens=statistics.mean(lengths),
        median_output_tokens=statistics.median(lengths),
        peak_hbm_mib=max(t["gpu"]["used_mib"] for t in telemetry),
        min_free_hbm_mib=min(t["gpu"]["free_mib"] for t in telemetry),
        peak_ram_bytes=max(t["rss_bytes"] for t in telemetry),
        kv_peak_occupancy=max(t["metrics"].get("vllm:kv_cache_usage_perc", 0) for t in telemetry),
        prefix_cache_hit_rate=hits / queries if queries else None,
        prefix_cache_hits=hits,
        prefix_cache_queries=queries,
        finish_reasons=dict(Counter(r["finish_reason"] for r in rows)),
        truncations=sum(r["finish_reason"] == "length" for r in rows),
        request_failures=0,
        ooms=0,
        requests=[
            {
                k: r.get(k)
                for k in (
                    "rollout_id",
                    "fork_idx",
                    "sample_idx",
                    "seed",
                    "prompt_token_count",
                    "generated_token_count",
                    "finish_reason",
                    "runtime_seconds",
                    "ttft_seconds",
                    "decode_seconds",
                    "decode_tokens_per_second",
                )
            }
            for r in rows
        ],
        planning_cost_usd=wall / 3600 * RATE,
        price_usd_per_hour=RATE,
        rate_source="user planning rate; actual billed rate unavailable",
        roofline_tokens_per_second={
            str(b): {str(c): roofline(b, c) for c in (2750, MAX_CONTEXT)} for b in (1, 2, 3)
        },
    )


def execute(root, run, startup):
    import resource

    config = load_config(CONFIG)
    manifest = check_resume(run, REPO, config)
    specs, _, hashes = plan(root, config)
    if hashes != manifest["input_hashes"]:
        raise ValueError("Input hash mismatch")
    stage = run / "stages/stage5"
    record = read_json(stage / "plan.json")
    benchmark = record["namespace"] == "benchmark"
    if benchmark:
        specs = [s for s in specs if s.sample_idx == 0]
    if [s.rollout_id for s in specs] != [j["rollout_id"] for j in record["jobs"]]:
        raise ValueError("Plan changed")
    admission = read_json(stage / "runtime_admission.json")
    n = admission["concurrency"]
    current = server_admission(n, startup)
    if current["kv_capacity_tokens"] < admission["kv_capacity_tokens"]:
        raise ValueError("Server KV capacity decreased")
    projection = None
    if not benchmark:
        projection = read_json(stage / "resource_projection_a6000.json")
        if (
            projection["code_commit"] != manifest["source"]["git_sha"]
            or projection["concurrency"] != n
            or projection["config_hash"] != config.resolved_hash()
        ):
            raise ValueError("Stale or mismatched projection")
    store = CompletionStore(stage / ("benchmark_units" if benchmark else "units"))
    backend = StreamingVLLMBackend(image_digest=DIGEST)
    rows, pending = [], []
    for spec in specs:
        if (store.raw_units.root / spec.rollout_id).exists():
            # Fail closed for a purported completed payload; UnitStore can repair partials.
            directory = store.raw_units.root / spec.rollout_id
            if (directory / "SUCCESS.json").exists():
                try:
                    store.raw_units.validate(spec.rollout_id)
                except (ValueError, OSError, KeyError, TypeError):
                    pending.append(spec)
                    continue
                row, _ = store.raw_one(spec, backend, max_tokens=16384, temperature=0.6, top_p=0.95)
                rows.append(row)
                continue
        pending.append(spec)
    if benchmark and rows:
        raise ValueError("Benchmark interrupted; retain outputs and use a new benchmark run")
    initial_count = len(rows)
    before = metrics()
    telemetry = []
    started = time.monotonic()
    last_sample = last_backup = -math.inf
    window_checked = False
    attempt = uuid.uuid4().hex
    with lock(run / ".run.lock"):
        tracking, tracking_record = start_tracking(run, manifest, config, stage="stage5")
        manifest["wandb"].append(tracking_record)
        manifest["stages"]["stage5"]["state"] = "running"
        save_manifest(run, manifest)
        try:
            with Runtime(run, "stage5", len(specs), tracking) as runtime:
                runtime.completed = len(rows)

                def tick():
                    nonlocal last_sample, last_backup
                    now = time.monotonic()
                    if not runtime.stop_signal:
                        runtime.pulse()
                    if now - last_sample >= 5:
                        sample = dict(
                            time=utc_now(),
                            elapsed=now - started,
                            gpu=gpu(),
                            metrics=metrics(),
                            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        )
                        telemetry.append(sample)
                        atomic_bytes(stage / "telemetry.json", json_bytes(sample), replace=True)
                        write_json(
                            stage / "telemetry" / attempt / f"{len(telemetry):06d}.json", sample
                        )
                        last_sample = now
                    if now - last_backup >= 900 and not benchmark:
                        manifest["backup"]["latest_primary_checkpoint"] = checkpoint(run, rows)
                        save_manifest(run, manifest)
                        last_backup = now

                def work(spec):
                    return store.raw_one(
                        spec, backend, max_tokens=16384, temperature=0.6, top_p=0.95
                    )

                tick()
                for spec, row, _created in continuous(
                    pending, work, n, tick=tick, stopping=lambda: bool(runtime.stop_signal)
                ):
                    rows.append(row)
                    runtime.completed = len(rows)
                    runtime.last_output = str(store.raw_units.root / spec.rollout_id)
                    fresh = rows[initial_count:]
                    elapsed = time.monotonic() - started
                    tokens = sum(r["generated_token_count"] for r in fresh)
                    rate = tokens / elapsed
                    progress = dict(
                        updated_at=utc_now(),
                        completed=len(rows),
                        total=len(specs),
                        by_fork=dict(Counter(r["fork_idx"] for r in rows)),
                        session_seconds=elapsed,
                        session_tokens=tokens,
                        aggregate_tokens_per_second=rate,
                        peak_hbm_mib=max(t["gpu"]["used_mib"] for t in telemetry),
                        eta_seconds=(len(specs) - len(rows))
                        * statistics.mean([r["generated_token_count"] for r in fresh])
                        / rate
                        if rate
                        else None,
                        session_cost_usd=elapsed / 3600 * RATE,
                        truncations=sum(r["finish_reason"] == "length" for r in rows),
                    )
                    atomic_bytes(stage / "progress.json", json_bytes(progress), replace=True)
                    print(json.dumps(progress), flush=True)
                    if not benchmark and len(fresh) >= 24 and not window_checked:
                        ratio = rate / projection["aggregate_tokens_per_second"]
                        write_json(
                            stage / f"first_window_{attempt}.json",
                            {**progress, "benchmark_ratio": ratio},
                        )
                        window_checked = True
                        if ratio < 0.8:
                            raise RuntimeError("THROUGHPUT_BELOW_BENCHMARK_REQUIRES_DIAGNOSIS")
                    if not runtime.stop_signal:
                        runtime.pulse(force=True)
                if runtime.stop_signal:
                    raise StageInterrupted(runtime.stop_signal)
                wall = time.monotonic() - started
                tick()
                if len(rows) != len(specs) or len({r["rollout_id"] for r in rows}) != len(specs):
                    raise ValueError("Completion count mismatch")
                for spec in specs:
                    store.raw_units.validate(spec.rollout_id)
                physical_ids = {p.parent.name for p in store.raw_units.root.glob("*/SUCCESS.json")}
                if physical_ids != {s.rollout_id for s in specs}:
                    raise ValueError("Unexpected physical job or namespace contamination")
                summary = summarize(rows[initial_count:], wall, telemetry, before, metrics())
                summary.update(
                    concurrency=n,
                    max_num_seqs=n,
                    namespace=record["namespace"],
                    run_id=run.name,
                    resumed_prior_completions=initial_count,
                    warnings=warning_lines(startup),
                    completed_total=len(rows),
                    by_fork=dict(Counter(r["fork_idx"] for r in rows)),
                    duplicate_rollout_ids=0,
                    duplicate_physical_jobs=0,
                    benchmark_contamination=0,
                    valid_checksums=len(rows),
                    code_commit=manifest["source"]["git_sha"],
                )
                summary["measured_artifact_bytes"] = sum(
                    p.stat().st_size for p in stage.rglob("*") if p.is_file()
                )
                write_json(stage / "summary.json", summary)
                manifest["stages"]["stage5"]["state"] = "generated"
                manifest["stage_timings"]["stage5_generation_seconds"] = wall
                runtime.pulse(force=True, state="generated")
        except BaseException as exc:
            failure = dict(
                time=utc_now(),
                error_type=type(exc).__name__,
                completed=len(rows),
                reason=str(exc)
                if isinstance(exc, (ValueError, RuntimeError, StageInterrupted))
                else "See log",
            )
            write_json(stage / "failures" / (attempt + ".json"), failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"]["stage5"]["state"] = "stopped"
            raise
        finally:
            if not benchmark:
                manifest["backup"]["latest_primary_checkpoint"] = checkpoint(run, rows)
            if tracking:
                tracking.finish()
            save_manifest(run, manifest)
    return run


def select(root, benchmark_ids, startup):
    config = load_config(CONFIG)
    candidates = []
    for run_id in benchmark_ids:
        directory = root / "artifacts/runs" / run_id
        manifest = load_manifest(directory)
        if manifest["identity"]["resolved_config_hash"] != config.resolved_hash():
            raise ValueError("Benchmark config differs from current scientific config")
        if manifest["stages"].get("stage5", {}).get("namespace") != "benchmark":
            raise ValueError("Candidate is not a benchmark run")
        stage = directory / "stages/stage5"
        summary = read_json(stage / "summary.json")
        admission = read_json(stage / "runtime_admission.json")
        if (
            summary["namespace"] != "benchmark"
            or summary["completed"] != 8
            or sorted(r["fork_idx"] for r in summary["requests"]) != FORKS
        ):
            raise ValueError("Benchmark workload mismatch")
        store = CompletionStore(stage / "benchmark_units")
        for row in summary["requests"]:
            store.raw_units.validate(row["rollout_id"])
        if summary["warnings"] or summary["request_failures"] or summary["ooms"]:
            continue
        if summary["truncations"] / summary["completed"] > 0.05:
            continue
        if summary["min_free_hbm_mib"] < 2048:
            raise ValueError("Less than 2 GiB measured physical headroom")
        if admission["kv_capacity_tokens"] < required_capacity(summary["concurrency"]):
            raise ValueError("KV safety margin failed")
        candidates.append((summary, admission, stage))
    candidates.sort(key=lambda c: c[0]["concurrency"])
    chosen = candidates[0]
    if len(candidates) == 2:
        a, b = candidates
        if [a[0]["concurrency"], b[0]["concurrency"]] != [2, 3]:
            raise ValueError("Expected B2/B3 comparison")
        if b[0]["aggregate_tokens_per_second"] >= 1.1 * a[0]["aggregate_tokens_per_second"]:
            chosen = b
    summary, admission, stage = chosen
    n = summary["concurrency"]
    run = initialize(root, "production", n, startup)
    total_tokens = summary["mean_output_tokens"] * 256
    seconds = total_tokens / summary["aggregate_tokens_per_second"]
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 20 * 3600:
        raise ValueError("Projection outside order-of-magnitude sanity range; diagnosis required")
    projection = dict(
        schema_version=1,
        code_commit=manifest["source"]["git_sha"],
        config_hash=config.resolved_hash(),
        unit_count=256,
        benchmark_units=8,
        concurrency=n,
        max_num_seqs=n,
        aggregate_tokens_per_second=summary["aggregate_tokens_per_second"],
        mean_per_request_decode_tokens_per_second=summary[
            "mean_per_request_decode_tokens_per_second"
        ],
        mean_output_tokens=summary["mean_output_tokens"],
        median_output_tokens=summary["median_output_tokens"],
        projected_generated_tokens=total_tokens,
        projected_seconds=seconds,
        projected_gpu_hours=seconds / 3600,
        projected_cost_usd=seconds / 3600 * RATE,
        price_usd_per_hour=RATE,
        rate_source=summary["rate_source"],
        measured_seconds_per_unit=summary["wall_seconds"] / 8,
        peak_vram_mib=summary["peak_hbm_mib"],
        peak_ram_bytes=summary["peak_ram_bytes"],
        kv_capacity_tokens=admission["kv_capacity_tokens"],
        kv_headroom_tokens=admission["kv_headroom_tokens"],
        measured_benchmark_artifact_bytes=summary["measured_artifact_bytes"],
        disk_growth_bytes_upper_estimate=summary["measured_artifact_bytes"] * 32 * 4,
        base_authorized=True,
        adaptive_authorized=False,
        selected_benchmark_run=summary["run_id"],
        evidence=[
            {"path": str(c[2] / "summary.json"), "sha256": file_hash(c[2] / "summary.json")}
            for c in candidates
        ],
        selection_rule="B3 only with >=10% aggregate throughput improvement and safety gates",
        created_at=utc_now(),
    )
    for name in ("resource_projection_a6000.json", "resource_projection.json"):
        write_json(run / "stages/stage5" / name, projection)
    print(json.dumps({"production_run_id": run.name, "projection": projection}), flush=True)
    return run


def finalize(root, run):
    import pyarrow as pa

    config = load_config(CONFIG)
    manifest = check_resume(run, REPO, config)
    if gpu()["used_mib"] > 1024:
        raise ValueError("Stop vLLM and verify memory release before scoring")
    specs, references, hashes = plan(root, config)
    if hashes != manifest["input_hashes"]:
        raise ValueError("Input mismatch")
    stage = run / "stages/stage5"
    summary = read_json(stage / "summary.json")
    if summary["namespace"] != "production" or summary["completed_total"] != 256:
        raise ValueError("Need exactly 256 production completions")
    store = CompletionStore(stage / "units")
    for spec in specs:
        store.raw_units.validate(spec.rollout_id)
    scorer = MiniLMSimilarityScorer(device="cpu")

    class NoGeneration:
        def complete(self, *args, **kwargs):
            raise RuntimeError("Scoring must never generate")

    rows = []
    for spec in specs:
        row, _ = store.run_one(
            spec,
            NoGeneration(),
            max_tokens=16384,
            temperature=0.6,
            top_p=0.95,
            similarity_scorer=scorer,
            reference_texts=references,
        )
        rows.append(row)
    support = support_report(rows)
    deficient = [int(f) for f, counts in support.items() if counts["dissimilar_valid"] < 10]
    report = dict(
        schema_version=1,
        stage="exploratory",
        preregistered=True,
        physical_jobs=256,
        valid_completions=256,
        valid_checksums=256,
        duplicate_rollout_ids=0,
        duplicate_physical_jobs=0,
        benchmark_contamination=0,
        by_fork=dict(Counter(r["fork_idx"] for r in rows)),
        support=support,
        deficient_forks=deficient,
        adaptive_extension_required=bool(deficient),
        adaptive_authorized=False,
    )
    write_json(stage / "base_validation.json", report)
    if not deficient:
        labels = []
        for target in (6, 12, 13, 14, 46):
            without = []
            for row in rows:
                if row["fork_idx"] != target:
                    continue
                association = (
                    next(
                        a
                        for a in row["association_similarity"]
                        if a["target_sentence_idx"] == target and a["condition"] == "without"
                    )
                    if (row["association_similarity"])
                    else None
                )
                if association and association["is_dissimilar"]:
                    without.append(answer_category(row["raw_answer"]))
            with_answers = [
                answer_category(r["raw_answer"]) for r in rows if r["fork_idx"] == target + 1
            ]
            labels.append(
                dict(
                    sentence_idx=target,
                    n_without=len(without),
                    n_with=len(with_answers),
                    importance_canonical=canonical_kl(without, with_answers),
                    importance_legacy=legacy_kl(without, with_answers),
                    stage="exploratory",
                    preregistered=True,
                )
            )
        write_parquet(stage / "fresh_labels.parquet", pa.Table.from_pylist(labels))
    manifest["stages"]["stage5"]["state"] = "adaptive_gate" if deficient else "labels_constructed"
    manifest["backup"]["latest_primary_checkpoint"] = checkpoint(run, rows)
    save_manifest(run, manifest)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("benchmark", "select", "execute", "finalize"))
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--startup", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--benchmarks", nargs="+")
    args = parser.parse_args()
    root = Settings().persistent_root()
    if args.action in ("benchmark", "select", "execute") and args.startup is None:
        parser.error("--startup required")
    if args.action == "benchmark":
        run = initialize(root, "benchmark", args.concurrency, args.startup)
        execute(root, run, args.startup)
    elif args.action == "select":
        select(root, args.benchmarks, args.startup)
    else:
        if not args.run_id or Path(args.run_id).name != args.run_id:
            parser.error("Explicit safe run ID required")
        run = root / "artifacts/runs" / args.run_id
        if args.action == "execute":
            execute(root, run, args.startup)
        else:
            finalize(root, run)


if __name__ == "__main__":
    main()
