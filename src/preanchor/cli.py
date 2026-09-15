"""Command-line entry point; unimplemented stages are not advertised as runnable."""

import json
from pathlib import Path
from typing import Annotated

import typer

from preanchor.backup.cli import backup_app, restore_app
from preanchor.config import Settings, load_config
from preanchor.control.cli import agent_app, lease_app
from preanchor.doctor import run_doctor
from preanchor.logging import configure_logging, redact
from preanchor.manifest import create_run, load_manifest
from preanchor.runtime import Runtime
from preanchor.smoke import run_smoke
from preanchor.storage import persistent_mount

app = typer.Typer(no_args_is_help=True, help="Passive offline thought-anchor experiment.")
app.add_typer(backup_app, name="backup")
app.add_typer(restore_app, name="restore")
app.add_typer(agent_app, name="agent")
app.add_typer(lease_app, name="lease")


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def doctor(
    config: Annotated[Path, typer.Option(help="Experiment YAML.")] = Path("configs/mvp.yaml"),
    cpu_only: Annotated[bool, typer.Option(help="Bootstrap checks only; never GPU admission.")]
    = False,
) -> None:
    """Check environment readiness without loading model weights."""
    try:
        loaded = load_config(config)
        report = run_doctor(loaded, Settings(), Path(__file__).resolve().parents[2],
                            cpu_only=cpu_only)
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(json.dumps({"ok": False, "error": type(exc).__name__}))
        raise typer.Exit(1) from None
    typer.echo(json.dumps(redact(report), indent=2, sort_keys=True))
    if not report["ok"]:
        raise typer.Exit(1)


def run_path(run_id: str) -> Path:
    if not run_id or any(c not in "0123456789abcdefghijklmnopqrstuvwxyzTZ-" for c in run_id):
        raise typer.BadParameter("Invalid run ID")
    root = Settings().persistent_root()
    persistent_mount(root)
    directory = root / "artifacts/runs" / run_id
    if not directory.is_dir() or directory.is_symlink():
        raise typer.BadParameter("Run does not exist")
    return directory


@app.command("init-run")
def init_run(config: Annotated[Path, typer.Option()] = Path("configs/smoke.yaml")) -> None:
    """Create provenance and initial heartbeat; no model or dataset access."""
    root = Settings().persistent_root()
    persistent_mount(root)
    loaded = load_config(config)
    directory = create_run(root, Path(__file__).resolve().parents[2], loaded,
                           [f"preanchor init-run --config {config}"])
    Runtime(directory, "initialized", 0).pulse(force=True, state="initialized")
    typer.echo(json.dumps({"run_id": directory.name, "directory": str(directory)}))


@app.command("io-smoke")
def io_smoke(
    run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/smoke.yaml"),
    resume: Annotated[bool, typer.Option()] = False,
    units: Annotated[int, typer.Option(min=1, max=8)] = 3,
    delay_seconds: Annotated[float, typer.Option(min=0, max=60)] = 0,
) -> None:
    """Validate Stage 1 artifact formats and resume on tiny synthetic CPU units."""
    result = run_smoke(run_path(run_id), Path(__file__).resolve().parents[2], load_config(config),
                       resume=resume, units=units, delay_seconds=delay_seconds)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command()
def status(
    root: Annotated[Path | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Read heartbeats and recent durable events; stale progress never kills a job."""
    import time

    runs = (root or Settings().persistent_root()) / "artifacts/runs"
    directories = sorted(runs.iterdir()) if runs.exists() else []
    for directory in directories:
        if not directory.is_dir() or (run_id and directory.name != run_id):
            continue
        manifest = load_manifest(directory)
        heartbeat = directory / "heartbeat.json"
        heartbeat_data = json.loads(heartbeat.read_bytes()) if heartbeat.exists() else None
        events = sorted((directory / "logs/events").glob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime)[-5:]
        typer.echo(json.dumps({"run_id": directory.name, "stages": manifest["stages"],
                               "heartbeat": heartbeat_data,
                               "stale": (heartbeat_data.get("state") == "running"
                                         and time.time() - heartbeat.stat().st_mtime > 90)
                               if heartbeat_data else True,
                               "recent_logs": [p.read_text() for p in events]}))



@app.command("fetch")
def fetch_command(
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    discover: Annotated[bool, typer.Option()] = False,
    run_id: Annotated[str | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option()] = False,
    smoke: Annotated[bool, typer.Option()] = False,
    delay_seconds: Annotated[float, typer.Option(min=0, max=60)] = 0,
) -> None:
    """Discover exact pinned paths or fetch them into immutable CPU-only source units."""
    import os

    from preanchor.data.fetch import discover as discover_files
    from preanchor.data.fetch import save_discovery
    from preanchor.data.stage import run_fetch

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Settings().persistent_root()
    persistent_mount(root)
    os.environ['HF_HOME'] = str(root / 'cache/huggingface')
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    if discover:
        if resume or run_id or smoke:
            raise typer.BadParameter('Discovery cannot resume a fetch run')
        found = discover_files(loaded)
        save_discovery(found, repo)
        typer.echo(json.dumps({'files': len(found['files']),
                              'bytes': sum(r['size'] for r in found['files'])}))
        return
    if resume and not run_id:
        raise typer.BadParameter('--resume requires --run-id')
    directory = run_path(run_id) if run_id else create_run(root, repo, loaded, [
        f'preanchor fetch --config {config} --smoke={smoke}'])
    result = run_fetch(directory, repo, loaded, root / 'cache/huggingface/hub',
                       resume=resume, smoke=smoke, delay_seconds=delay_seconds)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("validate-download")
def validate_download_command(
    run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
) -> None:
    """Recheck exact source coverage, upstream hashes and completed-unit checksums."""
    from preanchor.data.stage import validate_download

    result = validate_download(run_path(run_id), Path(__file__).resolve().parents[2],
                               load_config(config))
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("import-data")
def import_data_command(
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    run_id: Annotated[str, typer.Option()] = "20260914T183017Z-1c085746e3-08bc1e7c",
) -> None:
    """Import exact released traces and sequentially align their chunks."""
    from preanchor.data.importer import import_all
    from preanchor.io import read_json

    root = Settings().persistent_root()
    run = run_path(run_id)  # Validated charset and existence; never a raw path join.
    manifest = read_json(run / "stages/stage2/source_manifest.json")
    loaded = load_config(config)
    output = root / "data/processed/stage3"
    result = import_all(manifest, loaded.dataset.problem_ids, output)
    typer.echo(json.dumps({"trajectories": len(result["trajectories"]),
                           "sentences": len(result["sentences"]), "unresolved": 0}))


@app.command("align")
def align_command(
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    run_id: Annotated[str, typer.Option()] = "20260914T183017Z-1c085746e3-08bc1e7c",
) -> None:
    """Validate the persisted Stage 3 alignment output."""
    from preanchor.io import validate_parquet

    root = Settings().persistent_root()
    output = root / "data/processed/stage3"
    trajectories = validate_parquet(output / "trajectories.parquet")
    sentences = validate_parquet(output / "sentences.parquet")
    if trajectories.num_rows != 40 or sentences.num_rows == 0:
        raise typer.BadParameter("Stage 3 grain is incomplete")
    typer.echo(json.dumps({"trajectories": trajectories.num_rows,
                           "sentences": sentences.num_rows, "unresolved": 0}))


@app.command("validate-data")
def validate_data_command() -> None:
    """Check Stage 3 checksummed Parquet outputs and exact span slices."""
    from preanchor.io import validate_parquet

    root = Settings().persistent_root()
    output = root / "data/processed/stage3"
    trajectories = validate_parquet(output / "trajectories.parquet").to_pylist()
    sentences = validate_parquet(output / "sentences.parquet").to_pylist()
    by_id = {r["trajectory_id"]: r for r in trajectories}
    if any(not r["alignment_ok"] or not r["eligible"] for r in sentences):
        raise typer.BadParameter("Unresolved or ineligible Stage 3 primary row")
    if any(by_id[r["trajectory_id"]]["full_cot"][r["char_start"]:r["char_end"]] != r["text"]
           for r in sentences):
        raise typer.BadParameter("Sentence span does not match exact trace text")
    typer.echo(json.dumps({"trajectories": len(trajectories), "sentences": len(sentences),
                           "unresolved": 0}))


@app.command("fetch-raw-resamples")
def fetch_raw_resamples_command(
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    discover: Annotated[bool, typer.Option()] = False,
    run_id: Annotated[str | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option()] = False,
    workers: Annotated[int, typer.Option(min=1, max=8)] = 4,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Stage 4 fallback: pin and fetch every raw fork file for the 40 trajectories."""
    import os

    from preanchor.labels.stage4 import discover_raw, run_raw_fetch, save_raw_discovery

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Settings().persistent_root()
    persistent_mount(root)
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    if discover:
        if resume or run_id or limit:
            raise typer.BadParameter("Discovery cannot resume a fetch run")
        found = discover_raw(loaded)
        save_raw_discovery(found, repo)
        typer.echo(json.dumps({"files": len(found["files"]),
                               "bytes": sum(r["size"] for r in found["files"]),
                               "trajectories": len(found["forks_per_trajectory"])}))
        return
    if resume and not run_id:
        raise typer.BadParameter("--resume requires --run-id")
    # The shared hub cache can be root-owned (created through the vLLM container mount);
    # transit downloads then use a sibling cache the user owns. Units hold the bytes.
    cache_dir = root / "cache/huggingface/hub"
    if not os.access(cache_dir, os.W_OK):
        cache_dir = root / "cache/huggingface/datasets-hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
    directory = run_path(run_id) if run_id else create_run(root, repo, loaded, [
        f"preanchor fetch-raw-resamples --config {config} --workers {workers} --limit {limit}"
        f" --cache-dir {cache_dir}"])
    result = run_raw_fetch(directory, repo, loaded, cache_dir,
                           resume=resume, workers=workers, limit=limit)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("validate-raw-resamples")
def validate_raw_resamples_command(
    run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    require_identity: Annotated[bool, typer.Option()] = False,
) -> None:
    """Recheck a raw-resample run's coverage, upstream hashes and unit checksums."""
    from preanchor.labels.stage4 import validate_raw_download

    result = validate_raw_download(run_path(run_id), Path(__file__).resolve().parents[2],
                                   load_config(config), require_identity=require_identity)
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("audit-upstream")
def audit_upstream_command(
    raw_run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    run_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Problem 4682 released-versus-canonical/legacy audit (SPEC §3.6 gate)."""
    from preanchor.labels.stage4 import audit_4682

    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    raw_run = run_path(raw_run_id)
    directory = run_path(run_id) if run_id else create_run(root, repo, loaded, [
        f"preanchor audit-upstream --config {config} --raw-run-id {raw_run_id}"])
    report = audit_4682(directory, repo, loaded, root, raw_run, device=device)
    typer.echo(json.dumps({"run_id": directory.name, **report}, sort_keys=True, default=str))


@app.command("build-labels")
def build_labels_command(
    run_id: Annotated[str, typer.Option()],
    raw_run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    device: Annotated[str, typer.Option()] = "cpu",
    publish: Annotated[bool, typer.Option()] = True,
) -> None:
    """Build canonical/legacy/primary labels for all 40 trajectories after the audit."""
    from preanchor.labels.stage4 import build_labels

    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    summary = build_labels(run_path(run_id), repo, loaded, root, run_path(raw_run_id),
                           device=device, publish=publish)
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "per_trajectory"},
                          sort_keys=True, default=str))


@app.command("pilot-labels")
def pilot_labels_command(
    config: Annotated[Path, typer.Option()] = Path("configs/pilot.yaml"),
    benchmark_only: Annotated[int | None, typer.Option("--benchmark-only", min=1)] = None,
    resume: Annotated[bool, typer.Option()] = False,
    base_only: Annotated[bool, typer.Option()] = False,
    run_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Generate resumable Problem 4682 completions; benchmark outputs are isolated.

    Benchmark: ``--benchmark-only N`` measures N completions (resumable with
    ``--run-id``) and writes a measured resource projection. Production:
    ``--resume --base-only --run-id ID`` on a run prepared from a benchmark run.
    """
    if benchmark_only is None and not resume:
        raise typer.BadParameter("production pilot requires --resume after benchmark projection")
    if benchmark_only is None:
        if not base_only or not run_id or Path(run_id).name != run_id:
            raise typer.BadParameter("Production requires --base-only --resume --run-id")
        from preanchor.labels.pilot import execute

        root = Settings().persistent_root()
        execute(root, Path(__file__).resolve().parents[2], config, run_path(run_id))
        return
    result = run_benchmark(config, benchmark_only, run_id=run_id, resume=resume)
    typer.echo(json.dumps(result, sort_keys=True))


def run_benchmark(config: Path, count: int, *, run_id: str | None, resume: bool) -> dict:
    """Measured benchmark: exact tokenizer parity, digest-checked backend, telemetry."""
    import resource
    import statistics
    import subprocess
    import time

    from transformers import AutoTokenizer

    from preanchor.io import atomic_bytes, json_bytes
    from preanchor.labels.generate import (
        ADD_SPECIAL_TOKENS,
        CompletionStore,
        VLLMBackend,
        benchmark_specs,
        prompt_hash,
    )
    from preanchor.labels.pilot import DIGEST, hourly_rate, inputs, tokenizer_check
    from preanchor.manifest import check_resume, load_manifest, save_manifest, utc_now

    if resume and not run_id:
        raise typer.BadParameter("--resume requires --run-id")
    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    rate = hourly_rate()
    trajectory, targets, _references, hashes = inputs(root, repo)
    target_indices = [int(x["sentence_idx"]) for x in targets]
    if run_id:
        run = run_path(run_id)
        manifest = check_resume(run, repo, loaded)
        if manifest["input_hashes"] != hashes:
            raise typer.BadParameter("Benchmark run inputs differ from the current inputs")
        if manifest["stages"].get("stage5", {}).get("namespace") != "benchmark":
            raise typer.BadParameter("Run is not a benchmark run")
    else:
        run = create_run(root, repo, loaded, [f"preanchor pilot-labels --benchmark-only {count}"])
        manifest = load_manifest(run)
        manifest["input_hashes"] = hashes
        manifest["identity"]["input_hashes"] = hashes
        manifest["stages"]["stage5"] = {"state": "benchmark_prepared", "namespace": "benchmark"}
        save_manifest(run, manifest)
    stage = run / "stages/stage5"
    backend = VLLMBackend(endpoint=loaded.generation.get("endpoint", "http://127.0.0.1:8000"),
                          image_digest=DIGEST)
    inspection = json.loads(subprocess.check_output(
        ["sudo", "-n", "docker", "inspect", "preanchor-vllm"], text=True))[0]
    image = json.loads(subprocess.check_output(
        ["sudo", "-n", "docker", "image", "inspect", inspection["Image"]], text=True))[0]
    if not any(d.endswith("@" + DIGEST) for d in image["RepoDigests"]):
        raise typer.BadParameter("vLLM digest mismatch")
    if any(p["HostIp"] != "127.0.0.1"
           for p in inspection["HostConfig"]["PortBindings"]["8000/tcp"]):
        raise typer.BadParameter("Backend is not localhost only")
    tokenizer = AutoTokenizer.from_pretrained(
        loaded.model.id, revision=loaded.model.tokenizer_revision, local_files_only=True)
    specs = benchmark_specs(trajectory, target_indices, loaded.seed_root,
                            loaded.resolved_hash(), config_hash_of(hashes), count)
    expected_tokens, records = {}, []
    for prompt in sorted({s.prompt for s in specs}, key=prompt_hash):
        ids = tokenizer(prompt, add_special_tokens=ADD_SPECIAL_TOKENS)["input_ids"]
        record = tokenizer_check(ids, backend.tokenize_ids(prompt))
        expected_tokens[prompt_hash(prompt)] = len(ids)
        records.append({**record, "prompt_sha256": prompt_hash(prompt)})
    # Attempt-level records: a resumed benchmark rewrites them (manifest keeps history).
    atomic_bytes(stage / "tokenizer_validation.json", json_bytes({
        "tokenizer_id": loaded.model.id, "tokenizer_revision": loaded.model.tokenizer_revision,
        "add_special_tokens": ADD_SPECIAL_TOKENS, "chat_template": None,
        "per_unit_prompt_token_assertion": True, "prompts": records}), replace=True)
    store = CompletionStore(stage / "benchmark_units")
    manifest["commands"].append({"command": "pilot-labels --benchmark-only", "count": count,
                                 "resume": resume})
    manifest["stages"]["stage5"]["state"] = "benchmark_running"
    manifest["vllm"] = {"image": loaded.generation["vllm_image"], "digest": DIGEST}
    save_manifest(run, manifest)
    peak_vram = 0
    started = time.perf_counter()
    rows, created_count = [], 0
    for spec in specs:
        row, created = store.run_one(spec, backend, max_tokens=loaded.generation["max_new_tokens"],
                                     temperature=loaded.generation["temperature"],
                                     top_p=loaded.generation["top_p"],
                                     expected_prompt_tokens=expected_tokens[prompt_hash(spec.prompt)])
        rows.append(row)
        created_count += int(created)
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True).strip()
        peak_vram = max(peak_vram, int(smi.split("\n")[0]))
        atomic_bytes(stage / "telemetry.json", json_bytes({"updated_at": utc_now(),
                     "peak_hbm_mib": peak_vram, "completed": len(rows)}), replace=True)
    elapsed = time.perf_counter() - started
    rates = [r["generated_token_count"] / max(float(r.get("runtime_seconds", 0.0)), 1e-9)
             for r in rows]
    seconds_per_unit = elapsed / created_count if created_count else None
    projection = {"schema_version": 2, "status": "benchmark_complete", "run_id": run.name,
                  "code_commit": manifest["source"]["git_sha"],
                  "config_hash": loaded.resolved_hash(), "input_hashes": hashes,
                  "model": {"id": loaded.model.id, "revision": loaded.model.revision,
                            "dtype": loaded.model.dtype},
                  "benchmark_completion_count": len(rows), "created_this_session": created_count,
                  "elapsed_seconds": elapsed, "wall_seconds": elapsed,
                  "measured_seconds_per_unit": seconds_per_unit,
                  "generated_tokens": sum(r["generated_token_count"] for r in rows),
                  "decode_tokens_per_second": {
                      "median": statistics.median(rates) if rates else 0.0},
                  "peak_vram_mib": peak_vram,
                  "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "truncations": sum(r["finish_reason"] == "length" for r in rows),
                  "price_usd_per_hour": rate, "rate_source": "PREANCHOR_HOURLY_RATE",
                  "projected_base_seconds": seconds_per_unit * 256 if seconds_per_unit else None,
                  "projected_base_cost_usd": (seconds_per_unit * 256 / 3600 * rate
                                              if seconds_per_unit else None),
                  "production_started": False}
    atomic_bytes(stage / "resource_projection.json", json_bytes(projection), replace=True)
    manifest["stages"]["stage5"]["state"] = "benchmark_complete"
    manifest["output_hashes"]["stage5_benchmark_projection"] = prompt_hash(
        json.dumps(projection, sort_keys=True))
    save_manifest(run, manifest)
    return {"run_id": run.name, "stage": str(stage), "rows": len(rows),
            "created": created_count, "elapsed_seconds": elapsed,
            "projection": str(stage / "resource_projection.json")}


def config_hash_of(value: dict) -> str:
    from preanchor.config import config_hash

    return config_hash(value)


if __name__ == "__main__":
    app()


@app.command("extract-activations")
def extract_activations_command(
    config: Annotated[Path, typer.Option()] = Path("configs/smoke.yaml"),
    trajectory: Annotated[list[str] | None, typer.Option(
        help="Trajectory id such as 330:correct; repeatable.")] = None,
    limit_sentences: Annotated[int | None, typer.Option("--limit-sentences", min=1)] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    resume: Annotated[bool, typer.Option()] = False,
    pause_seconds: Annotated[float, typer.Option(min=0, max=60,
                                                 help="Interruption drills only.")] = 0.0,
) -> None:
    """Replay trajectories once and store boundary activations as resumable units.

    Without ``--run-id`` a new run is prepared from the current commit and executed.
    ``--resume --run-id ID`` validates completed units and reruns only missing work.
    SPEC Stage 7 (all 40 trajectories under configs/mvp.yaml) needs separate authorisation.
    """
    from preanchor.features.extract import execute, load_inputs, prepare, select_work

    loaded = load_config(config)
    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    if run_id is None:
        if resume:
            raise typer.BadParameter("--resume requires --run-id")
        inputs, hashes = load_inputs(root)
        work = select_work(loaded, inputs, trajectory, limit_sentences)
        command = (f"preanchor extract-activations --config {config}"
                   + "".join(f" --trajectory {t}" for t in trajectory or [])
                   + (f" --limit-sentences {limit_sentences}" if limit_sentences else ""))
        directory = prepare(root, repo, loaded, command, work, hashes)
    else:
        if not resume:
            raise typer.BadParameter("--run-id requires --resume")
        directory = run_path(run_id)
    result = execute(directory, repo, loaded, trajectories=trajectory,
                     limit_sentences=limit_sentences, resume=resume, pause_seconds=pause_seconds)
    typer.echo(json.dumps(redact({"run_id": directory.name, "created": result["created"],
                                  "skipped": result["skipped"],
                                  "n_rows": result["summary"]["n_rows"],
                                  "diagnostics": result.get("diagnostics")}), sort_keys=True))


@app.command("build-dataset")
def build_dataset_command(
    run_id: Annotated[str, typer.Option(help="Completed extract-activations run.")],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    device: Annotated[str | None, typer.Option(
        help="Device for the float32 norm+lm_head statistics; default cuda:0 if available.")]
    = None,
    publish: Annotated[bool | None, typer.Option(
        help="Publish to data/processed/stage7; default: only for execution mode mvp.")]
    = None,
) -> None:
    """SPEC Stage 7: strict observables joined to Stage 4 labels by sentence_id.

    Reads the run's validated activation shards, tokenises each stored trace once, computes
    the SPEC §3.9 observables in float32 from stored layer-48 boundary states and writes
    features/features.parquet plus feature_audit.json; publishes to
    data/processed/stage7/features.parquet.
    """
    from preanchor.features.dataset import build_dataset

    loaded = load_config(config)
    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    result = build_dataset(run_path(run_id), repo, loaded, root, device=device, publish=publish)
    audit = result["audit"]
    typer.echo(json.dumps(redact({"run_id": run_id, "n_rows": result["n_rows"],
                                  "features_sha256": result["features_sha256"],
                                  "device": audit["device"], "units": audit["units"],
                                  "eligible_rows": audit["eligible_rows"],
                                  "anchor_rows": audit["anchor_rows"],
                                  "published": audit["published"],
                                  "superseded_publication":
                                      audit.get("superseded_publication")}), sort_keys=True))


@app.command("validate-features")
def validate_features_command(
    run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/mvp.yaml"),
    sample: Annotated[int, typer.Option(min=0, help="Rows recomputed from shards.")] = 64,
    device: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Re-validate the Stage 7 feature table: coverage, joins, allow-list, invariants,
    published-copy identity and an exact recomputation of a seeded row sample."""
    from preanchor.features.dataset import validate_features

    loaded = load_config(config)
    root = Settings().persistent_root()
    persistent_mount(root)
    report = validate_features(run_path(run_id), loaded, root, sample=sample, device=device)
    typer.echo(json.dumps(redact({k: report[k] for k in (
        "ok", "run_id", "rows", "units", "features_sha256", "published_identical",
        "recomputed", "split_rows", "eligible_rows", "anchor_rows")}), sort_keys=True))


@app.command("validate-activations")
def validate_activations_command(
    run_id: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/smoke.yaml"),
) -> None:
    """Re-validate activation units, index and coverage without loading the model."""
    from preanchor.features.extract import validate_run

    typer.echo(json.dumps(validate_run(run_path(run_id), load_config(config)), sort_keys=True))
