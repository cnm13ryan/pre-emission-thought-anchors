"""Command-line entry point; unimplemented stages are not advertised as runnable."""

import json
from pathlib import Path
from typing import Annotated

import typer

from preanchor.config import Settings, load_config
from preanchor.doctor import run_doctor
from preanchor.logging import configure_logging, redact
from preanchor.manifest import create_run, load_manifest
from preanchor.runtime import Runtime
from preanchor.smoke import run_smoke
from preanchor.storage import persistent_mount

app = typer.Typer(no_args_is_help=True, help="Passive offline thought-anchor experiment.")


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

    root = Settings().persistent_root()
    run = root / "artifacts/runs" / run_id
    manifest = json.loads((run / "stages/stage2/source_manifest.json").read_bytes())
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


@app.command("pilot-labels")
def pilot_labels_command(
    config: Annotated[Path, typer.Option()] = Path("configs/pilot.yaml"),
    benchmark_only: Annotated[int | None, typer.Option("--benchmark-only", min=1)] = None,
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Generate resumable Problem 4682 completions; benchmark outputs are isolated."""
    if benchmark_only is None and not resume:
        raise typer.BadParameter("production pilot requires --resume after benchmark projection")
    import statistics
    import time

    import pyarrow.parquet as pq
    import yaml

    from preanchor.io import write_json
    from preanchor.labels.generate import (
        CompletionStore,
        VLLMBackend,
        benchmark,
        prompt_hash,
        tokenizer_identity_check,
    )

    root = Settings().persistent_root()
    persistent_mount(root)
    repo = Path(__file__).resolve().parents[2]
    loaded = load_config(config)
    trajectory_rows = pq.read_table(root / "data/processed/stage3/trajectories.parquet").to_pylist()
    trajectory = next((r for r in trajectory_rows if r["trajectory_id"] == "4682:correct"), None)
    if trajectory is None:
        raise typer.BadParameter("missing processed 4682:correct trajectory")
    sentences = pq.read_table(root / "data/processed/stage3/sentences.parquet").to_pylist()
    trajectory["sentence_boundaries"] = {
        str(r["sentence_idx"]): int(r["char_start"])
        for r in sentences if r["trajectory_id"] == "4682:correct"
    }
    targets = yaml.safe_load((repo / "configs/pilot_targets_4682.yaml").read_text())["targets"]
    target_indices = [int(x["sentence_idx"]) for x in targets]
    run = create_run(root, repo, loaded, ["preanchor pilot-labels --benchmark-only"])
    stage = run / "stages/stage5"
    unit_root = stage / ("benchmark_units" if benchmark_only is not None else "units")
    backend = VLLMBackend(endpoint=loaded.generation.get("endpoint", "http://127.0.0.1:8000"))
    if benchmark_only is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            loaded.model.id, revision=loaded.model.revision, trust_remote_code=False)
        check_prompt = trajectory["prompt"]
        # vLLM's raw completion endpoint tokenizes the supplied prompt without
        # injecting BOS/EOS. Validate that exact behavior; never alter prompts
        # or apply a chat template to hide a mismatch.
        host_count = len(tokenizer(check_prompt, add_special_tokens=False)["input_ids"])
        backend_count = backend.tokenize(check_prompt)
        token_record = tokenizer_identity_check(
            host_count=host_count, backend_count=backend_count,
            tokenizer_id=loaded.model.id,
            tokenizer_revision=loaded.model.tokenizer_revision,
            prompt_sha256=prompt_hash(check_prompt),
            special_tokens={"add_special_tokens": False, "chat_template": None})
        write_json(stage / "tokenizer_validation.json", token_record)
    started = time.perf_counter()
    rows = benchmark(backend, CompletionStore(unit_root), trajectory, target_indices,
                     count=benchmark_only or 0, seed_root=loaded.seed_root,
                     config_hash=loaded.resolved_hash(), input_hash=trajectory["full_cot_sha256"],
                     max_tokens=loaded.generation["max_new_tokens"],
                     temperature=loaded.generation["temperature"], top_p=loaded.generation["top_p"])
    elapsed = time.perf_counter() - started
    rates = [r["generated_token_count"] / max(float(r.get("runtime_seconds", 0.0)), 1e-9)
             for r in rows]
    projection = {"schema_version": 1, "status": "benchmark_complete", "run_id": run.name,
                  "code_config_hash": loaded.resolved_hash(), "model": {
                      "id": loaded.model.id, "revision": loaded.model.revision,
                      "dtype": loaded.model.dtype}, "benchmark_completion_count": len(rows),
                  "elapsed_seconds": elapsed,
                  "generated_tokens": sum(r["generated_token_count"] for r in rows),
                  "decode_tokens_per_second": {
                      "median": statistics.median(rates) if rates else 0.0},
                  "analytical_roofline": {"hbm_tb_per_s": 1.555, "ideal_decode_tokens_per_s": 52.6},
                  "production_started": False}
    write_json(stage / "resource_projection.json", projection)
    typer.echo(json.dumps({"run_id": run.name, "stage": str(stage), "rows": len(rows),
                           "elapsed_seconds": elapsed,
                           "projection": str(stage / "resource_projection.json")}, sort_keys=True))


if __name__ == "__main__":
    app()
