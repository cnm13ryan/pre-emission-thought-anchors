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


if __name__ == "__main__":
    app()
