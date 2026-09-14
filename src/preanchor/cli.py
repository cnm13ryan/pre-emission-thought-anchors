"""Command-line entry point; unimplemented stages are not advertised as runnable."""

import json
from pathlib import Path
from typing import Annotated

import typer

from preanchor.config import Settings, load_config
from preanchor.doctor import run_doctor
from preanchor.logging import configure_logging, redact

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


if __name__ == "__main__":
    app()
