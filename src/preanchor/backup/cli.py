"""``preanchor backup ...`` and ``preanchor restore ...`` Typer sub-apps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from preanchor.backup import prepare as prepare_module
from preanchor.backup import restore as restore_module
from preanchor.backup import verify as verify_module
from preanchor.backup.bundle import list_bundles
from preanchor.backup.legacy import import_legacy
from preanchor.backup.prepare import portable_root
from preanchor.backup.receipts import list_receipts
from preanchor.config import Settings
from preanchor.logging import redact

backup_app = typer.Typer(no_args_is_help=True,
                         help="Content-addressed portable backup of canonical state.")
restore_app = typer.Typer(no_args_is_help=True,
                          help="Fail-closed restore and scientific-equivalence verification.")

REPO = Path(__file__).resolve().parents[3]
RootOption = Annotated[Path | None, typer.Option(
    help="Persistent root (default: PREANCHOR_ROOT, which must be a mounted /lambda/nfs path).")]
RepoOption = Annotated[Path, typer.Option(help="Repository checkout for code identity.")]
StoreOption = Annotated[Path | None, typer.Option(
    help="Portable store directory (default: <root>/artifacts/portable).")]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output only.")]


def resolve_root(root: Path | None) -> Path:
    if root is not None:
        return root.resolve()
    from preanchor.storage import persistent_mount

    resolved = Settings().persistent_root()
    persistent_mount(resolved)
    return resolved


def emit(value: dict, as_json: bool, text: str | None = None) -> None:
    if as_json or text is None:
        typer.echo(json.dumps(redact(value), sort_keys=True, indent=2))
    else:
        typer.echo(text)


def fail(exc: Exception, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps({"ok": False, "error": type(exc).__name__,
                               "detail": str(exc)}))
    else:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
    raise typer.Exit(1) from None


def _size(value: int | None) -> str:
    value = value or 0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value} B"
        value /= 1024
    return str(value)


def render_plan(summary: dict) -> str:
    lines = [f"Backup plan for stage {summary['for_stage']} "
             f"(kind={summary['kind']}, highest protected stage="
             f"{summary['highest_protected_stage']})"]
    for stage, info in summary["stages"].items():
        extra = ""
        if stage == "stage5" and info.get("status") != "absent":
            extra = (f" run={info.get('run_id')} units={info.get('protected_completed_units')}/"
                     f"{info.get('planned_units')} resumable={info.get('resumable')}")
        if stage == "stage6" and info.get("status") != "absent":
            extra = (f" runs={len(info.get('run_ids') or [])} units={info.get('total_units')} "
                     f"rows={info.get('total_rows')}")
        lines.append(f"  {stage}: {info.get('status')}{extra}")
    lines.append("Classification:")
    for name, row in summary["classification"].items():
        lines.append(f"  {name:<20} {row['files']:>7} items  {_size(row['bytes']):>10}")
    objects = summary["objects"]
    lines.append(f"Logical bundle: {summary['logical_files']} files, "
                 f"{_size(summary['logical_bytes'])}")
    lines.append(f"Objects: {objects['unique']} unique, {objects['existing']} already in store, "
                 f"{objects['new']} new ({_size(objects['new_bytes'])} incremental transfer)")
    lines.append("Reacquire (pinned, not bundled):")
    for item in summary["reacquire"]:
        lines.append(f"  - {item['kind']}: {item['resource']} {json.dumps(item['pin'])}"
                     + (f" {_size(item['bytes'])}" if item.get("bytes") else ""))
    if summary["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in summary["warnings"])
    if summary["blockers"]:
        lines.append("BLOCKERS:")
        lines.extend(f"  - {b}" for b in summary["blockers"])
    else:
        lines.append("No blockers.")
    return "\n".join(lines)


@backup_app.command("plan")
def backup_plan(
    for_stage: Annotated[int, typer.Option(help="Continuation target stage.")] = 5,
    run_id: Annotated[str | None, typer.Option(help="Disambiguate the production run.")] = None,
    include_historical: Annotated[bool, typer.Option(
        help="Also bundle historical/failed runs (metadata only).")] = False,
    root: RootOption = None, repo: RepoOption = REPO, as_json: JsonOption = False,
) -> None:
    """Read-only: classify, validate and size the canonical state for a stage."""
    try:
        summary, _ = prepare_module.plan(resolve_root(root), repo, for_stage=for_stage,
                                         run_id=run_id, include_historical=include_historical)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    emit(summary, as_json, render_plan(summary))
    if summary["blockers"]:
        raise typer.Exit(2)


@backup_app.command("prepare")
def backup_prepare(
    for_stage: Annotated[int, typer.Option()] = 5,
    run_id: Annotated[str | None, typer.Option()] = None,
    include_historical: Annotated[bool, typer.Option()] = False,
    allow_dirty: Annotated[bool, typer.Option(
        help="Allow a dirty preparing checkout (recorded in the producer identity).")] = True,
    root: RootOption = None, repo: RepoOption = REPO, as_json: JsonOption = False,
) -> None:
    """Validate, ingest new objects, write an immutable bundle manifest and receipt."""
    try:
        result = prepare_module.prepare(resolve_root(root), repo, for_stage=for_stage,
                                        run_id=run_id, include_historical=include_historical,
                                        allow_dirty=allow_dirty)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    text = (f"Prepared bundle {result['bundle_id']} ({result['kind']}): "
            f"{result['entries']} entries, {_size(result['logical_bytes'])} logical, "
            f"{result['objects_created']} new objects ({_size(result['bytes_created'])}), "
            f"{result['objects_reused']} reused.\n"
            f"Manifest: {result['manifest']}\nReceipt: {result['receipt']}\n"
            "Primary-only. Not a verified secondary copy until the MacBook pull "
            "receipt exists.")
    emit(result, as_json, text)


@backup_app.command("verify")
def backup_verify(
    bundle: Annotated[str, typer.Option(help="Bundle ID, hash prefix or 'latest'.")] = "latest",
    root: RootOption = None, store: StoreOption = None, as_json: JsonOption = False,
) -> None:
    """Rehash every object a bundle references in the local store."""
    try:
        portable = store or portable_root(resolve_root(root))
        result = prepare_module.verify_bundle(portable, bundle)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    emit(result, as_json, f"{result['bundle_id']}: {result['objects']} objects, "
         f"{_size(result['verified_bytes'])} rehashed, missing {len(result['missing'])}, "
         f"corrupt {len(result['corrupt'])} -> {'OK' if result['ok'] else 'FAILED'}")
    if not result["ok"]:
        raise typer.Exit(1)


@backup_app.command("list")
def backup_list(root: RootOption = None, store: StoreOption = None,
                as_json: JsonOption = False) -> None:
    """List bundle manifests, the latest pointer and receipts."""
    try:
        portable = store or portable_root(resolve_root(root))
        bundles = list_bundles(portable)
        latest = None
        if (portable / "latest.json").is_file():
            latest = json.loads((portable / "latest.json").read_bytes())
        result = {"store": str(portable), "latest": latest, "bundles": bundles,
                  "receipts": list_receipts(portable)}
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    lines = [f"Store: {portable}", f"Latest: {latest['bundle_id'] if latest else None}"]
    for row in bundles:
        if row["valid"]:
            lines.append(f"  {row['bundle_id']}  kind={row['kind']} for_stage={row['for_stage']} "
                         f"protected={row['highest_protected_stage']} entries={row['entries']} "
                         f"{_size(row['logical_bytes'])}")
        else:
            lines.append(f"  {row['bundle_id']}  INVALID: {row['error']}")
    lines.append("Receipts:")
    for receipt in result["receipts"]:
        lines.append(f"  {receipt['path']}  {receipt.get('kind')} {receipt.get('outcome', '')}")
    emit(result, as_json, "\n".join(lines))


@backup_app.command("import-legacy")
def backup_import_legacy(
    snapshot: Annotated[Path, typer.Argument(help="Legacy checksums.json snapshot directory.")],
    root: RootOption = None, repo: RepoOption = REPO, as_json: JsonOption = False,
) -> None:
    """Verify a legacy checksums.json snapshot and ingest it as historical evidence."""
    try:
        result = import_legacy(resolve_root(root), repo, snapshot)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    emit(result, as_json, f"Imported {snapshot} as {result['bundle_id']}: "
         f"{result['entries']} entries, {result['objects_created']} new objects.")


@backup_app.command("store-path")
def backup_store_path(root: RootOption = None, as_json: JsonOption = False) -> None:
    """Print the portable store directory (used by the MacBook wrapper)."""
    try:
        portable = portable_root(resolve_root(root))
    except (ValueError, OSError) as exc:
        fail(exc, as_json)
    emit({"store": str(portable), "root": str(portable.parents[1])}, as_json, str(portable))


@restore_app.command("plan")
def restore_plan_command(
    bundle: Annotated[str, typer.Option()] = "latest",
    for_stage: Annotated[int | None, typer.Option()] = None,
    required_only: Annotated[bool, typer.Option()] = False,
    root: RootOption = None, store: StoreOption = None, as_json: JsonOption = False,
) -> None:
    """Read-only: which entries would be installed, conflicts and missing objects."""
    try:
        resolved = resolve_root(root)
        result = restore_module.restore_plan(resolved, store or portable_root(resolved), bundle,
                                             for_stage=for_stage, required_only=required_only)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    text = (f"Restore plan {result['bundle_id']} -> {result['root']}: {result['entries']} "
            f"entries, {_size(result['bytes'])}; states {json.dumps(result['states'])}")
    if result["details"]:
        text += "\n" + "\n".join(f"  {d['state']}: {d['path']}" for d in result["details"][:40])
    emit(result, as_json, text)
    if not result["ok"]:
        raise typer.Exit(2)


@restore_app.command("apply")
def restore_apply_command(
    bundle: Annotated[str, typer.Option()] = "latest",
    for_stage: Annotated[int | None, typer.Option()] = None,
    required_only: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
    root: RootOption = None, store: StoreOption = None, repo: RepoOption = REPO,
    as_json: JsonOption = False,
) -> None:
    """Install verified objects atomically; refuse to overwrite different bytes."""
    try:
        resolved = resolve_root(root)
        result = restore_module.restore_apply(resolved, store or portable_root(resolved), bundle,
                                              repo=repo, for_stage=for_stage,
                                              required_only=required_only, dry_run=dry_run)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    text = (f"Restore {'(dry run) ' if dry_run else ''}{result['bundle_id']} -> "
            f"{result['root']}: states {json.dumps(result['states'])} -> "
            f"{'OK' if result['ok'] else 'FAILED'}")
    if result["failures"]:
        text += "\n" + "\n".join(f"  {v}: {k}" for k, v in list(result["failures"].items())[:40])
    emit(result, as_json, text)
    if not result["ok"]:
        raise typer.Exit(1)


@restore_app.command("verify")
def restore_verify_command(
    bundle: Annotated[str, typer.Option()] = "latest",
    for_stage: Annotated[int, typer.Option()] = 5,
    root: RootOption = None, store: StoreOption = None, repo: RepoOption = REPO,
    as_json: JsonOption = False,
) -> None:
    """Scientific-equivalence gate: identity, inputs, plan/rollout IDs, units, settings."""
    try:
        resolved = resolve_root(root)
        result = verify_module.restore_verify(resolved, store or portable_root(resolved), bundle,
                                              repo=repo, for_stage=for_stage)
    except (ValueError, OSError, KeyError) as exc:
        fail(exc, as_json)
    emit(result, as_json, verify_module.render(result))
    if not result["verified"]:
        raise typer.Exit(1)
