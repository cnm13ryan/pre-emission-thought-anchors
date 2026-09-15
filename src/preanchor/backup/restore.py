"""Fail-closed, idempotent restoration of a bundle onto any root (new UUID, host, region)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import stage_closure
from preanchor.backup.receipts import write_receipt
from preanchor.backup.store import ObjectStore
from preanchor.ids import file_hash
from preanchor.io import UnitStore, validate_artifact, validate_parquet
from preanchor.manifest import load_manifest

STATES = ("restored", "already_present_identical", "missing", "mismatch", "validation_failed",
          "conflict", "planned")


def select_entries(manifest: dict, for_stage: int | None, *, required_only: bool) -> list[dict]:
    if for_stage is None:
        entries = manifest["entries"]
    else:
        stages = {f"stage{s}" for s in stage_closure(for_stage)}
        entries = [e for e in manifest["entries"]
                   if e["stage"] in stages or e["stage"] in ("provenance", "evidence")
                   or not required_only]
    if required_only:
        entries = [e for e in entries if e["required"]]
    return entries


def safe_target(root: Path, rel: str) -> Path:
    """The restore destination for a bundle entry, refusing anything that could resolve
    outside ``root``: absolute paths, ``..``/empty/``.`` components, and symlinked
    ancestors between ``root`` and the target."""
    parts = Path(rel).parts
    if Path(rel).is_absolute() or not parts or any(p in ("..", ".", "") for p in parts):
        raise ValueError(f"unsafe bundle path {rel!r}")
    target = root / rel
    ancestor = target.parent
    ancestors = []
    while ancestor != root and root in ancestor.parents:
        ancestors.append(ancestor)
        ancestor = ancestor.parent
    if ancestor != root:
        raise ValueError(f"unsafe bundle path {rel!r}")
    for ancestor in reversed(ancestors):
        if ancestor.is_symlink():
            raise ValueError(f"symlinked ancestor {ancestor} for bundle path {rel!r}")
    resolved_root = root.resolve()
    if target.parent.exists() and not target.parent.resolve().is_relative_to(resolved_root):
        raise ValueError(f"bundle path {rel!r} traverses a symlink out of the root")
    return target


def destination_state(store: ObjectStore, root: Path, entry: dict) -> str:
    try:
        target = safe_target(root, entry["path"])
    except ValueError:
        return "conflict"
    if target.is_symlink():
        return "conflict"
    if target.exists():
        return "already_present_identical" if (
            target.is_file() and file_hash(target) == entry["sha256"]) else "conflict"
    if not store.has(entry["sha256"]):
        return "missing"
    return "planned"


def restore_plan(root: Path, portable: Path, selector: str, *, for_stage: int | None = None,
                 required_only: bool = False) -> dict:
    root, portable = Path(root), Path(portable)
    name, manifest = load_bundle(portable, selector)
    store = ObjectStore(portable)
    entries = select_entries(manifest, for_stage, required_only=required_only)
    states = Counter()
    details = []
    for entry in entries:
        state = destination_state(store, root, entry)
        states[state] += 1
        if state != "planned":
            details.append({"path": entry["path"], "state": state})
    return {"bundle_id": name, "root": str(root), "store": str(portable),
            "entries": len(entries), "bytes": sum(e["size"] for e in entries),
            "states": dict(states), "details": details,
            "reacquire": manifest["reacquire"], "ok": states["missing"] == 0
            and states["conflict"] == 0}


def _validate_installed(root: Path, entries: list[dict]) -> dict[str, str]:
    """Artifact-specific validators after installation; unit directories as a whole."""
    failures: dict[str, str] = {}
    units: dict[str, list[dict]] = {}
    for entry in entries:
        path = root / entry["path"]
        try:
            if entry["validation"] == "sha256_sidecar":
                validate_artifact(path)
            elif entry["validation"] == "parquet":
                validate_parquet(path)
            elif entry["validation"] == "run_manifest":
                load_manifest(path.parent)
            elif entry["validation"] == "unit_file":
                units.setdefault(entry["unit"], []).append(entry)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            failures[entry["path"]] = str(exc)
    for unit, members in units.items():
        directory = root / unit
        try:
            UnitStore(directory.parent).validate(directory.name)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            for entry in members:
                failures[entry["path"]] = f"unit: {exc}"
    return failures


def restore_apply(root: Path, portable: Path, selector: str, *, repo: Path | None = None,
                  for_stage: int | None = None, required_only: bool = False,
                  dry_run: bool = False) -> dict:
    """Install every selected entry from verified objects; never overwrite different bytes;
    never regenerate missing canonical state; write an immutable receipt."""
    root, portable = Path(root), Path(portable)
    name, manifest = load_bundle(portable, selector)
    store = ObjectStore(portable)
    entries = select_entries(manifest, for_stage, required_only=required_only)
    per_entry: dict[str, str] = {}
    installed: list[dict] = []
    for entry in entries:
        if dry_run:
            per_entry[entry["path"]] = destination_state(store, root, entry)
            continue
        try:
            target = safe_target(root, entry["path"])
        except ValueError as exc:
            per_entry[entry["path"]] = f"conflict:{exc}"
            continue
        try:
            store.verify(entry["sha256"])
        except FileNotFoundError:
            per_entry[entry["path"]] = "missing"
            continue
        except ValueError:
            per_entry[entry["path"]] = "mismatch"
            continue
        try:
            state = store.install(entry["sha256"], target)
        except FileExistsError:
            per_entry[entry["path"]] = "conflict"
            continue
        except ValueError as exc:
            per_entry[entry["path"]] = f"mismatch:{exc}"
            continue
        per_entry[entry["path"]] = state
        installed.append(entry)
    if not dry_run:
        for path, reason in _validate_installed(root, installed).items():
            per_entry[path] = f"validation_failed:{reason}"
    states = Counter(state.split(":", 1)[0] for state in per_entry.values())
    ok = all(state in ("restored", "already_present_identical", "planned")
             for state in states)
    result = {"bundle_id": name, "manifest_sha256": name.rsplit("-", 1)[1],
              "root": str(root), "store": str(portable), "dry_run": dry_run,
              "entries": len(entries), "states": dict(states),
              "failures": {k: v for k, v in per_entry.items()
                           if v.split(":", 1)[0] not in ("restored",
                                                         "already_present_identical", "planned")},
              "run_ids": manifest["scientific_identity"].get("run_ids", []),
              "reacquire": manifest["reacquire"], "ok": ok,
              "physical_origin": manifest["physical_origin"]}
    if not dry_run:
        from preanchor.backup.prepare import physical_origin

        result["physical_destination"] = physical_origin(root)
        receipt = write_receipt(portable, "restore", {
            "bundle_id": name, "manifest_sha256": result["manifest_sha256"],
            "run_ids": result["run_ids"], "source": str(portable), "destination": str(root),
            "object_count": len({e["sha256"] for e in entries}),
            "verified_bytes": sum(e["size"] for e in installed), "states": dict(states),
            "failures": result["failures"], "outcome": "restored" if ok else "failed",
            "physical_destination": result["physical_destination"]})
        result["receipt"] = str(receipt)
    return result
