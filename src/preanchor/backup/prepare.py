"""Read-only backup plans and immutable content-addressed bundle preparation."""

from __future__ import annotations

import os
import platform
import subprocess
from collections import Counter
from pathlib import Path

from preanchor import manifest as run_manifests
from preanchor.backup import classify
from preanchor.backup.bundle import SCHEMA_VERSION, manifest_digest, write_bundle
from preanchor.backup.classify import (
    CACHE_EXCLUDE,
    HISTORICAL_EVIDENCE,
    MUST_BACK_UP,
    REACQUIRE,
    SECRET,
    SHOULD_BACK_UP,
    Inventory,
    discover,
    validate_entry,
)
from preanchor.backup.receipts import write_receipt
from preanchor.backup.store import ObjectStore
from preanchor.config import config_hash
from preanchor.control.leases import WRITER_LEASE, require
from preanchor.manifest import utc_now


def portable_root(root: Path) -> Path:
    return Path(root) / "artifacts/portable"


def producer_identity(repo: Path) -> dict:
    """Code identity of the checkout that prepared the bundle (operational provenance)."""
    source, _ = run_manifests.provenance(repo)   # module attribute: test-patchable
    upstream = (source.get("upstream") or {}).get("repositories") or {}
    return {"git_sha": source["git_sha"], "dirty": bool(source.get("dirty")),
            "dirty_diff_hash": source["dirty_diff_hash"], "lock_hash": source["lock_hash"],
            "upstream": {name: entry["revision"] for name, entry in upstream.items()},
            "python": source.get("python"), "os": source.get("os")}


def physical_origin(root: Path) -> dict:
    """Where the bytes physically lived. Provenance only; never part of scientific equality."""
    root = Path(root)
    origin = {"root": str(root), "hostname": platform.node(), "mount": None, "fstype": None,
              "filesystem": None, "gpu": None, "region": os.environ.get("LAMBDA_REGION") or None}
    try:
        from preanchor.storage import persistent_mount

        mount = persistent_mount(root)
        origin.update(mount=mount.get("target"), fstype=mount.get("fstype"),
                      filesystem=mount.get("source"))
    except (ValueError, OSError, subprocess.SubprocessError, KeyError):
        pass
    try:
        origin["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return origin


def hash_entries(root: Path, inventory: Inventory) -> None:
    """Validate every source artifact and fix its SHA-256. Required failures block."""
    kept = []
    for entry in inventory.entries:
        try:
            entry.sha256 = validate_entry(root, entry)
            entry.size = (root / entry.path).stat().st_size
        except (ValueError, OSError, KeyError, TypeError) as exc:
            message = f"{entry.path}: {exc}"
            if entry.classification == MUST_BACK_UP:
                inventory.blockers.append(message)
            else:
                inventory.warnings.append(f"dropped {message}")
                continue
        kept.append(entry)
    inventory.entries = kept


def summarize(root: Path, inventory: Inventory, store: ObjectStore) -> dict:
    counts: Counter = Counter()
    sizes: Counter = Counter()
    for entry in inventory.entries:
        counts[entry.classification] += 1
        sizes[entry.classification] += entry.size or 0
    for item in inventory.excluded:
        counts[item["classification"]] += 1
        sizes[item["classification"]] += item.get("bytes") or 0
    digests: dict[str, int] = {}
    for entry in inventory.entries:
        if entry.sha256:
            digests.setdefault(entry.sha256, entry.size or 0)
    existing = {d for d in digests if store.has(d)}
    new_bytes = sum(size for d, size in digests.items() if d not in existing)
    return {
        "for_stage": inventory.for_stage, "required_stages": list(inventory.required_stages),
        "kind": classify.bundle_kind(inventory),
        "highest_protected_stage": classify.highest_protected_stage(inventory),
        "stages": inventory.stages, "runs": inventory.runs,
        "classification": {name: {"files": counts.get(name, 0), "bytes": sizes.get(name, 0)}
                           for name in (MUST_BACK_UP, SHOULD_BACK_UP, REACQUIRE, CACHE_EXCLUDE,
                                        HISTORICAL_EVIDENCE, SECRET)},
        "logical_files": len(inventory.entries),
        "logical_bytes": sum(e.size or 0 for e in inventory.entries),
        "objects": {"unique": len(digests), "existing": len(existing),
                    "new": len(digests) - len(existing), "new_bytes": new_bytes,
                    "store": str(store.root)},
        "reacquire": inventory.reacquire, "blockers": inventory.blockers,
        "warnings": inventory.warnings, "unit_counts": inventory.unit_counts,
        "scientific_identity_hash": config_hash(inventory.scientific_identity),
    }


def plan(root: Path, repo: Path, *, for_stage: int, run_id: str | None = None,
         include_historical: bool = False) -> tuple[dict, Inventory]:
    """Read-only: discover, classify, validate, hash and estimate the incremental transfer."""
    root, repo = Path(root), Path(repo)
    inventory = discover(root, repo, for_stage=for_stage, run_id=run_id,
                         include_historical=include_historical)
    hash_entries(root, inventory)
    store = ObjectStore(portable_root(root))
    return summarize(root, inventory, store), inventory


def prepare(root: Path, repo: Path, *, for_stage: int, run_id: str | None = None,
            include_historical: bool = False, allow_dirty: bool = True,
            lease_tokens: dict[str, str] | None = None) -> dict:
    """Ingest validated payloads, write an immutable manifest, repoint latest, receipt it.

    Mutates the shared store, so the ``portable-store:writer`` lease must be held and its
    token presented (``lease_tokens`` or ``PREANCHOR_LEASE_TOKENS``). The read-only plan
    and its blockers come first; the gate sits immediately before the first store write.
    (The CLI takes the lease before calling, so a contended writer still fails fast.)"""
    root, repo = Path(root), Path(repo)
    summary, inventory = plan(root, repo, for_stage=for_stage, run_id=run_id,
                              include_historical=include_historical)
    producer = producer_identity(repo)
    if producer["dirty"] and not allow_dirty:
        inventory.blockers.append("Producer checkout is dirty; commit or pass --allow-dirty")
    if inventory.blockers:
        raise ValueError("Backup blocked:\n  " + "\n  ".join(inventory.blockers))
    if not any(e.classification == MUST_BACK_UP for e in inventory.entries):
        raise ValueError("Nothing canonical to protect")
    require([WRITER_LEASE], root=root, tokens=lease_tokens)   # gate before any store write
    portable = portable_root(root)
    store = ObjectStore(portable)
    created = reused = created_bytes = 0
    seen: set[str] = set()
    for entry in inventory.entries:
        digest, new = store.put_file(root / entry.path, entry.sha256)
        if digest in seen:
            continue
        seen.add(digest)
        if new:
            created += 1
            created_bytes += entry.size or 0
        else:
            reused += 1
    stages = {name: {k: v for k, v in value.items() if k != "unit_counts"}
              for name, value in inventory.stages.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": {"created_at": utc_now(), "kind": summary["kind"],
                   "for_stage": for_stage, "required_stages": list(inventory.required_stages),
                   "highest_protected_stage": summary["highest_protected_stage"],
                   "stages": stages, "runs": inventory.runs},
        "producer": producer,
        "scientific_identity": inventory.scientific_identity,
        "scientific_identity_hash": summary["scientific_identity_hash"],
        "physical_origin": physical_origin(root),
        "entries": [e.record() for e in inventory.entries],
        "reacquire": inventory.reacquire,
        "excluded": inventory.excluded,
        "warnings": inventory.warnings,
        "stats": {"entries": len(inventory.entries),
                  "logical_bytes": summary["logical_bytes"],
                  "unique_objects": len(seen), "objects_created": created,
                  "objects_reused": reused, "bytes_created": created_bytes,
                  "classification": summary["classification"]},
    }
    digest = manifest_digest(manifest)
    name = write_bundle(portable, manifest)
    receipt = write_receipt(portable, "prepare", {
        "bundle_id": name, "manifest_sha256": digest, "bundle_kind": summary["kind"],
        "for_stage": for_stage, "run_ids": inventory.scientific_identity.get("run_ids", []),
        "source": str(root), "destination": str(portable),
        "object_count": len(seen), "objects_created": created, "objects_reused": reused,
        "verified_bytes": summary["logical_bytes"], "outcome": "prepared_primary_only",
        "secondary_verified": False})
    return {"bundle_id": name, "manifest_sha256": digest, "kind": summary["kind"],
            "store": str(portable), "manifest": str(portable / "manifests" / f"{name}.json"),
            "receipt": str(receipt), "entries": len(inventory.entries),
            "logical_bytes": summary["logical_bytes"], "unique_objects": len(seen),
            "objects_created": created, "objects_reused": reused,
            "bytes_created": created_bytes, "stages": stages,
            "highest_protected_stage": summary["highest_protected_stage"],
            "warnings": inventory.warnings, "secondary_verified": False}


def verify_bundle(portable: Path, selector: str) -> dict:
    """Rehash every object a bundle references (the primary-side counterpart of the
    MacBook pull verification)."""
    from preanchor.backup.bundle import load_bundle

    name, manifest = load_bundle(portable, selector)
    store = ObjectStore(portable)
    missing, corrupt, verified_bytes = [], [], 0
    seen: set[str] = set()
    for entry in manifest["entries"]:
        digest = entry["sha256"]
        if digest in seen:
            continue
        seen.add(digest)
        try:
            verified_bytes += store.verify(digest)
        except FileNotFoundError:
            missing.append(digest)
        except ValueError:
            corrupt.append(digest)
    return {"bundle_id": name, "manifest_sha256": name.rsplit("-", 1)[1],
            "objects": len(seen), "verified_bytes": verified_bytes,
            "missing": missing, "corrupt": corrupt, "ok": not missing and not corrupt}
