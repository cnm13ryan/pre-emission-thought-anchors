"""Ingest a legacy ``checksums.json`` snapshot (prepare_*_backup.py) into the object store."""

from __future__ import annotations

import json
from pathlib import Path

from preanchor.backup.bundle import SCHEMA_VERSION, manifest_digest, write_bundle
from preanchor.backup.classify import HISTORICAL_EVIDENCE, SECRET, secret_reason
from preanchor.backup.prepare import (
    WRITER_LEASE,
    physical_origin,
    portable_root,
    producer_identity,
)
from preanchor.backup.receipts import write_receipt
from preanchor.backup.store import ObjectStore
from preanchor.control.leases import require
from preanchor.ids import file_hash
from preanchor.manifest import utc_now


def verify_legacy_snapshot(snapshot: Path) -> dict[str, str]:
    snapshot = Path(snapshot).resolve()
    checksums = snapshot / "checksums.json"
    if not checksums.is_file() or checksums.is_symlink():
        raise ValueError("Legacy snapshot lacks checksums.json")
    entries = json.loads(checksums.read_bytes())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Empty or malformed legacy checksum manifest")
    for name, expected in entries.items():
        path = snapshot / name
        if (path.is_symlink() or not path.resolve().is_relative_to(snapshot)
                or not path.is_file()):
            raise ValueError(f"Unsafe or missing legacy entry: {name}")
        if file_hash(path) != expected:
            raise ValueError(f"Legacy checksum mismatch: {name}")
    return entries


def import_legacy(root: Path, repo: Path, snapshot: Path, *,
                  lease_tokens: dict[str, str] | None = None) -> dict:
    root, snapshot = Path(root), Path(snapshot).resolve()
    entries = verify_legacy_snapshot(snapshot)
    prefix = f"legacy/{snapshot.name}"
    for name in list(entries) + ["checksums.json"]:
        reason = secret_reason(snapshot / name, f"{prefix}/{name}")
        if reason:
            raise ValueError(f"Refusing legacy entry {name}: {reason}")
    require([WRITER_LEASE], root=root, tokens=lease_tokens)   # gate before any store write
    portable = portable_root(root)
    store = ObjectStore(portable)
    records, created, reused, seen = [], 0, 0, set()
    for name in sorted(list(entries) + ["checksums.json"]):
        path = snapshot / name
        digest, new = store.put_file(path)
        if digest not in seen:
            seen.add(digest)
            created += int(new)
            reused += int(not new)
        records.append({"path": f"{prefix}/{name}", "sha256": digest,
                        "size": path.stat().st_size, "classification": HISTORICAL_EVIDENCE,
                        "stage": "evidence", "run_id": None, "required": False,
                        "validation": "bytes", "unit": None})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": {"created_at": utc_now(), "kind": "legacy_import", "for_stage": 0,
                   "required_stages": [], "highest_protected_stage": None,
                   "stages": {}, "runs": {},
                   "legacy_snapshot": {"name": snapshot.name, "path": str(snapshot),
                                       "checksums_sha256": file_hash(snapshot / "checksums.json")}},
        "producer": producer_identity(repo),
        "scientific_identity": {"legacy_snapshot": snapshot.name,
                                "checksums_sha256": file_hash(snapshot / "checksums.json")},
        "scientific_identity_hash": None,
        "physical_origin": physical_origin(root),
        "entries": records, "reacquire": [], "excluded": [], "warnings": [],
        "stats": {"entries": len(records), "logical_bytes": sum(r["size"] for r in records),
                  "unique_objects": len(seen), "objects_created": created,
                  "objects_reused": reused, "bytes_created": None,
                  "classification": {HISTORICAL_EVIDENCE: {"files": len(records)},
                                     SECRET: {"files": 0}}},
    }
    digest = manifest_digest(manifest)
    name = write_bundle(portable, manifest)
    receipt = write_receipt(portable, "legacy_import", {
        "bundle_id": name, "manifest_sha256": digest, "source": str(snapshot),
        "destination": str(portable), "object_count": len(seen),
        "verified_bytes": manifest["stats"]["logical_bytes"], "outcome": "imported",
        "secondary_verified": False})
    return {"bundle_id": name, "manifest_sha256": digest, "entries": len(records),
            "objects_created": created, "objects_reused": reused, "receipt": str(receipt)}
