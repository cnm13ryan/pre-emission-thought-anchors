"""Immutable, self-hashed bundle manifests and the replaceable ``latest.json`` pointer."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from preanchor.io import atomic_bytes, json_bytes, write_artifact
from preanchor.logging import redact

SCHEMA_VERSION = 1
BUNDLE_NAME = re.compile(r"^(\d{8}T\d{6}Z)-([0-9a-f]{64})$")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def manifest_digest(manifest: dict) -> str:
    """The manifest identity is the SHA-256 of its canonical JSON bytes."""
    return hashlib.sha256(json_bytes(manifest)).hexdigest()


def bundle_id(stamp: str, digest: str) -> str:
    return f"{stamp}-{digest}"


def write_bundle(portable: Path, manifest: dict, *, stamp: str | None = None) -> str:
    """Persist an immutable manifest revision and repoint ``latest.json``. Returns its ID."""
    payload = json_bytes(manifest)
    # Value-level check: an environment credential must never be serialised anywhere.
    text = payload.decode()
    if redact(text) != text:
        raise ValueError("Refusing to write a bundle manifest containing a credential")
    digest = hashlib.sha256(payload).hexdigest()
    name = bundle_id(stamp or utc_stamp(), digest)
    write_artifact(portable / "manifests" / f"{name}.json", payload)
    pointer = {"schema_version": SCHEMA_VERSION, "bundle_id": name, "manifest_sha256": digest,
               "kind": manifest["bundle"]["kind"],
               "for_stage": manifest["bundle"]["for_stage"],
               "highest_protected_stage": manifest["bundle"]["highest_protected_stage"],
               "created_at": manifest["bundle"]["created_at"]}
    atomic_bytes(portable / "latest.json", json_bytes(pointer), replace=True)
    return name


def resolve_bundle(portable: Path, selector: str) -> str:
    """``latest``, a full bundle ID, or a unique manifest-hash prefix."""
    manifests = portable / "manifests"
    if selector == "latest":
        pointer = json.loads((portable / "latest.json").read_bytes())
        return pointer["bundle_id"]
    if BUNDLE_NAME.match(selector) and (manifests / f"{selector}.json").is_file():
        return selector
    if not re.fullmatch(r"[0-9a-fA-F]{6,64}", selector):
        raise ValueError(f"Unknown bundle: {selector}")
    matches = sorted(p.stem for p in manifests.glob("*.json")
                     if BUNDLE_NAME.match(p.stem)
                     and BUNDLE_NAME.match(p.stem).group(2).startswith(selector.lower()))
    if len(matches) != 1:
        raise ValueError(f"Bundle selector matches {len(matches)} manifests: {selector}")
    return matches[0]


def load_bundle(portable: Path, selector: str) -> tuple[str, dict]:
    """Load a manifest, proving its bytes hash to the name and sidecar (tamper detection)."""
    name = resolve_bundle(portable, selector)
    match = BUNDLE_NAME.match(name)
    if not match:
        raise ValueError(f"Malformed bundle name: {name}")
    path = portable / "manifests" / f"{name}.json"
    if path.is_symlink():
        raise ValueError("Unsafe bundle manifest")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    if digest != match.group(2) or sidecar.read_text().strip() != digest:
        raise ValueError(f"Bundle manifest tampered or corrupt: {name}")
    manifest = json.loads(payload)
    if json_bytes(manifest) != payload:
        raise ValueError(f"Bundle manifest is not canonical JSON: {name}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported bundle schema")
    return name, manifest


def list_bundles(portable: Path) -> list[dict]:
    rows = []
    for path in sorted((portable / "manifests").glob("*.json")) if (
            portable / "manifests").is_dir() else []:
        if not BUNDLE_NAME.match(path.stem):
            continue
        try:
            name, manifest = load_bundle(portable, path.stem)
            rows.append({"bundle_id": name, "valid": True, **manifest["bundle"],
                         "entries": len(manifest["entries"]),
                         "logical_bytes": manifest["stats"]["logical_bytes"]})
        except (ValueError, OSError, KeyError) as exc:
            rows.append({"bundle_id": path.stem, "valid": False, "error": str(exc)})
    return rows
