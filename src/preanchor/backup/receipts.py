"""Append-only, checksummed receipts for prepare, pull-verification, restore and verify."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from preanchor.io import write_json
from preanchor.logging import redact

KINDS = ("prepare", "pull_verification", "restore", "restore_verification", "legacy_import")


def write_receipt(portable: Path, kind: str, record: dict) -> Path:
    if kind not in KINDS:
        raise ValueError(f"Unknown receipt kind: {kind}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = str(record.get("manifest_sha256", ""))[:12] or "none"
    path = portable / "receipts" / f"{stamp}-{kind}-{digest}-{uuid.uuid4().hex[:8]}.json"
    payload = redact({"schema_version": 1, "kind": kind,
                      "created_at": datetime.now(UTC).isoformat(), **record})
    write_json(path, payload)
    return path


def list_receipts(portable: Path, *, manifest_sha256: str | None = None) -> list[dict]:
    rows = []
    directory = portable / "receipts"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if manifest_sha256 and record.get("manifest_sha256") != manifest_sha256:
            continue
        rows.append({"path": path.name, **record})
    rows.sort(key=lambda r: (str(r.get("created_at", "")), r["path"]))
    return rows
