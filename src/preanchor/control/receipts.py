"""Immutable, checksummed lifecycle receipts (append-only audit trail)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from preanchor.io import write_json
from preanchor.logging import redact
from preanchor.manifest import utc_now

KINDS = ("session_registered", "session_closed", "lease_acquired", "lease_released",
         "lease_reclaimed")


def write_receipt(control: Path, kind: str, subject: str, record: dict) -> Path:
    if kind not in KINDS:
        raise ValueError(f"Unknown receipt kind: {kind}")
    stamp = utc_now().replace("-", "").replace(":", "")[:15] + "Z"
    safe_subject = "".join(c if c.isalnum() or c in "._-" else "_" for c in subject)[:80]
    path = control / "receipts" / f"{stamp}-{kind}-{safe_subject}-{uuid.uuid4().hex[:8]}.json"
    payload = {"schema_version": 1, "kind": kind, "subject": subject,
               "created_at": utc_now(), "record": redact(record)}
    write_json(path, payload)   # tmp + rename + .sha256 sidecar; never replaces
    return path


def list_receipts(control: Path, *, kind: str | None = None,
                  subject: str | None = None) -> list[dict]:
    rows = []
    directory = control / "receipts"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            record = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if kind and record.get("kind") != kind:
            continue
        if subject and record.get("subject") != subject:
            continue
        rows.append({"path": path.name, **record})
    rows.sort(key=lambda r: (str(r.get("created_at", "")), r["path"]))
    return rows
