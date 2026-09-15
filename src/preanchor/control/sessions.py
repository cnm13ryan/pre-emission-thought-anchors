"""Agent session registry: who is working where, with a replaceable heartbeat."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from preanchor.control import (
    STALE_AFTER_SECONDS,
    ControlError,
    git_facts,
    hostname,
    refuse_symlinks,
    safe_name,
)
from preanchor.control.receipts import write_receipt
from preanchor.io import atomic_bytes, json_bytes
from preanchor.manifest import utc_now

ACTIVE, CLOSED = "active", "closed"


def session_dir(control: Path, session_id: str) -> Path:
    path = control / "sessions" / safe_name(session_id)
    refuse_symlinks(path, control)
    return path


def _write(path: Path, value: dict) -> None:
    atomic_bytes(path, json_bytes(value), replace=True)


def load_session(control: Path, session_id: str) -> dict:
    path = session_dir(control, session_id) / "session.json"
    if not path.is_file():
        raise ControlError(f"Unknown session: {session_id}")
    return json.loads(path.read_bytes())


def register(control: Path, *, task: str, role: str, worktree: Path | None = None,
             allowed_write_roots: list[str] | None = None,
             forbidden_write_roots: list[str] | None = None,
             base_commit: str | None = None) -> dict:
    facts = git_facts(worktree)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{stamp}-{safe_name(role)}-{uuid.uuid4().hex[:8]}"
    record = {"schema_version": 1, "session_id": session_id, "task": task, "role": role,
              "hostname": hostname(), "pid": os.getpid(), "worktree": facts["worktree"],
              "branch": facts["branch"], "base_commit": base_commit or facts["commit"],
              "current_commit": facts["commit"],
              "allowed_write_roots": list(allowed_write_roots or []),
              "forbidden_write_roots": list(forbidden_write_roots or []),
              "status": ACTIVE, "created_at": utc_now(), "heartbeat": utc_now(),
              "closed_at": None, "outcome": None}
    directory = session_dir(control, session_id)
    directory.mkdir(parents=True, exist_ok=False)   # ids carry a uuid: never collide
    _write(directory / "session.json", record)
    _write(directory / "heartbeat.json", {"schema_version": 1, "updated_at": record["heartbeat"],
                                          "note": "registered"})
    write_receipt(control, "session_registered", session_id, record)
    return record


def heartbeat(control: Path, session_id: str, *, note: str | None = None) -> dict:
    record = load_session(control, session_id)
    if record["status"] != ACTIVE:
        raise ControlError(f"Session {session_id} is {record['status']}")
    facts = git_facts(Path(record["worktree"]) if record["worktree"] else None)
    record["heartbeat"] = utc_now()
    record["current_commit"] = facts["commit"] or record["current_commit"]
    record["branch"] = facts["branch"] or record["branch"]
    directory = session_dir(control, session_id)
    _write(directory / "session.json", record)
    _write(directory / "heartbeat.json", {"schema_version": 1, "updated_at": record["heartbeat"],
                                          "note": note})
    return record


def close(control: Path, session_id: str, *, outcome: str = "closed") -> dict:
    record = load_session(control, session_id)
    if record["status"] == CLOSED:
        raise ControlError(f"Session {session_id} is already closed")
    record.update(status=CLOSED, closed_at=utc_now(), outcome=outcome)
    _write(session_dir(control, session_id) / "session.json", record)
    write_receipt(control, "session_closed", session_id, record)
    return record


def heartbeat_age(record: dict, now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    return (now - datetime.fromisoformat(record["heartbeat"])).total_seconds()


def status(control: Path, *, stale_after: float = STALE_AFTER_SECONDS,
           include_closed: bool = False) -> list[dict]:
    """Operator view. ``stale`` is a diagnosis trigger, never a liveness verdict."""
    rows = []
    directory = control / "sessions"
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.is_symlink() or not (path / "session.json").is_file():
            continue
        try:
            record = json.loads((path / "session.json").read_bytes())
        except (OSError, ValueError):
            continue
        if record.get("status") == CLOSED and not include_closed:
            continue
        age = heartbeat_age(record)
        state = record["status"]
        if state == ACTIVE and age > stale_after:
            state = "stale"
        rows.append({**record, "heartbeat_age_seconds": round(age, 1), "state": state})
    return rows


def render_status(rows: list[dict]) -> str:
    if not rows:
        return "No registered sessions."
    lines = []
    for row in rows:
        lines.append(f"{row['session_id']}  [{row['state']}]  role={row['role']}  "
                     f"heartbeat {row['heartbeat_age_seconds']:.0f}s ago")
        lines.append(f"    task:     {row['task']}")
        lines.append(f"    branch:   {row['branch']}  @ {(row['current_commit'] or '')[:12]}")
        lines.append(f"    worktree: {row['worktree']}  ({row['hostname']} pid {row['pid']})")
    return "\n".join(lines)
