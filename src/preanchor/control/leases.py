"""Persistent single-owner leases on named shared resources.

Layout: ``control/leases/<safe-resource>/{owner.json,heartbeat.json}``. Ownership is the
existence of that directory, created with one ``mkdir`` (atomic on the project's
virtiofs/NFS persistent root: exactly one of two concurrent creators succeeds). Release
and reclaim move the whole directory into ``control/leases/.archive/`` with one
``rename``, so ownership disappears atomically and the evidence is kept. ``owner.json``
stores only the SHA-256 of the owner token; the plaintext is returned once to the
acquirer and is the only proof accepted for heartbeat and release.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from preanchor.control import (
    STALE_AFTER_SECONDS,
    ControlError,
    control_root,
    git_facts,
    hostname,
    presented_tokens,
    process_alive,
    refuse_symlinks,
    safe_name,
)
from preanchor.control.receipts import write_receipt
from preanchor.control.sessions import ACTIVE, CLOSED, heartbeat_age, load_session
from preanchor.io import atomic_bytes, json_bytes
from preanchor.manifest import utc_now

GPU_LEASE, VLLM_LEASE, WRITER_LEASE = "gpu:h100", "service:vllm", "portable-store:writer"
KNOWN_RESOURCES = (GPU_LEASE, VLLM_LEASE, WRITER_LEASE,
                   "canonical-publish:stageN", "stage-finalize:stageN")


class LeaseHeld(ControlError):
    """Acquisition refused; ``info`` describes the current owner."""

    def __init__(self, info: dict):
        self.info = info
        super().__init__("Lease held:\n" + render_owner(info))


class LeaseRequired(ControlError):
    """An enforced operation ran without valid proof of the required lease(s)."""


def lease_dir(control: Path, resource: str) -> Path:
    path = control / "leases" / safe_name(resource)
    refuse_symlinks(path, control)
    return path


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _read(path: Path) -> dict | None:
    try:
        if path.is_symlink():
            raise ControlError(f"Unsafe symlinked control file: {path}")
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def describe(control: Path, resource: str, *, stale_after: float = STALE_AFTER_SECONDS) -> dict:
    """Current ownership view; ``state`` is held | stale | incomplete | free."""
    directory = lease_dir(control, resource)
    info = {"resource": resource, "state": "free", "owner": None, "heartbeat_at": None,
            "heartbeat_age_seconds": None, "path": str(directory)}
    if not directory.is_dir():
        return info
    owner = _read(directory / "owner.json")
    # A concurrent winner writes owner.json milliseconds after its mkdir; give a very
    # young directory a moment before calling it an interrupted acquisition.
    for _ in range(20):
        if owner is not None or time.time() - directory.stat().st_mtime > 2:
            break
        time.sleep(0.05)
        owner = _read(directory / "owner.json")
    pulse = _read(directory / "heartbeat.json")
    now = datetime.now(UTC)
    if owner is None:
        info["state"] = "incomplete"    # mkdir succeeded, owner.json never landed
        age = now.timestamp() - directory.stat().st_mtime
        info["heartbeat_age_seconds"] = round(age, 1)
        return info
    info["owner"] = owner
    stamp = (pulse or {}).get("updated_at") or owner["acquired_at"]
    age = (now - datetime.fromisoformat(stamp)).total_seconds()
    info.update(heartbeat_at=stamp, heartbeat_age_seconds=round(age, 1),
                state="stale" if age > stale_after else "held")
    return info


def render_owner(info: dict) -> str:
    owner = info.get("owner") or {}
    lines = [f"resource: {info['resource']}", f"state: {info['state']}"]
    if owner:
        lines += [f"owner session: {owner.get('session_id')}",
                  f"branch: {owner.get('branch')}", f"worktree: {owner.get('worktree')}",
                  f"host/pid: {owner.get('hostname')}/{owner.get('pid')}",
                  f"purpose: {owner.get('purpose')}", f"acquired: {owner.get('acquired_at')}"]
    if info.get("heartbeat_age_seconds") is not None:
        lines.append(f"heartbeat age: {info['heartbeat_age_seconds']:.0f}s")
    if info["state"] == "stale":
        lines.append("stale leases are never taken over implicitly: "
                     "`preanchor lease reclaim` with a reason")
    return "\n".join(lines)


def acquire(control: Path, resource: str, *, session_id: str, purpose: str) -> dict:
    session = load_session(control, session_id)
    if session["status"] != ACTIVE:
        raise ControlError(f"Session {session_id} is {session['status']}")
    directory = lease_dir(control, resource)
    directory.parent.mkdir(parents=True, exist_ok=True)
    refuse_symlinks(directory.parent, control)
    try:
        os.mkdir(directory)          # the one atomic ownership primitive
    except FileExistsError:
        raise LeaseHeld(describe(control, resource)) from None
    token = secrets.token_hex(32)
    facts = git_facts(Path(session["worktree"]) if session["worktree"] else None)
    owner = {"schema_version": 1, "resource": resource, "session_id": session_id,
             "owner_token_sha256": _token_digest(token), "hostname": hostname(),
             "pid": os.getpid(), "worktree": session["worktree"],
             "branch": facts["branch"] or session["branch"],
             "commit": facts["commit"] or session["current_commit"], "purpose": purpose,
             "acquired_at": utc_now(), "lease_id": uuid.uuid4().hex}
    atomic_bytes(directory / "owner.json", json_bytes(owner))
    atomic_bytes(directory / "heartbeat.json",
                 json_bytes({"schema_version": 1, "updated_at": owner["acquired_at"]}))
    write_receipt(control, "lease_acquired", resource, owner)
    return {**owner, "owner_token": token}


def _verify(control: Path, resource: str, token: str) -> tuple[Path, dict]:
    directory = lease_dir(control, resource)
    owner = _read(directory / "owner.json") if directory.is_dir() else None
    if owner is None:
        raise LeaseRequired(f"No lease is held on {resource}")
    if not secrets.compare_digest(owner["owner_token_sha256"], _token_digest(str(token))):
        raise LeaseRequired(f"Owner token does not match the holder of {resource}\n"
                            + render_owner(describe(control, resource)))
    return directory, owner


def heartbeat(control: Path, resource: str, token: str) -> dict:
    directory, owner = _verify(control, resource, token)
    atomic_bytes(directory / "heartbeat.json",
                 json_bytes({"schema_version": 1, "updated_at": utc_now()}), replace=True)
    return describe(control, resource)


def _archive(control: Path, directory: Path, suffix: str) -> Path:
    archive = control / "leases" / ".archive"
    archive.mkdir(exist_ok=True)
    refuse_symlinks(archive, control)
    target = archive / f"{directory.name}.{suffix}.{uuid.uuid4().hex[:8]}"
    os.rename(directory, target)   # one atomic operation ends the ownership
    return target


def release(control: Path, resource: str, token: str, *, reason: str | None = None) -> dict:
    directory, owner = _verify(control, resource, token)
    record = {**owner, "reason": reason, "released_at": utc_now()}
    receipt = write_receipt(control, "lease_released", resource, record)
    archived = _archive(control, directory, "released")
    return {**record, "receipt": str(receipt), "archived": str(archived)}


def reclaim_evidence(control: Path, resource: str, *,
                     stale_after: float = STALE_AFTER_SECONDS) -> dict:
    """Safely available evidence about the holder. Heartbeat age alone proves nothing."""
    info = describe(control, resource, stale_after=stale_after)
    owner = info["owner"]
    evidence = {"state": info["state"], "heartbeat_age_seconds": info["heartbeat_age_seconds"],
                "stale": info["state"] in {"stale", "incomplete"}
                and (info["heartbeat_age_seconds"] or 0) > stale_after,
                "same_host": None, "owner_pid_alive": None, "owner_session_status": None,
                "owner_session_heartbeat_age_seconds": None, "checked_from_host": hostname()}
    if owner:
        evidence["same_host"] = owner.get("hostname") == hostname()
        if evidence["same_host"] and isinstance(owner.get("pid"), int):
            # Advisory only: the acquiring pid is usually a finished CLI call. A *live*
            # pid blocks reclaim; a dead one proves nothing about the owning session.
            evidence["owner_pid_alive"] = process_alive(owner["pid"])
        try:
            session = load_session(control, owner["session_id"])
            evidence["owner_session_status"] = session["status"]
            evidence["owner_session_heartbeat_age_seconds"] = round(
                heartbeat_age(session), 1)
        except (ControlError, KeyError, ValueError):
            evidence["owner_session_status"] = "unknown"
    return {"info": info, "evidence": evidence}


def reclaim(control: Path, resource: str, *, session_id: str, reason: str,
            force: bool = False, stale_after: float = STALE_AFTER_SECONDS) -> dict:
    """End a lease the caller does not own. Never transfers ownership: the reclaimer
    must ``acquire`` afterwards like anyone else.

    Allowed without ``force`` only on positive evidence that the holder is gone: the
    owner session is closed, or the directory is an incomplete acquisition older than
    ``stale_after``. A fresh lease heartbeat, or a live owner process on this host, is
    always refused. Anything else (a stale heartbeat alone, whatever the host) is
    ambiguous: it needs ``force`` plus the reason, and the operator's own check of the
    owner host is what the reason must describe. Process death is never inferred from
    heartbeat age.
    """
    if not reason or not reason.strip():
        raise ControlError("Reclaim requires an explicit --reason")
    reclaimer = load_session(control, session_id)
    if reclaimer["status"] != ACTIVE:
        raise ControlError(f"Session {session_id} is {reclaimer['status']}")
    directory = lease_dir(control, resource)
    if not directory.is_dir():
        raise ControlError(f"No lease is held on {resource}")
    facts = reclaim_evidence(control, resource, stale_after=stale_after)
    evidence, info = facts["evidence"], facts["info"]
    if evidence["owner_pid_alive"] is True:
        raise ControlError("Refusing reclaim: the owner process is alive on this host; "
                           "it must release the lease itself\n" + render_owner(info))
    positive = (evidence["owner_session_status"] == CLOSED
                or (info["state"] == "incomplete" and evidence["stale"]))
    if not positive:
        if not evidence["stale"]:
            raise ControlError("Refusing reclaim: the holder's heartbeat is fresh\n"
                               + render_owner(info))
        if not force:
            raise ControlError("Refusing reclaim: the lease is stale but nothing proves the "
                               "holder is gone (heartbeat age is not proof of death); "
                               "check the owner host, then pass --force with the reason\n"
                               + render_owner(info))
    record = {"resource": resource, "previous_owner": info["owner"], "evidence": evidence,
              "reason": reason, "force": force, "reclaimed_by_session": session_id,
              "reclaimed_at": utc_now(), "ownership_transferred": False}
    receipt = write_receipt(control, "lease_reclaimed", resource, record)
    archived = _archive(control, directory, "reclaimed")
    return {**record, "receipt": str(receipt), "archived": str(archived)}


def list_leases(control: Path, *, stale_after: float = STALE_AFTER_SECONDS) -> list[dict]:
    rows = []
    directory = control / "leases"
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir():
            continue
        rows.append(describe(control, path.name.replace("__", ":"), stale_after=stale_after))
    return rows


def check(control: Path, resource: str, token: str) -> dict:
    """Proof check used by enforcement points: returns the owner record or raises."""
    return _verify(control, resource, token)[1]


def require(resources: list[str] | tuple[str, ...], *, root: Path | None = None,
            tokens: dict[str, str] | None = None) -> dict[str, dict]:
    """Fail closed unless every resource is leased and its token is presented.

    Tokens come from ``tokens`` or ``PREANCHOR_LEASE_TOKENS``. This is the single gate
    that real GPU, vLLM and portable-store mutation paths call before doing anything.
    """
    control = control_root(root)
    presented = presented_tokens() if tokens is None else tokens
    owners = {}
    for resource in resources:
        token = presented.get(resource)
        if not token:
            raise LeaseRequired(
                f"{resource} lease required: acquire it with `preanchor lease acquire "
                f"{resource} --session <id> --purpose ...` and export its token in "
                f"PREANCHOR_LEASE_TOKENS\n" + render_owner(describe(control, resource)))
        owners[resource] = check(control, resource, token)
    return owners
