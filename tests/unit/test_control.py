"""Control plane v0.1: session registry, persistent leases, reclaim evidence, receipts."""

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from preanchor.control import (
    TOKENS_ENV,
    ControlError,
    control_root,
    leases,
    presented_tokens,
    safe_name,
    sessions,
)
from preanchor.control.leases import LeaseHeld, LeaseRequired
from preanchor.control.receipts import list_receipts
from preanchor.io import validate_artifact

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def control(tmp_path, monkeypatch):
    monkeypatch.delenv(TOKENS_ENV, raising=False)
    monkeypatch.setenv("PREANCHOR_ROOT", str(tmp_path / "root"))
    return control_root()


def session(control, role="tests", worktree=ROOT):
    return sessions.register(control, task=f"{role} task", role=role, worktree=worktree)


def backdate(path: Path, seconds: float) -> None:
    record = json.loads(path.read_bytes())
    stamp = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    for key in ("updated_at", "heartbeat"):
        if key in record:
            record[key] = stamp
    path.write_bytes(json.dumps(record).encode())


# ---- sessions --------------------------------------------------------------------------

def test_register_captures_git_facts_heartbeats_and_closes(control):
    record = session(control)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True)
    assert record["status"] == "active" and record["hostname"] and record["pid"] == os.getpid()
    assert record["worktree"] == str(ROOT) and record["current_commit"] == head.strip()
    assert record["base_commit"] == head.strip() and record["branch"]
    assert control == Path(os.environ["PREANCHOR_ROOT"]).resolve() / "control"
    later = sessions.heartbeat(control, record["session_id"], note="still here")
    assert later["heartbeat"] >= record["heartbeat"]
    closed = sessions.close(control, record["session_id"], outcome="done")
    assert closed["status"] == "closed" and closed["outcome"] == "done"
    with pytest.raises(ControlError, match="closed"):
        sessions.heartbeat(control, record["session_id"])
    kinds = [r["kind"] for r in list_receipts(control, subject=record["session_id"])]
    assert kinds == ["session_registered", "session_closed"]
    for receipt in (control / "receipts").glob("*.json"):
        validate_artifact(receipt)          # immutable + checksummed


def test_two_sessions_in_different_worktrees_are_independent(control, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "agent/x", str(other)], check=True)
    subprocess.run(["git", "-C", str(other), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    one, two = session(control, "implementation"), session(control, "tests", other)
    assert one["session_id"] != two["session_id"]
    assert {r["branch"] for r in (one, two)} >= {"agent/x"}
    assert two["worktree"] == str(other.resolve())
    sessions.close(control, one["session_id"])
    assert sessions.load_session(control, two["session_id"])["status"] == "active"
    rows = sessions.status(control)
    assert [r["session_id"] for r in rows] == [two["session_id"]]


def test_stale_heartbeat_is_reported_never_dead(control):
    record = session(control)
    backdate(control / "sessions" / record["session_id"] / "session.json", 1000)
    (row,) = sessions.status(control, stale_after=300)
    assert row["state"] == "stale" and row["status"] == "active"
    assert "stale" in sessions.render_status([row])
    assert sessions.load_session(control, record["session_id"])["status"] == "active"


# ---- leases ----------------------------------------------------------------------------

def test_acquire_release_roundtrip_and_token_proof(control):
    owner = session(control)
    lease = leases.acquire(control, "gpu:h100", session_id=owner["session_id"],
                           purpose="unit test")
    assert len(lease["owner_token"]) == 64
    stored = json.loads((control / "leases/gpu__h100/owner.json").read_bytes())
    assert "owner_token" not in stored and stored["session_id"] == owner["session_id"]
    assert stored["branch"] == owner["branch"] and stored["commit"] == owner["current_commit"]
    with pytest.raises(LeaseRequired, match="token does not match"):
        leases.release(control, "gpu:h100", "0" * 64)
    with pytest.raises(LeaseRequired):
        leases.heartbeat(control, "gpu:h100", "wrong")
    assert leases.heartbeat(control, "gpu:h100", lease["owner_token"])["state"] == "held"
    released = leases.release(control, "gpu:h100", lease["owner_token"], reason="done")
    assert leases.describe(control, "gpu:h100")["state"] == "free"
    assert Path(released["archived"]).is_dir() and Path(released["receipt"]).is_file()
    kinds = [r["kind"] for r in list_receipts(control, subject="gpu:h100")]
    assert kinds == ["lease_acquired", "lease_released"]
    assert all('"owner_token"' not in json.dumps(r) for r in list_receipts(control))


def test_second_acquirer_fails_with_owner_info_and_other_resources_are_independent(control):
    a, b = session(control, "a"), session(control, "b")
    leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="extract")
    with pytest.raises(LeaseHeld) as excinfo:
        leases.acquire(control, "gpu:h100", session_id=b["session_id"], purpose="vllm")
    text = str(excinfo.value)
    assert a["session_id"] in text and "purpose: extract" in text and "worktree:" in text
    assert excinfo.value.info["owner"]["session_id"] == a["session_id"]
    other = leases.acquire(control, "service:vllm", session_id=b["session_id"], purpose="x")
    assert other["resource"] == "service:vllm"
    assert {r["resource"] for r in leases.list_leases(control)} == {"gpu:h100", "service:vllm"}


def test_stale_lease_is_never_taken_over_implicitly(control):
    a, b = session(control, "a"), session(control, "b")
    leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    backdate(control / "leases/gpu__h100/heartbeat.json", 10_000)
    assert leases.describe(control, "gpu:h100")["state"] == "stale"
    with pytest.raises(LeaseHeld, match="never taken over implicitly"):
        leases.acquire(control, "gpu:h100", session_id=b["session_id"], purpose="p")
    assert leases.describe(control, "gpu:h100")["owner"]["session_id"] == a["session_id"]


def test_reclaim_rules(control):
    a, b = session(control, "a"), session(control, "b")
    leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    reclaim = lambda **kw: leases.reclaim(control, "gpu:h100", session_id=b["session_id"],  # noqa: E731
                                          reason=kw.pop("reason", "owner host checked"), **kw)
    owner_path = control / "leases/gpu__h100/owner.json"
    owner = json.loads(owner_path.read_bytes())
    gone = subprocess.Popen(["true"])
    gone.wait()
    dead = gone.pid                                # reaped: os.kill(pid, 0) fails
    owner_path.write_bytes(json.dumps({**owner, "pid": dead}).encode())
    with pytest.raises(ControlError, match="requires an explicit --reason"):
        reclaim(reason="  ")
    with pytest.raises(ControlError, match="heartbeat is fresh"):
        reclaim()
    with pytest.raises(ControlError, match="heartbeat is fresh"):
        reclaim(force=True)
    backdate(control / "leases/gpu__h100/heartbeat.json", 10_000)
    with pytest.raises(ControlError, match="nothing proves the holder is gone"):
        reclaim()                                  # stale alone is ambiguous
    owner_path.write_bytes(json.dumps({**owner, "pid": os.getpid()}).encode())
    with pytest.raises(ControlError, match="owner process is alive"):
        reclaim(force=True)                        # a live holder blocks even --force
    owner_path.write_bytes(json.dumps({**owner, "pid": dead}).encode())
    record = reclaim(force=True)
    assert record["ownership_transferred"] is False and record["force"] is True
    assert record["evidence"]["stale"] and record["evidence"]["owner_session_status"] == "active"
    assert leases.describe(control, "gpu:h100")["state"] == "free"
    receipt = list_receipts(control, kind="lease_reclaimed", subject="gpu:h100")
    assert len(receipt) == 1 and receipt[0]["record"]["reason"] == "owner host checked"
    assert receipt[0]["record"]["previous_owner"]["session_id"] == a["session_id"]
    validate_artifact(control / "receipts" / receipt[0]["path"])
    # Positive evidence (closed owner session) needs no force even without staleness.
    leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    owner_path.write_bytes(json.dumps({**json.loads(owner_path.read_bytes()),
                                       "pid": dead}).encode())
    sessions.close(control, a["session_id"])
    assert reclaim()["evidence"]["owner_session_status"] == "closed"


def test_interrupted_acquisition_leaves_one_incomplete_owner(control):
    a, b = session(control, "a"), session(control, "b")

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(leases, "atomic_bytes", crash)
        with pytest.raises(KeyboardInterrupt):
            leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    info = leases.describe(control, "gpu:h100")
    assert info["state"] == "incomplete" and info["owner"] is None
    with pytest.raises(LeaseHeld, match="incomplete"):
        leases.acquire(control, "gpu:h100", session_id=b["session_id"], purpose="p")
    with pytest.raises(LeaseRequired):
        leases.require(["gpu:h100"], tokens={"gpu:h100": "x"})
    with pytest.raises(ControlError, match="heartbeat is fresh"):
        leases.reclaim(control, "gpu:h100", session_id=b["session_id"], reason="crashed")
    old = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(control / "leases/gpu__h100", (old, old))
    leases.reclaim(control, "gpu:h100", session_id=b["session_id"], reason="crashed")
    leases.acquire(control, "gpu:h100", session_id=b["session_id"], purpose="p")


@pytest.mark.parametrize("name", ["", "../x", "gpu/h100", "a:", ":a", "gpu:..", "x" * 70,
                                  "gpu__h100",
                                  "gpu h100", "/abs", "a:b:c:d:e"])
def test_malformed_resource_names_fail_closed(control, name):
    with pytest.raises(ControlError, match="Malformed"):
        safe_name(name)
    with pytest.raises(ControlError):
        leases.acquire(control, name, session_id="s", purpose="p")
    assert not (control / "leases").exists() or not list((control / "leases").iterdir())


def test_symlinked_lease_paths_fail_closed(control, tmp_path):
    a = session(control)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (control / "leases").mkdir(parents=True)
    (control / "leases" / "gpu__h100").symlink_to(elsewhere)
    with pytest.raises(ControlError, match="symlink"):
        leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    with pytest.raises(ControlError, match="symlink"):
        leases.describe(control, "gpu:h100")
    (control / "leases" / "gpu__h100").unlink()
    (control / "leases").rmdir()
    (control / "leases").symlink_to(elsewhere)
    with pytest.raises(ControlError, match="symlink"):
        leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")


def test_require_reads_presented_tokens(control, monkeypatch):
    a = session(control)
    with pytest.raises(LeaseRequired, match="gpu:h100 lease required"):
        leases.require(["gpu:h100"])
    lease = leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    monkeypatch.setenv(TOKENS_ENV, f"service:vllm=abc,gpu:h100={lease['owner_token']}")
    assert presented_tokens() == {"service:vllm": "abc", "gpu:h100": lease["owner_token"]}
    assert leases.require(["gpu:h100"])["gpu:h100"]["session_id"] == a["session_id"]
    with pytest.raises(LeaseRequired):
        leases.require(["gpu:h100", "service:vllm"])   # vllm token presented but not held
    monkeypatch.setenv(TOKENS_ENV, "gpu:h100=")
    with pytest.raises(ControlError, match="Malformed"):
        presented_tokens()


def test_status_reports_active_and_stale_ownership(control):
    a = session(control)
    leases.acquire(control, "gpu:h100", session_id=a["session_id"], purpose="p")
    leases.acquire(control, "service:vllm", session_id=a["session_id"], purpose="q")
    backdate(control / "leases/service__vllm/heartbeat.json", 10_000)
    states = {r["resource"]: r["state"] for r in leases.list_leases(control)}
    assert states == {"gpu:h100": "held", "service:vllm": "stale"}
    assert "heartbeat age" in leases.render_owner(leases.describe(control, "service:vllm"))
