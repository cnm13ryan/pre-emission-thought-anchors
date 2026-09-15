"""Leases at the real mutation points: concurrent processes, the model/head loaders, the
vLLM launcher, the portable-store writer and the unchanged MacBook wrapper."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from backup_fixtures import ROOT, make_world

from preanchor.backup import prepare as prepare_module
from preanchor.backup.legacy import import_legacy
from preanchor.config import load_config
from preanchor.control import TOKENS_ENV, control_root, leases, presented_tokens, sessions
from preanchor.control.leases import WRITER_LEASE, LeaseRequired
from preanchor.control.receipts import list_receipts
from preanchor.ids import file_hash

CLI = [sys.executable, "-m", "preanchor.cli"]


def cli_env(root: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k != TOKENS_ENV}
    env.update({"PREANCHOR_ROOT": str(root), "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(ROOT / "src")}, **extra)
    return env


def test_concurrent_acquisitions_have_exactly_one_winner(tmp_path):
    root = tmp_path / "root"
    control = control_root(root)
    env = cli_env(root)
    ids = []
    for i in range(4):
        out = subprocess.run([*CLI, "agent", "register", "--task", f"racer {i}", "--role",
                              "tests", "--worktree", str(ROOT), "--json"],
                             env=env, capture_output=True, text=True, check=True).stdout
        ids.append(json.loads(out)["session_id"])
    procs = [subprocess.Popen([*CLI, "lease", "acquire", "test:race", "--session", sid,
                               "--purpose", "race", "--json"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for sid in ids * 2]                      # 8 real processes, one resource
    results = [(p.wait(timeout=120), p.communicate()[0]) for p in procs]
    winners = [json.loads(out) for code, out in results if code == 0]
    losers = [json.loads(out) for code, out in results if code == 3]
    assert len(winners) == 1 and len(losers) == 7, results
    assert all(lost["error"] == "LeaseHeld" for lost in losers)
    # A loser racing the winner's owner.json write may report "incomplete"; nobody else.
    seen = {(lost["owner"]["owner"] or {}).get("session_id") for lost in losers}
    assert seen <= {winners[0]["session_id"], None}
    assert leases.describe(control, "test:race")["owner"]["session_id"] == winners[0]["session_id"]
    assert len(list_receipts(control, kind="lease_acquired")) == 1
    # Different resources in parallel all succeed.
    procs = [subprocess.Popen([*CLI, "lease", "acquire", f"test:r{i}", "--session", ids[i],
                               "--purpose", "p", "--json"], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for i in range(4)]
    assert [p.wait(timeout=120) for p in procs] == [0] * 4
    status = subprocess.run([*CLI, "agent", "status"], env=env, capture_output=True,
                            text=True, check=True).stdout
    assert "test:race" in status and all(sid in status for sid in ids)


def test_model_and_head_loaders_reject_before_loading_without_the_gpu_lease(tmp_path,
                                                                            monkeypatch):
    from preanchor.features.observables import load_head
    from preanchor.model.load import load_model

    monkeypatch.setenv("PREANCHOR_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv(TOKENS_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)     # import would fail loudly
    config = load_config(ROOT / "configs/mvp.yaml")
    with pytest.raises(LeaseRequired, match="gpu:h100 lease required"):
        load_model(config)
    with pytest.raises(LeaseRequired, match="gpu:h100"):
        load_head(config, "cuda:0")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(Exception) as excinfo:
        load_head(config, "cpu")                                 # CPU path is not gated
    assert not isinstance(excinfo.value, LeaseRequired)
    control = control_root()
    session = sessions.register(control, task="t", role="tests", worktree=None)
    token = leases.acquire(control, "gpu:h100", session_id=session["session_id"],
                           purpose="p")["owner_token"]
    monkeypatch.setenv(TOKENS_ENV, f"gpu:h100={token}")
    with pytest.raises(Exception) as excinfo:
        load_model(config)                     # passes the gate, fails at the import stub
    assert not isinstance(excinfo.value, LeaseRequired)


def test_start_vllm_refuses_without_both_leases_before_touching_docker(tmp_path):
    root = tmp_path / "root"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "sudo.log"
    (binaries / "sudo").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 1\n")
    (binaries / "sudo").chmod(0o755)
    env = cli_env(root, PATH=f"{binaries}:{os.environ['PATH']}",
                  PREANCHOR_CMD=" ".join(CLI), HF_HOME=str(tmp_path))

    def start(extra_env=None):
        return subprocess.run(["bash", str(ROOT / "scripts/start_vllm.sh"),
                               "configs/lambda/h100_pcie_80.yaml"],
                              env={**env, **(extra_env or {})}, cwd=ROOT,
                              capture_output=True, text=True, check=False)

    refused = start()
    assert refused.returncode == 1 and "leases are required" in refused.stderr
    assert not log.exists()
    control = control_root(root)
    session = sessions.register(control, task="vllm", role="tests", worktree=None)
    gpu = leases.acquire(control, "gpu:h100", session_id=session["session_id"], purpose="p")
    half = start({TOKENS_ENV: f"gpu:h100={gpu['owner_token']}"})
    assert half.returncode == 1 and "leases are required" in half.stderr and not log.exists()
    vllm = leases.acquire(control, "service:vllm", session_id=session["session_id"],
                          purpose="p")
    both = start({TOKENS_ENV: f"gpu:h100={gpu['owner_token']},service:vllm={vllm['owner_token']}"})
    assert "leases are required" not in both.stderr and both.returncode == 1
    assert log.exists() and "docker" in log.read_text()   # gate passed; stub docker failed


def test_portable_store_writer_lease_is_enforced_and_auto_taken_by_the_cli(tmp_path,
                                                                            monkeypatch):
    root = tmp_path / "lambda-nfs/preanchor"
    make_world(root, monkeypatch, units=6)
    control = control_root(root)
    held = presented_tokens()[WRITER_LEASE]
    monkeypatch.delenv(TOKENS_ENV)                 # drop the fixture's writer lease token
    with pytest.raises(LeaseRequired, match="portable-store:writer"):
        prepare_module.prepare(root, ROOT, for_stage=5)
    snapshot = tmp_path / "legacy"
    snapshot.mkdir()
    (snapshot / "a.txt").write_bytes(b"a")
    (snapshot / "checksums.json").write_text(json.dumps({"a.txt": file_hash(snapshot / "a.txt")}))
    with pytest.raises(LeaseRequired, match="portable-store:writer"):
        import_legacy(root, ROOT, snapshot)
    assert not (root / "artifacts/portable").exists()
    assert not (root / "artifacts/portable/latest.json").exists()
    # The fixture's session still holds the lease: the CLI (the wrapper's remote call)
    # must fail fast as a second writer and mutate nothing.
    env = cli_env(root)
    second = subprocess.run([*CLI, "backup", "prepare", "--for-stage", "5", "--root", str(root),
                             "--repo", str(ROOT), "--json"], env=env, capture_output=True,
                            text=True)
    assert second.returncode == 3 and json.loads(second.stdout)["error"] == "LeaseHeld"
    assert not (root / "artifacts/portable/latest.json").exists()
    holder = leases.describe(control, WRITER_LEASE)["owner"]
    leases.release(control, WRITER_LEASE, held)
    # Free store: the CLI registers a short-lived session, leases, prepares, releases.
    env["PREANCHOR_ROOT"] = str(root)
    first = subprocess.run([*CLI, "backup", "prepare", "--for-stage", "5", "--root", str(root),
                            "--repo", str(ROOT), "--json"], env=env, capture_output=True,
                           text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (root / "artifacts/portable/latest.json").is_file()
    assert leases.describe(control, WRITER_LEASE)["state"] == "free"
    writers = [r for r in sessions.status(control, include_closed=True)
               if r["role"] == "backup-writer"]
    # One short-lived session per CLI call (the refused one and the successful one), both
    # closed again; neither is the fixture's holder session.
    assert len(writers) == 2 and all(w["status"] == "closed" for w in writers)
    assert holder["session_id"] not in {w["session_id"] for w in writers}
    kinds = [r["kind"] for r in list_receipts(control, subject=WRITER_LEASE)]
    assert kinds[-2:] == ["lease_acquired", "lease_released"]
    # Read-only commands never need a lease.
    for command in (["backup", "plan", "--for-stage", "5"], ["backup", "list"],
                    ["backup", "verify"], ["restore", "plan"]):
        result = subprocess.run([*CLI, *command, "--root", str(root)], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, (command, result.stderr)
