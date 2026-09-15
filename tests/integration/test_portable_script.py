"""MacBook wrapper on fake ssh/rsync: incremental transfer, independent rehash, receipts,
macOS-compatible rsync arguments, and restore onto a second root through the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from backup_fixtures import ROOT, make_world

SCRIPT = ROOT / "scripts/preanchor_portable.sh"
ALLOWED_RSYNC_FLAGS = {"-a", "--ignore-existing", "--"}


def write_fakes(binaries: Path, log: Path) -> None:
    """ssh runs the remote command locally; rsync copies files, honouring the flag subset
    that macOS rsync 2.6.9 supports, and logs every argument vector."""
    binaries.mkdir()
    ssh = binaries / "ssh"
    ssh.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys\n"
        f"open({str(log)!r}, 'a').write('ssh ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "host, command = sys.argv[1], sys.argv[2]\n"
        "assert host == 'lambda-alias', host\n"
        "sys.exit(subprocess.call(['bash', '-c', command]))\n")
    ssh.chmod(0o755)
    uv = binaries / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys\n"
        "# 'uv run --frozen preanchor <args>' -> the CLI against the fake remote root\n"
        "args = sys.argv[4:]\n"
        "args += ['--root', os.environ['FAKE_REMOTE_ROOT'], '--repo', os.environ['FAKE_REPO']]\n"
        "if args[0] in ('agent', 'lease') "
        "or args[0] in ('backup', 'restore') and args[1] in ('list', 'verify', 'store-path') "
        "or args[0] == 'restore' and args[1] == 'plan':\n"
        "    args = [a for i, a in enumerate(args) if not (a == '--repo' or "
        "(i > 0 and args[i-1] == '--repo'))]\n"
        f"sys.exit(subprocess.call([{sys.executable!r}, '-m', 'preanchor.cli'] + args))\n")
    uv.chmod(0o755)
    rsync = binaries / "rsync"
    rsync.write_text(
        "#!/usr/bin/env python3\n"
        "import os, shutil, sys\n"
        f"open({str(log)!r}, 'a').write('rsync ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "args = sys.argv[1:]\n"
        "ignore_existing = '--ignore-existing' in args\n"
        "files_from = next((a.split('=', 1)[1] for a in args\n"
        "                   if a.startswith('--files-from=')), None)\n"
        "positional = [a for a in args if not a.startswith('-')]\n"
        "sources, dest = positional[:-1], positional[-1]\n"
        "strip = lambda p: p.split(':', 1)[1] if ':' in p else p\n"
        "dest = strip(dest)\n"
        "def copy(src, dst):\n"
        "    if ignore_existing and os.path.exists(dst):\n"
        "        return\n"
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "    shutil.copyfile(src, dst)\n"
        "if files_from:\n"
        "    base = strip(sources[0])\n"
        "    for line in open(files_from):\n"
        "        rel = line.strip()\n"
        "        if rel:\n"
        "            copy(os.path.join(base, rel), os.path.join(dest, rel))\n"
        "else:\n"
        "    for src in sources:\n"
        "        src = strip(src)\n"
        "        if dest.endswith('/') or os.path.isdir(dest):\n"
        "            copy(src, os.path.join(dest, os.path.basename(src)))\n"
        "        else:\n"
        "            copy(src, dest)\n")
    rsync.chmod(0o755)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    remote_root = tmp_path / "lambda-nfs/preanchor"
    run, specs = make_world(remote_root, monkeypatch, units=6)
    remote_repo = tmp_path / "remote-repo"
    remote_repo.mkdir()
    (remote_repo / ".env").write_text("PREANCHOR_HOURLY_RATE=1\n")
    log = tmp_path / "calls.log"
    binaries = tmp_path / "bin"
    write_fakes(binaries, log)
    backup_dir = tmp_path / "macbook/preanchor-backups/portable"
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
           "PYTHONPATH": str(ROOT / "src"), "FAKE_REMOTE_ROOT": str(remote_root),
           "FAKE_REPO": str(ROOT), "PREANCHOR_BACKUP_DIR": str(backup_dir),
           "PREANCHOR_REMOTE_REPO": str(remote_repo),
           "PREANCHOR_REMOTE_CMD": "uv run --frozen preanchor"}
    return {"root": remote_root, "run": run, "specs": specs, "env": env, "log": log,
            "backup_dir": backup_dir, "tmp": tmp_path}


def run_script(env, *args, timeout=300):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                          text=True, timeout=timeout)


def rsync_calls(log: Path) -> list[list[str]]:
    return [line.split()[1:] for line in log.read_text().splitlines()
            if line.startswith("rsync ")]


def test_backup_transfers_incrementally_verifies_independently_and_receipts(harness):
    env, backup_dir, log = harness["env"], harness["backup_dir"], harness["log"]
    dry = run_script(env, "backup", "lambda-alias", "--for-stage", "5", "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert "BACKUP VERIFIED" not in dry.stdout and "dry run" in dry.stdout
    assert not list(backup_dir.glob("manifests/*.json"))
    first = run_script(env, "backup", "lambda-alias", "--for-stage", "5")
    assert first.returncode == 0, first.stderr + first.stdout
    assert first.stdout.rstrip().splitlines()[-2] == "BACKUP VERIFIED"
    manifests = list(backup_dir.glob("manifests/*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_bytes())
    objects = {e["sha256"] for e in manifest["entries"]}
    local_objects = {p.name for p in backup_dir.glob("objects/sha256/*/*/*")}
    assert objects == local_objects
    receipts = [json.loads(p.read_bytes()) for p in backup_dir.glob("receipts/*.json")]
    assert len(receipts) == 1 and receipts[0]["kind"] == "pull_verification"
    assert receipts[0]["outcome"] == "secondary_copy_verified"
    assert receipts[0]["object_count"] == len(objects)
    assert receipts[0]["manifest_sha256"] == manifests[0].stem.rsplit("-", 1)[1]
    # The receipt was copied back to Lambda by the Mac-initiated connection.
    remote_receipts = harness["root"] / "artifacts/portable/receipts"
    assert any(json.loads(p.read_bytes())["kind"] == "pull_verification"
               for p in remote_receipts.glob("*.json"))
    for call in rsync_calls(log):
        flags = {a.split("=", 1)[0] for a in call if a.startswith("-")}
        assert flags <= ALLOWED_RSYNC_FLAGS | {"--files-from"}, call
        assert "--info" not in " ".join(call) and "--mkpath" not in call
    for line in log.read_text().splitlines():
        if line.startswith("ssh"):
            assert "lambda-alias" in line
    # Second backup: nothing new on Lambda, zero objects transferred, still verified.
    log.write_text("")
    second = run_script(env, "backup", "lambda-alias", "--for-stage", "5")
    assert second.returncode == 0, second.stderr + second.stdout
    assert "transferred 0 new objects" in second.stdout
    assert "BACKUP VERIFIED" in second.stdout
    assert len(list(backup_dir.glob("manifests/*.json"))) == 2
    listing = run_script(env, "list")
    assert listing.returncode == 0 and "verified secondary copies" in listing.stdout
    assert listing.stdout.count("verified_at") == 2


def test_verify_fails_on_corrupted_local_object_and_never_prints_verified(harness):
    env, backup_dir = harness["env"], harness["backup_dir"]
    assert run_script(env, "backup", "lambda-alias", "--for-stage", "5").returncode == 0
    victim = next(backup_dir.glob("objects/sha256/*/*/*"))
    victim.write_bytes(b"corrupted")
    result = run_script(env, "verify")
    assert result.returncode != 0
    assert "BACKUP VERIFIED" not in result.stdout
    assert "verification FAILED" in result.stderr + result.stdout
    # Corruption is also caught during a fresh pull: --ignore-existing keeps the bad file.
    pulled = run_script(env, "backup", "lambda-alias", "--for-stage", "5")
    assert pulled.returncode != 0 and "BACKUP VERIFIED" not in pulled.stdout


def test_restore_pushes_bundle_and_verifies_on_a_new_root(harness):
    env = harness["env"]
    assert run_script(env, "backup", "lambda-alias", "--for-stage", "5").returncode == 0
    new_root = harness["tmp"] / "new-region/mount-1234/preanchor"
    new_root.mkdir(parents=True)
    env = {**env, "FAKE_REMOTE_ROOT": str(new_root)}
    dry = run_script(env, "restore", "lambda-alias", "--for-stage", "5", "--dry-run")
    assert dry.returncode == 0, dry.stderr + dry.stdout
    assert not (new_root / "data").exists()
    result = run_script(env, "restore", "lambda-alias", "--for-stage", "5")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "RESTORE VERIFIED" in result.stdout
    assert "Scientifically equivalent portable state: PASS" in result.stdout
    run = harness["run"]
    assert (new_root / "artifacts/runs" / run.name / "manifest.json").is_file()
    assert (new_root / "data/processed/stage3/trajectories.parquet").read_bytes() == (
        harness["root"] / "data/processed/stage3/trajectories.parquet").read_bytes()
    receipts = [json.loads(p.read_bytes())
                for p in (new_root / "artifacts/portable/receipts").glob("*.json")]
    assert {r["kind"] for r in receipts} == {"restore", "restore_verification"}
    assert all(r["destination"] == str(new_root) for r in receipts)
    # The push held portable-store:writer on the destination: acquired before the first
    # rsync towards it, released after the remote verification, session closed, lease free.
    from preanchor.control import control_root, leases, sessions
    from preanchor.control.receipts import list_receipts

    lines = harness["log"].read_text().splitlines()   # dry run, then the real restore
    store = str(new_root / "artifacts/portable")
    acquire = max(i for i, line in enumerate(lines)
                  if "lease acquire portable-store:writer" in line)
    release = max(i for i, line in enumerate(lines)
                  if "lease release portable-store:writer" in line)
    pushes = [i for i, line in enumerate(lines) if i > acquire and line.startswith("rsync ")
              and (line.rstrip("/").endswith(store) or f"{store}/manifests" in line)]
    assert len(pushes) == 2 and max(pushes) < release
    control = control_root(new_root)
    assert leases.describe(control, "portable-store:writer")["state"] == "free"
    kinds = [r["kind"] for r in list_receipts(control, subject="portable-store:writer")]
    assert kinds == ["lease_acquired", "lease_released"] * 2      # dry run + real restore
    pushers = [s for s in sessions.status(control, include_closed=True)
               if s["role"] == "backup-restore"]
    assert len(pushers) == 2 and all(s["status"] == "closed" for s in pushers)


def test_restore_refuses_to_push_while_another_writer_holds_the_store(harness):
    from preanchor.control import control_root, leases, sessions

    env = harness["env"]
    assert run_script(env, "backup", "lambda-alias", "--for-stage", "5").returncode == 0
    new_root = harness["tmp"] / "new-region/mount-1234/preanchor"
    new_root.mkdir(parents=True)
    control = control_root(new_root)
    holder = sessions.register(control, task="concurrent backup", role="tests", worktree=None)
    held = leases.acquire(control, "portable-store:writer", session_id=holder["session_id"],
                          purpose="p")
    env = {**env, "FAKE_REMOTE_ROOT": str(new_root)}
    result = run_script(env, "restore", "lambda-alias", "--for-stage", "5")
    assert result.returncode != 0 and "held by another writer" in result.stderr
    assert "RESTORE VERIFIED" not in result.stdout
    assert not (new_root / "artifacts/portable/objects").exists()
    assert not (new_root / "artifacts/portable/manifests").exists()
    assert not (new_root / "data").exists()
    # The holder keeps its lease; the wrapper's own session is closed again.
    owner = leases.describe(control, "portable-store:writer")["owner"]
    assert owner["session_id"] == holder["session_id"]
    pushers = [s for s in sessions.status(control, include_closed=True)
               if s["role"] == "backup-restore"]
    assert len(pushers) == 1 and pushers[0]["status"] == "closed"
    leases.release(control, "portable-store:writer", held["owner_token"])
    result = run_script(env, "restore", "lambda-alias", "--for-stage", "5")
    assert result.returncode == 0 and "RESTORE VERIFIED" in result.stdout


def test_script_rejects_unsafe_hosts(harness):
    result = run_script(harness["env"], "backup", "evil;rm -rf /", "--for-stage", "5")
    assert result.returncode != 0 and "plain SSH host alias" in result.stderr
