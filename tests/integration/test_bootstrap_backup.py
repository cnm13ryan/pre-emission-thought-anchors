import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corrupt", [False, True])
def test_mac_pull_checks_bytes_before_issuing_receipt(tmp_path, corrupt):
    fixture = tmp_path / "source"
    fixture.mkdir()
    data = b"committed-source-fixture"
    (fixture / "preanchor.bundle").write_bytes(b"tampered" if corrupt else data)
    (fixture / "checksums.json").write_text(json.dumps({
        "preanchor.bundle": hashlib.sha256(data).hexdigest()}))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    rsync = binaries / "rsync"
    rsync.write_text("#!/usr/bin/env python3\nimport os, shutil, sys\n"
                     "shutil.copytree(os.environ['BACKUP_FIXTURE'], sys.argv[-1], "
                     "dirs_exist_ok=True)\n")
    rsync.chmod(0o755)
    destination = tmp_path / "backups"
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
           "BACKUP_FIXTURE": str(fixture)}
    args = ["bash", str(ROOT / "scripts/pull_artifacts.sh"), "lambda-alias",
            "/lambda/nfs/preanchor/preanchor/artifacts/bootstrap/stage0-fixture",
            str(destination)]
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
    receipt = destination / "stage0-fixture/SECONDARY_VERIFIED.json"
    if corrupt:
        assert result.returncode != 0
        assert not receipt.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(receipt.read_text())["verified_files"] == 1
        before = receipt.read_bytes()
        repeated = subprocess.run(args, env=env, capture_output=True, text=True, timeout=10)
        assert repeated.returncode != 0
        assert receipt.read_bytes() == before


def test_pull_rejects_remote_shell_injection(tmp_path):
    result = subprocess.run([
        "bash", str(ROOT / "scripts/pull_artifacts.sh"), "host; echo unsafe",
        "/lambda/nfs/snapshot", str(tmp_path / "backup")],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path / "backup").exists()
