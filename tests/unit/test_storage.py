import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from preanchor.storage import persistent_mount


@pytest.mark.parametrize("fstype", ["nfs", "nfs4", "virtiofs"])
def test_persistent_mount_accepts_native_and_codex_exposure(monkeypatch, fstype):
    mount = {"source": "fixture-filesystem", "fstype": fstype,
             "target": "/lambda/nfs/preanchor", "options": "rw,nosuid,nodev"}
    monkeypatch.setattr("preanchor.storage.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"filesystems": [mount]})))
    assert persistent_mount(Path("/lambda/nfs/preanchor/preanchor")) == mount


@pytest.mark.parametrize("change", [
    {"fstype": "ext4"}, {"target": "/"}, {"target": "/lambda/nfs"},
    {"target": "/lambda/nfs/other"}, {"options": "ro,nosuid"},
])
def test_persistent_mount_rejects_ephemeral_or_readonly_exposure(monkeypatch, change):
    mount = {"source": "fixture-filesystem", "fstype": "virtiofs",
             "target": "/lambda/nfs/preanchor", "options": "rw,nosuid,nodev", **change}
    monkeypatch.setattr("preanchor.storage.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"filesystems": [mount]})))
    with pytest.raises(ValueError):
        persistent_mount(Path("/lambda/nfs/preanchor/preanchor"))
