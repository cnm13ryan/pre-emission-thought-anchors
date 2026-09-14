import hashlib
from pathlib import Path

import pytest

from preanchor.config import load_config
from preanchor.data.fetch import fetch_file, source_key
from preanchor.data.stage import run_fetch, validate_download
from preanchor.io import UnitStore
from preanchor.manifest import create_run, load_manifest

ROOT = Path(__file__).resolve().parents[2]


def fixture_data():
    payload = b'{"full_cot":"exact trace"}'
    row = {
        "path": "selected/base_solution.json",
        "role": "base_solution",
        "problem_id": 4682,
        "trace_kind": "correct",
        "size": len(payload),
        "lfs_sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob_sha1": "a" * 40,
    }
    return payload, row


def test_exact_download_resume_and_quarantine(tmp_path):
    payload, row = fixture_data()
    d = {"dataset_id": "pinned/dataset", "revision": "b" * 40}
    snapshot = tmp_path / "snapshot"
    source = snapshot / row["path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return snapshot

    store = UnitStore(tmp_path / "units")
    args = (store, row, d, {"code": "fixed"}, tmp_path / "cache")
    first = fetch_file(*args, download=download)
    p = Path(first["local_path"])
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    assert not fetch_file(*args, download=download)["created"]
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before
    assert len(calls) == 1
    assert calls[0]["allow_patterns"] == [row["path"]]
    assert calls[0]["revision"] == d["revision"]
    assert calls[0]["repo_type"] == "dataset"
    p.write_bytes(b"corrupt")
    assert fetch_file(*args, download=download)["created"]
    assert len(list((store.root / "quarantine").iterdir())) == 1
    assert source_key(row, d, {"code": "changed"}) != first["unit_key"]


def test_failure_has_no_accepted_output(tmp_path):
    payload, row = fixture_data()
    d = {"dataset_id": "pinned/dataset", "revision": "b" * 40}

    def download(**kwargs):
        raise ConnectionError("synthetic interruption")

    store = UnitStore(tmp_path / "units")
    with pytest.raises(ConnectionError):
        fetch_file(store, row, d, {}, tmp_path / "cache", download=download)
    with pytest.raises(ValueError, match="Missing or unsafe"):
        store.validate(source_key(row, d, {}))


def test_stage_resume_and_input_isolation(tmp_path, monkeypatch):
    payload, row = fixture_data()
    discovery = {"dataset_id": "fixture", "revision": "b" * 40, "files": [row]}
    monkeypatch.setattr("preanchor.data.stage.load_discovery", lambda *a: discovery)
    source = {
        "git_sha": "a" * 40,
        "dirty_diff_hash": "b" * 64,
        "lock_hash": "c" * 64,
        "package_freeze": [],
    }
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (source.copy(), b""))
    config = load_config(ROOT / "configs/mvp.yaml")
    directory = create_run(tmp_path, ROOT, config, ["fixture"])
    snapshot = tmp_path / "snapshot"
    p = snapshot / row["path"]
    p.parent.mkdir(parents=True)
    p.write_bytes(payload)

    def download(**kwargs):
        return snapshot

    args = (directory, ROOT, config, tmp_path / "cache")

    def timeout_download(**kwargs):
        raise TimeoutError("synthetic network timeout")

    with pytest.raises(TimeoutError, match="synthetic network"):
        run_fetch(*args, tracking=False, download=timeout_download)
    assert load_manifest(directory)["stages"]["stage2"]["state"] == "failed"
    assert (directory / "failure.json").is_file()
    assert run_fetch(*args, resume=True, tracking=False, download=download)["created"] == 1
    assert run_fetch(*args, resume=True, tracking=False, download=download)["skipped"] == 1
    assert validate_download(directory, ROOT, config)["state"] == "passed"
    assert load_manifest(directory)["stages"]["stage2"]["state"] == "complete"
    discovery["revision"] = "changed"
    with pytest.raises(ValueError, match="Changed input"):
        run_fetch(*args, resume=True, tracking=False, download=download)
