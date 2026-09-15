"""Stage 6 extraction on synthetic inputs: identity, no-op resume, SIGKILL, corruption."""

import hashlib
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from preanchor.config import load_config
from preanchor.features.extract import (
    execute,
    load_inputs,
    prepare,
    select_work,
    validate_run,
)
from preanchor.io import UnitStore, read_json, validate_parquet
from preanchor.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
sys.path.insert(0, str(FIXTURES))
from tiny_qwen2 import tiny_loader, write_synthetic_root  # noqa: E402

SOURCE = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
          "package_freeze": []}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
    config = load_config(ROOT / "configs/smoke.yaml")
    write_synthetic_root(tmp_path, config)
    inputs, hashes = load_inputs(tmp_path)
    work = select_work(config, inputs, ["330:correct", "2189:incorrect"], 6)
    run = prepare(tmp_path, ROOT, config, "test", work, hashes)
    return tmp_path, run, config


def unit_state(run):
    store = UnitStore(run / "activations/units")
    keys = sorted(p.name for p in store.root.iterdir() if len(p.name) == 64)
    return {k: {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (store.root / k).iterdir()} for k in keys}


def test_complete_resume_and_validation(prepared):
    root, run, config = prepared
    result = execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                     limit_sentences=6, tracking=False, loader=tiny_loader)
    assert result["created"] == 2 and result["summary"]["n_rows"] == 12
    assert result["diagnostics"]["primary_pass"] and result["diagnostics"]["n_boundaries"] == 12
    manifest = load_manifest(run)
    assert manifest["stages"]["stage6"]["state"] == "complete"
    assert set(manifest["input_hashes"]) == {"stage3/trajectories.parquet",
                                             "stage3/sentences.parquet",
                                             "stage4/labels.parquet"}
    index = validate_parquet(run / "activations/index.parquet").to_pylist()
    assert [r["sentence_idx"] for r in index] == list(range(6)) * 2
    assert [r["eligible"] for r in index] == [True] * 12 and index[0]["n_layers"] == 5
    assert index[0]["final_prev_position_available"]
    features = validate_parquet(run / "stages/stage6/logit_features.parquet").to_pylist()
    assert len(features) == 12 and "diagnostic__first_target_token_logprob" in features[0]
    assert "first_target_token_logprob" not in {
        c for c in features[0] if not c.startswith("diagnostic__")}
    projection = read_json(run / "stages/stage6/resource_projection.json")
    assert projection["unit_count"] == 40 and projection["row_count"] == 40 * 17
    before = unit_state(run)
    with pytest.raises(ValueError, match="requires --resume"):
        execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                limit_sentences=6, tracking=False, loader=tiny_loader)
    again = execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                    limit_sentences=6, resume=True, tracking=False, loader=tiny_loader)
    assert again["created"] == 0 and again["skipped"] == 2
    assert unit_state(run) == before
    assert validate_run(run, config)["rows"] == 12
    with pytest.raises(ValueError, match="scope differs"):
        execute(run, ROOT, config, trajectories=["330:correct"], limit_sentences=6,
                resume=True, tracking=False, loader=tiny_loader)


def test_changed_input_hash_is_refused(prepared, monkeypatch):
    root, run, config = prepared
    manifest = load_manifest(run)
    monkeypatch.setitem(manifest["input_hashes"], "stage4/labels.parquet", "0" * 64)
    from preanchor.manifest import save_manifest

    manifest["identity"]["input_hashes"] = manifest["input_hashes"]
    save_manifest(run, manifest)
    with pytest.raises(ValueError, match="differ"):
        execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                limit_sentences=6, tracking=False, loader=tiny_loader)


def test_corrupt_shard_is_quarantined_and_regenerated(prepared):
    root, run, config = prepared
    execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
            limit_sentences=6, tracking=False, loader=tiny_loader)
    before = unit_state(run)
    key = next(iter(before))
    shard = next(p for p in (run / "activations/units" / key).iterdir()
                 if p.suffix == ".safetensors")
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        validate_run(run, config)
    again = execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                    limit_sentences=6, resume=True, tracking=False, loader=tiny_loader)
    assert again["created"] == 1
    assert unit_state(run) == before  # regenerated unit is byte-identical to the original
    assert len(list((run / "activations/units/quarantine").iterdir())) == 1
    assert validate_run(run, config)["units"] == 2


def test_sigkill_mid_unit_leaves_no_accepted_partial_and_resumes(prepared):
    root, run, config = prepared
    code = f'''
import sys
sys.path.insert(0, {str(FIXTURES)!r})
from pathlib import Path
import preanchor.manifest as m
m.provenance = lambda repo: ({SOURCE!r}, b"")
from preanchor.config import load_config
from preanchor.features.extract import execute
import preanchor.features.extract as ex
from tiny_qwen2 import tiny_loader
original = ex.replay
def slow(model, ids, boundaries, **kw):
    capture = original(model, ids, boundaries, **kw)
    print("unit-started", flush=True)
    return capture
ex.replay = slow
config = load_config(Path({str(ROOT)!r}) / "configs/smoke.yaml")
execute(Path({str(run)!r}), Path({str(ROOT)!r}), config,
        trajectories=["330:correct", "2189:incorrect"], limit_sentences=6, tracking=False,
        loader=tiny_loader, pause_seconds=30)
'''
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 60)[0]
        assert process.stdout.readline().strip() == "unit-started"
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
    store = UnitStore(run / "activations/units")
    partial = [p for p in store.root.iterdir() if len(p.name) == 64]
    assert len(partial) == 1 and not (partial[0] / "SUCCESS.json").exists()
    assert load_manifest(run)["stages"]["stage6"]["state"] == "running"
    result = execute(run, ROOT, config, trajectories=["330:correct", "2189:incorrect"],
                     limit_sentences=6, resume=True, tracking=False, loader=tiny_loader)
    assert result["created"] == 2
    assert (run / "stages/stage6/failures").exists()
    assert len(list((run / "activations/units/quarantine").iterdir())) == 1
    assert validate_run(run, config)["rows"] == 12
