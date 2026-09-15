"""prepare -> content-addressed bundle -> empty second root -> restore -> verify.

Scientific identity (Stage 3 hashes, run ID, manifest revision chain, plan and rollout IDs,
frozen settings) must survive a move to a different physical root exactly; every failure
mode must fail closed."""

import json
import shutil
from pathlib import Path

import pytest
from backup_fixtures import ROOT, SOURCE, fake_provenance, grant_writer_lease, make_world

from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import MUST_BACK_UP, SECRET, SHOULD_BACK_UP
from preanchor.backup.legacy import import_legacy
from preanchor.backup.prepare import plan, prepare, verify_bundle
from preanchor.backup.receipts import list_receipts
from preanchor.backup.restore import restore_apply, restore_plan
from preanchor.backup.store import ObjectStore
from preanchor.backup.verify import render, restore_verify
from preanchor.config import config_hash
from preanchor.ids import file_hash
from preanchor.io import UnitStore, atomic_bytes, json_bytes, read_json
from preanchor.labels.generate import CompletionStore, FakeBackend
from preanchor.manifest import load_manifest, save_manifest


def portable(root: Path) -> Path:
    return root / "artifacts/portable"


# ---------------------------------------------------------------------------
# Plan and classification
# ---------------------------------------------------------------------------
def test_plan_classifies_current_state_and_reports_sizes(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, specs = make_world(root, monkeypatch, units=4)
    (root / "cache/huggingface/hub").mkdir(parents=True)
    (root / "cache/huggingface/hub/blob").write_bytes(b"x" * 100)
    (root / ".env").write_text("HF_TOKEN=hf_" + "Q" * 30 + "\n")
    summary, inventory = plan(root, ROOT, for_stage=5)
    paths = {e.path: e for e in inventory.entries}
    assert summary["blockers"] == []
    assert summary["kind"] == "live_checkpoint" and summary["highest_protected_stage"] == 3
    assert summary["stages"]["stage3"]["status"] == "complete"
    assert summary["stages"]["stage5"]["run_id"] == run.name
    assert summary["stages"]["stage5"]["protected_completed_units"] == 4
    assert paths["data/processed/stage3/trajectories.parquet"].classification == MUST_BACK_UP
    assert paths["data/processed/stage3/trajectories.parquet"].validation == "parquet"
    assert paths[f"artifacts/runs/{run.name}/stages/stage5/plan.json"].required
    unit = f"artifacts/runs/{run.name}/stages/stage5/units_raw/{specs[0].rollout_id}"
    assert paths[f"{unit}/response.json"].validation == "unit_file"
    assert paths[f"artifacts/runs/{run.name}/heartbeat.json"].classification == SHOULD_BACK_UP
    assert not any(p.endswith(".env") or "cache/" in p for p in paths)
    excluded = {(e["path"], e["classification"]) for e in inventory.excluded}
    assert (".env", SECRET) in excluded and ("cache", "CACHE_EXCLUDE") in excluded
    assert {r["kind"] for r in summary["reacquire"]} >= {"model_weights", "minilm_weights",
                                                         "vllm_image"}
    assert summary["objects"]["new"] == summary["objects"]["unique"] > 0
    assert summary["logical_bytes"] == sum(e.size for e in inventory.entries)
    assert summary["classification"][MUST_BACK_UP]["files"] > 20


def test_plan_reports_blockers_for_missing_canonical_state(tmp_path, monkeypatch):
    fake_provenance(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("Stage 3" in b for b in summary["blockers"])
    with pytest.raises(ValueError, match="Backup blocked"):
        prepare(root, ROOT, for_stage=5)


def test_secret_inside_a_run_blocks_prepare(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, _ = make_world(root, monkeypatch, units=2)
    (run / "logs/notes.txt").write_text("hf_" + "Z" * 30)
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("credential-like content" in b for b in summary["blockers"])


# ---------------------------------------------------------------------------
# Incremental preparation
# ---------------------------------------------------------------------------
def test_repeated_prepare_adds_zero_objects_and_new_unit_adds_only_its_files(
        tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, specs = make_world(root, monkeypatch, units=3)
    first = prepare(root, ROOT, for_stage=5)
    assert first["objects_created"] == first["unique_objects"] > 0
    store = ObjectStore(portable(root))
    count = len(store.digests())
    second = prepare(root, ROOT, for_stage=5)
    assert second["objects_created"] == 0 and second["objects_reused"] == first["unique_objects"]
    assert len(store.digests()) == count
    assert second["bundle_id"] != first["bundle_id"]   # new timestamped manifest, same objects
    stage = run / "stages/stage5"
    CompletionStore(stage / "units").raw_one(specs[3], FakeBackend(["A new one. Next."]),
                                             max_tokens=16384, temperature=0.6, top_p=0.95)
    third = prepare(root, ROOT, for_stage=5)
    new_files = sorted(p for p in (stage / "units_raw" / specs[3].rollout_id).iterdir())
    assert third["objects_created"] == len({file_hash(p) for p in new_files}
                                           - store.digests() | {file_hash(p) for p in new_files}
                                           & set()) or third["objects_created"] <= len(new_files)
    assert third["objects_created"] > 0
    assert len(store.digests()) == count + third["objects_created"]
    assert verify_bundle(portable(root), "latest")["ok"]
    receipts = list_receipts(portable(root))
    assert [r["kind"] for r in receipts] == ["prepare"] * 3
    assert all(r["secondary_verified"] is False for r in receipts)


# ---------------------------------------------------------------------------
# Restore onto a different root and prove scientific identity
# ---------------------------------------------------------------------------
def test_round_trip_to_different_root_preserves_scientific_identity(tmp_path, monkeypatch):
    root = tmp_path / "region-a/preanchor"
    run, specs = make_world(root, monkeypatch, units=256)
    result = prepare(root, ROOT, for_stage=5)
    name, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["scientific_identity"]["stage5"]["run_id"] == run.name
    assert manifest["scientific_identity"]["stage5"]["rollout_id_set_sha256"] == config_hash(
        sorted(s.rollout_id for s in specs))
    assert manifest["physical_origin"]["root"] == str(root)
    assert manifest["producer"]["git_sha"] == SOURCE["git_sha"]

    other = tmp_path / "region-b/mount-9f1c/preanchor"    # empty, different physical root
    other.mkdir(parents=True)
    planned = restore_plan(other, portable(root), name, for_stage=5)
    assert planned["ok"] and planned["states"] == {"planned": planned["entries"]}
    applied = restore_apply(other, portable(root), name, repo=ROOT, for_stage=5)
    assert applied["ok"], applied["failures"]
    assert set(applied["states"]) == {"restored"}
    assert applied["physical_destination"]["root"] == str(other) != applied[
        "physical_origin"]["root"]
    restored_run = other / "artifacts/runs" / run.name
    assert load_manifest(restored_run)["run_id"] == run.name
    assert sorted(p.name for p in (restored_run / "manifests").glob("*.json")) == sorted(
        p.name for p in (run / "manifests").glob("*.json"))
    for spec in specs:
        UnitStore(restored_run / "stages/stage5/units_raw").validate(spec.rollout_id)
    for name_ in ("trajectories.parquet", "sentences.parquet"):
        assert file_hash(other / "data/processed/stage3" / name_) == file_hash(
            root / "data/processed/stage3" / name_)
    verified = restore_verify(other, portable(root), name, repo=ROOT, for_stage=5)
    text = render(verified)
    assert verified["verified"], verified["failed"]
    assert verified["stage5"]["rollout_ids_identical"] and verified["stage5"][
        "planned_units"] == 256
    assert verified["ready_to_continue"], text     # fake provenance == producing source
    assert "RESTORE VERIFIED" in text and "Ready to continue Stage 5: YES" in text
    assert "Scientifically equivalent portable state: PASS" in text
    names = {c["name"] for c in verified["checks"] if c["status"] == "pass"}
    assert "recomputed production_specs rollout IDs == plan (sorted)" in names
    assert "fresh input hashes == manifest.input_hashes" in names
    assert "Stage 5 frozen scientific settings" in names
    assert f"{run.name} run_id preserved" in names
    # Idempotent: a second apply changes nothing and reports identical presence.
    again = restore_apply(other, portable(root), name, repo=ROOT, for_stage=5)
    assert again["ok"] and set(again["states"]) == {"already_present_identical"}
    kinds = [r["kind"] for r in list_receipts(portable(root))]
    assert kinds.count("restore") == 2 and kinds.count("restore_verification") == 1


def test_restore_verify_fails_when_stage3_bytes_change(tmp_path, monkeypatch):
    root = tmp_path / "a"
    run, _ = make_world(root, monkeypatch, units=2)
    result = prepare(root, ROOT, for_stage=5)
    other = tmp_path / "b"
    restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5)
    # A "scientifically equivalent" recompute with different bytes must not verify.
    from backup_fixtures import write_stage3

    shutil.rmtree(other / "data/processed/stage3")
    write_stage3(other, sentences=49)
    verified = restore_verify(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5)
    assert not verified["verified"]
    failed = {c["name"] for c in verified["failed"]}
    assert "Stage 3 tables hash to the recorded identity" in failed
    assert "RESTORE NOT VERIFIED" in render(verified)


# ---------------------------------------------------------------------------
# Fail-closed restore
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure", ["missing", "corrupt", "conflict"])
def test_restore_fails_closed(tmp_path, monkeypatch, failure):
    root = tmp_path / "a"
    run, specs = make_world(root, monkeypatch, units=2)
    result = prepare(root, ROOT, for_stage=5)
    store = ObjectStore(portable(root))
    plan_path = f"artifacts/runs/{run.name}/stages/stage5/plan.json"
    digest = file_hash(root / plan_path)
    other = tmp_path / "b"
    if failure == "missing":
        store.path(digest).unlink()
    elif failure == "corrupt":
        store.path(digest).write_bytes(b"{}")
    else:
        (other / plan_path).parent.mkdir(parents=True)
        (other / plan_path).write_bytes(b"different")
    applied = restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5)
    assert not applied["ok"]
    state = applied["failures"][plan_path].split(":")[0]
    assert state == {"missing": "missing", "corrupt": "mismatch", "conflict": "conflict"}[failure]
    if failure == "conflict":
        assert (other / plan_path).read_bytes() == b"different"
    else:
        assert not (other / plan_path).exists()
    # Everything else was installed and individually verified; success is still refused.
    assert (other / "data/processed/stage3/trajectories.parquet").is_file()
    receipt = list_receipts(portable(root))[-1]
    assert receipt["kind"] == "restore" and receipt["outcome"] == "failed"
    dry = restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5,
                        dry_run=True)
    assert dry["dry_run"] and "receipt" not in dry


# ---------------------------------------------------------------------------
# Source validation before ingestion
# ---------------------------------------------------------------------------
def test_checksum_sidecar_mismatch_blocks_prepare(tmp_path, monkeypatch):
    root = tmp_path / "root"
    make_world(root, monkeypatch, units=1)
    (root / "data/processed/stage3/import_summary.json.sha256").write_text("0" * 64 + "\n")
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("import_summary.json: Checksum mismatch" in b for b in summary["blockers"])


def test_parquet_schema_mismatch_blocks_prepare(tmp_path, monkeypatch):
    root = tmp_path / "root"
    make_world(root, monkeypatch, units=1)
    schema = root / "data/processed/stage3/sentences.parquet.schema.json"
    metadata = read_json(schema)
    metadata["row_count"] += 1
    payload = json_bytes(metadata)
    atomic_bytes(schema, payload, replace=True)
    import hashlib

    atomic_bytes(Path(str(schema) + ".sha256"),
                 (hashlib.sha256(payload).hexdigest() + "\n").encode(), replace=True)
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("sentences.parquet: Parquet schema, grain or checksum mismatch" in b
               for b in summary["blockers"])


def test_run_manifest_current_view_mismatch_blocks_prepare(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, _ = make_world(root, monkeypatch, units=1)
    manifest = load_manifest(run)
    manifest["stages"]["stage5"]["state"] = "generated"
    manifest["commands"].append("edited without save_manifest")
    atomic_bytes(run / "manifest.json", json_bytes(manifest), replace=True)
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any(f"{run.name}: unreadable manifest" in w for w in summary["warnings"])
    assert any("inconsistent manifest" in b and run.name in b for b in summary["blockers"])


def test_units_outside_the_plan_and_corrupt_units_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, specs = make_world(root, monkeypatch, units=256)
    raw = run / "stages/stage5/units_raw"
    (raw / specs[0].rollout_id / "response.json").write_bytes(b"{}")
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("corrupt unit" in b for b in summary["blockers"])
    foreign = raw / ("f" * 64)
    shutil.copytree(raw / specs[1].rollout_id, foreign)
    summary, _ = plan(root, ROOT, for_stage=5)
    assert any("not in the run plan" in b for b in summary["blockers"])
    # A live run only warns about a corrupt unit (the runner quarantines and regenerates).
    shutil.rmtree(foreign)
    manifest = load_manifest(run)
    manifest["stages"]["stage5"]["state"] = "running"
    save_manifest(run, manifest)
    summary, inventory = plan(root, ROOT, for_stage=5)
    assert summary["blockers"] == [] and any("corrupt unit" in w for w in summary["warnings"])
    assert not any(specs[0].rollout_id in e.path for e in inventory.entries)


def test_dirty_producer_is_recorded_and_can_be_refused(tmp_path, monkeypatch):
    root = tmp_path / "root"
    make_world(root, monkeypatch, units=1)
    dirty = {**SOURCE, "dirty": True, "dirty_diff_hash": "d" * 64}
    fake_provenance(monkeypatch, dirty)
    with pytest.raises(ValueError, match="dirty"):
        prepare(root, ROOT, for_stage=5, allow_dirty=False)
    result = prepare(root, ROOT, for_stage=5)
    _, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["producer"]["dirty"] is True
    assert manifest["producer"]["dirty_diff_hash"] == "d" * 64
    # The producing run's own identity is untouched: canonical run manifests are not rewritten.
    assert manifest["scientific_identity"]["dirty_diff_hashes"] == {
        r: SOURCE["dirty_diff_hash"] for r in manifest["scientific_identity"]["run_ids"]}


# ---------------------------------------------------------------------------
# Live Stage 5 checkpoint
# ---------------------------------------------------------------------------
def test_live_checkpoint_round_trip_then_canonical_bundle(tmp_path, monkeypatch):
    root = tmp_path / "root"
    run, specs = make_world(root, monkeypatch, units=8, state="running")
    raw = run / "stages/stage5/units_raw"
    partial = raw / specs[8].rollout_id
    partial.mkdir()
    (partial / "response.json").write_bytes(b'{"in": "progress"}')
    (raw / ".locks").mkdir(exist_ok=True)
    summary, inventory = plan(root, ROOT, for_stage=5)
    assert summary["blockers"] == [] and summary["kind"] == "live_checkpoint"
    assert summary["highest_protected_stage"] == 3   # Stage 4 absent, Stage 5 live
    stage5 = summary["stages"]["stage5"]
    assert stage5["status"] == "running" and stage5["protected_completed_units"] == 8
    assert stage5["resumable"] is True and stage5["finalized"] is False
    assert not any(specs[8].rollout_id in e.path for e in inventory.entries)
    live = prepare(root, ROOT, for_stage=5)
    assert live["kind"] == "live_checkpoint"
    other = tmp_path / "other"
    applied = restore_apply(other, portable(root), live["bundle_id"], repo=ROOT, for_stage=5)
    assert applied["ok"]
    verified = restore_verify(other, portable(root), live["bundle_id"], repo=ROOT, for_stage=5)
    assert verified["verified"] and verified["live_checkpoint"]
    assert "live checkpoint, not a finalized canonical stage" in render(verified)
    for spec in specs[:8]:
        UnitStore(other / "artifacts/runs" / run.name / "stages/stage5/units_raw").validate(
            spec.rollout_id)
    # Later completions: only their objects are new; finalization yields a canonical bundle.
    shutil.rmtree(partial)
    store = CompletionStore(run / "stages/stage5/units")
    for spec in specs[8:]:
        store.raw_one(spec, FakeBackend(), max_tokens=16384, temperature=0.6, top_p=0.95)
    manifest = load_manifest(run)
    manifest["stages"]["stage5"]["state"] = "generated"
    save_manifest(run, manifest)
    final = prepare(root, ROOT, for_stage=5)
    assert final["kind"] == "canonical" and final["highest_protected_stage"] == 5
    assert final["objects_reused"] >= live["unique_objects"] - 4    # heartbeat/manifest changed
    assert 0 < final["objects_created"] <= 248 * 6 + 8


# ---------------------------------------------------------------------------
# Secrets never enter a bundle
# ---------------------------------------------------------------------------
def test_bundle_contains_no_secret_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "S" * 30)
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key-77")
    root = tmp_path / "root"
    run, _ = make_world(root, monkeypatch, units=1)
    (root / ".env").write_text("HF_TOKEN=hf_" + "S" * 30 + "\nWANDB_API_KEY=wandb-key-77\n")
    result = prepare(root, ROOT, for_stage=5)
    text = (portable(root) / "manifests" / f"{result['bundle_id']}.json").read_text()
    assert "hf_" + "S" * 30 not in text and "wandb-key-77" not in text
    store = ObjectStore(portable(root))
    for digest in store.digests():
        assert b"wandb-key-77" not in store.path(digest).read_bytes()
    _, manifest = load_bundle(portable(root), result["bundle_id"])
    assert all(not e["path"].endswith(".env") for e in manifest["entries"])


# ---------------------------------------------------------------------------
# Legacy snapshot import
# ---------------------------------------------------------------------------
def test_import_legacy_verifies_checksums_before_ingesting(tmp_path, monkeypatch):
    fake_provenance(monkeypatch)
    root = tmp_path / "root"
    snapshot = root / "artifacts/backups/stage2-fixture"
    (snapshot / "run").mkdir(parents=True)
    (snapshot / "run/manifest.json").write_bytes(b"{}\n")
    (snapshot / "preanchor.bundle").write_bytes(b"bundle-bytes")
    entries = {"run/manifest.json": file_hash(snapshot / "run/manifest.json"),
               "preanchor.bundle": "0" * 64}
    (snapshot / "checksums.json").write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="checksum mismatch"):
        import_legacy(root, ROOT, snapshot)
    assert not portable(root).exists()
    grant_writer_lease(root, monkeypatch)      # ingest mutates the store: writer lease
    entries["preanchor.bundle"] = file_hash(snapshot / "preanchor.bundle")
    (snapshot / "checksums.json").write_text(json.dumps(entries))
    result = import_legacy(root, ROOT, snapshot)
    _, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["bundle"]["kind"] == "legacy_import"
    assert {e["path"] for e in manifest["entries"]} == {
        "legacy/stage2-fixture/run/manifest.json", "legacy/stage2-fixture/preanchor.bundle",
        "legacy/stage2-fixture/checksums.json"}
    assert import_legacy(root, ROOT, snapshot)["objects_created"] == 0


# ---------------------------------------------------------------------------
# Absolute-path compatibility of Stage 2/4 validators after a move
# ---------------------------------------------------------------------------
def test_validate_download_accepts_a_moved_root(tmp_path, monkeypatch):
    from test_fetch_resume import fixture_data

    from preanchor.config import load_config
    from preanchor.data.stage import run_fetch, validate_download

    payload, row = fixture_data()
    discovery = {"dataset_id": "fixture", "revision": "b" * 40, "files": [row]}
    monkeypatch.setattr("preanchor.data.stage.load_discovery", lambda *a: discovery)
    fake_provenance(monkeypatch)
    config = load_config(ROOT / "configs/mvp.yaml")
    from preanchor.manifest import create_run

    old_root = tmp_path / "old"
    directory = create_run(old_root, ROOT, config, ["fixture"])
    snapshot = tmp_path / "snapshot"
    (snapshot / row["path"]).parent.mkdir(parents=True)
    (snapshot / row["path"]).write_bytes(payload)
    run_fetch(directory, ROOT, config, tmp_path / "cache", tracking=False,
              download=lambda **kwargs: snapshot)
    assert validate_download(directory, ROOT, config)["state"] == "passed"
    recorded = read_json(directory / "stages/stage2/source_manifest.json")["files"][0]
    assert recorded["local_path"].startswith(str(old_root))   # legacy absolute path kept
    new_root = tmp_path / "new-region"
    moved = new_root / "artifacts/runs" / directory.name
    shutil.copytree(directory, moved)
    assert validate_download(moved, ROOT, config)["state"] == "passed"
    wrong = new_root / "artifacts/runs" / ("x" + directory.name[1:])
    shutil.copytree(directory, wrong)
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        validate_download(wrong, ROOT, config)
