"""Stage 6 portable backup: classification, incremental prepare, round trip, fail-closed."""

import shutil
from pathlib import Path

import pytest
from backup_fixtures import ROOT, make_stage6, make_stage6_world

from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import MUST_BACK_UP, SHOULD_BACK_UP, stage_closure
from preanchor.backup.prepare import plan, prepare, verify_bundle
from preanchor.backup.restore import restore_apply
from preanchor.backup.store import ObjectStore
from preanchor.backup.verify import render, restore_verify
from preanchor.ids import file_hash
from preanchor.io import read_json
from preanchor.manifest import load_manifest, save_manifest


def portable(root: Path) -> Path:
    return root / "artifacts/portable"


def stage6_paths(run: Path) -> list[Path]:
    """Identity-bearing Stage 6 files: shards, index, records, sidecars and schemas."""
    files = [p for p in (run / "activations").rglob("*")
             if p.is_file() and not {".locks", "quarantine"} & set(p.parts)]
    files += [p for p in (run / "stages/stage6").iterdir() if p.is_file()]
    return sorted(files)


def test_stage6_closure_and_plan_classification(tmp_path, monkeypatch):
    root = tmp_path / "root"
    production, _, run = make_stage6_world(root, monkeypatch)
    assert stage_closure(6) == (3, 4, 5, 6)
    # A prepared-but-never-executed run and a stale-input run must stay historical.
    from preanchor.config import load_config
    from preanchor.features.extract import load_inputs, select_work
    from preanchor.features.extract import prepare as prepare_run

    config = load_config(ROOT / "configs/smoke.yaml")
    inputs, hashes = load_inputs(root)
    idle = prepare_run(root, ROOT, config, "idle", select_work(config, inputs, None, 2), hashes)
    stale = make_stage6(root, trajectories=("330:incorrect",), limit=2)
    manifest = load_manifest(stale)
    manifest["input_hashes"]["stage4/labels.parquet"] = "0" * 64
    manifest["identity"]["input_hashes"] = manifest["input_hashes"]
    save_manifest(stale, manifest)
    quarantine = run / "activations/units/quarantine/deadbeef"
    quarantine.mkdir(parents=True)
    (quarantine / "junk.safetensors").write_bytes(b"partial")

    summary, inventory = plan(root, ROOT, for_stage=6)
    assert summary["blockers"] == [], summary["blockers"]
    assert summary["kind"] == "canonical" and summary["highest_protected_stage"] == 6
    stage6 = summary["stages"]["stage6"]
    assert stage6["status"] == "complete" and stage6["run_ids"] == [run.name]
    assert stage6["total_units"] == 2 and stage6["total_rows"] == 8
    assert summary["runs"][run.name] == {"role": "stage6_smoke", "stage": "stage6"}
    assert summary["runs"][idle.name]["reason"] == "stage6_state_prepared"
    assert summary["runs"][stale.name]["reason"] == "input_hashes_differ_from_current_stage3_4"
    assert summary["runs"][production.name]["role"] == "stage5_production"
    paths = {e.path: e for e in inventory.entries}
    prefix = f"artifacts/runs/{run.name}"
    for name in ("summary.json", "invariance_report.json", "resource_projection.json",
                 "boundary_audit.json"):
        entry = paths[f"{prefix}/stages/stage6/{name}"]
        assert entry.classification == MUST_BACK_UP and entry.required
        assert entry.validation == "sha256_sidecar"
    assert paths[f"{prefix}/stages/stage6/logit_features.parquet"].validation == "parquet"
    assert paths[f"{prefix}/activations/index.parquet"].validation == "parquet"
    assert paths[f"{prefix}/activations/index.parquet"].required
    shards = [e for p, e in paths.items() if p.startswith(f"{prefix}/activations/units/")
              and p.endswith(".safetensors")]
    assert len(shards) == 2 and all(e.validation == "unit_file" and e.required for e in shards)
    assert paths[f"{prefix}/heartbeat.json"].classification == SHOULD_BACK_UP
    assert not any("quarantine" in p for p in paths)
    assert not any(p.startswith(f"artifacts/runs/{idle.name}/") for p in paths)
    assert not any(p.startswith(f"artifacts/runs/{stale.name}/") for p in paths)
    excluded = {e["path"] for e in inventory.excluded}
    assert f"{prefix}/activations/units/quarantine" in excluded
    assert f"artifacts/runs/{idle.name}" in excluded and f"artifacts/runs/{stale.name}" in excluded
    identity = inventory.scientific_identity["stage6"]
    assert identity["runs"][run.name]["invariance"]["primary_pass"] is True
    assert identity["runs"][run.name]["index_sha256"] == file_hash(
        run / "activations/index.parquet")
    # Every identity-bearing Stage 6 file is an entry with its exact hash.
    for path in stage6_paths(run):
        logical = f"{prefix}/{path.relative_to(run).as_posix()}"
        assert paths[logical].classification == MUST_BACK_UP, logical
    # A Stage 5 plan on the same world is unchanged: Stage 6 is carried but not required.
    five, inv5 = plan(root, ROOT, for_stage=5)
    assert five["required_stages"] == [3, 5] and five["blockers"] == []
    assert not any(e.required for e in inv5.entries if e.stage == "stage6")
    assert five["stages"]["stage5"]["run_id"] == production.name


def test_missing_stage6_evidence_blocks_a_stage6_bundle(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _, _, run = make_stage6_world(root, monkeypatch)
    (run / "stages/stage6/invariance_report.json").unlink()
    summary, _ = plan(root, ROOT, for_stage=6)
    assert any("Stage 6 records invalid" in b for b in summary["blockers"])
    with pytest.raises(ValueError, match="Backup blocked"):
        prepare(root, ROOT, for_stage=6)


def test_unchanged_prepare_reuses_every_object(tmp_path, monkeypatch):
    root = tmp_path / "root"
    make_stage6_world(root, monkeypatch, stage5_units=2)
    first = prepare(root, ROOT, for_stage=6)
    assert first["kind"] == "canonical" and first["highest_protected_stage"] == 6
    assert first["objects_created"] == first["unique_objects"] > 0
    second = prepare(root, ROOT, for_stage=6)
    assert second["objects_created"] == 0
    assert second["objects_reused"] == first["unique_objects"]
    assert verify_bundle(portable(root), "latest")["ok"]


def test_round_trip_restores_stage6_byte_for_byte_and_verifies(tmp_path, monkeypatch):
    root = tmp_path / "region-a/preanchor"
    production, _, run = make_stage6_world(root, monkeypatch)
    result = prepare(root, ROOT, for_stage=6)
    name, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["scientific_identity"]["stage6"]["run_ids"] == [run.name]
    other = tmp_path / "region-b/mount-77/preanchor"
    other.mkdir(parents=True)
    applied = restore_apply(other, portable(root), name, repo=ROOT, for_stage=6)
    assert applied["ok"], applied["failures"]
    restored = other / "artifacts/runs" / run.name
    for path in stage6_paths(run):
        twin = restored / path.relative_to(run)
        assert twin.read_bytes() == path.read_bytes(), path
    assert file_hash(other / "data/processed/stage4/labels.parquet") == file_hash(
        root / "data/processed/stage4/labels.parquet")
    verified = restore_verify(other, portable(root), name, repo=ROOT, for_stage=6)
    text = render(verified)
    assert verified["verified"], verified["failed"]
    assert verified["stage6"]["verified_runs"] == [run.name]
    assert verified["stage5"]["rollout_ids_identical"]
    assert "RESTORE VERIFIED" in text and "Ready to continue Stage 6: YES" in text
    passed = {c["name"] for c in verified["checks"] if c["status"] == "pass"}
    assert f"{run.name} validate-activations on restored units and index" in passed
    assert f"{run.name} primary invariance gate (same-shape mutation) and logit identity pass" \
        in passed
    assert "restored Stage 3/4 tables hash to the Stage 6 input identity" in passed
    # Tampering with a restored shard fails the verification gate.
    shard = next(restored.glob("activations/units/*/330_correct.safetensors"))
    shard.write_bytes(b"tampered")
    tampered = restore_verify(other, portable(root), name, repo=ROOT, for_stage=6)
    assert not tampered["verified"]
    assert any("required entries hash" in c["name"] for c in tampered["failed"])


@pytest.mark.parametrize("failure", ["missing", "corrupt", "conflict"])
def test_stage6_restore_fails_closed(tmp_path, monkeypatch, failure):
    root = tmp_path / "a"
    _, _, run = make_stage6_world(root, monkeypatch, stage5_units=2)
    result = prepare(root, ROOT, for_stage=6)
    store = ObjectStore(portable(root))
    index = f"artifacts/runs/{run.name}/activations/index.parquet"
    digest = file_hash(root / index)
    other = tmp_path / "b"
    if failure == "missing":
        store.path(digest).unlink()
    elif failure == "corrupt":
        store.path(digest).write_bytes(b"not parquet")
    else:
        (other / index).parent.mkdir(parents=True)
        (other / index).write_bytes(b"different")
    applied = restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=6)
    assert not applied["ok"]
    assert applied["failures"][index].split(":")[0] == {
        "missing": "missing", "corrupt": "mismatch", "conflict": "conflict"}[failure]
    if failure != "conflict":
        assert not (other / index).exists()
    # Independent of the failed entry, the bundle itself still rehashes closed.
    check = verify_bundle(portable(root), result["bundle_id"])
    assert check["ok"] == (failure == "conflict")


def test_stage6_entries_are_dropped_from_a_stage5_restore_verify(tmp_path, monkeypatch):
    """Stage 3–5 behaviour is unchanged: a --for-stage 5 verify ignores Stage 6 evidence."""
    root = tmp_path / "a"
    make_stage6_world(root, monkeypatch, stage5_units=2)
    result = prepare(root, ROOT, for_stage=6)
    other = tmp_path / "b"
    restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5)
    shutil.rmtree(other / "artifacts/runs" / read_json(
        root / "artifacts/portable/manifests" / f"{result['bundle_id']}.json")
        ["scientific_identity"]["stage6"]["run_ids"][0], ignore_errors=True)
    verified = restore_verify(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=5)
    assert verified["verified"] and verified["stage6"] == {}
