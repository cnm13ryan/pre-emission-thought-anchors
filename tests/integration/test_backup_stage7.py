"""Stage 7 portable backup: classification, incremental prepare, round trip, fail-closed."""

from pathlib import Path

import pytest
from backup_fixtures import ROOT, make_stage7_world

from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import MUST_BACK_UP, SHOULD_BACK_UP, stage_closure
from preanchor.backup.prepare import plan, prepare, verify_bundle
from preanchor.backup.restore import restore_apply
from preanchor.backup.store import ObjectStore
from preanchor.backup.verify import render, restore_verify
from preanchor.features.observables import feature_path
from preanchor.ids import file_hash
from preanchor.io import read_json, validate_parquet
from preanchor.manifest import load_manifest, save_manifest


def portable(root: Path) -> Path:
    return root / "artifacts/portable"


def stage7_paths(run: Path) -> list[Path]:
    """Identity-bearing Stage 7 files: shards, index, records, feature table, sidecars."""
    files = [p for p in (run / "activations").rglob("*")
             if p.is_file() and not {".locks", "quarantine"} & set(p.parts)]
    files += [p for p in (run / "stages/stage7").rglob("*") if p.is_file()]
    files += [p for p in (run / "features").rglob("*") if p.is_file()]
    return sorted(files)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One synthetic Stage 3–7 world per module: extraction of 40 tiny trajectories is the
    expensive part and every test below reads it without modifying it."""
    root = tmp_path_factory.mktemp("region-a") / "preanchor"
    with pytest.MonkeyPatch.context() as monkeypatch:
        production, _, stage6, stage7 = make_stage7_world(root, monkeypatch)
    return root, production, stage6, stage7


@pytest.fixture
def provenance(world, monkeypatch):
    from backup_fixtures import fake_provenance, grant_writer_lease

    fake_provenance(monkeypatch)
    grant_writer_lease(world[0], monkeypatch)


def test_stage7_closure_and_plan_classification(world, provenance):
    root, production, stage6, run = world
    assert stage_closure(7) == (3, 4, 5, 6, 7)
    summary, inventory = plan(root, ROOT, for_stage=7)
    assert summary["blockers"] == [], summary["blockers"]
    assert summary["kind"] == "canonical" and summary["highest_protected_stage"] == 7
    stage7 = summary["stages"]["stage7"]
    assert stage7["status"] == "complete" and stage7["run_id"] == run.name
    rows = validate_parquet(root / "data/processed/stage4/labels.parquet").num_rows
    assert stage7["n_units"] == 40 and stage7["n_rows"] == rows
    assert stage7["features_sha256"] == file_hash(run / "features/features.parquet") \
        == stage7["published_features_sha256"] == file_hash(feature_path(root))
    assert stage7["feature_columns"]["primary"][0] == "sentence_idx"
    assert summary["runs"][run.name] == {"role": "stage7_production", "stage": "stage7"}
    assert summary["runs"][stage6.name]["role"] == "stage6_smoke"
    assert summary["runs"][production.name]["role"] == "stage5_production"
    paths = {e.path: e for e in inventory.entries}
    prefix = f"artifacts/runs/{run.name}"
    for logical, validation in ((f"{prefix}/stages/stage7/summary.json", "sha256_sidecar"),
                                (f"{prefix}/stages/stage7/resource_projection.json",
                                 "sha256_sidecar"),
                                (f"{prefix}/activations/index.parquet", "parquet"),
                                (f"{prefix}/features/features.parquet", "parquet"),
                                (f"{prefix}/features/feature_audit.json", "sha256_sidecar"),
                                (f"{prefix}/features/feature_validation.json",
                                 "sha256_sidecar"),
                                ("data/processed/stage7/features.parquet", "parquet"),
                                ("data/processed/stage7/features.parquet.schema.json",
                                 "sha256_sidecar"),
                                ("data/processed/stage7/feature_audit.json", "sha256_sidecar"),
                                ("data/processed/stage7/features.parquet.sha256", "bytes")):
        entry = paths[logical]
        assert entry.classification == MUST_BACK_UP and entry.required, logical
        assert entry.validation == validation, logical
        assert entry.stage == "stage7"
    shards = [e for p, e in paths.items() if p.startswith(f"{prefix}/activations/units/")
              and p.endswith(".safetensors")]
    assert len(shards) == 40 and all(e.validation == "unit_file" and e.required for e in shards)
    assert paths[f"{prefix}/heartbeat.json"].classification == SHOULD_BACK_UP
    for path in stage7_paths(run):
        logical = f"{prefix}/{path.relative_to(run).as_posix()}"
        assert paths[logical].classification == MUST_BACK_UP, logical
    # Stage 6 evidence is carried as a required dependency of a Stage 7 bundle.
    assert paths[f"artifacts/runs/{stage6.name}/activations/index.parquet"].required
    identity = inventory.scientific_identity["stage7"]
    assert identity["run_id"] == run.name and identity["index_sha256"] == file_hash(
        run / "activations/index.parquet")
    assert identity["head_weights_sha256"] == read_json(
        run / "features/feature_audit.json")["head"]["weights_sha256"]
    # Stage 3–6 plans on the same world are unchanged: Stage 7 is carried but not required.
    six, inv6 = plan(root, ROOT, for_stage=6)
    assert six["required_stages"] == [3, 4, 5, 6] and six["blockers"] == []
    assert six["stages"]["stage6"]["run_ids"] == [stage6.name]
    assert not any(e.required for e in inv6.entries if e.stage == "stage7")
    five, inv5 = plan(root, ROOT, for_stage=5)
    assert five["required_stages"] == [3, 5] and five["blockers"] == []
    assert not any(e.required for e in inv5.entries if e.stage in ("stage6", "stage7"))


def test_stage7_run_that_did_not_publish_is_historical_and_stale_scope_blocks(
        world, provenance, tmp_path):
    root, _, _, run = world
    manifest = load_manifest(run)
    original = dict(manifest["output_hashes"])
    try:
        manifest["output_hashes"]["features/features.parquet"] = "0" * 64
        save_manifest(run, manifest)
        summary, _ = plan(root, ROOT, for_stage=7)
        assert summary["runs"][run.name]["reason"] == "did_not_publish_current_stage7_features"
        assert any("No complete Stage 7 run publishes" in b for b in summary["blockers"])
        with pytest.raises(ValueError, match="Backup blocked"):
            prepare(root, ROOT, for_stage=7)
        # Without Stage 7 required, the same world still plans a Stage 6 bundle.
        assert plan(root, ROOT, for_stage=6)[0]["blockers"] == []
    finally:
        manifest["output_hashes"] = original
        save_manifest(run, manifest)
    assert plan(root, ROOT, for_stage=7)[0]["blockers"] == []


def test_missing_stage7_validation_record_blocks(world, provenance):
    root, _, _, run = world
    record = run / "features/feature_validation.json"
    payload = record.read_bytes()
    try:
        record.unlink()
        summary, _ = plan(root, ROOT, for_stage=7)
        assert any("Stage 7 records invalid" in b for b in summary["blockers"])
    finally:
        record.write_bytes(payload)


def test_unchanged_prepare_reuses_every_object(world, provenance):
    root, _, _, _ = world
    first = prepare(root, ROOT, for_stage=7)
    assert first["kind"] == "canonical" and first["highest_protected_stage"] == 7
    assert first["unique_objects"] > 0
    second = prepare(root, ROOT, for_stage=7)
    assert second["objects_created"] == 0
    assert second["objects_reused"] == first["unique_objects"]
    assert verify_bundle(portable(root), "latest")["ok"]


def test_round_trip_restores_stage7_byte_for_byte_and_verifies(world, provenance, tmp_path):
    root, production, stage6, run = world
    result = prepare(root, ROOT, for_stage=7)
    name, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["scientific_identity"]["stage7"]["run_id"] == run.name
    assert manifest["bundle"]["required_stages"] == [3, 4, 5, 6, 7]
    other = tmp_path / "region-b/mount-99/preanchor"
    other.mkdir(parents=True)
    applied = restore_apply(other, portable(root), name, repo=ROOT, for_stage=7)
    assert applied["ok"], applied["failures"]
    restored = other / "artifacts/runs" / run.name
    for path in stage7_paths(run):
        twin = restored / path.relative_to(run)
        assert twin.read_bytes() == path.read_bytes(), path
    for name7 in ("features.parquet", "features.parquet.schema.json", "features.parquet.sha256",
                  "feature_audit.json", "feature_audit.json.sha256"):
        assert (other / "data/processed/stage7" / name7).read_bytes() == (
            root / "data/processed/stage7" / name7).read_bytes()
    assert validate_parquet(other / "data/processed/stage7/features.parquet").equals(
        validate_parquet(feature_path(root)))
    verified = restore_verify(other, portable(root), name, repo=ROOT, for_stage=7)
    text = render(verified)
    assert verified["verified"], verified["failed"]
    assert verified["stage7"]["verified"] and verified["stage7"]["rows"] == validate_parquet(
        root / "data/processed/stage4/labels.parquet").num_rows
    assert verified["stage6"]["verified_runs"] == [stage6.name]
    assert verified["stage5"]["rollout_ids_identical"]
    assert "RESTORE VERIFIED" in text and "Ready to continue Stage 7: YES" in text
    passed = {c["name"] for c in verified["checks"] if c["status"] == "pass"}
    assert f"{run.name} validate-activations on restored units and index" in passed
    assert f"{run.name} validate-features on the restored table (coverage, joins, " \
           "allow-list, invariants, published copy)" in passed
    assert "published data/processed/stage7/features.parquet == the run's feature table " \
           "and bundle identity" in passed
    # Verification is read-only: every restored Stage 7 byte is still the original.
    for path in stage7_paths(run):
        assert (restored / path.relative_to(run)).read_bytes() == path.read_bytes(), path
    assert restore_verify(other, portable(root), name, repo=ROOT, for_stage=7)["verified"]
    # Tampering with the restored published table fails the gate.
    (other / "data/processed/stage7/features.parquet").write_bytes(b"tampered")
    tampered = restore_verify(other, portable(root), name, repo=ROOT, for_stage=7)
    assert not tampered["verified"]
    assert any("required entries hash" in c["name"] for c in tampered["failed"])


@pytest.mark.parametrize("failure", ["missing", "corrupt", "conflict"])
def test_stage7_restore_fails_closed(world, provenance, tmp_path, failure):
    root, _, _, run = world
    result = prepare(root, ROOT, for_stage=7)
    store = ObjectStore(portable(root))
    target = f"artifacts/runs/{run.name}/features/features.parquet"
    digest = file_hash(root / target)
    payload = store.path(digest).read_bytes()
    other = tmp_path / "b"
    try:
        if failure == "missing":
            store.path(digest).unlink()
        elif failure == "corrupt":
            store.path(digest).write_bytes(b"not parquet")
        else:
            (other / target).parent.mkdir(parents=True)
            (other / target).write_bytes(b"different")
        applied = restore_apply(other, portable(root), result["bundle_id"], repo=ROOT,
                                for_stage=7)
        assert not applied["ok"]
        assert applied["failures"][target].split(":")[0] == {
            "missing": "missing", "corrupt": "mismatch", "conflict": "conflict"}[failure]
        if failure != "conflict":
            assert not (other / target).exists()
        check = verify_bundle(portable(root), result["bundle_id"])
        assert check["ok"] == (failure == "conflict")
    finally:
        store.path(digest).parent.mkdir(parents=True, exist_ok=True)
        store.path(digest).write_bytes(payload)
    assert verify_bundle(portable(root), result["bundle_id"])["ok"]


def test_stage7_entries_are_dropped_from_a_stage6_restore_verify(world, provenance, tmp_path):
    """Stage 3–6 behaviour is unchanged: a --for-stage 6 verify ignores Stage 7 state."""
    root, _, stage6, run = world
    result = prepare(root, ROOT, for_stage=7)
    other = tmp_path / "b"
    applied = restore_apply(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=6,
                            required_only=True)
    assert applied["ok"], applied["failures"]
    assert not (other / "artifacts/runs" / run.name).exists()
    assert not (other / "data/processed/stage7").exists()
    verified = restore_verify(other, portable(root), result["bundle_id"], repo=ROOT, for_stage=6)
    assert verified["verified"], verified["failed"]
    assert verified["stage7"] == {} and verified["stage6"]["verified_runs"] == [stage6.name]
