"""Stage 8/9 portable backup: classification, identity, round trip and fail-closed verify."""

import sys
from pathlib import Path

import pytest
from backup_fixtures import (
    ROOT,
    fake_provenance,
    make_production,
    make_stage4_runs,
    make_stage6,
    make_stage7,
)

from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import MUST_BACK_UP, stage_closure
from preanchor.backup.prepare import plan, prepare
from preanchor.backup.restore import restore_apply
from preanchor.backup.verify import render, restore_verify
from preanchor.config import load_config
from preanchor.ids import file_hash
from preanchor.probes import fit, select
from preanchor.probes import stage9 as S
from preanchor.probes.validate import validate_probes

sys.path.insert(0, str(ROOT / "tests/fixtures"))
from synthetic_pipeline import relabel  # noqa: E402
from tiny_qwen2 import write_synthetic_root  # noqa: E402


def portable(root: Path) -> Path:
    return root / "artifacts/portable"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Stage 3-7 synthetic world (mvp configuration, tiny model) plus a finalized Stage 8
    run and a frozen, test-evaluated Stage 9 run under the smoke configuration."""
    root = tmp_path_factory.mktemp("region-a") / "preanchor"
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake_provenance(monkeypatch)
        write_synthetic_root(root, load_config(ROOT / "configs/mvp.yaml"))
        relabel(root)
        make_stage4_runs(root)
        make_production(root, units=2)
        stage6 = make_stage6(root)
        stage7 = make_stage7(root)
        config = load_config(ROOT / "configs/smoke.yaml")
        stage8 = fit.prepare_run(root, ROOT, config, stage7, "fit")
        fit.execute(stage8, ROOT, config, tracking=False, workers=2)
        select.execute(stage8, ROOT, config, workers=2)
        validate_probes(stage8, config, root)
        stage9 = S.prepare_run(root, ROOT, config, stage8)
        S.calibrate(stage9, ROOT, config)
        sha = S.freeze(stage9, ROOT, config)["freeze_sha256"]
        S.evaluate_test(stage9, ROOT, config, sha)
        # A second, unfrozen Stage 9 run and a smoke-sized Stage 8 fit stay historical.
        unfrozen = S.prepare_run(root, ROOT, config, stage8)
        S.calibrate(unfrozen, ROOT, config)
        smoke8 = fit.prepare_run(root, ROOT, config, stage7, "fit")
        fit.execute(smoke8, ROOT, config, tracking=False, workers=2, limit_sites=2)
        select.execute(smoke8, ROOT, config, workers=2)
        validate_probes(smoke8, config, root)
    return root, stage6, stage7, stage8, stage9, unfrozen, smoke8


@pytest.fixture
def provenance(world, monkeypatch):
    from backup_fixtures import grant_writer_lease

    fake_provenance(monkeypatch)
    grant_writer_lease(world[0], monkeypatch)   # prepare is gated by portable-store:writer


def test_stage9_closure_classification_and_identity(world, provenance):
    root, stage6, stage7, stage8, stage9, unfrozen, smoke8 = world
    assert stage_closure(9) == (3, 4, 5, 6, 7, 8, 9)
    summary, inventory = plan(root, ROOT, for_stage=9)
    assert summary["blockers"] == [], summary["blockers"]
    assert summary["kind"] == "canonical" and summary["highest_protected_stage"] == 9
    s8, s9 = summary["stages"]["stage8"], summary["stages"]["stage9"]
    assert s8["status"] == "complete" and s8["run_id"] == stage8.name
    assert s8["n_units"] == 5 + 2 * 5 + 1 and s8["binary"]["probe_name"] == \
        "activation_plus_observables"
    assert s9["status"] == "complete" and s9["run_id"] == stage9.name
    assert s9["freeze_sha256"] == file_hash(stage9 / "frozen_selection.json")
    assert s9["test_evaluated"] and s9["test_attempts"] == 1 and s9["calibration_fitted"]
    assert s9["test_receipt"]["n_problems"] == 4
    assert summary["runs"][stage9.name] == {"role": "stage9_production", "stage": "stage9"}
    assert summary["runs"][unfrozen.name]["reason"].startswith("stage9_calibrate_complete_freeze")
    assert summary["runs"][smoke8.name]["reason"] == "site_count_differs_from_stage7"
    paths = {e.path: e for e in inventory.entries}
    p8, p9 = f"artifacts/runs/{stage8.name}", f"artifacts/runs/{stage9.name}"
    for logical, validation in ((f"{p8}/probe_models/cv_results.parquet", "parquet"),
                                (f"{p8}/probe_models/selected_model.joblib", "sha256_sidecar"),
                                (f"{p8}/probe_models/frozen_selection.pending.json",
                                 "sha256_sidecar"),
                                (f"{p8}/probe_models/probe_validation.json", "sha256_sidecar"),
                                (f"{p8}/stages/stage8/summary.json", "sha256_sidecar"),
                                (f"{p9}/frozen_selection.json", "sha256_sidecar"),
                                (f"{p9}/frozen_selection.json.sha256", "bytes"),
                                (f"{p9}/metrics/analysis_registry.json", "sha256_sidecar"),
                                (f"{p9}/probe_models/calibrator.json", "sha256_sidecar"),
                                (f"{p9}/predictions/validation.parquet", "parquet"),
                                (f"{p9}/predictions/test.parquet", "parquet"),
                                (f"{p9}/metrics/test.json", "sha256_sidecar"),
                                (f"{p9}/metrics/test_receipt.json", "sha256_sidecar")):
        entry = paths[logical]
        assert entry.classification == MUST_BACK_UP and entry.required, logical
        assert entry.validation == validation, logical
    units = [e for p, e in paths.items() if p.startswith(f"{p8}/probe_models/units/")
             and p.endswith("/cv_results.parquet")]
    assert len(units) == 16 and all(e.validation == "unit_file" for e in units)
    identity = inventory.scientific_identity
    assert identity["stage9"]["probe_run_id"] == stage8.name
    assert identity["stage8"]["pending_sha256"] == file_hash(
        stage8 / "probe_models/frozen_selection.pending.json")
    # A Stage 8 bundle carries Stage 9 as unrequired; a Stage 7 bundle ignores both.
    eight, inv8 = plan(root, ROOT, for_stage=8)
    assert eight["blockers"] == [] and eight["highest_protected_stage"] == 8
    assert not any(e.required for e in inv8.entries if e.stage == "stage9")
    assert any(e.required for e in inv8.entries if e.stage == "stage8")
    seven, inv7 = plan(root, ROOT, for_stage=7)
    assert seven["blockers"] == [] and not any(
        e.required for e in inv7.entries if e.stage in ("stage8", "stage9"))


def test_round_trip_restores_stage8_9_and_verifies_read_only(world, provenance, tmp_path):
    root, stage6, stage7, stage8, stage9, unfrozen, smoke8 = world
    result = prepare(root, ROOT, for_stage=9)
    assert result["highest_protected_stage"] == 9
    name, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["bundle"]["required_stages"] == [3, 4, 5, 6, 7, 8, 9]
    other = tmp_path / "region-b/preanchor"
    other.mkdir(parents=True)
    applied = restore_apply(other, portable(root), name, repo=ROOT, for_stage=9)
    assert applied["ok"], applied["failures"]
    files = [p for p in stage9.rglob("*") if p.is_file() and ".run.lock" not in p.name]
    restored = other / "artifacts/runs" / stage9.name
    for path in files:
        assert (restored / path.relative_to(stage9)).read_bytes() == path.read_bytes(), path
    verified = restore_verify(other, portable(root), name, repo=ROOT, for_stage=9)
    assert verified["verified"], verified["failed"]
    assert verified["stage8"]["verified"] and verified["stage9"]["verified"]
    assert verified["stage9"]["test_evaluated"]
    text = render(verified)
    assert "RESTORE VERIFIED" in text
    for path in files:   # verification never touched restored bytes
        assert (restored / path.relative_to(stage9)).read_bytes() == path.read_bytes(), path
    assert not (restored / "stages/stage9/validation.json").exists() or (
        restored / "stages/stage9/validation.json").read_bytes() == (
        stage9 / "stages/stage9/validation.json").read_bytes()
    # Tampering with the restored freeze fails the gate.
    (restored / "frozen_selection.json").write_bytes(b"{}")
    tampered = restore_verify(other, portable(root), name, repo=ROOT, for_stage=9)
    assert not tampered["verified"]


def test_unchanged_prepare_reuses_objects_and_second_stage9_blocks(world, provenance):
    root, stage6, stage7, stage8, stage9, unfrozen, smoke8 = world
    first = prepare(root, ROOT, for_stage=9)
    second = prepare(root, ROOT, for_stage=9)
    assert second["objects_created"] == 0 and second["objects_reused"] == first["unique_objects"]
    config = load_config(ROOT / "configs/smoke.yaml")
    S.freeze(unfrozen, ROOT, config)
    try:
        summary, _ = plan(root, ROOT, for_stage=9)
        assert any("both frozen Stage 9 runs" in b for b in summary["blockers"])
    finally:
        pass
