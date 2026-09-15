"""Portable backup of the Stage 10 closure: classification of the canonical run on the
canonical Stage 9 run, round trip into a new root and read-only verification."""

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
from preanchor.probes import stage9 as S9
from preanchor.probes import stage10 as S10
from preanchor.probes.validate import validate_probes

sys.path.insert(0, str(ROOT / "tests/fixtures"))
from synthetic_pipeline import relabel  # noqa: E402
from tiny_qwen2 import write_synthetic_root  # noqa: E402


def portable(root: Path) -> Path:
    return root / "artifacts/portable"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("region-a") / "preanchor"
    with pytest.MonkeyPatch.context() as monkeypatch:
        fake_provenance(monkeypatch)
        original = S10.clustered_bootstrap
        monkeypatch.setattr(S10, "clustered_bootstrap",
                            lambda **kw: original(**{**kw, "replicates": 20}))
        write_synthetic_root(root, load_config(ROOT / "configs/mvp.yaml"))
        relabel(root, train_quantile_threshold=True)
        make_stage4_runs(root)
        make_production(root, units=2)
        make_stage6(root)
        stage7 = make_stage7(root)
        config = load_config(ROOT / "configs/smoke.yaml")
        stage8 = fit.prepare_run(root, ROOT, config, stage7, "fit")
        fit.execute(stage8, ROOT, config, tracking=False, workers=2)
        select.execute(stage8, ROOT, config, workers=2)
        validate_probes(stage8, config, root)
        stage9 = S9.prepare_run(root, ROOT, config, stage8)
        S9.calibrate(stage9, ROOT, config)
        sha = S9.freeze(stage9, ROOT, config)["freeze_sha256"]
        S9.evaluate_test(stage9, ROOT, config, sha)
        stage10 = S10.prepare_run(root, ROOT, config, stage9)
        S10.execute(stage10, ROOT, config, workers=2)
        # An interrupted Stage 10 run stays historical evidence.
        partial = S10.prepare_run(root, ROOT, config, stage9)
        with pytest.raises(KeyboardInterrupt):
            S10.execute(partial, ROOT, config, workers=2, interrupt_after=1)
    return root, stage9, stage10, partial


@pytest.fixture
def provenance(world, monkeypatch):
    from backup_fixtures import grant_writer_lease

    fake_provenance(monkeypatch)
    grant_writer_lease(world[0], monkeypatch)


def test_stage10_closure_classification_and_identity(world, provenance):
    root, stage9, stage10, partial = world
    assert stage_closure(10) == (3, 4, 5, 6, 7, 8, 9, 10)
    summary, inventory = plan(root, ROOT, for_stage=10)
    assert summary["blockers"] == [], summary["blockers"]
    assert summary["kind"] == "canonical" and summary["highest_protected_stage"] == 10
    s10 = summary["stages"]["stage10"]
    assert s10["status"] == "complete" and s10["run_id"] == stage10.name
    assert s10["stage9_run_id"] == stage9.name
    assert s10["freeze_sha256"] == file_hash(stage9 / "frozen_selection.json")
    assert s10["n_units"] == len(S10.unit_specs(S10.check_spec(
        load_config(ROOT / "configs/smoke.yaml"))))
    assert summary["runs"][stage10.name] == {"role": "stage10_production", "stage": "stage10"}
    assert summary["runs"][partial.name]["reason"] == "stage10_sensitivities_failed"
    paths = {e.path: e for e in inventory.entries}
    p10 = f"artifacts/runs/{stage10.name}"
    for logical, validation in ((f"{p10}/metrics/sensitivities.parquet", "parquet"),
                                (f"{p10}/metrics/permutations.parquet", "parquet"),
                                (f"{p10}/predictions/lead_time.parquet", "parquet"),
                                (f"{p10}/figures/permutation_null.html", "sha256_sidecar"),
                                (f"{p10}/figures/lead_time.json", "sha256_sidecar"),
                                (f"{p10}/stages/stage10/summary.json", "sha256_sidecar")):
        entry = paths[logical]
        assert entry.classification == MUST_BACK_UP and entry.required, logical
        assert entry.validation == validation, logical
    units = [e for p, e in paths.items() if p.startswith(f"{p10}/stages/stage10/units/")
             and p.endswith("/rows.parquet")]
    assert len(units) == s10["n_units"] and all(e.validation == "unit_file" for e in units)
    assert inventory.scientific_identity["stage10"]["stage9_run_id"] == stage9.name
    # A Stage 9 bundle carries Stage 10 as unrequired.
    nine, inv9 = plan(root, ROOT, for_stage=9)
    assert nine["blockers"] == [] and nine["highest_protected_stage"] == 9
    assert not any(e.required for e in inv9.entries if e.stage == "stage10")
    assert any(e.required for e in inv9.entries if e.stage == "stage9")


def test_round_trip_restores_stage10_and_verifies_read_only(world, provenance, tmp_path):
    root, stage9, stage10, partial = world
    result = prepare(root, ROOT, for_stage=10)
    assert result["highest_protected_stage"] == 10
    name, manifest = load_bundle(portable(root), result["bundle_id"])
    assert manifest["bundle"]["required_stages"] == [3, 4, 5, 6, 7, 8, 9, 10]
    other = tmp_path / "region-b/preanchor"
    other.mkdir(parents=True)
    applied = restore_apply(other, portable(root), name, repo=ROOT, for_stage=10)
    assert applied["ok"], applied["failures"]
    # Operational files (run lock, unit locks, heartbeat) are not part of the bundle.
    files = [p for p in stage10.rglob("*") if p.is_file() and ".run.lock" not in p.name
             and "/.locks/" not in p.as_posix() and p.name != "heartbeat.json"]
    restored = other / "artifacts/runs" / stage10.name
    for path in files:
        assert (restored / path.relative_to(stage10)).read_bytes() == path.read_bytes(), path
    verified = restore_verify(other, portable(root), name, repo=ROOT, for_stage=10)
    assert verified["verified"], verified["failed"]
    assert verified["stage9"]["verified"] and verified["stage10"]["verified"]
    assert "RESTORE VERIFIED" in render(verified)
    for path in files:   # verification never touched restored bytes
        assert (restored / path.relative_to(stage10)).read_bytes() == path.read_bytes(), path
    # Tampering with a restored sensitivity table fails the gate.
    (restored / "metrics/sensitivities.parquet").write_bytes(b"corrupt")
    tampered = restore_verify(other, portable(root), name, repo=ROOT, for_stage=10)
    assert not tampered["verified"]
