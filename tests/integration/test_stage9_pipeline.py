"""Stage 9 end to end on the synthetic lineage: validation scoring, fitted validation-only
calibration, registry, freeze, exactly-once test evaluation, validation report, and the
byte-determinism the exactly-once protocol relies on."""

import sys
from pathlib import Path

import numpy as np
import pytest

from preanchor.config import config_hash, load_config
from preanchor.io import read_json, validate_artifact, validate_parquet
from preanchor.manifest import load_manifest
from preanchor.probes import stage9 as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from synthetic_pipeline import SOURCE, TEST, VALIDATION, build_lineage  # noqa: E402


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
        config = load_config(ROOT / "configs/smoke.yaml")
        root, s7, s8 = build_lineage(tmp_path_factory.mktemp("lineage"), ROOT, config)
        yield config, root, s7, s8


def stage8_fingerprint(s8):
    return {n: validate_artifact(s8 / n) for n in (S.PENDING_FILE, S.MODEL_FILE, *S.PROBE_TABLES)}


def test_calibrate_freeze_evaluate_once(lineage, monkeypatch):
    config, root, s7, s8 = lineage
    before = stage8_fingerprint(s8)
    seen = []
    original = S.fit_platt

    def spy(score, y):
        seen.append((np.array(score), np.array(y)))
        return original(score, y)

    monkeypatch.setattr(S, "fit_platt", spy)
    run = S.prepare_run(root, ROOT, config, s8)
    manifest = load_manifest(run)
    assert manifest["input_hashes"]["probe_run_id"] == s8.name
    assert manifest["input_hashes"][S.PENDING_FILE] == before[S.PENDING_FILE]
    with pytest.raises(S.TestLockError, match="does not exist"):
        S.evaluate_test(run, ROOT, config, "0" * 64)
    result = S.calibrate(run, ROOT, config)
    cal = result["calibrator"]
    val = validate_parquet(run / "predictions/validation.parquet")
    n_val = len(set(val.column("sentence_id").to_pylist()))
    assert set(val.column("split").to_pylist()) == {"validation"}
    assert set(val.column("problem_id").to_pylist()) == set(VALIDATION)
    assert cal["fitted"] and cal["n_positive"] == 24 and cal["n_rows"] == n_val
    assert len(seen) == 1 and len(seen[0][1]) == n_val and seen[0][1].sum() == 24
    assert cal["validation_rows_sha256"] == config_hash(
        sorted({s for s in val.column("sentence_id").to_pylist()}, key=lambda s: s) and
        [r["sentence_id"] for r in sorted(
            {r["sentence_id"]: r for r in val.to_pylist()}.values(),
            key=lambda r: (r["trajectory_id"], int(r["sentence_id"].rsplit(":", 1)[1])))])
    calibrated = [p for p in val.column("probability_calibrated").to_pylist() if p is not None]
    assert calibrated and min(calibrated) >= 0 and max(calibrated) <= 1
    metrics = result["metrics"]
    assert metrics["split"] == "validation" and metrics["stage"] == "exploratory"
    assert metrics["primary"]["model"] == "activation_plus_observables:binary"
    assert metrics["bootstrap"]["replicates_requested"] == 2000
    assert "calibrated" in metrics["models"]["activation_plus_observables:binary"]
    assert (run / "predictions/test.parquet").exists() is False
    # Validation diagnostics never touch the Stage 8 selection.
    assert stage8_fingerprint(s8) == before
    with pytest.raises(ValueError, match="once per run"):
        S.calibrate(run, ROOT, config)
    with pytest.raises(S.TestLockError, match="does not exist"):
        S.evaluate_test(run, ROOT, config, "0" * 64)
    frozen = S.freeze(run, ROOT, config)
    sha = frozen["freeze_sha256"]
    assert validate_artifact(run / S.FREEZE_FILE) == sha
    record = read_json(run / S.FREEZE_FILE)
    pending = read_json(s8 / S.PENDING_FILE)
    assert record["binary"] == pending["binary"] and record["continuous"] == pending["continuous"]
    assert record["anchor_threshold"] == pending["anchor_threshold"] == 0.5
    assert record["calibration"]["fitted"] and record["calibration"]["parameters"]["A"]
    assert record["split"]["test"] == TEST and record["input_hashes"] == manifest["input_hashes"]
    registry = read_json(run / S.REGISTRY_FILE)
    assert validate_artifact(run / S.REGISTRY_FILE) == record["analysis_registry_sha256"]
    assert registry["primary_endpoint"]["metric"] == "average_precision"
    assert all("value" not in a for a in registry["analyses"])
    confirmatory = [a for a in registry["analyses"] if a["confirmatory"]]
    assert {a["id"] for a in confirmatory} >= {"primary_test_aucpr", "paired_baseline_differences"}
    assert any(a["stage"] == "stage10" and not a["confirmatory"] for a in registry["analyses"])
    with pytest.raises(ValueError, match="already frozen"):
        S.freeze(run, ROOT, config)
    # The single evaluation.
    out = S.evaluate_test(run, ROOT, config, sha)
    receipt = out["receipt"]
    assert receipt["freeze_sha256"] == sha and receipt["n_problems"] == 4
    test = validate_parquet(run / "predictions/test.parquet")
    assert set(test.column("split").to_pylist()) == {"test"}
    assert set(test.column("problem_id").to_pylist()) == set(TEST)
    assert not set(test.column("sentence_id").to_pylist()) & set(
        val.column("sentence_id").to_pylist())
    assert test.schema.equals(S.EVAL_SCHEMA)
    tm = read_json(run / "metrics/test.json")
    assert tm["partition"]["anchor_threshold"] == 0.5 and tm["partition"]["n_problems"] == 4
    assert tm["freeze_sha256"] == sha and tm["bootstrap"]["n_problems"] == 4
    assert tm["bootstrap"]["replicates_requested"] == 2000
    assert set(tm["paired_differences_point"]) == {
        f"activation_plus_observables-minus-{f}" for f in (
            "prevalence", "position", "context", "uncertainty", "observable_combined",
            "activation_only", "residualized_activation")}
    with pytest.raises(S.TestLockError, match="already evaluated"):
        S.evaluate_test(run, ROOT, config, sha)
    report = S.validate_stage9(run, ROOT, config)
    assert report["ok"], report["problems"]
    assert report["test"]["receipt"] and report["test_summary"]["attempts"] == 1
    assert stage8_fingerprint(s8) == before
    for name in S.STAGE10_OUTPUTS:
        assert not (run / name).exists()


def test_outputs_are_byte_deterministic_across_runs(lineage):
    config, root, s7, s8 = lineage
    runs = []
    for _ in range(2):
        run = S.prepare_run(root, ROOT, config, s8)
        S.calibrate(run, ROOT, config)
        runs.append(run)
    a, b = (validate_parquet(r / "predictions/validation.parquet") for r in runs)
    assert a.equals(b)
    ma, mb = (read_json(r / "metrics/validation.json") for r in runs)
    for m in (ma, mb):
        m.pop("run_id")
        m["calibration"].pop("run_id")
    assert ma == mb
