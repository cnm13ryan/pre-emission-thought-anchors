"""SPEC §10.5 test lock: evaluate-test fails before the freeze, with a wrong or malformed
hash, after any post-freeze change to the selection, inputs, code or calibrator, a second
time, and with another run's freeze; an interrupted attempt has defined recovery."""

import hashlib
import sys
from pathlib import Path

import pytest

from preanchor.config import load_config
from preanchor.features.extract import write_current_json
from preanchor.io import json_bytes, read_json, validate_parquet, write_parquet
from preanchor.manifest import load_manifest
from preanchor.probes import stage9 as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from synthetic_pipeline import SOURCE, build_lineage  # noqa: E402


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
        config = load_config(ROOT / "configs/smoke.yaml")
        root, s7, s8 = build_lineage(tmp_path_factory.mktemp("lineage"), ROOT, config)
        yield config, root, s7, s8


@pytest.fixture
def frozen(lineage):
    config, root, s7, s8 = lineage
    run = S.prepare_run(root, ROOT, config, s8)
    S.calibrate(run, ROOT, config)
    return run, S.freeze(run, ROOT, config)["freeze_sha256"]


def sidecars(path: Path):
    return [Path(str(path) + s) for s in ("", ".sha256", ".schema.json", ".schema.json.sha256")]


def rewrite_parquet(path: Path, mutate):
    table = validate_parquet(path)
    rows = table.to_pylist()
    mutate(rows)
    originals = {p: p.read_bytes() for p in sidecars(path) if p.exists()}
    for p in originals:
        p.unlink()
    write_parquet(path, __import__("pyarrow").Table.from_pylist(rows, schema=table.schema))
    return originals


def restore(originals):
    for p, data in originals.items():
        p.write_bytes(data)


def test_refuses_before_freeze_and_malformed_or_wrong_hash(lineage, frozen):
    config, root, s7, s8 = lineage
    run, sha = frozen
    fresh = S.prepare_run(root, ROOT, config, s8)
    with pytest.raises(S.TestLockError, match="does not exist"):
        S.evaluate_test(fresh, ROOT, config, sha)
    S.calibrate(fresh, ROOT, config)
    with pytest.raises(S.TestLockError, match="does not exist"):
        S.evaluate_test(fresh, ROOT, config, sha)
    for bad in ("", "abc", sha.upper(), sha[:-1] + ("0" if sha[-1] != "0" else "1"),
                "0" * 64):
        with pytest.raises(S.TestLockError, match="exact lowercase|does not equal"):
            S.evaluate_test(run, ROOT, config, bad)
    with pytest.raises(S.TestLockError, match="required"):
        S.evaluate_test(run, ROOT, config, None)
    assert not (run / "predictions/test.parquet").exists()
    assert S.TEST_STAGE not in load_manifest(run)["stages"]


def test_freeze_hash_tracks_selection_relevant_fields(frozen):
    run, sha = frozen
    record = read_json(run / S.FREEZE_FILE)
    assert hashlib.sha256(json_bytes(record)).hexdigest() == sha   # stable serialisation
    changed = dict(record)
    changed["binary"] = {**record["binary"], "C": 12345.0}
    assert hashlib.sha256(json_bytes(changed)).hexdigest() != sha
    changed = dict(record)
    changed["anchor_threshold"] = 0.6
    assert hashlib.sha256(json_bytes(changed)).hexdigest() != sha


def test_unrelated_run_cannot_use_another_freeze(lineage, frozen):
    config, root, s7, s8 = lineage
    run_a, sha_a = frozen
    run_b = S.prepare_run(root, ROOT, config, s8)
    S.calibrate(run_b, ROOT, config)
    sha_b = S.freeze(run_b, ROOT, config)["freeze_sha256"]
    assert sha_a != sha_b
    with pytest.raises(S.TestLockError, match="does not equal"):
        S.evaluate_test(run_a, ROOT, config, sha_b)
    # Copying B's freeze into A: the file hash is right for the argument but not for A.
    for suffix in ("", ".sha256"):
        (run_a / (S.FREEZE_FILE + suffix)).write_bytes(
            (run_b / (S.FREEZE_FILE + suffix)).read_bytes())
    with pytest.raises(S.TestLockError, match="not the one recorded"):
        S.evaluate_test(run_a, ROOT, config, sha_b)


def test_code_change_after_freeze_is_refused(lineage, frozen, monkeypatch):
    config, root, s7, s8 = lineage
    run, sha = frozen
    monkeypatch.setattr("preanchor.manifest.provenance",
                        lambda repo: ({**SOURCE, "git_sha": "d" * 40}, b""))
    with pytest.raises(S.TestLockError, match="identity check failed"):
        S.evaluate_test(run, ROOT, config, sha)


def test_calibrator_change_after_freeze_is_refused(lineage, frozen):
    config, root, s7, s8 = lineage
    run, sha = frozen
    path = run / S.CALIBRATOR_FILE
    original = {p: p.read_bytes() for p in (path, Path(str(path) + ".sha256"))}
    record = read_json(path)
    record["parameters"]["A"] += 1.0
    write_current_json(path, record)
    with pytest.raises(S.TestLockError, match="calibrator.json changed after the freeze"):
        S.evaluate_test(run, ROOT, config, sha)
    restore(original)
    assert not S.validate_stage9(run, ROOT, config)["problems"]


def test_selection_change_after_freeze_invalidates(lineage, frozen):
    config, root, s7, s8 = lineage
    run, sha = frozen
    path = s8 / S.PENDING_FILE
    original = {p: p.read_bytes() for p in (path, Path(str(path) + ".sha256"))}
    record = read_json(path)
    record["binary"]["layer"] = (record["binary"]["layer"] + 1) % 5
    write_current_json(path, record)
    with pytest.raises(S.TestLockError, match="no longer verify"):
        S.evaluate_test(run, ROOT, config, sha)
    report = S.validate_stage9(run, ROOT, config)
    assert not report["ok"] and not report["frozen"]
    restore(original)
    assert S.validate_stage9(run, ROOT, config)["ok"]


def test_input_change_after_freeze_is_refused(lineage, frozen):
    config, root, s7, s8 = lineage
    run, sha = frozen
    labels = root / "data/processed/stage4/labels.parquet"

    def bump(rows):
        rows[0]["importance_log1p"] += 1.0

    originals = rewrite_parquet(labels, bump)
    try:
        with pytest.raises(S.TestLockError, match="no longer verify"):
            S.evaluate_test(run, ROOT, config, sha)
    finally:
        restore(originals)
    assert S.validate_stage9(run, ROOT, config)["ok"]


def test_interrupted_attempt_recovers_and_stays_exactly_once(lineage, frozen, monkeypatch):
    config, root, s7, s8 = lineage
    run, sha = frozen
    # Crash after the claim, before any output: state failed, no outputs.
    monkeypatch.setattr(S, "split_metrics", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("simulated crash")))
    with pytest.raises(RuntimeError, match="simulated"):
        S.evaluate_test(run, ROOT, config, sha)
    state = S.test_state(run, load_manifest(run))
    assert state["state"] == "failed" and not state["receipt"]
    assert not any(state["outputs"].values()) and len(state["attempts"]) == 1
    monkeypatch.undo()
    # Crash after predictions were written, before the receipt.
    original = S.write_json

    def crash_on_receipt(path, value):
        if path.name == "test_receipt.json":
            raise RuntimeError("simulated crash before receipt")
        return original(path, value)

    monkeypatch.setattr(S, "write_json", crash_on_receipt)
    with pytest.raises(RuntimeError, match="before receipt"):
        S.evaluate_test(run, ROOT, config, sha)
    monkeypatch.undo()
    state = S.test_state(run, load_manifest(run))
    assert state["outputs"]["predictions/test.parquet"] and not state["receipt"]
    assert len(state["attempts"]) == 2
    interrupted = S.validate_stage9(run, ROOT, config)
    assert interrupted["ok"] and interrupted["test_summary"] == {"interrupted_claim": True,
                                                                  "attempts": 2}
    partial = (run / "predictions/test.parquet").read_bytes()
    # Retry under the same freeze reproduces identical bytes and publishes once.
    out = S.evaluate_test(run, ROOT, config, sha)
    assert (run / "predictions/test.parquet").read_bytes() == partial
    assert len(out["receipt"]["attempts"]) == 3 and out["receipt"]["attempts"][2]["resumed"]
    with pytest.raises(S.TestLockError, match="already evaluated"):
        S.evaluate_test(run, ROOT, config, sha)
    report = S.validate_stage9(run, ROOT, config)
    assert report["ok"] and report["test_summary"]["attempts"] == 3


def test_test_outputs_without_a_claim_are_refused(lineage, frozen):
    config, root, s7, s8 = lineage
    run, sha = frozen
    (run / "predictions/test.parquet").write_bytes(b"planted")
    with pytest.raises(S.TestLockError, match="without a recorded claim"):
        S.evaluate_test(run, ROOT, config, sha)


def test_calibrator_is_only_fitted_inside_calibrate():
    src = (ROOT / "src/preanchor").rglob("*.py")
    callers = [p for p in src if "fit_platt(" in p.read_text() and p.name != "calibrate.py"]
    assert [p.name for p in callers] == ["stage9.py"]
    text = (ROOT / "src/preanchor/probes/stage9.py").read_text()
    body = text.split("def calibrate(")[1].split("\ndef ")[0]
    assert text.count("fit_platt(") == 1 and "fit_platt(" in body
    assert "TEST" not in body.replace("TEST_STAGE", "")
