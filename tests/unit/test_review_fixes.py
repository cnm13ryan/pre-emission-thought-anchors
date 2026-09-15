"""Regression tests for the PR #3 review findings: receipt-crash recovery, nonconverged
selection, and restore path containment. (The stage 10 per-family layer fix is exercised by
the stage 10 pipeline tests on a torch-capable machine.)"""

import os

import pytest

from preanchor.backup.restore import safe_target
from preanchor.ids import file_hash
from preanchor.io import write_json
from preanchor.probes import stage9 as S
from preanchor.probes.select import _best


def _row(**kw):
    base = {"task": "binary", "average_precision": 0.5, "hyperparameter": 1.0,
            "layer": 1, "converged": True, "spearman": None, "rmse": None}
    return {**base, **kw}


def test_best_skips_nonconverged_candidates():
    winner_if_allowed = _row(average_precision=0.9, layer=1, converged=False)
    converged = _row(average_precision=0.5, layer=2)
    assert _best([winner_if_allowed, converged], "binary")["layer"] == 2


def test_best_refuses_when_every_candidate_is_nonconverged():
    with pytest.raises(ValueError, match="nonconverged"):
        _best([_row(converged=False)], "binary")


def test_safe_target_rejects_escaping_and_symlinked_paths(tmp_path):
    assert safe_target(tmp_path, "runs/a/b.json") == tmp_path / "runs/a/b.json"
    for bad in ("/etc/passwd", "../x", "a/../../x", "."):
        with pytest.raises(ValueError):
            safe_target(tmp_path, bad)
    (tmp_path / "real").mkdir()
    os.symlink(tmp_path / "real", tmp_path / "link")
    with pytest.raises(ValueError):
        safe_target(tmp_path, "link/f.txt")


@pytest.fixture
def receipted_run(tmp_path):
    """A run that crashed after the receipt (commit point) but before the manifest revision."""
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "predictions/test.parquet", {"p": 1})
    write_json(run / "metrics/test.json", {"aucpr": 0.1})
    receipt = {"run_id": run.name, "stage": S.TEST_STAGE, "attempt": "a1",
               "freeze_sha256": "f" * 64, "n_rows": 10, "n_positive": 2, "n_problems": 4,
               "predictions_sha256": file_hash(run / "predictions/test.parquet"),
               "metrics_sha256": file_hash(run / "metrics/test.json")}
    write_json(run / S.RECEIPT_FILE, receipt)
    manifest = {"output_hashes": {},
                "stages": {S.TEST_STAGE: {"state": "claimed",
                                          "attempts": [{"attempt": "a1"}]}}}
    return run, manifest


def test_receipt_crash_state_is_finalized_not_refused(receipted_run, monkeypatch):
    run, manifest = receipted_run
    saved = {}
    monkeypatch.setattr(S, "save_manifest", lambda r, m: saved.update(m))
    out = S._complete_from_receipt(run, manifest, "f" * 64)
    assert out["recovered"] is True
    stage = saved["stages"][S.TEST_STAGE]
    assert stage["state"] == "complete" and stage["recovered_from_receipt"] is True
    assert saved["output_hashes"][S.RECEIPT_FILE] == file_hash(run / S.RECEIPT_FILE)


def test_receipt_recovery_refuses_wrong_freeze_unclaimed_attempt_and_tampering(
        receipted_run, monkeypatch):
    run, manifest = receipted_run
    monkeypatch.setattr(S, "save_manifest", lambda r, m: None)
    with pytest.raises(S.TestLockError):
        S._complete_from_receipt(run, manifest, "e" * 64)
    unclaimed = {**manifest, "stages": {S.TEST_STAGE: {"state": "claimed", "attempts": []}}}
    with pytest.raises(S.TestLockError):
        S._complete_from_receipt(run, unclaimed, "f" * 64)
    (run / "metrics/test.json").unlink()
    with pytest.raises(S.TestLockError):
        S._complete_from_receipt(run, manifest, "f" * 64)
