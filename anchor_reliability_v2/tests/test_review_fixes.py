"""Regression tests for Codex review findings on PR #5 (s02 resume receipts, s12 counts)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02_fetch_rollouts")
s12 = _load("s12_phase2_score")


def _fork(tmp_path: Path, rec: dict, rows: int = 3) -> Path:
    target = tmp_path / "1_correct/fork_0000.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({c: [None] * rows for c in s02.COLUMNS}), target)
    receipt = {"version": s02.RECEIPT_VERSION, "revision": s02.REVISION,
               "source_path": rec["path"], "source_sha256": rec["sha256"],
               "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "rows": rows, "columns": list(s02.COLUMNS)}
    s02.receipt_path(target).write_text(json.dumps(receipt))
    return target


def test_resume_skips_only_receipt_verified_forks(tmp_path):
    rec = {"path": "x/chunk_0/solutions.json", "sha256": "a" * 64}
    target = _fork(tmp_path, rec)
    assert s02.receipt_ok(target, rec)
    # A different source record, i.e. a fork written from another manifest, is not resumable.
    assert not s02.receipt_ok(target, {**rec, "sha256": "b" * 64})
    # Corrupt content no longer matches the receipt's Parquet hash.
    target.write_bytes(target.read_bytes() + b"x")
    assert not s02.receipt_ok(target, rec)
    # A fork without a receipt (written before receipts existed) is not resumable either.
    s02.receipt_path(target).unlink()
    assert not s02.receipt_ok(target, rec)


def test_quarantine_moves_files_aside_without_deleting(tmp_path):
    rec = {"path": "x/chunk_0/solutions.json", "sha256": "a" * 64}
    target = _fork(tmp_path, rec)
    s02.quarantine(target)
    assert not target.exists() and not s02.receipt_path(target).exists()
    moved = sorted(p.name for p in target.parent.iterdir())
    assert len(moved) == 2 and all(".quarantined-" in name for name in moved)


def test_common_counts_treat_a_missing_replicate_as_zero():
    ok = pd.DataFrame({"trajectory_id": ["t", "t", "t", "u", "u"],
                       "fork_idx": [0, 0, 1, 0, 0], "replicate": [1, 2, 1, 1, 2]})
    common = s12.common_counts(ok)
    assert common[("t", 1)] == 0  # replicate 2 has no OK samples: not 1
    assert common[("t", 0)] == 1 and common[("u", 0)] == 1
    # A planned fork with no OK rows in either replicate is listed with 0.
    forks = pd.DataFrame({"trajectory_id": ["t", "t", "u", "v"], "fork_idx": [0, 1, 0, 3]})
    assert s12.common_counts(ok, forks)[("v", 3)] == 0
