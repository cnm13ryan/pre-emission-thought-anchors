"""Stage 4 end to end on fixtures: raw units resume/quarantine, audit gate, label build."""

import hashlib
import json
from pathlib import Path

import pytest

from preanchor.config import load_config
from preanchor.io import UnitStore, validate_parquet
from preanchor.labels import stage4
from preanchor.labels.stage4 import (
    audit_4682,
    build_labels,
    fetch_raw_file,
    raw_unit_key,
    run_raw_fetch,
    validate_raw_download,
)
from preanchor.manifest import create_run, load_manifest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95"


def raw_record(problem, kind, fork, payload):
    return {"path": f"{PREFIX}/{kind}_base_solution/problem_{problem}/chunk_{fork}/solutions.json",
            "size": len(payload), "git_blob_sha1": "a" * 40,
            "lfs_sha256": hashlib.sha256(payload).hexdigest(), "role": "raw_resamples",
            "problem_id": problem, "trace_kind": kind}


def make_download(tmp_path, payloads):
    calls = []

    def download(*, repo_id, repo_type, revision, filename, cache_dir):
        calls.append(filename)
        target = tmp_path / "hfcache" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[filename])
        return target

    return download, calls


def test_raw_unit_resume_and_quarantine(tmp_path):
    payload = json.dumps([{"answer": "1"}]).encode()
    record = raw_record(4682, "correct", 0, payload)
    discovery = {"dataset_id": "pinned/dataset", "revision": "b" * 40}
    download, calls = make_download(tmp_path, {record["path"]: payload})
    store = UnitStore(tmp_path / "units")
    args = (store, record, discovery, {"code": "fixed"}, tmp_path / "cache")
    first = fetch_raw_file(*args, download=download)
    unit = Path(first["local_path"])
    before = (unit.read_bytes(), unit.stat().st_mtime_ns)
    assert not fetch_raw_file(*args, download=download)["created"]
    assert (unit.read_bytes(), unit.stat().st_mtime_ns) == before and len(calls) == 1
    unit.write_bytes(b"corrupt")
    assert fetch_raw_file(*args, download=download)["created"]
    assert len(list((store.root / "quarantine").iterdir())) == 1
    assert raw_unit_key(record, discovery, {"code": "changed"}) != first["unit_key"]


def test_raw_unit_rejects_upstream_hash_mismatch(tmp_path):
    payload = json.dumps([{"answer": "1"}]).encode()
    record = raw_record(4682, "correct", 0, payload)
    download, _ = make_download(tmp_path, {record["path"]: payload + b" "})
    with pytest.raises(ValueError, match="size mismatch"):
        fetch_raw_file(UnitStore(tmp_path / "units"), record, {"dataset_id": "d", "revision": "r"},
                       {}, tmp_path / "cache", download=download)


@pytest.fixture
def fixture_world(tmp_path, monkeypatch):
    """A tiny 40-trajectory world: Stage 3 tables, raw discovery and fake provenance."""
    config = load_config(ROOT / "configs/mvp.yaml")
    source = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
              "package_freeze": []}
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (source.copy(), b""))
    monkeypatch.setattr("preanchor.labels.stage4.start_tracking",
                        lambda *a, **k: (None, {"mode": "disabled_for_test"}))
    payloads, records, forks = {}, [], {}
    sentences, trajectories = [], []
    for kind in ("correct", "incorrect"):
        for problem in config.dataset.problem_ids:
            trajectory_id = f"{problem}:{kind}"
            n = 4
            forks[trajectory_id] = n
            trace = "<think>\n" + " ".join(f"S{i}." for i in range(n))
            trajectories.append({"trajectory_id": trajectory_id, "problem_id": problem,
                                 "trace_kind": kind, "full_cot": trace,
                                 "full_cot_sha256": hashlib.sha256(trace.encode()).hexdigest()})
            for i in range(n):
                text = f"S{i}."
                start = trace.index(text)
                sentences.append({"schema_version": 1, "sentence_id": f"{trajectory_id}:{i}",
                                  "trajectory_id": trajectory_id, "problem_id": problem,
                                  "trace_kind": kind, "sentence_idx": i, "text": text,
                                  "text_sha256": "x", "char_start": start,
                                  "char_end": start + len(text), "alignment_method": "exact",
                                  "alignment_ok": True, "eligible": True,
                                  "exclusion_reason": None,
                                  # Released label rises with i so ranks are comparable.
                                  "importance_released": 0.1 * (i + 1) * (problem % 7 + 1),
                                  "importance_legacy": None, "importance_canonical": None,
                                  "importance_primary": None, "label_source": "released"})
                samples = []
                for s in range(6):
                    # Later forks converge to answer "1"; earlier forks are mixed.
                    answer = "1" if (s + i) % 2 == 0 or i >= 2 else str(2 + s + problem % 3)
                    samples.append({"chunk_removed": text,
                                    "chunk_resampled": "same" if s == 5 else f"r{i}{s}{problem}",
                                    "answer": answer, "is_correct": answer == "1"})
                payload = json.dumps(samples).encode()
                record = raw_record(problem, kind, i, payload)
                payloads[record["path"]] = payload
                records.append(record)
    discovery = {"schema_version": 1, "dataset_id": config.dataset.id,
                 "revision": config.dataset.revision, "forks_per_trajectory": forks,
                 "files": sorted(records, key=lambda r: r["path"])}
    monkeypatch.setattr("preanchor.labels.stage4.load_raw_discovery", lambda *a: discovery)
    root = tmp_path / "root"
    stage3 = root / "data/processed/stage3"
    stage3.mkdir(parents=True)
    import pyarrow as pa

    from preanchor.io import write_parquet

    write_parquet(stage3 / "trajectories.parquet", pa.Table.from_pylist(trajectories))
    write_parquet(stage3 / "sentences.parquet", pa.Table.from_pylist(sentences))

    class FakeSimilarity:
        def __init__(self, *, device="cpu"):
            self.device = device

        def scores(self, rows):
            return [0.95 if r["chunk_resampled"] == "same" else 0.2 for r in rows]

        def clear(self):
            pass

        def identity(self):
            return {"model_id": "fake", "device": self.device}

    monkeypatch.setattr(stage4, "ReplacementSimilarity", FakeSimilarity)
    download, calls = make_download(tmp_path, payloads)
    return {"config": config, "root": root, "download": download, "calls": calls,
            "discovery": discovery, "records": records}


def test_raw_fetch_stage_resumes_and_isolates_inputs(fixture_world, tmp_path):
    world = fixture_world
    config, root = world["config"], world["root"]
    run = create_run(root, ROOT, config, ["fixture"])

    def flaky(**kwargs):
        if len(world["calls"]) >= 5:
            raise TimeoutError("synthetic network timeout")
        return world["download"](**kwargs)

    args = (run, ROOT, config, tmp_path / "cache")
    with pytest.raises(TimeoutError):
        run_raw_fetch(*args, tracking=False, download=flaky, workers=2)
    manifest = load_manifest(run)
    assert manifest["stages"]["stage4_raw"]["state"] == "failed"
    assert (run / "failure.json").is_file()
    result = run_raw_fetch(*args, resume=True, tracking=False, download=world["download"],
                           workers=3)
    assert result["state"] == "passed" and result["files"] == len(world["records"])
    assert result["created"] + result["skipped"] == len(world["records"]) and result["skipped"] >= 1
    again = run_raw_fetch(*args, resume=True, tracking=False, download=world["download"])
    assert again["created"] == 0 and again["skipped"] == len(world["records"])
    assert validate_raw_download(run, ROOT, config)["state"] == "passed"
    with pytest.raises(ValueError, match="requires --resume"):
        run_raw_fetch(*args, tracking=False, download=world["download"])
    world["discovery"]["revision"] = "changed"
    with pytest.raises(ValueError, match="Changed raw discovery|Dataset identity"):
        run_raw_fetch(*args, resume=True, tracking=False, download=world["download"])


def test_audit_and_build_on_fixture_world(fixture_world, tmp_path):
    world = fixture_world
    config, root = world["config"], world["root"]
    raw_run = create_run(root, ROOT, config, ["raw"])
    run_raw_fetch(raw_run, ROOT, config, tmp_path / "cache", tracking=False,
                  download=world["download"], workers=4)
    label_run = create_run(root, ROOT, config, ["labels"])
    report = audit_4682(label_run, ROOT, config, root, raw_run)
    assert report["rows"] == 4 and report["rows_compared"] >= 3
    assert report["label_source"] in {"released", "canonical_fallback"}
    assert (label_run / "stages/stage4/upstream_audit.parquet").is_file()
    gate_passed = load_manifest(label_run)["stages"]["stage4_audit"]["gate_passed"]
    assert gate_passed == report["gate"]["passed"]

    summary = build_labels(label_run, ROOT, config, root, raw_run)
    assert summary["rows"] == 160 and summary["label_source"] == report["label_source"]
    table = validate_parquet(label_run / "stages/stage4/labels.parquet").to_pylist()
    assert len(table) == 160
    assert {r["label_source"] for r in table} == {summary["label_source"]}
    assert all(r["importance_primary"] is None for r in table if not r["eligible"])
    assert all(r["dissimilar_count"] == 5 and r["similar_count"] == 1
               for r in table if r["sample_count"])
    assert "missing_adjacent_fork" in summary["exclusions"]
    assert set(summary["thresholds_train_only"]) == {"q80", "q90", "q95"}
    assert validate_parquet(root / "data/processed/stage4/labels.parquet").num_rows == 160
    # Identity checks: a second build under unchanged inputs is refused only if inputs moved.
    manifest = load_manifest(label_run)
    assert manifest["stages"]["stage4_labels"]["state"] == "complete"
    assert manifest["input_hashes"]["raw_run_id"] == raw_run.name


def test_build_refuses_without_audit_or_incomplete_raw(fixture_world, tmp_path):
    world = fixture_world
    config, root = world["config"], world["root"]
    raw_run = create_run(root, ROOT, config, ["raw"])
    run_raw_fetch(raw_run, ROOT, config, tmp_path / "cache", tracking=False,
                  download=world["download"], limit=3)
    label_run = create_run(root, ROOT, config, ["labels"])
    with pytest.raises(ValueError, match="limited"):
        audit_4682(label_run, ROOT, config, root, raw_run)
    with pytest.raises(ValueError, match="audit must complete"):
        build_labels(label_run, ROOT, config, root, raw_run)
