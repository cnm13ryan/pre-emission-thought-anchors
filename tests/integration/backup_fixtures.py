"""Synthetic canonical worlds for backup/restore tests: no GPU, network or NFS."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa

from preanchor.config import config_hash, load_config
from preanchor.io import write_json, write_parquet
from preanchor.labels.generate import CompletionStore, FakeBackend, production_specs
from preanchor.labels.pilot import DIGEST, assert_plan, inputs
from preanchor.manifest import create_run, load_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
          "package_freeze": [], "dirty": False, "python": "3.11", "os": "test",
          "upstream": {"repositories": {}}}


def fake_provenance(monkeypatch, source: dict | None = None) -> dict:
    source = dict(source or SOURCE)
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (dict(source), b""))
    return source


def write_stage3(root: Path, sentences: int = 48) -> None:
    """A 4682:correct trace with enough sentences for every frozen fork (up to 47)."""
    base = root / "data/processed/stage3"
    base.mkdir(parents=True, exist_ok=True)
    texts = [f"S{i}." for i in range(sentences)]
    trace = "P <think>\n" + " ".join(texts)
    trajectories = [{"trajectory_id": "4682:correct", "problem_id": 4682,
                     "trace_kind": "correct", "prompt": "P <think>\n", "full_cot": trace,
                     "full_cot_sha256": hashlib.sha256(trace.encode()).hexdigest(),
                     "source_path": str(root / "artifacts/runs/legacy/stages/stage2/x.json")}]
    rows = []
    offset = 0
    for i, text in enumerate(texts):
        start = trace.index(text, offset)
        offset = start + len(text)
        rows.append({"trajectory_id": "4682:correct", "sentence_idx": i, "text": text,
                     "char_start": start, "char_end": start + len(text)})
    write_parquet(base / "trajectories.parquet", pa.Table.from_pylist(trajectories))
    write_parquet(base / "sentences.parquet", pa.Table.from_pylist(rows))
    write_json(base / "import_summary.json", {"schema_version": 1, "sentences": sentences,
                                              "trajectories": 1, "unresolved": 0})


def make_production(root: Path, *, units: int = 256, state: str | None = None,
                    config_path: Path = ROOT / "configs/pilot.yaml") -> tuple[Path, list]:
    """A FakeBackend Stage 5 production run laid out exactly like ``a6000.initialize``.

    ``state`` defaults to ``generated`` for a complete plan and ``running`` otherwise."""
    state = state or ("generated" if units >= 256 else "running")
    config = load_config(config_path)
    trajectory, targets, references, hashes = inputs(root, ROOT)
    run = create_run(root, ROOT, config, ["python -m preanchor.labels.a6000 production"])
    manifest = load_manifest(run)
    manifest["input_hashes"] = hashes
    manifest["identity"]["input_hashes"] = hashes
    manifest["vllm"] = {"digest": DIGEST}
    manifest["stages"]["stage5"] = {"state": state, "namespace": "production"}
    save_manifest(run, manifest)
    specs = production_specs(trajectory, targets, config.seed_root, config.resolved_hash(),
                             config_hash(hashes))
    assert_plan(specs)
    stage = run / "stages/stage5"
    write_json(stage / "plan.json", {
        "schema_version": 1, "namespace": "production", "physical_jobs": len(specs),
        "references": references, "input_hashes": hashes,
        "jobs": [{**asdict(s), "rollout_id": s.rollout_id} for s in specs]})
    write_json(stage / "runtime_admission.json", {"schema_version": 1, "concurrency": 8,
                                                  "vllm_digest": DIGEST})
    write_json(stage / "tokenizer_validation.json", [
        {"fork_idx": fork, "count": len(next(s.prompt for s in specs if s.fork_idx == fork)
                                        .split()), "equal": True} for fork in
        sorted({s.fork_idx for s in specs})])
    (run / "heartbeat.json").write_bytes(b'{"state": "%s"}\n' % state.encode())
    store = CompletionStore(stage / "units")
    backend = FakeBackend(["A replacement sentence. Next."])
    for spec in specs[:units]:
        store.raw_one(spec, backend, max_tokens=16384, temperature=0.6, top_p=0.95)
    return run, specs


def make_world(root: Path, monkeypatch, **kwargs) -> tuple[Path, list]:
    fake_provenance(monkeypatch)
    write_stage3(root)
    return make_production(root, **kwargs)
