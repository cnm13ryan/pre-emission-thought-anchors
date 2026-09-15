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


_WRITER_TOKENS: dict[Path, str] = {}   # plaintext per control root; only the hash is on disk


def grant_writer_lease(root: Path, monkeypatch) -> str:
    """Hold ``portable-store:writer`` for the test and present its token through the
    environment, exactly as an operator who leased the store would."""
    from preanchor.control import TOKENS_ENV, control_root, presented_tokens
    from preanchor.control.leases import WRITER_LEASE, acquire, describe
    from preanchor.control.sessions import register

    control = control_root(root)
    if describe(control, WRITER_LEASE)["state"] == "free":
        session = register(control, task="backup test world", role="tests", worktree=None)
        _WRITER_TOKENS[control] = acquire(control, WRITER_LEASE, purpose="test",
                                          session_id=session["session_id"])["owner_token"]
    token = _WRITER_TOKENS[control]
    tokens = {**presented_tokens(), WRITER_LEASE: token}
    monkeypatch.setenv(TOKENS_ENV, ",".join(f"{k}={v}" for k, v in tokens.items()))
    return token


def make_world(root: Path, monkeypatch, **kwargs) -> tuple[Path, list]:
    fake_provenance(monkeypatch)
    grant_writer_lease(root, monkeypatch)
    write_stage3(root)
    return make_production(root, **kwargs)


def make_stage6(root: Path, *, trajectories=("330:correct", "2189:incorrect"),
                limit: int = 4) -> Path:
    """A real Stage 6 run (tiny Qwen2, character tokenizer) laid out by ``features.extract``.

    Requires ``write_synthetic_root`` (40 synthetic trajectories plus Stage 4 labels)."""
    import sys

    sys.path.insert(0, str(ROOT / "tests/fixtures"))
    from tiny_qwen2 import tiny_loader

    from preanchor.features.extract import execute, load_inputs, prepare, select_work

    config = load_config(ROOT / "configs/smoke.yaml")
    inputs, hashes = load_inputs(root)
    work = select_work(config, inputs, list(trajectories), limit)
    run = prepare(root, ROOT, config, "fixture", work, hashes)
    execute(run, ROOT, config, trajectories=list(trajectories), limit_sentences=limit,
            tracking=False, loader=tiny_loader)
    return run


def make_stage4_runs(root: Path) -> tuple[Path, Path]:
    """Provenance-only Stage 4 raw and build runs shaped like ``labels.stage4`` records them,
    publishing the synthetic ``data/processed/stage4/labels.parquet``."""
    from preanchor.ids import file_hash

    config = load_config(ROOT / "configs/mvp.yaml")
    raw = create_run(root, ROOT, config, ["preanchor fetch-raw-resamples"])
    digest = write_json(raw / "stages/stage4_raw/source_manifest.json",
                        {"schema_version": 1, "files": []})
    manifest = load_manifest(raw)
    manifest["input_hashes"] = {"raw_discovery": "r" * 64}
    manifest["identity"]["input_hashes"] = manifest["input_hashes"]
    manifest["output_hashes"]["stage4_raw_source_manifest"] = digest
    manifest["stages"]["stage4_raw"] = {"state": "complete", "attempt": "x", "limit": None}
    save_manifest(raw, manifest)
    build = create_run(root, ROOT, config, ["preanchor build-labels"])
    base = root / "data/processed"
    manifest = load_manifest(build)
    manifest["input_hashes"] = {
        "stage3_trajectories": file_hash(base / "stage3/trajectories.parquet"),
        "stage3_sentences": file_hash(base / "stage3/sentences.parquet"),
        "raw_run_id": raw.name, "raw_source_manifest": digest}
    manifest["identity"]["input_hashes"] = manifest["input_hashes"]
    manifest["output_hashes"]["stage4_labels"] = file_hash(base / "stage4/labels.parquet")
    manifest["stages"]["stage4_labels"] = {"state": "complete"}
    save_manifest(build, manifest)
    return raw, build


def make_stage6_world(root: Path, monkeypatch, *, stage5_units: int = 256, **kwargs):
    """Stage 3/4 synthetic tables, a finalized-looking Stage 5 production run and a complete
    Stage 6 smoke run: the full ``--for-stage 6`` closure without GPU or network."""
    import sys

    sys.path.insert(0, str(ROOT / "tests/fixtures"))
    from tiny_qwen2 import write_synthetic_root

    fake_provenance(monkeypatch)
    grant_writer_lease(root, monkeypatch)
    write_synthetic_root(root, load_config(ROOT / "configs/mvp.yaml"))
    make_stage4_runs(root)
    production, specs = make_production(root, units=stage5_units)
    stage6 = make_stage6(root, **kwargs)
    return production, specs, stage6


def make_stage7(root: Path) -> Path:
    """A complete Stage 7 run (tiny Qwen2, all 40 synthetic trajectories, mvp configuration)
    with ``build-dataset`` published to ``data/processed/stage7`` and ``validate-features``
    recorded, laid out exactly as ``features.extract`` and ``features.dataset`` write it."""
    import sys

    sys.path.insert(0, str(ROOT / "tests/fixtures"))
    from tiny_qwen2 import tiny_loader, tiny_model, tiny_tokenizer

    from preanchor.features.dataset import build_dataset, validate_features
    from preanchor.features.extract import execute, load_inputs, prepare, select_work
    from preanchor.features.observables import head_from_model

    config = load_config(ROOT / "configs/mvp.yaml")
    inputs, hashes = load_inputs(root)
    run = prepare(root, ROOT, config, "fixture", select_work(config, inputs, None, None), hashes)
    execute(run, ROOT, config, tracking=False, loader=tiny_loader, diagnostics=False)
    head_loader = lambda cfg, device: head_from_model(tiny_model(), device)  # noqa: E731
    tokenizer_loader = lambda cfg: tiny_tokenizer()  # noqa: E731
    build_dataset(run, ROOT, config, root, device="cpu", tracking=False,
                  head_loader=head_loader, tokenizer_loader=tokenizer_loader)
    validate_features(run, config, root, sample=0)
    return run


def make_stage7_world(root: Path, monkeypatch, *, stage5_units: int = 2, **kwargs):
    """The full ``--for-stage 7`` closure without GPU or network: Stage 3/4 synthetic tables
    and runs, a Stage 5 production run, a Stage 6 smoke run and the canonical Stage 7 run."""
    production, specs, stage6 = make_stage6_world(root, monkeypatch, stage5_units=stage5_units,
                                                  **kwargs)
    stage7 = make_stage7(root)
    return production, specs, stage6, stage7
