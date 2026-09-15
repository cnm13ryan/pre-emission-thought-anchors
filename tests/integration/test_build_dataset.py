"""Stage 7 dataset on synthetic inputs: joins, identity, determinism, leakage, validation."""

import hashlib
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from preanchor.config import load_config
from preanchor.features.dataset import build_dataset, validate_features
from preanchor.features.extract import execute, load_inputs, prepare, select_work
from preanchor.features.observables import (
    DIAGNOSTIC_FEATURES,
    PRIMARY_FEATURES,
    feature_path,
    head_from_model,
)
from preanchor.io import read_json, validate_parquet, write_parquet
from preanchor.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
sys.path.insert(0, str(FIXTURES))
from tiny_qwen2 import tiny_loader, tiny_model, tiny_tokenizer, write_synthetic_root  # noqa: E402

SOURCE = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
          "package_freeze": []}
WORK = ["330:correct", "2189:incorrect"]
LIMIT = 6


def head_loader(config, device):
    return head_from_model(tiny_model(), device)


def tokenizer_loader(config):
    return tiny_tokenizer()


def extract(root, config, work=WORK, limit=LIMIT):
    inputs, hashes = load_inputs(root)
    run = prepare(root, ROOT, config, "test", select_work(config, inputs, work, limit), hashes)
    execute(run, ROOT, config, trajectories=work, limit_sentences=limit, tracking=False,
            loader=tiny_loader, diagnostics=False)
    return run


def build(root, run, config, **kwargs):
    return build_dataset(run, ROOT, config, root, device="cpu", tracking=False,
                         head_loader=head_loader, tokenizer_loader=tokenizer_loader, **kwargs)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
    return load_config(ROOT / "configs/smoke.yaml")


def test_build_validate_and_deterministic_identity(tmp_path, config):
    root = tmp_path / "root"
    write_synthetic_root(root, config)
    run = extract(root, config)
    with pytest.raises(ValueError, match="must be complete"):
        build(root, prepare(root, ROOT, config, "idle",
                            select_work(config, load_inputs(root)[0], WORK, LIMIT),
                            load_inputs(root)[1]), config)
    result = build(root, run, config)
    assert result["n_rows"] == 12
    manifest = load_manifest(run)
    assert manifest["stages"]["stage6_features"]["state"] == "complete"
    assert manifest["output_hashes"]["features/features.parquet"] == result["features_sha256"]
    table = validate_parquet(run / "features/features.parquet")
    published = validate_parquet(feature_path(root))
    assert published.equals(table)
    rows = table.to_pylist()
    assert [r["sentence_id"] for r in rows] == [f"2189:incorrect:{i}" for i in range(6)] + \
        [f"330:correct:{i}" for i in range(6)]
    assert {r["split"] for r in rows if r["problem_id"] == 330} == {"validation"}
    assert {r["split"] for r in rows if r["problem_id"] == 2189} == {"train"}
    assert all(r["activation_run_id"] == run.name and r["eligible"] for r in rows)
    assert [r["anchor"] for r in rows if r["trajectory_id"] == "330:correct"] == \
        [False, False, False, True, False, False]
    index = {r["sentence_id"]: r for r in
             validate_parquet(run / "activations/index.parquet").to_pylist()}
    for r in rows:
        assert r["prefix_token_count"] == index[r["sentence_id"]]["boundary_token_idx"] + 1
        assert r["shard_path"] == index[r["sentence_id"]]["shard_path"]
        assert r["diagnostic__relative_completed_trace_position"] == r["sentence_idx"] / 17
    audit = read_json(run / "features/feature_audit.json")
    assert audit["columns"]["primary"] == PRIMARY_FEATURES
    assert audit["value_checks"]["violations"] == [] and audit["units"] == 2
    assert audit["head"]["weights_sha256"] == head_loader(config, "cpu").weights_sha256
    report = validate_features(run, config, root, sample=5, device="cpu",
                               head_loader=head_loader, tokenizer_loader=tokenizer_loader)
    assert report["ok"] and report["rows"] == 12 and report["recomputed"]["sample"] == 5
    assert report["published_identical"]
    assert (run / "features/feature_validation.json").is_file()
    # Deterministic identity: a second extraction run of the same inputs and code yields
    # byte-identical shards and a feature table that differs only in the run pointer.
    other = extract(root, config)
    with pytest.raises(FileExistsError):
        build(root, other, config)  # the published copy would differ in activation_run_id
    build(root, other, config, publish=False)
    again = validate_parquet(other / "features/features.parquet")
    drop = ["activation_run_id", "shard_path"]
    assert again.drop_columns(drop).equals(table.drop_columns(drop))
    assert again.column("shard_path").to_pylist() == table.column("shard_path").to_pylist()


def test_tampering_is_detected(tmp_path, config):
    root = tmp_path / "root"
    write_synthetic_root(root, config)
    run = extract(root, config)
    build(root, run, config)
    table = validate_parquet(run / "features/features.parquet")
    column = table.schema.get_field_index("next_token_entropy")
    tampered = table.set_column(column, "next_token_entropy",
                                pa.array([v + 1e-3 for v in
                                          table.column("next_token_entropy").to_pylist()]))
    path = run / "features/features.parquet"
    for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
        Path(str(path) + suffix).unlink()
    write_parquet(path, tampered)
    with pytest.raises(ValueError, match="hash differs"):
        validate_features(run, config, root, sample=0)
    manifest = load_manifest(run)
    manifest["output_hashes"]["features/features.parquet"] = hashlib.sha256(
        path.read_bytes()).hexdigest()
    audit = read_json(run / "features/feature_audit.json")
    audit["features_sha256"] = manifest["output_hashes"]["features/features.parquet"]
    from preanchor.features.extract import write_current_json
    from preanchor.manifest import save_manifest

    save_manifest(run, manifest)
    write_current_json(run / "features/feature_audit.json", audit)
    with pytest.raises(ValueError, match="Published feature table differs"):
        validate_features(run, config, root, sample=0)
    audit["published"] = None
    write_current_json(run / "features/feature_audit.json", audit)
    with pytest.raises(ValueError, match="Recomputed next_token_entropy"):
        validate_features(run, config, root, sample=12, device="cpu",
                          head_loader=head_loader, tokenizer_loader=tokenizer_loader)


def rewrite_future(root, config, trajectory_id, keep):
    """Replace everything after sentence `keep-1` of one trajectory with different text."""
    processed = root / "data/processed"
    tables = {n: validate_parquet(processed / n).to_pylist() for n in
              ("stage3/trajectories.parquet", "stage3/sentences.parquet",
               "stage4/labels.parquet")}
    kept = [s for s in tables["stage3/sentences.parquet"]
            if s["trajectory_id"] == trajectory_id and s["sentence_idx"] < keep]
    cut = max(s["char_end"] for s in kept)
    for t in tables["stage3/trajectories.parquet"]:
        if t["trajectory_id"] == trajectory_id:
            t["full_cot"] = t["full_cot"][:cut] + " Entirely different future text.\nEnd."
            t["full_cot_sha256"] = hashlib.sha256(t["full_cot"].encode()).hexdigest()
            t["n_chars"] = len(t["full_cot"])
    for name in ("stage3/sentences.parquet", "stage4/labels.parquet"):
        tables[name] = [r for r in tables[name]
                        if r["trajectory_id"] != trajectory_id or r["sentence_idx"] < keep]
    for name, rows in tables.items():
        for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
            (processed / (name + suffix)).unlink()
        write_parquet(processed / name, pa.Table.from_pylist(rows))


def test_primary_features_do_not_depend_on_target_or_future_tokens(tmp_path, config):
    """Same prefix, different future: every primary column is unchanged; only the
    completed-length diagnostic moves."""
    roots = []
    for variant in ("original", "mutated"):
        root = tmp_path / variant
        write_synthetic_root(root, config)
        if variant == "mutated":
            rewrite_future(root, config, "2189:incorrect", keep=LIMIT)
        run = extract(root, config)
        build(root, run, config)
        roots.append(validate_parquet(run / "features/features.parquet").to_pylist())
    original = {r["sentence_id"]: r for r in roots[0] if r["trajectory_id"] == "2189:incorrect"}
    mutated = {r["sentence_id"]: r for r in roots[1] if r["trajectory_id"] == "2189:incorrect"}
    assert set(original) == set(mutated) and len(original) == LIMIT
    assert original["2189:incorrect:0"]["trace_sha256"] != mutated["2189:incorrect:0"][
        "trace_sha256"]
    for sid, row in original.items():
        for c in PRIMARY_FEATURES:
            assert abs(float(row[c]) - float(mutated[sid][c])) <= 1e-5, (sid, c)
        assert abs(row["diagnostic__first_target_token_logprob"]
                   - mutated[sid]["diagnostic__first_target_token_logprob"]) <= 1e-5
    assert original["2189:incorrect:1"]["diagnostic__relative_completed_trace_position"] \
        != mutated["2189:incorrect:1"]["diagnostic__relative_completed_trace_position"]
    assert all(c not in DIAGNOSTIC_FEATURES for c in PRIMARY_FEATURES)
