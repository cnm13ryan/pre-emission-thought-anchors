"""Content-addressed store, bundle manifests, receipts, secrets and path portability."""

import json
import os
from pathlib import Path

import pytest

from preanchor.backup.bundle import (
    SCHEMA_VERSION,
    load_bundle,
    manifest_digest,
    resolve_bundle,
    write_bundle,
)
from preanchor.backup.classify import (
    CACHE_EXCLUDE,
    MUST_BACK_UP,
    SHOULD_BACK_UP,
    _run_file,
    secret_reason,
    stage_closure,
)
from preanchor.backup.receipts import list_receipts, write_receipt
from preanchor.backup.store import ObjectStore, is_digest
from preanchor.ids import file_hash
from preanchor.io import same_logical_path


def manifest_stub(entries=None):
    return {"schema_version": SCHEMA_VERSION,
            "bundle": {"created_at": "t", "kind": "canonical", "for_stage": 5,
                       "highest_protected_stage": 5, "stages": {}, "runs": {}},
            "entries": entries or [], "stats": {"logical_bytes": 0}}


def test_objects_are_named_by_sha256_and_deduplicated(tmp_path):
    store = ObjectStore(tmp_path / "portable")
    a = tmp_path / "a.bin"
    a.write_bytes(b"payload")
    b = tmp_path / "b.bin"
    b.write_bytes(b"payload")
    digest, created = store.put_file(a)
    assert created and is_digest(digest) and digest == file_hash(a)
    assert store.path(digest) == tmp_path / "portable/objects/sha256" / digest[:2] / digest[
        2:4] / digest
    assert store.put_file(b) == (digest, False)      # same bytes: zero new objects
    assert store.digests() == {digest} and store.verify(digest) == 7
    with pytest.raises(ValueError, match="Source changed"):
        store.put_file(a, "0" * 64)
    link = tmp_path / "link.bin"
    link.symlink_to(a)
    with pytest.raises(ValueError, match="symlink"):
        store.put_file(link)


def test_existing_object_is_never_overwritten(tmp_path):
    store = ObjectStore(tmp_path / "portable")
    source = tmp_path / "s"
    source.write_bytes(b"good")
    digest, _ = store.put_file(source)
    store.path(digest).write_bytes(b"evil")
    with pytest.raises(ValueError, match="does not match its name"):
        store.put_file(source)
    with pytest.raises(ValueError, match="Corrupt object"):
        store.verify(digest)
    assert store.path(digest).read_bytes() == b"evil"   # untouched, reported, never replaced


def test_install_states_and_conflicts(tmp_path):
    store = ObjectStore(tmp_path / "portable")
    source = tmp_path / "s"
    source.write_bytes(b"bytes")
    digest, _ = store.put_file(source)
    target = tmp_path / "root/data/file"
    assert store.install(digest, target) == "restored"
    assert store.install(digest, target) == "already_present_identical"
    target.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        store.install(digest, target)
    assert target.read_bytes() == b"different"
    with pytest.raises(FileNotFoundError):
        store.install("f" * 64, tmp_path / "root/other")
    assert not list((tmp_path / "root").glob("**/*.partial"))


def test_bundle_manifest_is_deterministic_self_hashed_and_tamper_evident(tmp_path):
    portable = tmp_path / "portable"
    manifest = manifest_stub()
    digest = manifest_digest(manifest)
    assert digest == manifest_digest(json.loads(json.dumps(manifest)))
    name = write_bundle(portable, manifest, stamp="20260101T000000Z")
    assert name == f"20260101T000000Z-{digest}"
    assert write_bundle(portable, manifest, stamp="20260101T000000Z") == name   # idempotent
    latest = json.loads((portable / "latest.json").read_bytes())
    assert latest["bundle_id"] == name and latest["manifest_sha256"] == digest
    assert load_bundle(portable, "latest")[0] == name
    assert resolve_bundle(portable, digest[:8]) == name
    path = portable / "manifests" / f"{name}.json"
    tampered = dict(manifest)
    tampered["entries"] = [{"path": "x"}]
    path.write_bytes(json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    with pytest.raises(ValueError, match="tampered"):
        load_bundle(portable, name)


def test_receipts_are_append_only(tmp_path):
    portable = tmp_path / "portable"
    first = write_receipt(portable, "prepare", {"manifest_sha256": "a" * 64, "outcome": "x"})
    second = write_receipt(portable, "prepare", {"manifest_sha256": "a" * 64, "outcome": "y"})
    assert first != second and Path(str(first) + ".sha256").is_file()
    rows = list_receipts(portable, manifest_sha256="a" * 64)
    assert [r["outcome"] for r in rows] == ["x", "y"]
    with pytest.raises(ValueError):
        write_receipt(portable, "bogus", {})
    third = write_receipt(portable, "prepare", {"manifest_sha256": "b" * 64, "api_key": "leak"})
    assert json.loads(third.read_bytes())["api_key"] == "[REDACTED]"


def test_secret_detection_by_name_and_content(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PREANCHOR_ROOT=/x\n")
    assert secret_reason(env, ".env") == "credential-like path"
    assert secret_reason(env, "artifacts/runs/r/.env.local") == "credential-like path"
    key = tmp_path / "id_rsa"
    key.write_text("x")
    assert secret_reason(key, "home/id_rsa")
    token = tmp_path / "notes.txt"
    token.write_text("token hf_" + "A" * 30 + " here")
    assert secret_reason(token, "notes.txt") == "credential-like content"
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret-value-123")
    leak = tmp_path / "log.txt"
    leak.write_text("key=wandb-secret-value-123")
    assert secret_reason(leak, "log.txt") == "credential-like content"
    clean = tmp_path / "tokenizer_validation.json"
    clean.write_text('{"count": 3}')
    assert secret_reason(clean, "stages/stage5/tokenizer_validation.json") is None


def test_run_file_classification_for_representative_artifacts():
    P = Path
    assert _run_file("stage5_production", P("manifest.json"), True) == (
        MUST_BACK_UP, "run_manifest", True)
    assert _run_file("stage5_production", P("manifests/abc.json"), True) == (
        MUST_BACK_UP, "sha256_sidecar", True)
    assert _run_file("stage5_production", P("stages/stage5/plan.json"), True) == (
        MUST_BACK_UP, "sha256_sidecar", True)
    assert _run_file("stage5_production", P("stages/stage5/fresh_labels.parquet"), True) == (
        MUST_BACK_UP, "parquet", True)
    assert _run_file("stage5_production", P("stages/stage5/telemetry/x/000001.json"),
                     True) == (SHOULD_BACK_UP, "bytes", False)
    assert _run_file("stage5_production", P("logs/wandb/run.wandb"), True) == (
        SHOULD_BACK_UP, "bytes", False)
    assert _run_file("stage5_production", P("heartbeat.json"), True) == (
        SHOULD_BACK_UP, "bytes", False)
    assert _run_file("stage5_production", P(".run.lock"), True) is None
    assert _run_file("stage4_raw", P("stages/stage4_raw/attempts/a.json"), True) == (
        MUST_BACK_UP, "sha256_sidecar", True)
    assert _run_file("stage4_raw", P("stages/stage4_raw/source_manifest.json"), True) == (
        MUST_BACK_UP, "sha256_sidecar", True)
    assert _run_file("stage2", P("stages/stage2/source_manifest.json"), True) == (
        SHOULD_BACK_UP, "sha256_sidecar", False)
    assert _run_file("historical", P("stages/stage2/source_manifest.json"), True)[0] \
        == "HISTORICAL_EVIDENCE"
    assert CACHE_EXCLUDE == "CACHE_EXCLUDE"


def test_stage_closure_follows_actual_dependencies():
    assert stage_closure(3) == (3,)
    assert stage_closure(4) == (3, 4)
    assert stage_closure(5) == (3, 5)          # Stage 5 hashes Stage 3, not Stage 4
    assert stage_closure(8) == (3, 4, 5, 6, 7, 8)
    assert stage_closure(9) == (3, 4, 5, 6, 7, 8, 9)
    with pytest.raises(ValueError):
        stage_closure(12)          # no Stage 12 entry: unknown stages still fail closed


def test_same_logical_path_accepts_moved_roots_and_rejects_other_runs():
    run = Path("/lambda/nfs/new/preanchor/artifacts/runs/20260915T134433Z-d7d055d8b7-05a81618")
    actual = run / "stages/stage4_raw/units" / ("a" * 64) / "solutions.json"
    old = ("/lambda/nfs/preanchor/preanchor/artifacts/runs/20260915T134433Z-d7d055d8b7-05a81618"
           "/stages/stage4_raw/units/" + "a" * 64 + "/solutions.json")
    assert same_logical_path(str(actual), actual, run)          # legacy exact equality
    assert same_logical_path(old, actual, run)                  # moved root, same run/unit
    assert not same_logical_path(old.replace("05a81618", "deadbeef"), actual, run)
    assert not same_logical_path(old.replace("a" * 64, "b" * 64), actual, run)
    assert not same_logical_path(old, Path("/elsewhere/solutions.json"), run)
    assert os.sep == "/"
