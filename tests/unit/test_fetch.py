import copy
import hashlib
import json
from pathlib import Path

import pytest

from preanchor.config import load_config
from preanchor.data.fetch import load_discovery, validate_discovery, validate_source

ROOT = Path(__file__).resolve().parents[2]


def test_pinned_discovery_coverage():
    config = load_config(ROOT / "configs/mvp.yaml")
    value = load_discovery(ROOT, config)
    assert len(value["files"]) == 306
    assert value["audit_forks"] == list(range(145))
    assert sum(r["size"] for r in value["files"]) == 339390844


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "wildcard",
        "other_model",
        "other_problem",
        "revision",
        "fork",
        "hash",
    ],
)
def test_reject_invalid_discovery(mutation):
    config = load_config(ROOT / "configs/mvp.yaml")
    d = copy.deepcopy(load_discovery(ROOT, config))
    row = next(r for r in d["files"] if r["role"] == "base_solution")
    if mutation == "duplicate":
        d["files"].append(row.copy())
    elif mutation == "missing":
        d["files"].remove(row)
    elif mutation == "wildcard":
        row["path"] = "data/*"
    elif mutation == "other_model":
        row["path"] = row["path"].replace("qwen-14b", "llama-8b")
    elif mutation == "other_problem":
        row["problem_id"] = 9999
    elif mutation == "revision":
        d["revision"] = "main"
    elif mutation == "fork":
        d["audit_forks"] = []
    else:
        row["git_blob_sha1"] = None
    with pytest.raises(ValueError):
        validate_discovery(d, config)


@pytest.mark.parametrize(
    "role,value",
    [
        ("base_solution", {"full_cot": "exact trace"}),
        ("chunks", {"chunks": ["sentence"]}),
        ("chunks_labeled", [{"chunk": "sentence"}]),
        ("problem", {"problem": "question"}),
        ("audit_resamples", [{"full_cot": "rollout"}]),
    ],
)
def test_source_shapes_and_upstream_hashes(tmp_path, role, value):
    p = tmp_path / "source.json"
    payload = json.dumps(value).encode()
    p.write_bytes(payload)
    row = {
        "role": role,
        "size": len(payload),
        "lfs_sha256": None,
        "git_blob_sha1": hashlib.sha1(
            b"blob " + str(len(payload)).encode() + b"\0" + payload
        ).hexdigest(),
    }
    expected = hashlib.sha256(payload).hexdigest()
    assert validate_source(p, row) == expected
    row["lfs_sha256"] = expected
    assert validate_source(p, row) == expected
    p.write_bytes(b" " * len(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_source(p, row)


def test_chunks_object_is_not_labeled_list(tmp_path):
    p = tmp_path / "chunks.json"
    payload = b"[]"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="chunk list"):
        validate_source(
            p, {"role": "chunks", "size": 2, "lfs_sha256": hashlib.sha256(payload).hexdigest()}
        )
