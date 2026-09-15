import json

from preanchor.data.importer import import_all
from preanchor.io import validate_parquet


def test_import_fixture(tmp_path):
    base = {"prompt": "P <think>\n", "solution": "A. B.",
            "full_cot": "P <think>\nA. B.", "answer": "2", "is_correct": True}
    problem = {"problem": "P", "gt_answer": "2"}
    labels = [{"counterfactual_importance_kl": 0.1}, {"counterfactual_importance_kl": 0.2}]
    source = tmp_path / "src"
    source.mkdir()
    paths = {}
    records = []
    for trace_kind in ("correct", "incorrect"):
        paths = {}
        for role, value in [("base_solution", base), ("chunks", {"chunks": ["A.", "B."]}),
                            ("chunks_labeled", labels), ("problem", problem)]:
            p = source / f"{trace_kind}-{role}.json"
            p.write_text(json.dumps(value))
            paths[role] = str(p)
        records.extend({"role": role, "path": role, "local_path": path,
                        "problem_id": 1, "trace_kind": trace_kind}
                       for role, path in paths.items())
    manifest = {"revision": "r", "files": records}
    result = import_all(manifest, [1], tmp_path / "out")
    assert len(result["trajectories"]) == 2 and len(result["sentences"]) == 4
    validate_parquet(tmp_path / "out/trajectories.parquet")
    validate_parquet(tmp_path / "out/sentences.parquet")


def _fixture_manifest(tmp_path, base_text="P <think>\nA. B.", checksummed=False):
    import hashlib

    from preanchor.io import write_json

    base = {"prompt": "P <think>\n", "solution": "A. B.", "full_cot": base_text,
            "answer": "2", "is_correct": True}
    problem = {"problem": "P", "gt_answer": "2"}
    labels = [{"counterfactual_importance_kl": 0.1}, {"counterfactual_importance_kl": 0.2}]
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    records = []
    for role, value in [("base_solution", base), ("chunks", {"chunks": ["A.", "B."]}),
                        ("chunks_labeled", labels), ("problem", problem)]:
        p = source / f"{role}.json"
        if checksummed:
            write_json(p, value)
        else:
            p.write_text(json.dumps(value))
        records.append({"role": role, "path": role, "local_path": str(p), "problem_id": 1,
                        "trace_kind": "correct",
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return {"revision": "r", "files": records}


def test_checksummed_source_corruption_is_rejected(tmp_path):
    import pytest

    from preanchor.data.importer import import_trajectory

    manifest = _fixture_manifest(tmp_path, checksummed=True)
    trajectory, sentences = import_trajectory(manifest, 1, "correct")
    assert len(sentences) == 2
    base = tmp_path / "src/base_solution.json"
    tampered = json.loads(base.read_bytes())
    tampered["full_cot"] = "P <think>\nA. B. tampered"
    base.write_text(json.dumps(tampered))  # Sidecar no longer matches.
    with pytest.raises(ValueError, match="Checksum mismatch"):
        import_trajectory(manifest, 1, "correct")


def test_alignment_starts_at_zero_when_think_marker_is_absent(tmp_path):
    from preanchor.data.importer import import_trajectory

    manifest = _fixture_manifest(tmp_path, base_text="A. B.")
    _, sentences = import_trajectory(manifest, 1, "correct")
    assert [(s["char_start"], s["char_end"]) for s in sentences] == [(0, 2), (3, 5)]
