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
