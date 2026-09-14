"""Stage 2 data selection gate only; later scientific gates remain unimplemented."""

from pathlib import Path

from preanchor.config import load_config
from preanchor.data.fetch import load_discovery


def test_both_trajectories_for_exact_problem_split():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/mvp.yaml")
    files = load_discovery(root, config)["files"]
    expected = {(p, k) for p in config.dataset.problem_ids for k in ("correct", "incorrect")}
    for role in ("base_solution", "chunks_labeled"):
        assert {(r["problem_id"], r["trace_kind"]) for r in files if r["role"] == role} == expected
    assert {
        (r["problem_id"], r["trace_kind"]) for r in files if r["role"] == "audit_resamples"
    } == {(4682, "correct")}


def test_pinned_selected_problems_first_twenty_match_frozen_scope():
    import json

    root = Path(__file__).resolve().parents[2]
    rows = json.loads((root / "vendor/thought-anchors/selected_problems.json").read_bytes())
    selected = [int(r["problem_idx"].removeprefix("problem_")) for r in rows[:20]]
    config = load_config(root / "configs/mvp.yaml")
    assert len(set(selected)) == 20
    assert set(selected) == set(config.dataset.problem_ids)
