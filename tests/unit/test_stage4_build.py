"""Stage 4 row assembly: eligibility masks, thresholds and validation on the real build path."""

import json

import pytest

from preanchor.labels.stage4 import (
    build_rows,
    final_answer_index,
    fork_accuracy,
    load_fork,
    score_trajectory,
    validate_raw_discovery,
)
from preanchor.labels.threshold import train_threshold
from preanchor.labels.validation import validate_label_rows


class FakeSimilarity:
    """Deterministic stand-in: replacement 'same' is similar, anything else dissimilar."""

    def scores(self, rows):
        return [0.95 if r["chunk_resampled"] == "same" else 0.1 for r in rows]

    def clear(self):
        pass

    def identity(self):
        return {"model_id": "fake"}


def sentence(idx, text, released=0.5, char_start=None):
    return {"schema_version": 1, "sentence_id": f"1:correct:{idx}", "trajectory_id": "1:correct",
            "problem_id": 1, "trace_kind": "correct", "sentence_idx": idx, "text": text,
            "text_sha256": "x", "char_start": 10 + idx if char_start is None else char_start,
            "char_end": 20 + idx, "alignment_method": "exact", "alignment_ok": True,
            "eligible": True, "exclusion_reason": None, "importance_released": released}


def write_forks(tmp_path, forks):
    paths = {}
    for idx, samples in forks.items():
        p = tmp_path / f"chunk_{idx}.json"
        p.write_text(json.dumps(samples))
        paths[idx] = p
    return paths


def sample(answer, resampled="diff", correct=None):
    return {"chunk_removed": "orig", "chunk_resampled": resampled, "answer": answer,
            "is_correct": (answer == "1") if correct is None else correct}


LABELS = {"smoothing_alpha": 1e-9, "smoothing_sensitivities": [0.5, 1.0],
          "similarity_threshold": 0.8, "convergence_consecutive": 5,
          "exclude_final_answer": True, "binary_train_quantile": 0.9}


def test_build_rows_masks_and_validates(tmp_path):
    sentences = [sentence(0, "Start."), sentence(1, "Think."), sentence(2, "More."),
                 sentence(3, r"So \boxed{1}."), sentence(4, "After.")]
    forks = write_forks(tmp_path, {
        0: [sample("1"), sample("2"), sample("1", "same")],
        1: [sample("1"), sample("1")],
        2: [sample("2", "same"), sample("2", "same")],   # no dissimilar samples
        3: [sample("1")],
        4: [sample("1")],
    })
    scored = score_trajectory(sentences, forks, FakeSimilarity(), LABELS)
    rows, summary = build_rows(sentences, scored, LABELS, "canonical_fallback")
    by_idx = {r["sentence_idx"]: r for r in rows}
    assert summary["final_answer_idx"] == 3
    assert by_idx[0]["eligible"] and by_idx[0]["importance_primary"] is not None
    assert by_idx[0]["importance_primary"] == by_idx[0]["importance_canonical"]
    assert by_idx[0]["dissimilar_count"] == 2 and by_idx[0]["similar_count"] == 1
    assert by_idx[1]["eligible"]
    assert by_idx[2]["exclusion_reason"] == "no_dissimilar_samples"
    assert by_idx[3]["exclusion_reason"] == "final_answer"
    assert by_idx[4]["exclusion_reason"] == "final_answer"
    assert all(r["importance_primary"] is None for r in rows if not r["eligible"])
    assert "importance_canonical_alpha_0_5" in by_idx[0]
    assert by_idx[0]["importance_legacy"] is not None
    validate_label_rows(rows, primary_source="canonical_fallback")


def test_missing_adjacent_fork_and_first_token(tmp_path):
    sentences = [sentence(0, "A.", char_start=0), sentence(1, "B."), sentence(2, "C.")]
    forks = write_forks(tmp_path, {0: [sample("1")], 1: [sample("1")], 2: [sample("1")]})
    scored = score_trajectory(sentences, forks, FakeSimilarity(), LABELS)
    rows, _ = build_rows(sentences, scored, LABELS, "canonical_fallback")
    assert rows[0]["exclusion_reason"] == "first_token"
    assert rows[1]["eligible"]
    assert rows[2]["exclusion_reason"] == "missing_adjacent_fork"


def test_convergence_mask_uses_fork_accuracy(tmp_path):
    sentences = [sentence(i, f"S{i}.") for i in range(8)]
    forks = write_forks(tmp_path, {
        i: [sample("1" if i >= 2 else "2"), sample("1" if i >= 2 else "3")] for i in range(9)})
    scored = score_trajectory(sentences, forks, FakeSimilarity(), LABELS)
    rows, summary = build_rows(sentences, scored, LABELS, "canonical_fallback")
    assert summary["convergence_idx"] == 2   # first of five consecutive accuracy-1 forks
    assert [r["exclusion_reason"] for r in rows[2:]] == ["convergence"] * 6
    assert rows[0]["eligible"] and rows[1]["eligible"]
    assert summary["convergence_k4_idx"] == 2


def test_released_primary_uses_released_values(tmp_path):
    sentences = [sentence(0, "A.", released=0.7), sentence(1, "B.", released=float("nan"))]
    forks = write_forks(tmp_path, {0: [sample("1")], 1: [sample("2")], 2: [sample("1")]})
    scored = score_trajectory(sentences, forks, FakeSimilarity(), LABELS)
    rows, _ = build_rows(sentences, scored, LABELS, "released")
    assert rows[0]["importance_primary"] == 0.7 and rows[0]["label_source"] == "released"
    assert rows[1]["exclusion_reason"] == "nonfinite_importance"
    validate_label_rows(rows, primary_source="released")


def test_threshold_is_training_only_and_ignores_ineligible():
    rows = [{"problem_id": 1, "importance_primary": 1.0, "eligible": True},
            {"problem_id": 1, "importance_primary": 3.0, "eligible": True},
            {"problem_id": 1, "importance_primary": 100.0, "eligible": False},
            {"problem_id": 2, "importance_primary": 100.0, "eligible": True}]
    assert train_threshold(rows, {1}, 0.5) == 2.0
    with pytest.raises(ValueError):
        train_threshold(rows, {3}, 0.5)


def test_helpers():
    assert final_answer_index([sentence(0, r"\boxed{1}"), sentence(1, "x"),
                               sentence(2, r"final \boxed{2}")]) == 2
    assert final_answer_index([sentence(0, "x")]) is None
    assert fork_accuracy([sample("1"), sample("2")]) == 0.5
    assert fork_accuracy([]) is None


def test_load_fork_rejects_non_lists(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"solutions": [{"answer": "1"}]}))
    assert load_fork(p) == [{"answer": "1"}]
    p.write_text(json.dumps({"other": 1}))
    with pytest.raises(ValueError, match="not a sample list"):
        load_fork(p)


def test_raw_discovery_validation_rejects_gaps_and_foreign_paths():
    from pathlib import Path

    from preanchor.config import load_config

    config = load_config(Path(__file__).resolve().parents[2] / "configs/mvp.yaml")
    prefix = "deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95"

    def record(problem, kind, fork):
        path = f"{prefix}/{kind}_base_solution/problem_{problem}/chunk_{fork}/solutions.json"
        return {"path": path,
                "size": 1, "git_blob_sha1": "a" * 40, "lfs_sha256": None,
                "role": "raw_resamples", "problem_id": problem, "trace_kind": kind}

    forks = {f"{p}:{k}": 1 for p in config.dataset.problem_ids for k in ("correct", "incorrect")}
    files = [record(p, k, 0) for p in config.dataset.problem_ids for k in ("correct", "incorrect")]
    good = {"schema_version": 1, "dataset_id": config.dataset.id,
            "revision": config.dataset.revision, "forks_per_trajectory": forks, "files": files}
    validate_raw_discovery(good, config)
    with pytest.raises(ValueError, match="Fork coverage"):
        validate_raw_discovery({**good, "forks_per_trajectory": {**forks, "4682:correct": 2}},
                               config)
    with pytest.raises(ValueError, match="outside the selected scope"):
        validate_raw_discovery({**good, "files": files + [record(9999, "correct", 0)],
                                "forks_per_trajectory": forks}, config)
    with pytest.raises(ValueError, match="Dataset identity"):
        validate_raw_discovery({**good, "revision": "other"}, config)
