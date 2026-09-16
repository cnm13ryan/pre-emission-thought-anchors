"""SPEC §10.7: the Stage 11 report rebuilds deterministically from predictions and metrics
alone — no model payloads, no activation shards — and every figure records its input
artifact hashes. Runs on a small synthetic root; the same code path runs against a
restored real root."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from preanchor.io import write_json, write_parquet
from preanchor.report.export import export_interface
from preanchor.report.render import render_report

S8, S9, S10 = "20990101T000000Z-fake-stage08", "20990101T000000Z-fake-stage09", \
    "20990101T000000Z-fake-stage10"
PRIMARY = "activation_plus_observables"
FAMILIES = (PRIMARY, "activation_only", "observable_combined", "position", "prevalence")


def _predictions(seed: int, split: str) -> pa.Table:
    rng = np.random.default_rng(seed)
    n = 60
    rows = []
    for family in FAMILIES:
        prob = rng.uniform(0.01, 0.9, n)
        for i in range(n):
            rows.append({"schema_version": 1, "sentence_id": f"{i}:{split}",
                         "problem_id": 1000 + i % 4, "trajectory_id": f"t{i % 8}",
                         "split": split, "lead_sentences": 0, "threshold_quantile": 0.9,
                         "layer": 44 if "activation" in family else None,
                         "probe_name": family, "task": "binary", "hyperparameter": 0.1,
                         "y_true": int(rng.random() < 0.1),
                         "importance_true": float(rng.random()),
                         "score_raw": float(prob[i] * 4 - 2),
                         "probability": float(prob[i]),
                         "probability_calibrated": float(prob[i]),
                         "importance_predicted": None})
    return pa.Table.from_pylist(rows)


def _metrics(split: str) -> dict:
    model = {"average_precision": 0.1, "auroc": 0.6, "brier": 0.07, "ece": 0.08,
             "log_loss": 0.33, "n": 60, "n_positive": 6, "prevalence": 0.1,
             "calibrated": True, "n_trajectories": 8, "n_trajectories_with_positive": 4,
             "ndcg_top10_percent": 0.2, "precision_top10_percent": 0.2, "per_trajectory": {},
             "reliability": []}
    return {"schema_version": 1, "split": split, "run_id": S9, "stage": "stage9",
            "freeze_sha256": "f" * 64,
            "models": {f"{f}:binary": dict(model) for f in FAMILIES},
            "primary": {"metric": "average_precision", "model": f"{PRIMARY}:binary",
                        "value": 0.1},
            "aucpr_gain_over_observable_combined": 0.03,
            "bootstrap": {"replicates_requested": 100,
                          "binary": {PRIMARY: {"average_precision":
                                               {"low": 0.03, "high": 0.16, "mean": 0.09,
                                                "n_valid": 100}}},
                          "aucpr_gain_over_baseline": {"low": -0.05, "high": 0.08,
                                                       "mean": 0.03, "n_valid": 100}},
            "calibration": {}, "partition": {"meets_minimum_anchors": True},
            "paired_differences_point": {}, "selected": {}, "preregistered": True,
            "held_out_description": "synthetic"}


@pytest.fixture
def root(tmp_path) -> Path:
    root = tmp_path / "root"
    for run, stage in ((S8, "stage8"), (S9, "stage9"), (S10, "stage10")):
        (root / "artifacts/runs" / run / "stages" / stage).mkdir(parents=True)
    s8 = root / "artifacts/runs" / S8
    cv = [{"schema_version": 1, "probe_name": f, "layer": layer, "task": "binary",
           "hyperparameter": 0.1, "fold": fold, "fold_seed": 1, "n_train": 40, "n_val": 20,
           "train_pos": 4, "val_pos": 2, "val_problems": "1,2",
           "average_precision": 0.05 + 0.01 * layer + 0.001 * fold, "auroc": 0.6,
           "spearman": None, "rmse": None, "converged": True}
          for f in (PRIMARY, "activation_only") for layer in (0, 22, 44)
          for fold in range(2)]
    write_parquet(s8 / "probe_models/cv_results.parquet", pa.Table.from_pylist(cv))
    write_json(s8 / "probe_models/frozen_selection.pending.json",
               {"binary": {"layer": 44, "C": 0.1, "probe_name": PRIMARY},
                "anchor_threshold": 1.35})
    s9 = root / "artifacts/runs" / S9
    for split in ("validation", "test"):
        write_parquet(s9 / f"predictions/{split}.parquet",
                      _predictions(len(split), split))
        write_json(s9 / f"metrics/{split}.json", _metrics(split))
    write_json(s9 / "metrics/test_receipt.json", {"run_id": S9, "freeze_sha256": "f" * 64})
    write_json(s9 / "probe_models/calibrator.json",
               {"method": "platt", "fitted_on": "validation",
                "parameters": {"A": -0.1, "B": 1.9}})
    write_json(s9 / "frozen_selection.json",
               {"binary": {"layer": 44, "C": 0.1, "probe_name": PRIMARY},
                "anchor_threshold": 1.35})
    s10 = root / "artifacts/runs" / S10
    leads = [{"schema_version": 1, "sentence_id": f"{i}:test", "problem_id": 1,
              "trajectory_id": "t1", "split": "test", "lead_sentences": lead,
              "threshold_quantile": 0.9, "layer": 44, "probe_name": PRIMARY,
              "task": "binary", "hyperparameter": 0.1, "model_source": "refit",
              "target_sentence_idx": i, "feature_sentence_id": f"{i}:test",
              "feature_sentence_idx": i, "feature_boundary_token_idx": i,
              "available": True, "unavailable_reason": None, "y_true": i % 2,
              "importance_true": 0.1, "score_raw": 0.2, "probability": 0.1 + 0.01 * i}
             for lead in (0, 1) for i in range(10)]
    write_parquet(s10 / "predictions/lead_time.parquet", pa.Table.from_pylist(leads))
    write_parquet(s10 / "metrics/sensitivities.parquet",
                  pa.Table.from_pylist([{"analysis": "threshold", "variant": "q90"}]))
    write_parquet(s10 / "metrics/permutations.parquet",
                  pa.Table.from_pylist([{"kind": "observed", "permutation_index": -1}]))
    labels = [{"schema_version": 1, "sentence_id": f"{i}:test", "trajectory_id": "t1",
               "problem_id": 1, "trace_kind": "correct", "sentence_idx": i,
               "text": f"sentence {i}", "eligible": True, "exclusion_reason": None,
               "importance_released": 0.1, "importance_legacy": 0.1,
               "importance_canonical": 0.1, "anchor": i % 2 == 0,
               "anchor_threshold": 1.35} for i in range(6)]
    write_parquet(root / "data/processed/stage4/labels.parquet",
                  pa.Table.from_pylist(labels))
    return root


def _render(root: Path, out: Path) -> dict:
    return render_report(root, out, stage9_run_id=S9, stage10_run_id=S10, stage8_run_id=S8)


def test_rebuild_is_deterministic_and_traceable(root, tmp_path, monkeypatch):
    import joblib

    def forbidden(*a, **k):  # the report must never load model payloads
        raise AssertionError("report accessed a joblib model payload")
    monkeypatch.setattr(joblib, "load", forbidden)

    first = _render(root, tmp_path / "a")
    second = _render(root, tmp_path / "b")
    # Regenerated report data, markdown and HTML are byte-identical.
    assert first["outputs"] == second["outputs"]
    # Plot-data specifications have identical hashes; PNGs are exempt by spec.
    for name, digest in first["report"]["figure_sha256"].items():
        if not name.endswith(".png"):
            assert second["report"]["figure_sha256"][name] == digest, name
    # Every figure records the hashes of its input artifacts.
    for name in first["report"]["figures"]:
        meta = json.loads((tmp_path / "a/figures" / f"{name}.json").read_bytes())[
            "layout"]["meta"]
        assert meta["input_sha256"], name
        for path, digest in meta["input_sha256"].items():
            assert first["report"]["input_sha256"][path] == digest
    # No model payload, joblib or activation shard was opened.
    for path in first["report"]["input_sha256"]:
        assert ".joblib" not in path and "activations/" not in path and "/units/" not in path


def test_inputs_are_untouched_and_sidecar_verified(root, tmp_path):
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    _render(root, tmp_path / "out")
    export_interface(root, tmp_path / "out/trajectory_exports", stage9_run_id=S9)
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    # A tampered input is refused via its sidecar.
    target = root / "artifacts/runs" / S9 / "metrics/test.json"
    target.write_bytes(target.read_bytes().replace(b"0.03", b"0.04", 1))
    with pytest.raises(ValueError):
        _render(root, tmp_path / "out2")


def test_export_interface_matches_vendored_convention(root, tmp_path):
    result = export_interface(root, tmp_path / "exports", stage9_run_id=S9)
    assert result["trajectories"] == 1
    steps = json.loads((tmp_path / "exports/problem_1/correct/t1/step_importance.json")
                       .read_bytes())
    assert [s["source_chunk_idx"] for s in steps] == list(range(6))
    assert all(s["target_impacts"] == [] for s in steps)  # pairwise impacts unmeasured
    assert steps[0]["source_chunk_text"] == "sentence 0"
    chunks = json.loads((tmp_path / "exports/problem_1/correct/t1/chunks_labeled.json")
                        .read_bytes())
    assert chunks[0]["importance"] == chunks[0]["importance_canonical"]
