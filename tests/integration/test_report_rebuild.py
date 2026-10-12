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


def _metrics(split: str, freeze: str | None) -> dict:
    model = {"average_precision": 0.1, "auroc": 0.6, "brier": 0.07, "ece": 0.08,
             "log_loss": 0.33, "n": 60, "n_positive": 6, "prevalence": 0.1,
             "calibrated": True, "n_trajectories": 8, "n_trajectories_with_positive": 4,
             "ndcg_top10_percent": 0.2, "precision_top10_percent": 0.2, "per_trajectory": {},
             "reliability": []}
    return {"schema_version": 1, "split": split, "run_id": S9, "stage": "stage9",
            "freeze_sha256": freeze,
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
    cv_sha = write_parquet(s8 / "probe_models/cv_results.parquet", pa.Table.from_pylist(cv))
    pending_sha = write_json(s8 / "probe_models/frozen_selection.pending.json",
                             {"binary": {"layer": 44, "C": 0.1, "probe_name": PRIMARY},
                              "anchor_threshold": 1.35})
    s9 = root / "artifacts/runs" / S9
    # Lineage the report checks, in the order Stage 9 writes it: validation artifacts and the
    # calibrator are frozen by hash into frozen_selection.json; the freeze hash is the hash of
    # that file; the test receipt binds the freeze and the test predictions and metrics.
    val_pred = write_parquet(s9 / "predictions/validation.parquet",
                             _predictions(len("validation"), "validation"))
    val_metrics = write_json(s9 / "metrics/validation.json", _metrics("validation", None))
    calibrator = write_json(s9 / "probe_models/calibrator.json",
                            {"method": "platt", "fitted_on": "validation",
                             "parameters": {"A": -0.1, "B": 1.9}})
    freeze = write_json(s9 / "frozen_selection.json",
                        {"binary": {"layer": 44, "C": 0.1, "probe_name": PRIMARY},
                         "anchor_threshold": 1.35, "probe_run_id": S8,
                         "cv_results_sha256": cv_sha, "pending_selection_sha256": pending_sha,
                         "validation_predictions_sha256": val_pred,
                         "validation_metrics_sha256": val_metrics,
                         "calibration": {"calibrator_sha256": calibrator}})
    test_pred = write_parquet(s9 / "predictions/test.parquet",
                              _predictions(len("test"), "test"))
    test_metrics = write_json(s9 / "metrics/test.json", _metrics("test", freeze))
    write_json(s9 / "metrics/test_receipt.json",
               {"run_id": S9, "freeze_sha256": freeze, "predictions_sha256": test_pred,
                "metrics_sha256": test_metrics})
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
    produced = {
        "predictions/lead_time.parquet": write_parquet(
            s10 / "predictions/lead_time.parquet", pa.Table.from_pylist(leads)),
        "metrics/sensitivities.parquet": write_parquet(
            s10 / "metrics/sensitivities.parquet",
            pa.Table.from_pylist([{"analysis": "threshold", "variant": "q90"}])),
        "metrics/permutations.parquet": write_parquet(
            s10 / "metrics/permutations.parquet",
            pa.Table.from_pylist([{"kind": "observed", "permutation_index": -1}])),
    }
    write_json(s10 / "manifest.json",
               {"run_id": S10, "output_hashes": produced,
                "input_hashes": {"stage9_run_id": S9, "probe_run_id": S8,
                                 "stage9/freeze_sha256": freeze}})
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


def _figure_values(figure):
    """Exclude labels/layout, retain every original plotted coordinate and error bar."""
    return [
        {
            k: v
            for k, v in trace.items()
            if k in ("x", "y", "error_x", "error_y")
            and not (isinstance(v, list) and any(isinstance(x, str) for x in v))
        }
        for trace in figure["data"]
    ]


def test_presentation_preserves_original_numbers(root, tmp_path, monkeypatch):
    import plotly.graph_objects as go

    # Image rendering is covered by the rebuild test; keep this numerical guard quick.
    monkeypatch.setattr(go.Figure, "to_image", lambda *a, **k: b"presentation-test")
    original = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "stage11_presentation_original.json").read_text()
    )
    report = _render(root, tmp_path / "report")["report"]
    for key in ("metric_table", "primary", "bootstrap_test", "gain_over_baseline", "calibration"):
        assert report[key] == original[key], key
    for name, values in original["figure_values"].items():
        figure = json.loads((tmp_path / "report/figures" / f"{name}.json").read_text())
        assert _figure_values(figure) == values, name
    markdown = (tmp_path / "report/reports/mvp_result.md").read_text()
    assert markdown.index("What did we learn?") < markdown.index("Technical details")
    assert "do not clearly establish an improvement" in markdown
    assert "Base rate (no skill)" in markdown
    for name in report["figures"]:
        html = (tmp_path / "report/figures" / f"{name}.html").read_text()
        for label in (
            "Question",
            "No useful signal",
            "Observed",
            "Uncertainty",
            "Can conclude",
            "Cannot conclude",
        ):
            assert f"{label}:" in html


def test_interval_language_handles_other_results():
    from preanchor.report.language import improvement

    assert "do not clearly establish" in improvement(-0.05, 0.08)
    assert "do not clearly establish" in improvement(0, 0.08)
    assert "below zero" in improvement(-0.1, -0.01)
    assert "above zero" in improvement(0.01, 0.08)
    assert "unknown" in improvement(None, None)


def test_controls_restyle_without_changing_values():
    from preanchor.report import figures as figures
    from preanchor.viz import stage10

    rows = []
    for split in ('test', 'validation'):
        for analysis, variant in (('threshold', 'q90'), ('position_quintile', '1'),
                                  ('entropy_quintile', '1'),
                                  ('lead_time', 'lead0_frozen_same_rows')):
            for model in (PRIMARY, 'observable_combined', 'prevalence'):
                rows.append({'analysis': analysis, 'variant': variant, 'split': split,
                             'model_source': 'frozen_stage9', 'probe_name': model,
                             'metric': 'average_precision', 'comparison': None,
                             'value': 0.2, 'boot_low': 0.1, 'boot_high': 0.3,
                             'n_rows': 20, 'n_positive': 2})
    perms = [{'kind': kind, 'split': split, 'average_precision': 0.2,
              'n_rows': 20, 'n_positive': 2}
             for split in ('test', 'validation') for kind in ('observed', 'permutation')]
    source = {'sensitivities': pa.Table.from_pylist(rows),
              'permutations': pa.Table.from_pylist(perms)}
    before = {
        'threshold_sensitivity': stage10.threshold_figure(rows),
        'lead_time_performance': stage10.lead_time_figure(rows),
        'position_entropy_controls': stage10.position_entropy_figure(rows),
        'permutation_null': stage10.permutation_figure(perms),
    }
    after = figures.control_figures(source, {'saved-input': 'a' * 64})
    repeated = figures.control_figures(source, {'saved-input': 'a' * 64})
    for name in before:
        assert _figure_values(json.loads(before[name].to_json())) == _figure_values(
            json.loads(after[name].to_json()))
        assert after[name].to_json() == repeated[name].to_json()
        assert 'base rate' in after[name].to_json()
    groups = after['position_entropy_controls']
    assert 'Sentence position group / test' in [a.text for a in groups.layout.annotations]
    assert 'quintile' not in groups.to_json()
    colors = {}
    for trace in groups.data:
        colors.setdefault(trace.name, set()).add(trace.marker.color)
    assert all(len(values) == 1 for values in colors.values())


def _replace_json(path: Path, value: dict) -> None:
    """Artifacts are write-once; tests that tamper must remove the file and its sidecar."""
    for stale in (path, Path(str(path) + ".sha256")):
        stale.unlink(missing_ok=True)
    write_json(path, value)


def test_mismatched_stage_runs_are_refused(root, tmp_path):
    # Codex review on PR #5: each run is sidecar-valid on its own, but runs from different
    # lineages must not be rendered as one report.
    other8 = "20990101T000000Z-other-stage08"
    src8, dst8 = root / "artifacts/runs" / S8, root / "artifacts/runs" / other8
    for path in src8.rglob("*"):
        if path.is_file() and path.suffix in (".parquet", ".json") \
                and not path.name.endswith(".schema.json"):
            target = dst8 / path.relative_to(src8)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".parquet":
                write_parquet(target, pa.Table.from_pylist([{"x": 1}]))
            else:
                write_json(target, {"other": True})
    with pytest.raises(ValueError, match="not one lineage"):
        render_report(root, tmp_path / "a", stage9_run_id=S9, stage10_run_id=S10,
                      stage8_run_id=other8)
    # A Stage 10 run that consumed a different Stage 9 freeze is refused too.
    manifest10 = json.loads((root / "artifacts/runs" / S10 / "manifest.json").read_bytes())
    _replace_json(root / "artifacts/runs" / S10 / "manifest.json",
                  {**manifest10, "input_hashes": {**manifest10["input_hashes"],
                                                  "stage9/freeze_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="different Stage 9 freeze"):
        _render(root, tmp_path / "b")


def test_freeze_hash_must_match_frozen_selection(root, tmp_path):
    s9 = root / "artifacts/runs" / S9
    receipt = json.loads((s9 / "metrics/test_receipt.json").read_bytes())
    _replace_json(s9 / "metrics/test_receipt.json", {**receipt, "freeze_sha256": "e" * 64})
    with pytest.raises(ValueError, match="test_receipt.json freeze hash"):
        _render(root, tmp_path / "out")


def test_frozen_validation_artifacts_must_match(root, tmp_path):
    # Codex re-review: sidecar-valid validation results from another run must be refused.
    s9 = root / "artifacts/runs" / S9
    other = json.loads((s9 / "metrics/validation.json").read_bytes())
    _replace_json(s9 / "metrics/validation.json", {**other, "run_id": "another-run"})
    with pytest.raises(ValueError, match="metrics/validation.json does not match"):
        _render(root, tmp_path / "out")


def test_dashboard_lineage_follows_stage10_manifest(root):
    from preanchor.report.loaders import lineage_from_stage10

    assert lineage_from_stage10(root, S10) == {"stage10": S10, "stage9": S9, "stage8": S8}


def test_pilot_status_reflects_stage5_artifacts(tmp_path):
    from preanchor.report import interpretation as I
    from preanchor.report.loaders import Sources, find_stage5_pilot, load_stage5_pilot

    assert find_stage5_pilot(tmp_path) is None
    assert I.pilot_status(None) == (False, I.PILOT_STATUS)
    run = "20990101T000000Z-fake-stage05"
    stage5 = tmp_path / "artifacts/runs" / run / "stages/stage5"
    # A resumed run: the final session produced 56 completions, 200 were durable before it.
    summary = {"run_id": run, "namespace": "production", "completed": 56,
               "resumed_prior_completions": 200, "completed_total": 256,
               "valid_checksums": 256, "request_failures": 1}
    write_json(stage5 / "summary.json", summary)
    write_json(stage5 / "plan.json", {"jobs": [{"fork_idx": i % 8} for i in range(256)]})
    write_parquet(stage5 / "fresh_labels.parquet",
                  pa.Table.from_pylist([{"sentence_idx": i} for i in range(5)]))
    assert find_stage5_pilot(tmp_path) == run
    pilot = {"run_id": run, **load_stage5_pilot(Sources(tmp_path), run)}
    verified, text = I.pilot_status(pilot)
    assert verified and "256 of 256" in text and "5 sentences" in text
    _replace_json(stage5 / "summary.json",
                  {**summary, "completed_total": 200, "valid_checksums": 200})
    pilot = {"run_id": run, **load_stage5_pilot(Sources(tmp_path), run)}
    assert I.pilot_status(pilot)[0] is False


def test_stage10_tables_must_be_the_manifest_outputs(root, tmp_path):
    # Codex re-review: sidecar-valid Stage 10 tables copied from another run are refused.
    s10 = root / "artifacts/runs" / S10
    table = s10 / "metrics/permutations.parquet"
    for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
        Path(str(table) + suffix).unlink(missing_ok=True)
    write_parquet(s10 / "metrics/permutations.parquet",
                  pa.Table.from_pylist([{"kind": "observed", "permutation_index": -2}]))
    with pytest.raises(ValueError, match="permutations.parquet does not match the manifest"):
        _render(root, tmp_path / "out")


def test_render_refuses_a_non_empty_output_directory(root, tmp_path):
    # Codex re-review: rerendering must not unlink a persisted report's figures.
    _render(root, tmp_path / "out")
    before = {p: p.read_bytes() for p in (tmp_path / "out").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="non-empty directory"):
        _render(root, tmp_path / "out")
    after = {p: p.read_bytes() for p in (tmp_path / "out").rglob("*") if p.is_file()}
    assert before == after
