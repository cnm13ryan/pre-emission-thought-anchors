"""Stage 8 end to end on a synthetic Stage 7 run: units, interruption, resume, corruption,
selection, serialisation, train-only rows and deterministic identity."""

import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from preanchor.config import load_config
from preanchor.features.dataset import build_dataset
from preanchor.features.extract import execute as extract_execute
from preanchor.features.extract import load_inputs, prepare, select_work
from preanchor.features.observables import head_from_model
from preanchor.io import read_json, validate_parquet, write_parquet
from preanchor.manifest import load_manifest
from preanchor.probes import fit, select
from preanchor.probes.validate import validate_probes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from tiny_qwen2 import tiny_loader, tiny_model, tiny_tokenizer, write_synthetic_root  # noqa: E402

SOURCE = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
          "package_freeze": []}
TRAIN = [2238, 1591, 4682, 6596, 3448]
VALIDATION = [330]
WORK = [f"{p}:{k}" for p in TRAIN + VALIDATION for k in ("correct", "incorrect")]
LIMIT = 8


def varied_labels(root):
    """Give the synthetic labels a non-constant continuous target (anchor unchanged)."""
    path = root / "data/processed/stage4/labels.parquet"
    table = validate_parquet(path)
    rows = table.to_pylist()
    for r in rows:
        r["importance_primary"] = 0.05 * r["sentence_idx"] + (1.0 if r["anchor"] else 0.0)
        r["importance_log1p"] = math.log1p(r["importance_primary"])
    for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
        Path(str(path) + suffix).unlink()
    write_parquet(path, pa.Table.from_pylist(rows, schema=table.schema))


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
    return load_config(ROOT / "configs/smoke.yaml")


@pytest.fixture
def stage7(tmp_path, config):
    root = tmp_path / "root"
    write_synthetic_root(root, config)
    varied_labels(root)
    inputs, hashes = load_inputs(root)
    run = prepare(root, ROOT, config, "test", select_work(config, inputs, WORK, LIMIT), hashes)
    extract_execute(run, ROOT, config, trajectories=WORK, limit_sentences=LIMIT,
                    tracking=False, loader=tiny_loader, diagnostics=False)
    build_dataset(run, ROOT, config, root, device="cpu", tracking=False,
                  head_loader=lambda c, d: head_from_model(tiny_model(), d),
                  tokenizer_loader=lambda c: tiny_tokenizer())
    return root, run


def unit_hashes(run):
    units = run / "probe_models/units"
    return {d.name: (d / "SUCCESS.json").read_bytes() for d in units.iterdir()
            if d.is_dir() and d.name not in ("quarantine", ".locks")}


def test_fit_resume_corruption_select_validate(stage7, config):
    root, s7 = stage7
    run = fit.prepare_run(root, ROOT, config, s7, "fit")
    assert load_manifest(run)["input_hashes"]["activation_run_id"] == s7.name
    with pytest.raises(KeyboardInterrupt):
        fit.execute(run, ROOT, config, tracking=False, workers=2, interrupt_after=3)
    assert load_manifest(run)["stages"]["stage8_fit"]["state"] == "failed"
    assert (run / "failure.json").is_file()
    before = unit_hashes(run)
    assert len(before) == 3
    with pytest.raises(ValueError, match="pass --resume"):
        fit.execute(run, ROOT, config, tracking=False, workers=2)
    summary = fit.execute(run, ROOT, config, tracking=False, workers=2, resume=True)
    after = unit_hashes(run)
    assert all(after[k] == v for k, v in before.items())
    n_sites = 5  # tiny model: embedding + 4 blocks
    assert len(after) == 5 + 2 * n_sites == len(summary["units"])
    assert sum(u["created"] for u in summary["units"]) == len(after) - 3
    assert summary["n_rows"] == len(TRAIN) * 2 * LIMIT and summary["n_positive"] == 10
    manifest = load_manifest(run)
    assert manifest["stages"]["stage8_fit"]["state"] == "complete"
    assert len(manifest["fold_seeds"]) == 4 and all(
        f["seed"] == summary["folds"][0]["seed"] for f in manifest["fold_seeds"])
    cv = validate_parquet(run / "probe_models/cv_results.parquet")
    pred = validate_parquet(run / "probe_models/fold_predictions.parquet")
    assert set(pred.column("split").to_pylist()) == {"train"}
    assert not {p for p in pred.column("problem_id").to_pylist()} & set(VALIDATION)
    prevalence = [r for r in cv.to_pylist() if r["probe_name"] == "prevalence"]
    assert len(prevalence) == 4 and all(
        abs(r["average_precision"] - r["val_pos"] / r["n_val"]) < 1e-12 for r in prevalence)
    n_hp = len(config.probes["logistic_C"]) + len(config.probes["ridge_alpha"])
    assert cv.num_rows == 4 * (1 + 4 * n_hp + 2 * n_sites * n_hp)
    # Corruption of a completed unit is quarantined and regenerated identically on resume.
    key = summary["units"][7]["key"]
    target = run / "probe_models/units" / key / "cv_results.parquet"
    original = target.read_bytes()
    target.write_bytes(original[:-16] + b"x" * 16)
    fit.execute(run, ROOT, config, tracking=False, workers=2, resume=True)
    quarantine = run / "probe_models/units/quarantine"
    assert len(list(quarantine.iterdir())) == 1
    assert target.read_bytes() == original
    assert unit_hashes(run)[key] == after[key]
    # Selection: primary family, residualised control, refit, pending freeze.
    pending = select.execute(run, ROOT, config, workers=2)
    assert pending["status"] == "pending" and pending["binary"]["probe_name"] == \
        "activation_plus_observables"
    assert pending["anchor_threshold"] == 0.5 and pending["threshold_quantile"] == 0.9
    assert 0 <= pending["binary"]["layer"] < n_sites
    assert set(pending["per_family_best"]) == {"prevalence", "position", "context",
                                               "uncertainty", "observable_only"} - {
        "observable_only"} | {"observable_combined", "activation_only",
                              "activation_plus_observables", "residualized_activation"}
    curve = pending["layer_curve"]["activation_plus_observables"]["binary"]
    assert [r["layer"] for r in curve] == list(range(n_sites))
    best = max(curve, key=lambda r: r["average_precision"])
    assert best["average_precision"] == pending["binary"]["mean_cv_average_precision"]
    with pytest.raises(ValueError, match="new run"):
        select.execute(run, ROOT, config, workers=2)
    report = validate_probes(run, config, root)
    assert report["ok"] and report["selected"] and report["units"] == len(after) + 1
    assert report["max_ap_recompute_diff"] < 1e-12
    # Serialisation round trip predicts valid probabilities on training inputs.
    import io

    import joblib

    payload = joblib.load(io.BytesIO((run / "probe_models/selected_model.joblib").read_bytes()))
    pipeline = payload["pipelines"]["activation_plus_observables:binary"]
    partition, loader, _ = fit.load_partition(root, config, s7)
    from preanchor.probes import models as M

    X = M.design("activation_plus_observables", partition, loader.site(pending["binary"]["layer"]))
    probs = pipeline.predict_proba(X)[:, 1]
    assert probs.shape == (len(partition),) and probs.min() >= 0 and probs.max() <= 1
    # Tampering with the pending record is detected.
    record = read_json(run / "probe_models/frozen_selection.pending.json")
    record["binary"]["C"] = 12345.0
    from preanchor.features.extract import write_current_json

    write_current_json(run / "probe_models/frozen_selection.pending.json", record)
    with pytest.raises(ValueError, match="binary selection"):
        validate_probes(run, config, root)


def test_deterministic_identity_across_runs(stage7, config):
    root, s7 = stage7
    tables = []
    for _ in range(2):
        run = fit.prepare_run(root, ROOT, config, s7, "fit")
        fit.execute(run, ROOT, config, tracking=False, workers=2, limit_sites=2)
        tables.append((validate_parquet(run / "probe_models/cv_results.parquet"),
                       validate_parquet(run / "probe_models/fold_predictions.parquet")))
    assert tables[0][0].equals(tables[1][0]) and tables[0][1].equals(tables[1][1])
    scores = np.array(tables[0][1].column("score_raw").to_pylist())
    assert np.isfinite(scores).all()


def test_refuses_changed_inputs_and_other_partitions(stage7, config):
    root, s7 = stage7
    run = fit.prepare_run(root, ROOT, config, s7, "fit")
    with pytest.raises(ValueError, match="No eligible rows|split"):
        fit.load_partition(root, config, s7, split="test")
    varied_labels(root)  # rewrites the labels file with identical content: same hash
    fit.execute(run, ROOT, config, tracking=False, workers=2, limit_sites=1)
    labels = root / "data/processed/stage4/labels.parquet"
    table = validate_parquet(labels)
    rows = table.to_pylist()
    rows[0]["importance_log1p"] += 1.0
    for suffix in ("", ".sha256", ".schema.json", ".schema.json.sha256"):
        Path(str(labels) + suffix).unlink()
    write_parquet(labels, pa.Table.from_pylist(rows, schema=table.schema))
    with pytest.raises(ValueError, match="labels changed"):
        fit.execute(run, ROOT, config, tracking=False, workers=2, resume=True, limit_sites=1)
