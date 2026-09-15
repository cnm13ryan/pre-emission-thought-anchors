"""Stage 10 end to end on the synthetic lineage (Stage 7 -> 8 -> 9 -> 10): every analysis,
the frozen-primary protection, resumable immutable units, determinism and validation."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from preanchor.config import load_config
from preanchor.io import read_json, validate_artifact, validate_parquet
from preanchor.manifest import load_manifest
from preanchor.probes import stage9 as S9
from preanchor.probes import stage10 as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from synthetic_pipeline import SOURCE, build_lineage  # noqa: E402


def tree_hashes(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != ".run.lock"}


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("preanchor.manifest.provenance", lambda repo: (SOURCE.copy(), b""))
        # Small paired bootstrap keeps the synthetic run fast; the design is unchanged.
        original = S.clustered_bootstrap
        mp.setattr(S, "clustered_bootstrap",
                   lambda **kw: original(**{**kw, "replicates": 40}))
        config = load_config(ROOT / "configs/smoke.yaml")
        root, s7, s8 = build_lineage(tmp_path_factory.mktemp("lineage"), ROOT, config,
                                     train_quantile_threshold=True)
        s9 = S9.prepare_run(root, ROOT, config, s8)
        S9.calibrate(s9, ROOT, config)
        sha = S9.freeze(s9, ROOT, config)["freeze_sha256"]
        S9.evaluate_test(s9, ROOT, config, sha)
        assert S9.validate_stage9(s9, ROOT, config)["ok"]
        yield config, root, s8, s9


def test_stage10_end_to_end_preserves_the_frozen_primary(lineage):
    config, root, s8, s9 = lineage
    before9, before8 = tree_hashes(s9), tree_hashes(s8)
    frozen = read_json(s9 / S9.FREEZE_FILE)
    run = S.prepare_run(root, ROOT, config, s9)
    manifest = load_manifest(run)
    assert manifest["input_hashes"]["stage9_run_id"] == s9.name
    assert manifest["input_hashes"]["stage9/freeze_sha256"] == validate_artifact(
        s9 / S9.FREEZE_FILE)
    summary = S.execute(run, ROOT, config, workers=2)
    # Nothing in the Stage 9 or Stage 8 run changed; the freeze is byte-identical.
    assert tree_hashes(s9) == before9 and tree_hashes(s8) == before8
    assert read_json(s9 / S9.FREEZE_FILE) == frozen
    assert S9.validate_stage9(s9, ROOT, config, write_record=False)["ok"]
    for name in ("frozen_selection.json", "predictions/test.parquet", "metrics/test.json"):
        assert not (run / name).exists()
    for name in S.STAGE11_MARKERS:
        assert not (run / name).exists()
    # Outputs.
    table = validate_parquet(run / S.SENSITIVITIES_FILE)
    perms = validate_parquet(run / S.PERMUTATIONS_FILE)
    leads = validate_parquet(run / S.LEAD_FILE)
    assert table.schema.equals(S.SENSITIVITY_SCHEMA)
    rows = table.to_pylist()
    analyses = {r["analysis"] for r in rows}
    assert analyses == {"threshold", "convergence", "replacement_quality",
                        "trajectory_subgroup", "position_quintile", "entropy_quintile",
                        "lead_time", "paired_baseline_differences"}
    for name in ("threshold_sensitivity", "lead_time", "permutation_null",
                 "position_entropy_controls"):
        assert (run / "figures" / f"{name}.html").is_file()
        assert (run / "figures" / f"{name}.json").is_file()
    # Thresholds: train-only quantiles; q90 reproduces the frozen numeric threshold.
    thresholds = summary["train_thresholds"]
    assert set(thresholds) == {"q80", "q90", "q95"}
    assert thresholds["q90"] == frozen["anchor_threshold"]
    train_rows = [r for r in validate_parquet(
        root / "data/processed/stage4/labels.parquet").to_pylist()
        if r["eligible"] and r["problem_id"] in config.splits.train and r["sentence_idx"] < 8]
    for q in (0.8, 0.9, 0.95):
        assert thresholds[f"q{int(round(q * 100))}"] == pytest.approx(float(np.quantile(
            [r["importance_primary"] for r in train_rows], q)))
    for r in rows:
        if r["analysis"] == "threshold":
            d = json.loads(r["detail"])
            assert d["threshold_source"] == "train_eligible_importance_quantile"
            assert r["anchor_threshold"] == thresholds[r["variant"]]
            assert r["threshold_quantile"] == float(r["variant"][1:]) / 100
        else:
            assert r["anchor_threshold"] == frozen["anchor_threshold"]
    # Only the frozen site and hyperparameter, for every activation model everywhere.
    layer, C = frozen["binary"]["layer"], frozen["binary"]["C"]
    import io

    import joblib
    payload = joblib.load(io.BytesIO((s8 / S9.MODEL_FILE).read_bytes()))
    frozen_sites = {k.split(":")[0]: v["layer"] for k, v in payload["models"].items()
                    if v["task"] == "binary" and v["layer"] is not None}
    for r in rows:
        if r["probe_name"] in frozen_sites:
            assert r["layer"] == frozen_sites[r["probe_name"]], r
    assert {r["layer"] for r in rows if r["probe_name"] == S.PRIMARY_MODEL} == {layer}
    assert {r["hyperparameter"] for r in rows if r["probe_name"] == S.PRIMARY_MODEL} == {C}
    assert {r["layer"] for r in perms.to_pylist()} == {layer}
    assert {r["model_source"] for r in rows} == {S.FROZEN, S.REFIT}
    # Convergence: K=4 loses exactly the last eligible sentence of every trajectory.
    k4 = next(json.loads(r["detail"]) for r in rows if r["variant"] == "k4")
    k5 = next(json.loads(r["detail"]) for r in rows if r["variant"] == "k5")
    for split, lost in k4["rows_lost_vs_k5"].items():
        assert lost == k4["eligible_rows"][split] / 7 and k5["rows_lost_vs_k5"][split] == 0
    # Replacement filter counts are reported for both the retained and the excluded rows.
    repl = {r["variant"]: json.loads(r["detail"]) for r in rows
            if r["analysis"] == "replacement_quality"}
    assert set(repl) == {"supported", "unsupported"}
    for split in ("validation", "test"):
        c = repl["supported"]["counts"][split]
        n_split = sum(1 for r in rows if r["analysis"] == "paired_baseline_differences"
                      and r["split"] == split and r["metric"] == "prevalence"
                      and r["probe_name"] == S.PRIMARY_MODEL)
        assert n_split == 1
        assert c["supported"] + c["excluded"] == next(
            r["n_rows"] for r in rows if r["analysis"] == "paired_baseline_differences"
            and r["split"] == split)
    # Subgroups partition the rows; quintiles cover every row exactly once.
    for split in ("validation", "test"):
        total = next(r["n_rows"] for r in rows if r["analysis"] == "paired_baseline_differences"
                     and r["split"] == split)
        for analysis, variants in (("trajectory_subgroup", ("correct", "incorrect")),
                                   ("position_quintile", tuple("12345")),
                                   ("entropy_quintile", tuple("12345"))):
            counts = {}
            for v in variants:
                group = [r for r in rows if r["analysis"] == analysis and r["variant"] == v
                         and r["split"] == split]
                assert group, (analysis, v, split)
                counts[v] = group[0]["n_rows"]
            assert sum(counts.values()) == total
    # Lead one: every target maps to S_(i-1); sentence zero is unavailable, never shifted.
    lead_rows = leads.to_pylist()
    one = [r for r in lead_rows if r["lead_sentences"] == 1 and r["probe_name"] == S.PRIMARY_MODEL]
    zero = [r for r in lead_rows if r["lead_sentences"] == 0]
    assert len(one) == len(zero) and {r["sentence_id"] for r in one} == {
        r["sentence_id"] for r in zero}
    for r in one:
        if r["target_sentence_idx"] == 0:
            assert not r["available"] and r["score_raw"] is None
            assert r["unavailable_reason"] == "first_sentence_has_no_predecessor"
        else:
            assert r["available"] and r["feature_sentence_idx"] == r["target_sentence_idx"] - 1
            expected_id = f"{r['trajectory_id']}:{r['target_sentence_idx'] - 1}"
            assert r["feature_sentence_id"] == expected_id
            assert r["score_raw"] is not None
    unavailable = summary["records"]["lead_time"]["unavailable"]
    assert unavailable == {"validation": 8, "test": 8}      # one sentence zero per trajectory
    assert {r["layer"] for r in one} == {layer}
    # Permutations: exactly 100 requested, derived seeds, prespecified strata, observed rows.
    prow = perms.to_pylist()
    indices = sorted({r["permutation_index"] for r in prow if r["kind"] == "permutation"})
    assert indices == list(range(100))
    assert all(r["seed"] == S.permutation_seed(config, r["permutation_index"])
               for r in prow if r["kind"] == "permutation")
    assert {r["strata"] for r in prow} == {"problem_id,trace_kind,position_quintile"}
    assert {r["split"] for r in prow if r["kind"] == "observed"} == {"validation", "test"}
    assert all(r["labels_changed"] > 0 for r in prow if r["kind"] == "permutation")
    p = summary["records"]["permutations"]
    assert p["requested"] == 100 and p["n_strata"] == p["n_permutable_strata"] + \
        p["n_single_class_strata"]
    for split in ("validation", "test"):
        assert p["splits"][split]["effective_permutations"] == 100
        assert p["splits"][split]["observed_average_precision"] is not None
    # Paired differences equal the Stage 9 point values (asserted inside the unit as well).
    test_metrics = read_json(s9 / "metrics/test.json")
    paired = {r["comparison"]: r["value"] for r in rows
              if r["analysis"] == "paired_baseline_differences" and r["split"] == "test"
              and r["metric"] == "average_precision" and r["comparison"]}
    for family, value in test_metrics["paired_differences_point"].items():
        assert paired[f"minus:{family.split('-minus-')[1]}"] == pytest.approx(
            value["average_precision"])
    # Validation record.
    report = S.validate_stage10(run, ROOT, config)
    assert report["ok"], report["problems"]
    assert report["selected_layer"] == layer and report["stage9"]["unchanged"]
    assert S9.validate_stage9(s9, ROOT, config, write_record=False)["ok"]
    assert tree_hashes(s9) == before9


def test_resume_reuses_valid_units_and_regenerates_corrupt_ones(lineage):
    config, root, s8, s9 = lineage
    run = S.prepare_run(root, ROOT, config, s9)
    with pytest.raises(KeyboardInterrupt):
        S.execute(run, ROOT, config, workers=2, interrupt_after=3)
    assert load_manifest(run)["stages"][S.RUN_STAGE]["state"] == "failed"
    assert (run / "failure.json").is_file()
    store = run / S.UNITS_DIR
    done = sorted(p for p in store.iterdir() if p.is_dir() and p.name != ".locks")
    assert len(done) == 3
    hashes_before = {p.name: tree_hashes(p) for p in done}
    # Corrupt one completed unit; it must be quarantined and regenerated, the others reused.
    victim = done[1]
    (victim / "rows.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pass --resume"):
        S.execute(run, ROOT, config, workers=2)
    S.execute(run, ROOT, config, workers=2, resume=True)
    assert load_manifest(run)["stages"][S.RUN_STAGE]["state"] == "complete"
    assert any(p.name.startswith(victim.name) for p in (store / "quarantine").iterdir())
    for p in done:
        if p.name != victim.name:
            assert tree_hashes(p) == hashes_before[p.name]
    regenerated = tree_hashes(victim)
    regenerated.pop("SUCCESS.json", None)
    regenerated.pop("SUCCESS.json.sha256", None)
    original = hashes_before[victim.name]
    assert regenerated["rows.parquet"] == original["rows.parquet"]
    summary = S.read_summary(run)
    assert summary["units"] and sum(u["created"] for u in summary["units"]) == len(
        S.unit_specs(S.check_spec(config))) - 2
    # A second complete resume is a no-op that reproduces the same aggregate bytes.
    digest = validate_artifact(run / S.SENSITIVITIES_FILE)
    S.execute(run, ROOT, config, workers=2, resume=True)
    assert validate_artifact(run / S.SENSITIVITIES_FILE) == digest
    assert S.validate_stage10(run, ROOT, config, write_record=False)["ok"]


def test_outputs_are_deterministic_across_runs(lineage):
    config, root, s8, s9 = lineage
    runs = [S.prepare_run(root, ROOT, config, s9) for _ in range(2)]
    for run in runs:
        S.execute(run, ROOT, config, workers=2)
    a, b = (validate_parquet(r / S.SENSITIVITIES_FILE) for r in runs)
    assert a.equals(b)
    pa_, pb_ = (validate_parquet(r / S.PERMUTATIONS_FILE) for r in runs)
    assert pa_.equals(pb_)
    la, lb = (validate_parquet(r / S.LEAD_FILE) for r in runs)
    assert la.equals(lb)


def test_stage10_refuses_a_stage9_run_without_a_test_receipt(lineage):
    config, root, s8, s9 = lineage
    unfinished = S9.prepare_run(root, ROOT, config, s8)
    with pytest.raises(S9.TestLockError):
        S.prepare_run(root, ROOT, config, unfinished)
    S9.calibrate(unfinished, ROOT, config)
    S9.freeze(unfinished, ROOT, config)
    with pytest.raises(ValueError, match="not complete"):
        S.prepare_run(root, ROOT, config, unfinished)
    # Discovery counts receipted runs only: the frozen-but-untested run is ignored.
    assert S.discover_stage9(root) == s9
