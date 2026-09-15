"""Scientific-equivalence gate for a restored bundle.

Physical destination (region, filesystem, host, GPU, mount path) is recorded and never
compared. Scientific identity is recomputed from restored bytes with the project's own
logic and compared with what the producing runs recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

from preanchor import manifest as run_manifests
from preanchor.backup.bundle import load_bundle
from preanchor.backup.classify import stage3_hashes, stage_closure
from preanchor.backup.receipts import write_receipt
from preanchor.config import config_hash, read_yaml
from preanchor.ids import file_hash
from preanchor.io import UnitStore, read_json, validate_artifact, validate_parquet
from preanchor.manifest import load_manifest

FROZEN_STAGE5 = {"model_revision": "1df8507178afcc1bef68cd8c393f61a886323761",
                 "dtype": "bfloat16", "seed_root": 20260914, "temperature": 0.6,
                 "top_p": 0.95, "max_new_tokens": 16384, "add_special_tokens": False,
                 "max_model_len": 20480, "chat_template": None, "quantization": None}


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, category: str, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks.append({"category": category, "name": name,
                            "status": "pass" if ok else "info" if ok is None else "fail",
                            "detail": detail})

    def guard(self, category: str, name: str, function, *, info: bool = False):
        try:
            value = function()
        except Exception as exc:   # noqa: BLE001 - every failure is a verification failure
            self.add(category, name, False, f"{type(exc).__name__}: {exc}")
            return None
        self.add(category, name, None if info else True)
        return value

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "fail"]


def _entries_for(manifest: dict, for_stage: int) -> list[dict]:
    stages = {f"stage{s}" for s in stage_closure(for_stage)}
    return [e for e in manifest["entries"] if e["required"] and e["stage"] in stages]


def verify_artifacts(root: Path, manifest: dict, for_stage: int, report: Report) -> None:
    entries = _entries_for(manifest, for_stage)
    missing, mismatched, failed = [], [], []
    units: dict[str, Path] = {}
    for entry in entries:
        path = root / entry["path"]
        if path.is_symlink() or not path.is_file():
            missing.append(entry["path"])
            continue
        if file_hash(path) != entry["sha256"]:
            mismatched.append(entry["path"])
            continue
        try:
            if entry["validation"] == "sha256_sidecar":
                validate_artifact(path)
            elif entry["validation"] == "parquet":
                validate_parquet(path)
            elif entry["validation"] == "unit_file":
                units[entry["unit"]] = path.parent if entry["unit"] == str(
                    Path(entry["path"]).parent.as_posix()) else root / entry["unit"]
        except (ValueError, OSError, KeyError) as exc:
            failed.append(f"{entry['path']}: {exc}")
    report.add("artifact_integrity", "required entries present",
               not missing, f"{len(entries)} required; missing {len(missing)}"
               + (": " + ", ".join(missing[:5]) if missing else ""))
    report.add("artifact_integrity", "required entries hash to the manifest",
               not mismatched, ", ".join(mismatched[:5]))
    report.add("artifact_integrity", "checksum sidecars and Parquet schemas validate",
               not failed, "; ".join(failed[:5]))
    corrupt = []
    for unit, directory in units.items():
        try:
            UnitStore(directory.parent).validate(directory.name)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            corrupt.append(f"{unit}: {exc}")
    report.add("unit_integrity", f"UnitStore.validate on {len(units)} restored units",
               not corrupt, "; ".join(corrupt[:5]))


def verify_runs(root: Path, manifest: dict, for_stage: int, report: Report) -> dict[str, dict]:
    stages = {f"stage{s}" for s in stage_closure(for_stage)}
    loaded: dict[str, dict] = {}
    for run_id, info in sorted(manifest["bundle"]["runs"].items()):
        if info["role"] == "historical" or info["stage"] not in stages | {
                "stage5", "stage6", "stage7", "stage8", "stage9", "stage10"}:
            continue
        directory = root / "artifacts/runs" / run_id
        if not directory.is_dir():
            if info["stage"] in stages:
                report.add("run_identity", f"{run_id} present", False, "run directory missing")
            continue
        run_manifest = report.guard("run_identity", f"{run_id} manifest current view == revision",
                                    lambda d=directory: load_manifest(d))
        if not isinstance(run_manifest, dict):
            continue
        loaded[run_id] = run_manifest
        report.add("run_identity", f"{run_id} run_id preserved",
                   run_manifest.get("run_id") == run_id)
        revisions = [e["path"] for e in manifest["entries"]
                     if e["run_id"] == run_id and e["path"].startswith(
                         f"artifacts/runs/{run_id}/manifests/") and e["path"].endswith(".json")]
        chain_ok = all((root / p).is_file() and file_hash(root / p) == p.rsplit("/", 1)[1][:-5]
                       for p in revisions)
        report.add("run_identity", f"{run_id} manifest revision chain ({len(revisions)})",
                   chain_ok)
    return loaded


def verify_stage3(root: Path, repo: Path, manifest: dict, report: Report) -> dict | None:
    identity = manifest["scientific_identity"]
    recorded = identity.get("stage3") or {}
    if not recorded:
        report.add("input_identity", "Stage 3 identity recorded", False)
        return None
    current = stage3_hashes(root, repo)
    report.add("input_identity", "Stage 3 tables hash to the recorded identity",
               current == recorded,
               "" if current == recorded else f"recorded {recorded} != current {current}")
    return current


def verify_stage4(root: Path, manifest: dict, runs: dict, stage3: dict | None,
                  report: Report) -> None:
    recorded = manifest["scientific_identity"].get("stage4") or {}
    labels = root / "data/processed/stage4/labels.parquet"
    if not recorded or not labels.is_file():
        report.add("artifact_integrity", "Stage 4 labels present", False)
        return
    from preanchor.labels.stage4 import labels_fingerprint

    report.add("artifact_integrity", "Stage 4 labels SHA-256",
               file_hash(labels) == recorded.get("labels_sha256"))
    fingerprint = report.guard("artifact_integrity", "Stage 4 labels validate (fingerprint)",
                               lambda: labels_fingerprint(labels))
    report.add("artifact_integrity", "Stage 4 label fingerprint == bundle identity",
               fingerprint == recorded.get("fingerprint"))
    build = runs.get(recorded.get("build_run_id"))
    if build is None:
        report.add("run_identity", "Stage 4 build run loaded", False)
        return
    report.add("input_identity", "build run output hash == published labels",
               build["output_hashes"].get("stage4_labels") == file_hash(labels))
    if stage3:
        report.add("input_identity", "build run consumed the restored Stage 3 tables",
                   build["input_hashes"].get("stage3_trajectories")
                   == stage3["stage3/trajectories.parquet"]
                   and build["input_hashes"].get("stage3_sentences")
                   == stage3["stage3/sentences.parquet"])
    raw_id = build["input_hashes"].get("raw_run_id")
    raw = runs.get(raw_id)
    if raw is None:
        report.add("run_identity", f"raw run {raw_id} provenance present", False)
    else:
        source_manifest = (root / "artifacts/runs" / raw_id
                           / "stages/stage4_raw/source_manifest.json")
        digest = report.guard("input_identity", "raw run source manifest validates",
                              lambda: validate_artifact(source_manifest))
        report.add("input_identity", "raw run source manifest == build run inputs",
                   digest == build["input_hashes"].get("raw_source_manifest"))
        units = root / "artifacts/runs" / raw_id / "stages/stage4_raw/units"
        report.add("reacquire", "Stage 4 raw payload", None,
                   "present" if units.is_dir() else "absent: reacquire (REACQUIRE, not required)")


def verify_stage5(root: Path, repo: Path, manifest: dict, runs: dict, stage3: dict | None,
                  report: Report) -> dict:
    from preanchor.labels.generate import production_specs
    from preanchor.labels.pilot import DIGEST, inputs

    recorded = manifest["scientific_identity"].get("stage5") or {}
    run_id = recorded.get("run_id")
    run_manifest = runs.get(run_id)
    outcome = {"run_id": run_id, "rollout_ids_identical": False}
    if not run_id or run_manifest is None:
        report.add("plan_identity", "Stage 5 production run loaded", False)
        return outcome
    run = root / "artifacts/runs" / run_id
    stage = run / "stages/stage5"
    resolved_path = run / "resolved_config.yaml"
    resolved = report.guard("config_identity", "resolved_config.yaml validates",
                            lambda: (validate_artifact(resolved_path), read_yaml(resolved_path))[1])
    if not isinstance(resolved, dict):
        return outcome
    from preanchor.config import ExperimentConfig

    config = report.guard("config_identity", "resolved config parses under current code",
                          lambda: ExperimentConfig.model_validate(resolved))
    if not isinstance(config, ExperimentConfig):
        return outcome
    report.add("config_identity", "resolved-config hash == run identity",
               config.resolved_hash() == run_manifest["identity"]["resolved_config_hash"])
    report.add("config_identity", "scientific-config hash == run identity",
               config.scientific_hash() == run_manifest["identity"]["scientific_config_hash"])
    report.add("config_identity", "recorded resolved-config hash == bundle identity",
               manifest["scientific_identity"]["resolved_config_hashes"].get(run_id)
               == run_manifest["identity"]["resolved_config_hash"])
    frozen = {"model_revision": config.model.revision, "dtype": config.model.dtype,
              "seed_root": config.seed_root,
              "temperature": config.generation["temperature"],
              "top_p": config.generation["top_p"],
              "max_new_tokens": config.generation["max_new_tokens"],
              "add_special_tokens": False, "max_model_len": 20480,
              "chat_template": config.model.chat_template,
              "quantization": config.model.quantization}
    report.add("frozen_settings", "Stage 5 frozen scientific settings", frozen == FROZEN_STAGE5,
               json.dumps({k: v for k, v in frozen.items() if FROZEN_STAGE5.get(k) != v}))
    report.add("frozen_settings", "frozen settings == bundle identity",
               frozen == recorded.get("frozen"))
    pins = run_manifest.get("revisions") or {}
    report.add("pinned_identity", "model/tokenizer/dataset/MiniLM revisions == run config",
               pins == {"model": config.model.revision,
                        "tokenizer": config.model.tokenizer_revision,
                        "dataset": config.dataset.revision,
                        "embedding": config.labels["similarity_revision"]},
               json.dumps(pins))
    report.add("pinned_identity", "revisions == bundle identity",
               pins == manifest["scientific_identity"].get("revisions"))
    report.add("pinned_identity", "vLLM digest == pinned DIGEST",
               (run_manifest.get("vllm") or {}).get("digest") in (DIGEST, None)
               and (run_manifest.get("vllm") or {}).get("digest") is not None,
               str((run_manifest.get("vllm") or {}).get("digest")))
    # ---- Input identity: rehash restored inputs with the project's own logic ----
    fresh = report.guard("input_identity", "pilot.inputs on restored Stage 3 + targets",
                         lambda: inputs(root, repo))
    if not fresh:
        return outcome
    trajectory, targets, _, hashes = fresh
    report.add("input_identity", "fresh input hashes == manifest.input_hashes",
               hashes == run_manifest["input_hashes"],
               "" if hashes == run_manifest["input_hashes"]
               else f"{hashes} != {run_manifest['input_hashes']}")
    # ---- Plan and rollout identity: the critical acceptance criterion ----
    plan_path = next((stage / n for n in ("plan.json", "production_plan.json")
                      if (stage / n).is_file()), None)
    def load_plan():
        if plan_path is None:
            raise ValueError("plan missing")
        return read_json(plan_path)

    plan = report.guard("plan_identity", "Stage 5 plan validates", load_plan)
    if not isinstance(plan, dict):
        return outcome
    specs = production_specs(trajectory, targets, config.seed_root, config.resolved_hash(),
                             config_hash(hashes))
    recomputed = [s.rollout_id for s in specs]
    planned = [j["rollout_id"] for j in plan["jobs"]]
    identical = sorted(recomputed) == sorted(planned) and len(planned) == len(set(planned))
    ordered = recomputed == planned
    report.add("plan_identity", "recomputed production_specs rollout IDs == plan (sorted)",
               identical, f"{len(planned)} planned, {len(recomputed)} recomputed")
    report.add("plan_identity", "plan order == production_specs order", ordered)
    report.add("plan_identity", "sorted rollout-ID-set hash == bundle identity",
               config_hash(sorted(planned)) == recorded.get("rollout_id_set_sha256"))
    outcome["rollout_ids_identical"] = identical
    outcome["planned_units"] = len(planned)
    # ---- Unit integrity against the plan ----
    raw_root = stage / "units_raw"
    raw_keys = {p.parent.name for p in raw_root.glob("*/SUCCESS.json")}
    status = manifest["bundle"]["stages"]["stage5"]["status"]
    if status in ("generated", "finalized"):
        report.add("unit_integrity", "units_raw keys == plan", raw_keys == set(planned),
                   f"{len(raw_keys)} raw units")
    else:
        report.add("unit_integrity", "units_raw keys ⊆ plan (live checkpoint)",
                   raw_keys <= set(planned), f"{len(raw_keys)} of {len(planned)} protected")
    corrupt = []
    for key in sorted(raw_keys):
        try:
            UnitStore(raw_root).validate(key)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            corrupt.append(f"{key[:12]}: {exc}")
    report.add("unit_integrity", f"UnitStore.validate on {len(raw_keys)} raw units",
               not corrupt, "; ".join(corrupt[:3]))
    scored = {p.parent.name for p in (stage / "units").glob("*/SUCCESS.json")}
    if scored:
        bad = []
        for key in sorted(scored):
            try:
                UnitStore(stage / "units").validate(key)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                bad.append(f"{key[:12]}: {exc}")
        report.add("unit_integrity", f"UnitStore.validate on {len(scored)} scored units",
                   not bad and scored <= set(planned), "; ".join(bad[:3]))
    if status == "finalized":
        report.guard("artifact_integrity", "base_validation.json",
                     lambda: validate_artifact(stage / "base_validation.json"))
        if (stage / "fresh_labels.parquet").is_file():
            report.guard("artifact_integrity", "fresh_labels.parquet",
                         lambda: validate_parquet(stage / "fresh_labels.parquet") is not None)
    for benchmark in recorded.get("benchmark_run_ids") or []:
        report.add("run_identity", f"benchmark {benchmark} present", benchmark in runs)
    outcome["status"] = status
    outcome["raw_units"] = len(raw_keys)
    return outcome


def verify_stage6(root: Path, manifest: dict, runs: dict, stage3: dict | None,
                  report: Report) -> dict:
    """Stage 7 admission evidence: every canonical Stage 6 run re-validates on the restored
    bytes, consumed exactly the restored Stage 3/4 tables, and still passes the primary
    invariance gate. Nothing here starts Stage 7."""
    from preanchor.backup.classify import STAGE6_INPUTS, stage6_run_summary
    from preanchor.config import ExperimentConfig
    from preanchor.features.extract import validate_run

    recorded = manifest["scientific_identity"].get("stage6") or {}
    outcome = {"run_ids": recorded.get("run_ids") or [], "verified_runs": []}
    if not recorded:
        report.add("stage6_admission", "Stage 6 identity recorded", False)
        return outcome
    processed = root / "data/processed"
    current = {name: file_hash(processed / name) for name in STAGE6_INPUTS
               if (processed / name).is_file()}
    report.add("input_identity", "restored Stage 3/4 tables hash to the Stage 6 input identity",
               current == recorded.get("input_hashes"),
               "" if current == recorded.get("input_hashes") else f"{current}")
    traces = {}
    if (processed / "stage3/trajectories.parquet").is_file():
        table = validate_parquet(processed / "stage3/trajectories.parquet")
        traces = dict(zip(table.column("trajectory_id").to_pylist(),
                          table.column("full_cot_sha256").to_pylist(), strict=True))
    for run_id in outcome["run_ids"]:
        run_manifest = runs.get(run_id)
        run = root / "artifacts/runs" / run_id
        if run_manifest is None:
            report.add("stage6_admission", f"{run_id} loaded", False)
            continue
        expected = (recorded.get("runs") or {}).get(run_id) or {}
        report.add("input_identity", f"{run_id} consumed the restored Stage 3/4 tables",
                   run_manifest.get("input_hashes") == current)
        summary = report.guard("stage6_admission", f"{run_id} Stage 6 records validate "
                               "(summary, invariance report, projection, audit, index)",
                               lambda r=run: stage6_run_summary(r))
        if not isinstance(summary, dict):
            continue
        report.add("stage6_admission", f"{run_id} index, shard hashes and rows == bundle identity",
                   summary["index_sha256"] == expected.get("index_sha256")
                   and summary["shard_sha256"] == expected.get("shard_sha256")
                   and summary["n_rows"] == expected.get("n_rows"))
        report.add("stage6_admission", f"{run_id} primary invariance gate (same-shape mutation)"
                   " and logit identity pass", bool(summary["invariance"]["primary_pass"])
                   and bool(summary["invariance"]["logit_identity_pass"]),
                   json.dumps(summary["invariance"]))
        report.add("stage6_admission", f"{run_id} activation site and dimensions == bundle "
                   "identity", summary["activation_site"] == "block_post_pre_final_norm"
                   and summary["n_layers"] == expected.get("n_layers")
                   and summary["d_model"] == expected.get("d_model"),
                   f"{summary['activation_site']} {summary['n_layers']}+1 x {summary['d_model']}")
        resolved_path = run / "resolved_config.yaml"
        config = report.guard("config_identity", f"{run_id} resolved config validates and "
                              "parses under current code",
                              lambda p=resolved_path: (validate_artifact(p),
                                                       ExperimentConfig.model_validate(
                                                           read_yaml(p)))[1])
        if not isinstance(config, ExperimentConfig):
            continue
        report.add("config_identity", f"{run_id} resolved-config hash == run identity",
                   config.resolved_hash() == run_manifest["identity"]["resolved_config_hash"])
        validated = report.guard("unit_integrity", f"{run_id} validate-activations on restored "
                                 "units and index", lambda r=run, c=config: validate_run(r, c))
        if isinstance(validated, dict):
            index = validate_parquet(run / "activations/index.parquet").to_pylist()
            report.add("input_identity", f"{run_id} index trace hashes == restored Stage 3 "
                       "traces and labels hash == restored Stage 4 labels",
                       all(traces.get(r["trajectory_id"]) == r["trace_sha256"]
                           and r["labels_sha256"] == current.get("stage4/labels.parquet")
                           for r in index))
            outcome["verified_runs"].append(run_id)
    return outcome


def verify_stage7(root: Path, manifest: dict, runs: dict, stage3: dict | None,
                  report: Report) -> dict:
    """The canonical Stage 7 run re-validates on restored bytes with the project's own
    validators (validate-activations, validate-features), consumed exactly the restored
    Stage 3/4 tables, and its published feature table is the run's table."""
    from preanchor.backup.classify import STAGE6_INPUTS, stage7_run_summary
    from preanchor.config import ExperimentConfig
    from preanchor.features.dataset import validate_features
    from preanchor.features.extract import validate_run
    from preanchor.features.observables import feature_path

    recorded = manifest["scientific_identity"].get("stage7") or {}
    run_id = recorded.get("run_id")
    outcome = {"run_id": run_id, "verified": False}
    if not recorded or not run_id:
        report.add("stage7_dataset", "Stage 7 identity recorded", False)
        return outcome
    run_manifest = runs.get(run_id)
    run = root / "artifacts/runs" / run_id
    if run_manifest is None:
        report.add("stage7_dataset", f"{run_id} loaded", False)
        return outcome
    processed = root / "data/processed"
    current = {name: file_hash(processed / name) for name in STAGE6_INPUTS
               if (processed / name).is_file()}
    report.add("input_identity", f"{run_id} consumed the restored Stage 3/4 tables",
               run_manifest.get("input_hashes") == current == recorded.get("input_hashes"))
    summary = report.guard("stage7_dataset", f"{run_id} Stage 7 records validate (summary, "
                           "projection, index, feature table, audit, validation record)",
                           lambda r=run: stage7_run_summary(r))
    if not isinstance(summary, dict):
        return outcome
    report.add("stage7_dataset", f"{run_id} index, shard, feature hashes and rows == bundle "
               "identity", summary["index_sha256"] == recorded.get("index_sha256")
               and summary["features_sha256"] == recorded.get("features_sha256")
               and summary["shard_sha256"] == recorded.get("shard_sha256")
               and summary["n_rows"] == recorded.get("n_rows")
               and summary["n_units"] == recorded.get("n_units"),
               f"{summary['n_units']} units, {summary['n_rows']} rows")
    report.add("stage7_dataset", f"{run_id} activation site, dimensions, feature columns and "
               "head weights == bundle identity",
               summary["activation_site"] == "block_post_pre_final_norm"
               and summary["n_layers"] == recorded.get("n_layers")
               and summary["d_model"] == recorded.get("d_model")
               and summary["feature_columns"] == recorded.get("feature_columns")
               and summary["head_weights_sha256"] == recorded.get("head_weights_sha256"))
    published = feature_path(root)
    published_ok = published.is_file() and file_hash(published) == summary["features_sha256"]
    report.add("stage7_dataset", "published data/processed/stage7/features.parquet == the run's "
               "feature table and bundle identity", published_ok
               and summary["features_sha256"] == recorded.get("published_features_sha256"))
    resolved_path = run / "resolved_config.yaml"
    config = report.guard("config_identity", f"{run_id} resolved config validates and parses "
                          "under current code",
                          lambda p=resolved_path: (validate_artifact(p),
                                                   ExperimentConfig.model_validate(
                                                       read_yaml(p)))[1])
    if not isinstance(config, ExperimentConfig):
        return outcome
    report.add("config_identity", f"{run_id} resolved-config hash == run identity",
               config.resolved_hash() == run_manifest["identity"]["resolved_config_hash"])
    report.add("frozen_settings", f"{run_id} frozen extraction settings (49 sites, float16 "
               "store, bf16 sdpa, pinned revision)",
               config.features.get("layers") == list(range(49))
               and config.features.get("activation_dtype") == "float16"
               and config.model.dtype == "bfloat16" and config.model.quantization is None
               and config.model.attention_implementation == "sdpa"
               and config.model.revision == (manifest["scientific_identity"].get("revisions")
                                             or {}).get("model"))
    validated = report.guard("unit_integrity", f"{run_id} validate-activations on restored "
                             "units and index", lambda r=run, c=config: validate_run(r, c))
    features = report.guard("unit_integrity", f"{run_id} validate-features on the restored "
                            "table (coverage, joins, allow-list, invariants, published copy)",
                            lambda r=run, c=config: validate_features(r, c, root, sample=0,
                                                                      write_record=False))
    if isinstance(validated, dict) and isinstance(features, dict):
        traces = {}
        if (processed / "stage3/trajectories.parquet").is_file():
            table = validate_parquet(processed / "stage3/trajectories.parquet")
            traces = dict(zip(table.column("trajectory_id").to_pylist(),
                              table.column("full_cot_sha256").to_pylist(), strict=True))
        index = validate_parquet(run / "activations/index.parquet").to_pylist()
        report.add("input_identity", f"{run_id} index trace hashes == restored Stage 3 traces "
                   "and labels hash == restored Stage 4 labels",
                   all(traces.get(r["trajectory_id"]) == r["trace_sha256"]
                       and r["labels_sha256"] == current.get("stage4/labels.parquet")
                       for r in index))
        outcome.update(verified=True, rows=features["rows"], units=validated["units"])
    return outcome


def verify_stage8(root: Path, manifest: dict, runs: dict, report: Report) -> dict:
    """The canonical Stage 8 run re-validates on restored bytes (records, hashes, units) and
    the Stage 9 handoff loader accepts it against the restored Stage 7 inputs."""
    from preanchor.backup.classify import stage8_run_summary
    from preanchor.config import ExperimentConfig
    from preanchor.probes.stage9 import load_probe_run

    recorded = manifest["scientific_identity"].get("stage8") or {}
    run_id = recorded.get("run_id")
    outcome = {"run_id": run_id, "verified": False}
    if not recorded or not run_id or runs.get(run_id) is None:
        report.add("stage8_selection", "Stage 8 identity recorded and run loaded", False)
        return outcome
    run = root / "artifacts/runs" / run_id
    summary = report.guard("stage8_selection", f"{run_id} Stage 8 records validate (summary, "
                           "selection outputs, pending record, validate-probes)",
                           lambda r=run: stage8_run_summary(r))
    if not isinstance(summary, dict):
        return outcome
    report.add("stage8_selection", f"{run_id} selection outputs, threshold, layer and "
               "regularisation == bundle identity",
               summary["output_hashes"] == recorded.get("output_hashes")
               and summary["binary"] == recorded.get("binary")
               and summary["continuous"] == recorded.get("continuous")
               and summary["anchor_threshold"] == recorded.get("anchor_threshold")
               and summary["unit_keys"] == recorded.get("unit_keys"))
    config = report.guard("config_identity", f"{run_id} resolved config validates and parses "
                          "under current code",
                          lambda p=run / "resolved_config.yaml": (
                              validate_artifact(p),
                              ExperimentConfig.model_validate(read_yaml(p)))[1])
    if not isinstance(config, ExperimentConfig):
        return outcome
    handoff = report.guard("stage8_selection", f"{run_id} Stage 9 handoff loader accepts the "
                           "restored run against the restored Stage 7 inputs",
                           lambda r=run, c=config: load_probe_run(root, c, r))
    if isinstance(handoff, dict):
        outcome.update(verified=True, units=summary["n_units"])
    return outcome


def verify_stage9(root: Path, repo: Path, manifest: dict, runs: dict, report: Report) -> dict:
    """The canonical Stage 9 run re-validates read-only: freeze integrity against every
    restored artifact, validation/test membership and the exactly-once receipt."""
    from preanchor.backup.classify import stage9_run_summary
    from preanchor.config import ExperimentConfig
    from preanchor.probes.stage9 import validate_stage9

    recorded = manifest["scientific_identity"].get("stage9") or {}
    run_id = recorded.get("run_id")
    outcome = {"run_id": run_id, "verified": False}
    if not recorded or not run_id or runs.get(run_id) is None:
        report.add("stage9_freeze", "Stage 9 identity recorded and run loaded", False)
        return outcome
    run = root / "artifacts/runs" / run_id
    summary = report.guard("stage9_freeze", f"{run_id} Stage 9 records validate (freeze, "
                           "registry, calibrator, validation outputs, test receipt)",
                           lambda r=run: stage9_run_summary(r))
    if not isinstance(summary, dict):
        return outcome
    report.add("stage9_freeze", f"{run_id} freeze hash, outputs and test state == bundle "
               "identity", summary["freeze_sha256"] == recorded.get("freeze_sha256")
               and summary["output_hashes"] == recorded.get("output_hashes")
               and summary["test_evaluated"] == recorded.get("test_evaluated")
               and summary["test_receipt"] == recorded.get("test_receipt"))
    config = report.guard("config_identity", f"{run_id} resolved config validates and parses "
                          "under current code",
                          lambda p=run / "resolved_config.yaml": (
                              validate_artifact(p),
                              ExperimentConfig.model_validate(read_yaml(p)))[1])
    if not isinstance(config, ExperimentConfig):
        return outcome
    validated = report.guard("stage9_freeze", f"{run_id} validate-evaluation on the restored "
                             "run (read-only)",
                             lambda r=run, c=config: validate_stage9(r, repo, c,
                                                                     write_record=False))
    if isinstance(validated, dict):
        report.add("stage9_freeze", f"{run_id} validate-evaluation ok", bool(validated["ok"]),
                   "; ".join(validated.get("problems") or []))
        outcome.update(verified=bool(validated["ok"]),
                       test_evaluated=summary["test_evaluated"])
    return outcome


def verify_stage10(root: Path, repo: Path, manifest: dict, runs: dict, report: Report) -> dict:
    """The canonical Stage 10 run re-validates read-only against its restored Stage 9 input:
    output hashes, train-only thresholds, frozen-site-only models, lead-one mapping and the
    permutation record (``validate_stage10``)."""
    from preanchor.backup.classify import stage10_run_summary
    from preanchor.config import ExperimentConfig
    from preanchor.probes.stage10 import validate_stage10

    recorded = manifest["scientific_identity"].get("stage10") or {}
    run_id = recorded.get("run_id")
    outcome = {"run_id": run_id, "verified": False}
    if not recorded or not run_id or runs.get(run_id) is None:
        report.add("stage10_controls", "Stage 10 identity recorded and run loaded", False)
        return outcome
    run = root / "artifacts/runs" / run_id
    summary = report.guard("stage10_controls", f"{run_id} Stage 10 records validate (tables, "
                           "summary, figures)", lambda r=run: stage10_run_summary(r))
    if not isinstance(summary, dict):
        return outcome
    report.add("stage10_controls", f"{run_id} outputs, Stage 9 input and freeze == bundle "
               "identity", summary["output_hashes"] == recorded.get("output_hashes")
               and summary["stage9_run_id"] == recorded.get("stage9_run_id")
               and summary["freeze_sha256"] == recorded.get("freeze_sha256"))
    config = report.guard("config_identity", f"{run_id} resolved config validates and parses "
                          "under current code",
                          lambda p=run / "resolved_config.yaml": (
                              validate_artifact(p),
                              ExperimentConfig.model_validate(read_yaml(p)))[1])
    if not isinstance(config, ExperimentConfig):
        return outcome
    validated = report.guard("stage10_controls", f"{run_id} validate-sensitivities on the "
                             "restored run (read-only)",
                             lambda r=run, c=config: validate_stage10(r, repo, c,
                                                                      write_record=False))
    if isinstance(validated, dict):
        report.add("stage10_controls", f"{run_id} validate-sensitivities ok",
                   bool(validated["ok"]), "; ".join(validated.get("problems") or []))
        outcome.update(verified=bool(validated["ok"]))
    return outcome


def verify_source(repo: Path, manifest: dict, runs: dict, for_stage: int,
                  report: Report) -> bool:
    """Is the *current* checkout the one that can continue the producing run?"""
    identity = manifest["scientific_identity"]
    stage_runs = [r for r, info in manifest["bundle"]["runs"].items()
                  if info["stage"] == f"stage{for_stage}" and info["role"].endswith(
                      ("production", "build", "smoke")) and r in runs]
    try:
        source, _ = run_manifests.provenance(repo)
    except (ValueError, OSError) as exc:
        report.add("source_identity", "current checkout provenance", False, str(exc))
        return False
    lock = {name: entry["revision"] for name, entry in
            ((source.get("upstream") or {}).get("repositories") or {}).items()}
    if lock:
        report.add("source_identity", "submodule/vendor revisions == bundle identity",
                   lock == identity.get("upstream"), json.dumps(lock))
    ready = True
    for run_id in stage_runs:
        producing = runs[run_id]["source"]
        same = (source["git_sha"] == producing["git_sha"]
                and source["dirty_diff_hash"] == producing["dirty_diff_hash"]
                and source["lock_hash"] == producing["lock_hash"])
        detail = (f"run built at {producing['git_sha'][:12]} (dirty={producing.get('dirty')}), "
                  f"checkout at {source['git_sha'][:12]} (dirty={source['dirty']})")
        report.add("source_identity", f"checkout can resume {run_id} (git, dirty diff, uv.lock)",
                   None, detail + ("; MATCH" if same else "; checkout the producing commit"))
        ready = ready and same
    return ready


def restore_verify(root: Path, portable: Path, selector: str, *, repo: Path,
                   for_stage: int) -> dict:
    root, portable, repo = Path(root), Path(portable), Path(repo)
    report = Report()
    name, manifest = load_bundle(portable, selector)
    report.add("bundle", f"manifest {name} self-hash verified", True)
    report.add("bundle", "bundle covers the requested stage",
               manifest["bundle"]["for_stage"] >= for_stage,
               f"bundle for_stage {manifest['bundle']['for_stage']}")
    verify_artifacts(root, manifest, for_stage, report)
    runs = verify_runs(root, manifest, for_stage, report)
    stage3 = verify_stage3(root, repo, manifest, report)
    closure = stage_closure(for_stage)
    if 4 in closure:
        verify_stage4(root, manifest, runs, stage3, report)
    outcome = {}
    if 5 in closure:
        outcome = verify_stage5(root, repo, manifest, runs, stage3, report)
    stage6_outcome = {}
    if 6 in closure:
        stage6_outcome = verify_stage6(root, manifest, runs, stage3, report)
    stage7_outcome = {}
    if 7 in closure:
        stage7_outcome = verify_stage7(root, manifest, runs, stage3, report)
    stage8_outcome = {}
    if 8 in closure:
        stage8_outcome = verify_stage8(root, manifest, runs, report)
    stage9_outcome = {}
    if 9 in closure:
        stage9_outcome = verify_stage9(root, repo, manifest, runs, report)
    stage10_outcome = {}
    if 10 in closure:
        stage10_outcome = verify_stage10(root, repo, manifest, runs, report)
    ready = verify_source(repo, manifest, runs, for_stage, report)
    from preanchor.backup.prepare import physical_origin

    origin, destination = manifest["physical_origin"], physical_origin(root)
    report.add("physical", "origin recorded, destination recorded, never compared", None,
               f"origin fs={origin.get('filesystem')} host={origin.get('hostname')} "
               f"root={origin.get('root')}; destination fs={destination.get('filesystem')} "
               f"host={destination.get('hostname')} root={destination.get('root')}")
    stage_status = manifest["bundle"]["stages"].get(f"stage{for_stage}", {}).get("status")
    live = stage_status in ("running", "stopped")
    verified = not report.failed
    result = {"bundle_id": name, "manifest_sha256": name.rsplit("-", 1)[1],
              "for_stage": for_stage, "root": str(root), "checks": report.checks,
              "failed": report.failed, "verified": verified,
              "scientifically_equivalent": verified,
              "stage_status": stage_status, "live_checkpoint": live,
              "ready_to_continue": verified and ready,
              "continuation_note": (
                  "checkout must equal the producing commit with a clean tree and the "
                  "same uv.lock (check_resume)" if not ready else "checkout matches"),
              "reacquire": manifest["reacquire"], "stage5": outcome,
              "stage6": stage6_outcome, "stage7": stage7_outcome,
              "stage8": stage8_outcome, "stage9": stage9_outcome,
              "stage10": stage10_outcome,
              "physical_origin": origin, "physical_destination": destination}
    receipt = write_receipt(portable, "restore_verification", {
        "bundle_id": name, "manifest_sha256": result["manifest_sha256"],
        "run_ids": manifest["scientific_identity"].get("run_ids", []),
        "destination": str(root), "for_stage": for_stage,
        "object_count": len({e["sha256"] for e in _entries_for(manifest, for_stage)}),
        "verified_bytes": sum(e["size"] for e in _entries_for(manifest, for_stage)),
        "outcome": "verified" if verified else "failed", "ready_to_continue": result[
            "ready_to_continue"], "failed_checks": [c["name"] for c in report.failed],
        "physical_destination": destination})
    result["receipt"] = str(receipt)
    return result


def render(result: dict) -> str:
    lines = []
    for check in result["checks"]:
        mark = {"pass": "PASS", "fail": "FAIL", "info": "info"}[check["status"]]
        lines.append(f"[{mark}] {check['category']}: {check['name']}"
                     + (f" — {check['detail']}" if check["detail"] else ""))
    lines.append("")
    if result["verified"]:
        lines.append("RESTORE VERIFIED")
        lines.append("Scientifically equivalent portable state: PASS")
    else:
        lines.append("RESTORE NOT VERIFIED")
        lines.append("Scientifically equivalent portable state: FAIL")
    stage = result["for_stage"]
    ready = "YES" if result["ready_to_continue"] else "NO"
    lines.append(f"Ready to continue Stage {stage}: {ready} ({result['continuation_note']})")
    if result["live_checkpoint"]:
        lines.append(f"Stage {stage} status: {result['stage_status']} (live checkpoint, "
                     "not a finalized canonical stage)")
    lines.append("Reacquirable dependencies (not restored, not required for identity):")
    for item in result["reacquire"]:
        lines.append(f"  - {item['kind']}: {item['resource']} pin={json.dumps(item['pin'])}")
    return "\n".join(lines)
