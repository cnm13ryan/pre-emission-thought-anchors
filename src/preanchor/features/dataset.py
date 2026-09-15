"""SPEC Stage 7 dataset: observables joined to labels through stable sentence IDs.

``build_dataset`` runs after a complete extraction stage in the same run: it re-validates
every activation unit, tokenises each stored trace once (the boundary token IDs are the
only tokens it reads), computes the SPEC §3.9 observables in float32 from the stored
``layer=48`` boundary states, joins the Stage 4 targets by ``sentence_id`` and writes
``features/features.parquet`` plus ``features/feature_audit.json``; the table is published
to ``data/processed/stage7/features.parquet``. ``validate_features`` re-checks the
published table without the model and recomputes a seeded sample from the shards.
"""

from __future__ import annotations

import math
import resource
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from preanchor.config import ExperimentConfig
from preanchor.features.extract import (
    INPUT_FILES,
    load_inputs,
    stage_name,
    validate_run,
    write_current_json,
)
from preanchor.features.observables import (
    DIAGNOSTIC_FEATURES,
    PRIMARY_FEATURES,
    UNCERTAINTY_FEATURES,
    check_feature_names,
    feature_path,
    features_table,
    load_head,
    prompt_token_count,
    trajectory_features,
)
from preanchor.features.store import read_shard
from preanchor.ids import derived_seed, file_hash
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    read_json,
    validate_parquet,
    write_json,
    write_parquet,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, load_manifest, save_manifest, utc_now
from preanchor.model.boundaries import token_ids_sha256, tokenize_trace
from preanchor.model.load import load_tokenizer
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking

BATCH_ROWS = 256


def features_stage(config: ExperimentConfig) -> str:
    return f"{stage_name(config)}_features"


def split_of(config: ExperimentConfig) -> dict[int, str]:
    return {**{p: "train" for p in config.splits.train},
            **{p: "validation" for p in config.splits.validation},
            **{p: "test" for p in config.splits.test}}


def default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _unit_rows(index: list[dict]) -> dict[str, list[dict]]:
    by_unit: dict[str, list[dict]] = {}
    for row in index:
        by_unit.setdefault(row["shard_path"].split("/")[1], []).append(row)
    for rows in by_unit.values():
        rows.sort(key=lambda r: r["row_idx"])
        if [r["row_idx"] for r in rows] != list(range(len(rows))):
            raise ValueError("Index rows of a unit are not contiguous")
    return by_unit


def _logits_in_batches(head, states: np.ndarray, batch: int = BATCH_ROWS):
    import torch

    pieces = [head.logits(torch.from_numpy(np.ascontiguousarray(states[i:i + batch])))
              for i in range(0, len(states), batch)]
    return torch.cat(pieces) if pieces else torch.empty((0, head.vocab_size))


def unit_features(head, tokenizer, config: ExperimentConfig, trajectory: dict, meta: dict,
                  tensors: dict, n_sentences: int) -> tuple[list[dict], dict]:
    """Observables for one stored unit from its shard and the once-tokenised trace."""
    ids, offsets = tokenize_trace(tokenizer, trajectory["full_cot"])
    if meta["token_ids_sha256"] != token_ids_sha256(ids):
        raise ValueError("Stored token-id hash differs from the current tokenisation")
    if meta["trace_sha256"] != trajectory["full_cot_sha256"]:
        raise ValueError("Stored trace hash differs from the Stage 3 trace")
    boundaries = meta["boundaries"]
    prompt_tokens = prompt_token_count(offsets, trajectory["prompt"], trajectory["full_cot"])
    if tokenizer.decode(ids[:prompt_tokens]) != trajectory["prompt"]:
        raise ValueError("Prompt token count does not reproduce the stored prompt")
    resid = tensors["resid"]
    final = resid[:, resid.shape[1] - 1]
    prev = tensors["final_prev_position"]
    available = [bool(x) for x in meta["final_prev_available"]]
    prev_states = np.where(np.asarray(available)[:, None], prev, 0).astype(np.float16)
    b_idx = [int(b["boundary_token_idx"]) for b in boundaries]
    rows = trajectory_features(
        boundaries, prompt_tokens=prompt_tokens, n_sentences=n_sentences,
        context_length=config.model.context_length,
        boundary_logits=_logits_in_batches(head, final),
        prev_logits=_logits_in_batches(head, prev_states), prev_available=available,
        boundary_token_ids=[ids[b] for b in b_idx], first_target_ids=[ids[b + 1] for b in b_idx])
    audit = {"n_tokens": len(ids), "prompt_tokens": prompt_tokens, "n_rows": len(rows),
             "final_prev_available": sum(available)}
    return rows, audit


def supersede_publication(target: Path, run_id: str, digest: str) -> dict | None:
    """Only a smoke publication may be displaced, and only by relocation (nothing under the
    persistent root is deleted); a canonical table with different content is never touched."""
    if not target.is_file():
        return None
    if file_hash(target) == digest:
        return {"status": "identical"}
    audit_path = target.parent / "feature_audit.json"
    previous = read_json(audit_path) if audit_path.is_file() else {}
    if previous.get("extraction_stage") == "stage7":
        raise FileExistsError(f"Canonical {target} exists with different content")
    superseded = target.parent / "superseded" / str(previous.get("run_id") or "unknown")
    superseded.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in sorted(target.parent.iterdir()):
        if path.is_file():
            path.rename(superseded / path.name)
            moved.append(path.name)
    return {"status": "relocated_smoke_publication", "previous_run_id": previous.get("run_id"),
            "previous_stage": previous.get("extraction_stage"), "moved_to": str(superseded),
            "files": moved, "by_run_id": run_id}


def build_dataset(run: Path, repo: Path, config: ExperimentConfig, root: Path, *,
                  device: str | None = None, publish: bool | None = None,
                  tracking: bool = True, head_loader=None, tokenizer_loader=None) -> dict:
    stage = features_stage(config)
    extraction = stage_name(config)
    device = device or default_device()
    if publish is None:
        publish = config.execution.mode == "mvp"
    if publish and config.execution.mode != "mvp":
        raise ValueError("Only the mvp configuration publishes to data/processed/stage7")
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        if manifest["stages"].get(extraction, {}).get("state") != "complete":
            raise ValueError(f"{extraction} must be complete before building the dataset")
        inputs, hashes = load_inputs(root)
        if manifest["input_hashes"] != hashes:
            raise ValueError("Stage 3/4 input hashes differ from the extraction run")
        validated = validate_run(run, config)
        index = validate_parquet(run / "activations" / "index.parquet").to_pylist()
        index_sha = file_hash(run / "activations" / "index.parquet")
        by_unit = _unit_rows(index)
        attempt = uuid.uuid4().hex
        stage_dir = run / "features"
        stage_dir.mkdir(exist_ok=True)
        session, record = (start_tracking(run, manifest, config, stage=stage) if tracking
                           else (None, {"mode": "disabled_for_test", "url": None}))
        manifest["wandb"].append(record)
        manifest["commands"].append({"command": "build-dataset", "attempt": attempt,
                                     "device": device, "publish": publish})
        manifest["stages"][stage] = {"state": "running", "attempt": attempt}
        save_manifest(run, manifest)
        runtime = Runtime(run, stage, len(by_unit), session)
        store = UnitStore(run / "activations" / "units")
        started = time.monotonic()
        splits = split_of(config)
        labels = {r["sentence_id"]: r for r in
                  validate_parquet(root / "data/processed/stage4/labels.parquet").to_pylist()}
        result: dict = {}
        try:
            with runtime:
                tokenizer = (tokenizer_loader or load_tokenizer)(config)
                head = (head_loader or load_head)(config, device)
                rows, per_unit = [], {}
                for key in sorted(by_unit):
                    unit_index = by_unit[key]
                    tid = unit_index[0]["trajectory_id"]
                    if any(r["trajectory_id"] != tid for r in unit_index):
                        raise ValueError("A unit mixes trajectories")
                    meta, tensors = read_shard(store.root / key)
                    trajectory = inputs["trajectories"][tid]
                    n_sentences = len(inputs["sentences"][tid])
                    built, audit = unit_features(head, tokenizer, config, trajectory, meta,
                                                 tensors, n_sentences)
                    if len(built) != len(unit_index):
                        raise ValueError("Feature rows differ from the unit's index rows")
                    for feature, ix in zip(built, unit_index, strict=True):
                        b = meta["boundaries"][ix["row_idx"]]
                        if (b["sentence_id"], feature["sentence_idx"]) != (
                                ix["sentence_id"], ix["sentence_idx"]):
                            raise ValueError("Feature row does not match its index row")
                        label = labels[ix["sentence_id"]]
                        rows.append({
                            "schema_version": 1, "sentence_id": ix["sentence_id"],
                            "trajectory_id": tid, "problem_id": int(ix["problem_id"]),
                            "trace_kind": ix["trace_kind"],
                            "split": splits[int(ix["problem_id"])], **feature,
                            "boundary_token_idx": int(ix["boundary_token_idx"]),
                            "target_token_start": int(ix["target_token_start"]),
                            "overlapping_token": bool(ix["overlapping_token"]),
                            "final_prev_position_available":
                                bool(ix["final_prev_position_available"]),
                            "activation_run_id": run.name, "shard_path": ix["shard_path"],
                            "row_idx": int(ix["row_idx"]), "trace_sha256": ix["trace_sha256"],
                            "token_ids_sha256": ix["token_ids_sha256"],
                            **{k: label[k] for k in ("eligible", "eligible_k4",
                                                     "exclusion_reason", "label_source",
                                                     "importance_released",
                                                     "importance_legacy",
                                                     "importance_canonical",
                                                     "importance_primary", "importance_log1p",
                                                     "anchor_threshold", "anchor")}})
                    per_unit[tid] = {"unit_key": key, **audit}
                    runtime.completed += 1
                    runtime.pulse(force=True)
                rows.sort(key=lambda r: (r["trajectory_id"], r["sentence_idx"]))
                table = features_table(rows)
                names = check_feature_names(config, table.column_names)
                checks = value_checks(table, config, head.vocab_size)
                if checks["violations"]:
                    raise ValueError(f"Feature value checks failed: {checks['violations']}")
                digest = write_parquet(stage_dir / "features.parquet", table)
                audit = {"schema_version": 1, "run_id": run.name, "stage": stage,
                         "attempt": attempt, "preregistered": True, "evidence": "exploratory",
                         "activation_run_id": run.name, "extraction_stage": extraction,
                         "index_sha256": index_sha, "index_rows": len(index),
                         "units": validated["units"], "n_rows": table.num_rows,
                         "input_hashes": hashes, "features_sha256": digest,
                         "columns": names, "device": device, "statistics_dtype": "float32",
                         "head": {"source": head.source, "weights_sha256": head.weights_sha256,
                                  "vocab_size": head.vocab_size, "eps": head.eps},
                         "tokenizer_revision": config.model.tokenizer_revision,
                         "value_checks": checks, "per_trajectory": per_unit,
                         "split_rows": _counts(table, "split"),
                         "eligible_rows": int(sum(table.column("eligible").to_pylist())),
                         "anchor_rows": int(sum(table.column("anchor").to_pylist())),
                         "published": str(feature_path(root)) if publish else None,
                         "superseded_publication": None,
                         "elapsed_seconds": time.monotonic() - started}
                if publish:
                    target = feature_path(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    audit["superseded_publication"] = supersede_publication(target, run.name,
                                                                            digest)
                audit_digest = write_current_json(stage_dir / "feature_audit.json", audit,
                                                  attempt)
                if publish:
                    if write_parquet(target, table) != digest:
                        raise ValueError("Published table differs from the run copy")
                    write_current_json(target.parent / "feature_audit.json", audit)
                manifest["output_hashes"]["features/features.parquet"] = digest
                manifest["output_hashes"]["features/feature_audit.json"] = audit_digest
                manifest["stages"][stage] = {"state": "complete", "attempt": attempt,
                                             "n_rows": table.num_rows,
                                             "n_units": validated["units"]}
                runtime.pulse(force=True, state="complete")
                result = {"audit": audit, "n_rows": table.num_rows, "features_sha256": digest}
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": manifest["run_id"], "stage": stage,
                              "attempt": attempt, "at": utc_now(),
                              "completed": runtime.completed, "identity": manifest["identity"],
                              "exception": type(exc).__name__,
                              "traceback": traceback.format_exc(),
                              "recovery": "build-dataset --run-id <id> after fixing the cause"})
            write_json(stage_dir / "failures" / f"{attempt}.json", failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"][stage]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            manifest["stage_timings"].setdefault(stage, []).append(
                {"schema_version": 1, "attempt": attempt, "device": device,
                 "elapsed_seconds": time.monotonic() - started,
                 "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
            if session:
                try:
                    session.finish(exit_code=0 if manifest["stages"][stage]["state"]
                                   == "complete" else 1)
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "offline records require backup"
            save_manifest(run, manifest)
        return result


def _counts(table, column: str) -> dict[str, int]:
    values = table.column(column).to_pylist()
    return {k: values.count(k) for k in sorted(set(values))}


def value_checks(table, config: ExperimentConfig, vocab_size: int) -> dict:
    """Definitional and range invariants of every primary and diagnostic column."""
    cols = {c: np.asarray(table.column(c).to_pylist()) for c in
            PRIMARY_FEATURES + DIAGNOSTIC_FEATURES + ["boundary_token_idx",
                                                      "final_prev_position_available"]}
    violations = []

    def check(name: str, ok: bool) -> None:
        if not ok:
            violations.append(name)

    n = table.num_rows
    if n:
        for c in PRIMARY_FEATURES + DIAGNOSTIC_FEATURES:
            check(f"{c}_finite", bool(np.isfinite(cols[c].astype(np.float64)).all()))
        prefix = cols["prefix_token_count"].astype(np.int64)
        check("prefix_is_boundary_plus_one",
              bool((prefix == cols["boundary_token_idx"].astype(np.int64) + 1).all()))
        check("log1p_prefix", bool(np.allclose(cols["log1p_prefix_token_count"],
                                               np.log1p(prefix), rtol=0, atol=1e-12)))
        check("context_fraction", bool(np.allclose(
            cols["prefix_context_fraction"], prefix / config.model.context_length,
            rtol=0, atol=1e-15)))
        check("generated_nonnegative", bool((cols["generated_token_count"] >= 0).all()))
        check("generated_at_most_prefix", bool((cols["generated_token_count"] <= prefix).all()))
        check("previous_sentence_nonnegative",
              bool((cols["previous_sentence_tokens"] >= 0).all()))
        check("previous_sentence_within_prefix",
              bool((cols["previous_sentence_tokens"] <= prefix).all()))
        check("sentence_idx_nonnegative", bool((cols["sentence_idx"] >= 0).all()))
        check("prev_logprob_nonpositive",
              bool((cols["previous_emitted_token_logprob"] <= 1e-6).all()))
        check("first_target_logprob_nonpositive",
              bool((cols["diagnostic__first_target_token_logprob"] <= 1e-6).all()))
        check("final_prev_available_everywhere",
              bool(cols["final_prev_position_available"].all()))
        ent, pmax = cols["next_token_entropy"], cols["next_token_max_probability"]
        margin, top5, top10 = (cols["next_token_probability_margin"],
                               cols["next_token_top5_mass"], cols["next_token_top10_mass"])
        check("entropy_range", bool((ent >= -1e-6).all() and (ent <= math.log(vocab_size)
                                                                + 1e-6).all()))
        check("max_probability_range", bool((pmax > 0).all() and (pmax <= 1 + 1e-6).all()))
        check("margin_range", bool((margin >= -1e-6).all() and (margin <= pmax + 1e-6).all()))
        check("topk_monotone", bool((pmax <= top5 + 1e-6).all() and (top5 <= top10 + 1e-6).all()
                                    and (top10 <= 1 + 1e-6).all()))
        rel = cols["diagnostic__relative_completed_trace_position"]
        check("relative_position_range", bool((rel >= 0).all() and (rel < 1).all()))
    return {"violations": violations, "rows": n,
            "ranges": {c: [float(np.min(cols[c])), float(np.max(cols[c]))] if n else None
                       for c in PRIMARY_FEATURES + DIAGNOSTIC_FEATURES}}


def validate_features(run: Path, config: ExperimentConfig, root: Path, *, sample: int = 64,
                      device: str | None = None, head_loader=None, tokenizer_loader=None,
                      write_record: bool = True) -> dict:
    """SPEC Stage 7 validation of the finalized table (no decoder replay).

    Row coverage and identity against the activation index and Stage 4 labels, allow-list
    and exclusion checks, definitional invariants, published-copy identity, and an exact
    float32 recomputation of a seeded sample of rows from the stored shards.
    ``write_record=False`` keeps the check read-only (restore verification must never
    touch restored bytes).
    """
    stage = features_stage(config)
    manifest = load_manifest(run)
    state = manifest["stages"].get(stage, {})
    if state.get("state") != "complete":
        raise ValueError("build-dataset has not completed for this run")
    audit = read_json(run / "features" / "feature_audit.json")
    table = validate_parquet(run / "features" / "features.parquet")
    digest = file_hash(run / "features" / "features.parquet")
    if digest != audit["features_sha256"] or digest != manifest["output_hashes"].get(
            "features/features.parquet"):
        raise ValueError("Feature table hash differs from its audit or manifest")
    published = feature_path(root)
    published_same = published.is_file() and (
        validate_parquet(published).equals(table) and file_hash(published) == digest)
    if audit.get("published") and not published_same:
        raise ValueError("Published feature table differs from the run copy")
    names = check_feature_names(config, table.column_names)
    inputs_now = {name: file_hash(root / "data/processed" / name) for name in INPUT_FILES}
    if inputs_now != manifest["input_hashes"]:
        raise ValueError("Stage 3/4 inputs changed since the dataset was built")
    validated = validate_run(run, config)
    index = validate_parquet(run / "activations" / "index.parquet").to_pylist()
    rows = table.to_pylist()
    if len(rows) != len(index) or len(rows) != state["n_rows"]:
        raise ValueError("Feature rows do not cover the activation index")
    by_id = {r["sentence_id"]: r for r in index}
    labels = {r["sentence_id"]: r for r in
              validate_parquet(root / "data/processed/stage4/labels.parquet").to_pylist()}
    if set(by_id) != {r["sentence_id"] for r in rows} or not set(by_id) <= set(labels):
        raise ValueError("Sentence IDs differ between features, index and labels")
    if config.execution.mode == "mvp" and set(by_id) != set(labels):
        raise ValueError("Stage 7 must cover every Stage 4 row exactly once")
    splits = split_of(config)
    for r in rows:
        ix, label = by_id[r["sentence_id"]], labels[r["sentence_id"]]
        for k in ("trajectory_id", "problem_id", "trace_kind", "sentence_idx", "shard_path",
                  "row_idx", "boundary_token_idx", "target_token_start", "overlapping_token",
                  "final_prev_position_available", "trace_sha256", "token_ids_sha256"):
            if r[k] != ix[k]:
                raise ValueError(f"Feature column {k} disagrees with the activation index")
        for k in ("eligible", "eligible_k4", "exclusion_reason", "label_source",
                  "importance_released", "importance_legacy", "importance_canonical",
                  "importance_primary", "importance_log1p", "anchor_threshold", "anchor"):
            a, b = r[k], label[k]
            same = (a == b) or (isinstance(a, float) and isinstance(b, float)
                                and math.isnan(a) and math.isnan(b))
            if not same:
                raise ValueError(f"Target column {k} disagrees with Stage 4 labels")
        if r["split"] != splits[int(r["problem_id"])] or r["activation_run_id"] != run.name:
            raise ValueError("Split or activation run differs from the configuration")
    vocab = int(audit["head"]["vocab_size"])
    checks = value_checks(table, config, vocab)
    if checks["violations"]:
        raise ValueError(f"Feature value checks failed: {checks['violations']}")
    # Exact float32 recomputation of a seeded row sample from the stored shards.
    recomputed = {"sample": 0, "max_abs_diff": 0.0}
    if sample:
        rng = np.random.default_rng(derived_seed(config.seed_root, "validate_features"))
        chosen = sorted(rng.choice(len(rows), size=min(sample, len(rows)), replace=False))
        inputs, _ = load_inputs(root)
        tokenizer = (tokenizer_loader or load_tokenizer)(config)
        head = (head_loader or load_head)(config, device or default_device())
        if head.weights_sha256 != audit["head"]["weights_sha256"]:
            raise ValueError("Head weights differ from the ones used to build the dataset")
        store = UnitStore(run / "activations" / "units")
        cache: dict[str, tuple[dict, dict]] = {}
        for i in chosen:
            r = rows[i]
            key = r["shard_path"].split("/")[1]
            if key not in cache:
                meta, tensors = read_shard(store.root / key)
                trajectory = inputs["trajectories"][r["trajectory_id"]]
                built, _ = unit_features(head, tokenizer, config, trajectory, meta, tensors,
                                         len(inputs["sentences"][r["trajectory_id"]]))
                cache[key] = (meta, {b["sentence_id"]: f for b, f in
                                     zip(meta["boundaries"], built, strict=True)})
            fresh = cache[key][1][r["sentence_id"]]
            for c in PRIMARY_FEATURES + DIAGNOSTIC_FEATURES:
                diff = abs(float(fresh[c]) - float(r[c]))
                tolerance = 1e-5 if c in UNCERTAINTY_FEATURES + DIAGNOSTIC_FEATURES else 0.0
                if not diff <= tolerance:
                    raise ValueError(f"Recomputed {c} differs for {r['sentence_id']}: {diff}")
                recomputed["max_abs_diff"] = max(recomputed["max_abs_diff"], diff)
            recomputed["sample"] += 1
        recomputed["device"] = head.device
    report = {"schema_version": 1, "ok": True, "run_id": run.name, "stage": stage,
              "rows": len(rows), "units": validated["units"], "features_sha256": digest,
              "published_identical": bool(published_same), "columns": names,
              "value_checks": checks, "recomputed": recomputed,
              "split_rows": _counts(table, "split"),
              "eligible_rows": int(sum(table.column("eligible").to_pylist())),
              "anchor_rows": int(sum(table.column("anchor").to_pylist())),
              "validated_at": utc_now()}
    if write_record:
        write_current_json(run / "features" / "feature_validation.json", report)
    return report
