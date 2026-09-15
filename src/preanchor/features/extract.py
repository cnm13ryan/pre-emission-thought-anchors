"""Stage 6/7 activation extraction: one causal replay per trajectory, resumable units."""

from __future__ import annotations

import hashlib
import os
import resource
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from preanchor.config import ExperimentConfig, config_hash
from preanchor.features.store import (
    ACTIVATION_SITE,
    index_rows,
    index_table,
    read_shard,
    write_shard,
)
from preanchor.ids import cache_key, file_hash
from preanchor.io import (
    UnitStore,
    atomic_bytes,
    json_bytes,
    lock,
    validate_parquet,
    write_json,
    write_parquet,
)
from preanchor.logging import redact
from preanchor.manifest import check_resume, create_run, load_manifest, save_manifest, utc_now
from preanchor.model.boundaries import (
    audit_boundaries,
    boundary_records,
    boundary_table,
    tokenize_trace,
)
from preanchor.model.hooks import replay
from preanchor.model.load import gpu_snapshot, load_model, load_tokenizer, validate_architecture
from preanchor.model.logits import (
    STATISTIC_COLUMNS,
    boundary_logits,
    logit_identity,
    next_token_statistics,
    token_logprob,
)
from preanchor.runtime import Runtime
from preanchor.tracking import start_tracking

INPUT_FILES = ("stage3/trajectories.parquet", "stage3/sentences.parquet",
               "stage4/labels.parquet")
SEGMENTATION_VERSION = "released_chunks_v1"
STAGE7_UNITS = 40


def write_current_json(path: Path, value: object, attempt: str | None = None) -> str:
    """Replaceable current view of a per-attempt record; the attempt copy is immutable."""
    if attempt:
        write_json(path.parent / "attempts" / f"{attempt}.{path.name}", value)
    data = json_bytes(value)
    digest = atomic_bytes(path, data, replace=True)
    atomic_bytes(Path(str(path) + ".sha256"), (digest + "\n").encode(), replace=True)
    return digest


def load_inputs(root: Path) -> tuple[dict, dict]:
    """Validated Stage 3/4 tables keyed by trajectory, plus their content hashes."""
    processed = root / "data/processed"
    hashes = {name: file_hash(processed / name) for name in INPUT_FILES}
    tables = {name: validate_parquet(processed / name) for name in INPUT_FILES}
    trajectories = {r["trajectory_id"]: r
                    for r in tables["stage3/trajectories.parquet"].to_pylist()}
    sentences = tables["stage3/sentences.parquet"].to_pylist()
    labels = {r["sentence_id"]: r for r in tables["stage4/labels.parquet"].to_pylist()}
    if len(trajectories) != 40 or len(sentences) != len(labels):
        raise ValueError("Unexpected Stage 3/4 grain")
    by_trajectory: dict[str, list[dict]] = {}
    for row in sentences:
        t = trajectories[row["trajectory_id"]]
        if hashlib.sha256(t["full_cot"].encode()).hexdigest() != t["full_cot_sha256"]:
            raise ValueError("Trace hash mismatch")
        if t["full_cot"][row["char_start"]:row["char_end"]] != row["text"]:
            raise ValueError(f"Span does not slice the trace: {row['sentence_id']}")
        label = labels[row["sentence_id"]]
        if (label["char_start"], label["char_end"]) != (row["char_start"], row["char_end"]):
            raise ValueError("Stage 4 row disagrees with Stage 3 span")
        by_trajectory.setdefault(row["trajectory_id"], []).append({**row, **{
            k: label[k] for k in ("eligible", "eligible_k4", "exclusion_reason",
                                  "label_source")}})
    return {"trajectories": trajectories, "sentences": by_trajectory}, hashes


def select_work(config: ExperimentConfig, inputs: dict, trajectories: list[str] | None,
                limit_sentences: int | None) -> list[tuple[str, list[dict]]]:
    if config.execution.mode == "smoke":
        chosen = trajectories or [config.execution.trajectory]
        limit = limit_sentences or config.execution.limit_sentences
    else:
        chosen = trajectories or sorted(inputs["trajectories"])
        limit = limit_sentences
    work = []
    for tid in chosen:
        if tid not in inputs["trajectories"]:
            raise ValueError(f"Unknown trajectory {tid}")
        rows = sorted(inputs["sentences"][tid], key=lambda r: r["sentence_idx"])
        work.append((tid, rows[:limit] if limit else rows))
    return work


def stage_name(config: ExperimentConfig) -> str:
    return "stage6" if config.execution.mode == "smoke" else "stage7"


def prepare(root: Path, repo: Path, config: ExperimentConfig, command: str,
            work: list[tuple[str, list[dict]]], hashes: dict) -> Path:
    run = create_run(root, repo, config, [command])
    manifest = load_manifest(run)
    manifest["input_hashes"] = dict(hashes)
    manifest["identity"]["input_hashes"] = dict(hashes)
    manifest["gpu_gate"] = "production_doctor_plus_measured_extraction_fit"
    manifest["stages"][stage_name(config)] = {
        "state": "prepared", "trajectories": [t for t, _ in work],
        "sentence_limit": max(len(r) for _, r in work),
        "n_units": len(work), "n_rows": sum(len(r) for _, r in work)}
    save_manifest(run, manifest)
    return run


def unit_key(config: ExperimentConfig, manifest: dict, trajectory: dict, rows: list[dict],
             token_hash: str, mapping_hash: str) -> str:
    return cache_key(stage="activations", scientific_config_hash=config.scientific_hash(),
                     model_revision=config.model.revision,
                     tokenizer_revision=config.model.tokenizer_revision,
                     trace_sha256=trajectory["full_cot_sha256"],
                     segmentation_version=SEGMENTATION_VERSION,
                     boundary_mapping_sha256=mapping_hash, generation_parameters={},
                     fork_idx=None, sample_seed=None,
                     code_version=manifest["identity"]["git_sha"],
                     input_hashes={**manifest["input_hashes"], "token_ids_sha256": token_hash,
                                   "sentence_ids": config_hash([r["sentence_id"]
                                                                for r in rows])},
                     unit_id=trajectory["trajectory_id"])


def default_loader(config: ExperimentConfig):
    tokenizer = load_tokenizer(config)
    started = time.perf_counter()
    model = load_model(config)
    return tokenizer, model, time.perf_counter() - started


def execute(run: Path, repo: Path, config: ExperimentConfig, *, trajectories=None,
            limit_sentences=None, resume: bool = False, tracking: bool = True, loader=None,
            pause_seconds: float = 0.0, diagnostics: bool | None = None) -> dict:
    """Extract every selected trajectory as an immutable unit; rebuild the index at the end."""
    stage = stage_name(config)
    root = run.parents[2]
    inputs, hashes = load_inputs(root)
    work = select_work(config, inputs, trajectories, limit_sentences)
    with lock(run / ".run.lock"):
        manifest = check_resume(run, repo, config)
        if manifest["input_hashes"] != hashes:
            raise ValueError("Stage 3/4 input hashes differ from the prepared run")
        previous = manifest["stages"].get(stage)
        if previous is None:
            raise ValueError("Run was not prepared for this stage")
        if previous["state"] != "prepared" and not resume:
            raise ValueError("Existing stage requires --resume")
        if previous["trajectories"] != [t for t, _ in work] or \
                previous["n_rows"] != sum(len(r) for _, r in work):
            raise ValueError("Resume scope differs from the prepared stage")
        attempt = uuid.uuid4().hex
        stage_dir = run / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        if previous["state"] == "running":
            write_json(stage_dir / "failures" / f"{previous['attempt']}.json", {
                "schema_version": 1, "exception": "UnfinishedAttempt", "at": utc_now(),
                "attempt": previous["attempt"], "signal": "unknown",
                "recovery": "validate completed units and quarantine incomplete units"})
        session, record = (start_tracking(run, manifest, config, stage=stage) if tracking
                           else (None, {"mode": "disabled_for_test", "url": None}))
        manifest["wandb"].append(record)
        manifest["commands"].append({"command": "extract-activations", "resume": resume,
                                     "attempt": attempt, "trajectories": [t for t, _ in work]})
        manifest["stages"][stage] = {**previous, "state": "running", "attempt": attempt}
        save_manifest(run, manifest)
        runtime = Runtime(run, stage, len(work), session)
        store = UnitStore(run / "activations" / "units")
        started = time.monotonic()
        loaded = {"tokenizer": None, "model": None, "load_seconds": None}
        units: list[dict] = []
        created = 0
        current = None

        def ensure_model():
            if loaded["model"] is None:
                loaded["tokenizer"], loaded["model"], loaded["load_seconds"] = (
                    loader or default_loader)(config)
                loaded["architecture"] = validate_architecture(
                    loaded["model"], config, strict=loader is None)
            return loaded["tokenizer"], loaded["model"]

        try:
            with runtime:
                for tid, rows in work:
                    trajectory = inputs["trajectories"][tid]
                    tokenizer, _ = ensure_model()
                    ids, offsets = tokenize_trace(tokenizer, trajectory["full_cot"])
                    table = boundary_table(offsets, inputs["sentences"][tid])[:len(rows)]
                    audit = audit_boundaries(ids, offsets, table)
                    current = unit_key(config, manifest, trajectory, rows,
                                       audit["token_ids_sha256"],
                                       audit["boundary_mapping_sha256"])

                    timing = {"forward_seconds": None, "peak_allocated_bytes": None}

                    def writer(part: Path, tid=tid, trajectory=trajectory, ids=ids,
                               table=table, audit=audit, key=current, timing=timing):
                        _, model = ensure_model()
                        capture = replay(model, ids, [b.boundary_token_idx for b in table])
                        timing.update(forward_seconds=capture.forward_seconds,
                                      peak_allocated_bytes=capture.peak_allocated_bytes)
                        # Unit contents are deterministic; timings live in attempt records.
                        write_shard(part, tid, capture.resid, capture.final_prev_position, {
                            "unit_key": key, "trace_sha256": trajectory["full_cot_sha256"],
                            "token_ids_sha256": audit["token_ids_sha256"],
                            "boundary_mapping_sha256": audit["boundary_mapping_sha256"],
                            "n_tokens": capture.n_tokens,
                            "boundaries": boundary_records(table),
                            "final_prev_available": capture.final_prev_available,
                            "audit": audit, "model": loaded["architecture"]})
                        # Interruption drills pause here, at a real incomplete-unit boundary.
                        runtime.pause(pause_seconds)

                    part, was_created = store.produce(current, writer)
                    created += was_created
                    arch = loaded["architecture"]
                    meta, _ = read_shard(part, expected_sites=arch["n_layers"] + 1,
                                         expected_d_model=arch["d_model"])
                    if meta["unit_key"] != current or meta["n_rows"] != len(rows):
                        raise ValueError("Completed unit does not match the planned work")
                    units.append({"trajectory_id": tid, "unit_key": current,
                                  "created": was_created,
                                  "n_rows": meta["n_rows"], "n_tokens": meta["n_tokens"],
                                  "bytes": meta["bytes"], **timing,
                                  "shard_sha256": meta["shard_sha256"]})
                    runtime.completed += 1
                    runtime.last_output = str(part / "SUCCESS.json")
                    runtime.pulse(force=True)
                result = finalize(run, config, manifest, inputs, hashes, units, loaded,
                                  stage_dir, attempt)
                if diagnostics if diagnostics is not None else config.execution.mode == "smoke":
                    tokenizer, model = ensure_model()
                    result["diagnostics"] = smoke_diagnostics(
                        run, stage_dir, config, inputs, work, tokenizer, model, attempt)
                manifest["stages"][stage]["state"] = "complete"
                runtime.pulse(force=True, state="complete")
        except BaseException as exc:
            failure = redact({"schema_version": 1, "run_id": manifest["run_id"], "stage": stage,
                              "attempt": attempt, "at": utc_now(), "unit_key": current,
                              "completed": runtime.completed, "identity": manifest["identity"],
                              "exception": type(exc).__name__,
                              "traceback": traceback.format_exc(),
                              "recovery": "extract-activations --resume --run-id <id>"})
            write_json(stage_dir / "failures" / f"{attempt}.json", failure)
            atomic_bytes(run / "failure.json", json_bytes(failure), replace=True)
            manifest["stages"][stage]["state"] = "failed"
            runtime.stop_signal = None
            runtime.last_error = {"attempt": attempt, "exception": type(exc).__name__}
            runtime.pulse(force=True, state="failed")
            raise
        finally:
            timing = {"schema_version": 1, "attempt": attempt, "created": created,
                      "completed": runtime.completed, "elapsed_seconds": time.monotonic() - started,
                      "model_load_seconds": loaded["load_seconds"],
                      "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      "gpu": gpu_snapshot(), "units": units}
            write_json(stage_dir / "attempts" / f"{attempt}.json", timing)
            manifest["stage_timings"].setdefault(stage, []).append(timing)
            manifest["source"]["hardware"] = {**manifest["source"].get("hardware", {}),
                                              **timing["gpu"]}
            if session:
                try:
                    session.finish(exit_code=0 if manifest["stages"][stage]["state"]
                                   == "complete" else 1)
                except Exception:
                    manifest["wandb"][-1]["finish_error"] = "offline records require backup"
            save_manifest(run, manifest)
        return {**result, "created": created, "skipped": len(work) - created}


def finalize(run: Path, config: ExperimentConfig, manifest: dict, inputs: dict, hashes: dict,
             units: list[dict], loaded: dict, stage_dir: Path, attempt: str) -> dict:
    """Validate every unit again, publish the index and the summary, project Stage 7."""
    store = UnitStore(run / "activations" / "units")
    rows = []
    model = {"model_id": config.model.id, "revision": config.model.revision,
             "tokenizer_revision": config.model.tokenizer_revision}
    for unit in units:
        store.validate(unit["unit_key"])
        part = store.root / unit["unit_key"]
        meta, _ = read_shard(part, expected_sites=loaded["architecture"]["n_layers"] + 1,
                             expected_d_model=loaded["architecture"]["d_model"])
        sentences = {r["sentence_id"]: r for r in inputs["sentences"][unit["trajectory_id"]]}
        rows.extend(index_rows(meta, f"units/{unit['unit_key']}/{meta['shard']}", sentences,
                               hashes["stage4/labels.parquet"], model))
    table = index_table(rows)
    index_digest = write_parquet(run / "activations" / "index.parquet", table)
    n_tokens = sum(u["n_tokens"] for u in units)
    measured = [u for u in units if u["forward_seconds"] is not None]
    forward = sum(u["forward_seconds"] for u in measured)
    measured_tokens = sum(u["n_tokens"] for u in measured)
    peak = max((u["peak_allocated_bytes"] for u in measured), default=0) / 2**20
    load_seconds = loaded["load_seconds"]
    basis = "measured in this attempt"
    if not measured and (stage_dir / "summary.json").is_file():
        # A no-op resume measures nothing; keep the previous attempt's measurements.
        from preanchor.io import read_json

        previous = read_json(stage_dir / "summary.json")
        measured = [{"n_tokens": previous["measured_tokens"]}] * previous["measured_units"]
        forward, measured_tokens = previous["forward_seconds"], previous["measured_tokens"]
        peak, load_seconds = previous["peak_allocated_mib"], previous["model_load_seconds"]
        basis = f"carried forward from attempt {previous['attempt']}"
    summary = {"schema_version": 1, "run_id": manifest["run_id"], "stage": stage_name(config),
               "preregistered": True, "evidence": "exploratory",
               "units": units, "n_units": len(units), "n_rows": table.num_rows,
               "attempt": attempt, "measurement_basis": basis,
               "n_tokens": n_tokens, "forward_seconds": forward, "measured_units": len(measured),
               "measured_tokens": measured_tokens,
               "tokens_per_second": measured_tokens / forward if forward else None,
               "bytes": sum(u["bytes"] for u in units),
               "bytes_per_row": sum(u["bytes"] for u in units) / max(table.num_rows, 1),
               "peak_allocated_mib": peak, "model_load_seconds": load_seconds,
               "architecture": loaded.get("architecture"),
               "activation_site": ACTIVATION_SITE, "index_sha256": index_digest,
               "index_columns": table.column_names}
    write_current_json(stage_dir / "summary.json", summary, attempt)
    manifest["output_hashes"]["activations/index.parquet"] = index_digest
    for unit in units:
        manifest["output_hashes"][f"activations/{unit['trajectory_id']}"] = unit["shard_sha256"]
    projection = stage7_projection(config, manifest, inputs, summary, loaded["tokenizer"])
    write_current_json(stage_dir / "resource_projection.json", projection, attempt)
    return {"summary": summary, "projection": projection}


def hourly_rate() -> float | None:
    value = os.environ.get("PREANCHOR_HOURLY_RATE", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def stage7_projection(config: ExperimentConfig, manifest: dict, inputs: dict, summary: dict,
                      tokenizer) -> dict:
    """AGENTS §8 projection for the 40-trajectory Stage 7 run from this run's measurements."""
    counts = {}
    for tid, t in sorted(inputs["trajectories"].items()):
        counts[tid] = len(tokenizer(t["full_cot"], add_special_tokens=False)["input_ids"])
    rows = sum(len(v) for v in inputs["sentences"].values())
    tokens = sum(counts.values())
    rate = hourly_rate()
    tps = summary["tokens_per_second"]
    bytes_total = summary["bytes_per_row"] * rows
    if tps:
        # Conservative: per-unit overhead observed here, doubled, plus one model load.
        per_unit = 2 * (summary["forward_seconds"] / max(summary["measured_units"], 1))
        seconds = ((summary["model_load_seconds"] or 0) + tokens / tps
                   + per_unit * STAGE7_UNITS + bytes_total / (100 * 2**20))
    else:
        seconds = None
    return {"schema_version": 1, "stage": "stage7", "code_commit": manifest["identity"]["git_sha"],
            "config_hash": config.resolved_hash(), "input_hashes": manifest["input_hashes"],
            "empirical_run": manifest["run_id"], "basis": "measured Stage 6 extraction on this GPU",
            "unit_count": STAGE7_UNITS, "row_count": rows, "token_count": tokens,
            "tokens_per_trajectory": counts, "measured_tokens_per_second": tps,
            "measured_seconds_per_unit": (summary["forward_seconds"]
                                          / max(summary["measured_units"], 1)),
            "measured_bytes_per_row": summary["bytes_per_row"],
            "measured_peak_allocated_mib": summary["peak_allocated_mib"],
            "model_load_seconds": summary["model_load_seconds"],
            "projected_seconds": seconds,
            "projected_gpu_hours": seconds / 3600 if seconds is not None else None,
            "projected_bytes": bytes_total, "disk_growth_bytes_upper_estimate": 2 * bytes_total,
            "price_usd_per_hour": rate, "rate_source": "PREANCHOR_HOURLY_RATE",
            "projected_cost_usd": (seconds / 3600 * rate if rate is not None
                                   and seconds is not None else None),
            "computable": seconds is not None and rate is not None,
            "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def mutate_future(ids: list[int], boundary: int, vocab_size: int, seed: int) -> list[int]:
    """Same length; every token strictly after the boundary replaced by a different token."""
    rng = np.random.default_rng(seed)
    future = np.array(ids[boundary + 1:])
    draw = rng.integers(0, vocab_size, size=future.shape)
    draw = np.where(draw == future, (draw + 1) % vocab_size, draw)
    return ids[: boundary + 1] + draw.tolist()


def invariance_checks(model, ids: list[int], boundaries: list[int], captured, *, atol: float,
                      seed: int, logit_positions: list[int] | None = None) -> dict:
    """Primary gate: same-shape future mutation. Diagnostic: truncated-prefix replay."""
    vocab = model.config.vocab_size
    per_boundary = []
    for j, b in enumerate(boundaries):
        reference = captured[j].float()
        mutated = replay(model, mutate_future(ids, b, vocab, seed + j), [b]).resid[0].float()
        truncated = replay(model, ids[: b + 1], [b]).resid[0].float()
        m = (mutated - reference).abs().amax(dim=1)
        t = (truncated - reference).abs().amax(dim=1)
        entry = {"boundary_token_idx": b, "prefix_length": b + 1,
                 "mutation_max_abs_by_site": [float(x) for x in m],
                 "mutation_max_abs": float(m.max()), "mutation_pass": bool(m.max() <= atol),
                 "truncated_prefix_max_abs_by_site": [float(x) for x in t],
                 "truncated_prefix_max_abs": float(t.max()),
                 "truncated_prefix_within_atol": bool(t.max() <= atol),
                 "reference_abs_max": float(reference.abs().max())}
        if logit_positions is not None and j in logit_positions:
            entry["logit_identity"] = logit_identity(model, ids, b, captured[j, -1])
        per_boundary.append(entry)
    return {"schema_version": 1, "atol": atol, "n_boundaries": len(boundaries),
            "primary_gate": "same_shape_future_token_mutation",
            "primary_pass": all(e["mutation_pass"] for e in per_boundary),
            "primary_max_abs": max(e["mutation_max_abs"] for e in per_boundary),
            "diagnostic": "truncated_prefix_replay_sequence_length_dependent",
            "diagnostic_within_atol_count": sum(e["truncated_prefix_within_atol"]
                                                for e in per_boundary),
            "diagnostic_max_abs": max(e["truncated_prefix_max_abs"] for e in per_boundary),
            "logit_identity_pass": all(e["logit_identity"]["argmax_equal"]
                                       and e["logit_identity"]["max_abs"] <= atol
                                       for e in per_boundary if "logit_identity" in e),
            "per_boundary": per_boundary}


def smoke_diagnostics(run: Path, stage_dir: Path, config: ExperimentConfig, inputs: dict,
                      work, tokenizer, model, attempt: str) -> dict:
    """SPEC Stage 6: invariance report, boundary audit and observable logit features."""
    import pyarrow as pa

    store = UnitStore(run / "activations" / "units")
    manifest = load_manifest(run)
    atol = float(config.features["invariance_atol_bfloat16"])
    reports, feature_rows, audits = {}, [], {}
    for tid, rows in work:
        trajectory = inputs["trajectories"][tid]
        ids, offsets = tokenize_trace(tokenizer, trajectory["full_cot"])
        table = boundary_table(offsets, inputs["sentences"][tid])[:len(rows)]
        audits[tid] = audit_boundaries(ids, offsets, table)
        key = unit_key(config, manifest, trajectory, rows, audits[tid]["token_ids_sha256"],
                       audits[tid]["boundary_mapping_sha256"])
        meta, tensors = read_shard(store.root / key)
        import torch

        captured = torch.from_numpy(tensors["resid"])  # float16 stored copy
        boundaries = [b.boundary_token_idx for b in table]
        # Compare against a fresh bf16 replay so the gate is not limited by fp16 storage.
        fresh = replay(model, ids, boundaries)
        stored_vs_fresh = float((captured.float() - fresh.resid.float()).abs().max())
        positions = sorted({0, len(boundaries) // 2, len(boundaries) - 1})
        report = invariance_checks(model, ids, boundaries, fresh.resid, atol=atol,
                                   seed=config.seed_root, logit_positions=positions)
        report["stored_fp16_vs_fresh_bf16_max_abs"] = stored_vs_fresh
        reports[tid] = report
        # Observable logit features (SPEC §3.9) from stored final-site states, float32.
        logits = boundary_logits(model, torch.from_numpy(tensors["resid"][:, -1]))
        stats = next_token_statistics(logits)
        prev_logits = boundary_logits(model, torch.from_numpy(tensors["final_prev_position"]))
        prev_logprob = token_logprob(prev_logits, [ids[b] for b in boundaries])
        first_target = token_logprob(logits, [ids[b + 1] for b in boundaries])
        prompt_tokens = len(tokenizer(trajectory["prompt"], add_special_tokens=False)["input_ids"])
        for j, b in enumerate(table):
            feature_rows.append({
                "sentence_id": b.sentence_id, "sentence_idx": b.sentence_idx,
                "generated_token_count": b.prefix_token_count - prompt_tokens,
                "prefix_token_count": b.prefix_token_count,
                "log1p_prefix_token_count": float(np.log1p(b.prefix_token_count)),
                "prefix_context_fraction": b.prefix_token_count / config.model.context_length,
                "previous_sentence_tokens": (b.target_token_start - table[j - 1].target_token_start
                                             if j else b.target_token_start - prompt_tokens),
                "previous_emitted_token_logprob": prev_logprob[j], **stats[j],
                "diagnostic__first_target_token_logprob": first_target[j]})
    columns = ["sentence_id", "sentence_idx", "generated_token_count", "prefix_token_count",
               "log1p_prefix_token_count", "prefix_context_fraction", "previous_sentence_tokens",
               "previous_emitted_token_logprob", *STATISTIC_COLUMNS,
               "diagnostic__first_target_token_logprob"]
    features = pa.Table.from_pylist([{c: r[c] for c in columns} for r in feature_rows])
    write_parquet(stage_dir / "logit_features.parquet", features)
    write_current_json(stage_dir / "boundary_audit.json",
                       {"schema_version": 1, "trajectories": audits}, attempt)
    overall = {"schema_version": 1, "atol": atol, "trajectories": reports,
               "primary_pass": all(r["primary_pass"] for r in reports.values()),
               "logit_identity_pass": all(r["logit_identity_pass"] for r in reports.values()),
               "diagnostic_max_abs": max(r["diagnostic_max_abs"] for r in reports.values()),
               "diagnostic_within_atol": sum(r["diagnostic_within_atol_count"]
                                             for r in reports.values()),
               "n_boundaries": sum(r["n_boundaries"] for r in reports.values())}
    write_current_json(stage_dir / "invariance_report.json", overall, attempt)
    return overall


def read_json_summary(path: Path) -> dict:
    from preanchor.io import read_json

    summary = read_json(path)
    return summary["architecture"]


def validate_run(run: Path, config: ExperimentConfig) -> dict:
    """Re-validate every unit, the index and coverage without the model."""
    store = UnitStore(run / "activations" / "units")
    index = validate_parquet(run / "activations" / "index.parquet").to_pylist()
    manifest = load_manifest(run)
    stage = manifest["stages"][stage_name(config)]
    arch = read_json_summary(run / "stages" / stage_name(config) / "summary.json")
    seen = {}
    for row in index:
        key = row["shard_path"].split("/")[1]
        if key not in seen:
            store.validate(key)
            seen[key], _ = read_shard(store.root / key, expected_sites=arch["n_layers"] + 1,
                                      expected_d_model=arch["d_model"])
        meta = seen[key]
        b = meta["boundaries"][row["row_idx"]]
        if (b["sentence_id"] != row["sentence_id"]
                or b["boundary_token_idx"] != row["boundary_token_idx"]):
            raise ValueError("Index row disagrees with its shard")
    if len(index) != stage["n_rows"] or len(seen) != stage["n_units"]:
        raise ValueError("Index coverage differs from the prepared stage")
    return {"ok": True, "rows": len(index), "units": len(seen), "run_id": manifest["run_id"]}
