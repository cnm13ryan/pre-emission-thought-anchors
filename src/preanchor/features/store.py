"""Per-trajectory activation shards and the activation index (SPEC §5, §11 Stage 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from preanchor.io import read_json, validate_artifact, write_json, write_safetensors

ACTIVATION_SITE = "block_post_pre_final_norm"
STORED_DTYPE = "float16"

INDEX_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("sentence_id", pa.string(), nullable=False),
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("problem_id", pa.int64(), nullable=False),
    pa.field("trace_kind", pa.string(), nullable=False),
    pa.field("sentence_idx", pa.int64(), nullable=False),
    pa.field("shard_path", pa.string(), nullable=False),
    pa.field("row_idx", pa.int64(), nullable=False),
    pa.field("n_layers", pa.int32(), nullable=False),
    pa.field("d_model", pa.int32(), nullable=False),
    pa.field("dtype", pa.string(), nullable=False),
    pa.field("activation_site", pa.string(), nullable=False),
    pa.field("boundary_token_idx", pa.int64(), nullable=False),
    pa.field("target_token_start", pa.int64(), nullable=False),
    pa.field("prefix_token_count", pa.int64(), nullable=False),
    pa.field("overlapping_token", pa.bool_(), nullable=False),
    pa.field("final_prev_position_available", pa.bool_(), nullable=False),
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("model_revision", pa.string(), nullable=False),
    pa.field("tokenizer_revision", pa.string(), nullable=False),
    pa.field("trace_sha256", pa.string(), nullable=False),
    pa.field("token_ids_sha256", pa.string(), nullable=False),
    # Eligibility is carried for selection only; labels stay in the Stage 4 table.
    pa.field("eligible", pa.bool_(), nullable=False),
    pa.field("eligible_k4", pa.bool_(), nullable=False),
    pa.field("exclusion_reason", pa.string(), nullable=True),
    pa.field("label_source", pa.string(), nullable=False),
    pa.field("labels_sha256", pa.string(), nullable=False),
])


def shard_name(trajectory_id: str) -> str:
    problem, kind = trajectory_id.split(":")
    if kind not in {"correct", "incorrect"} or not problem.isdigit():
        raise ValueError(f"Malformed trajectory id {trajectory_id!r}")
    return f"{problem}_{kind}.safetensors"


def write_shard(directory: Path, trajectory_id: str, resid, final_prev, meta: dict) -> dict:
    """`resid` [n, sites, d] and `final_prev` [n, d] are torch tensors; stored as float16."""
    resid16 = resid.to("cpu").to(_torch().float16).numpy()
    prev16 = final_prev.to("cpu").to(_torch().float16).numpy()
    if resid16.ndim != 3 or prev16.shape != (resid16.shape[0], resid16.shape[2]):
        raise ValueError("Shard tensors have inconsistent shapes")
    if np.isnan(resid16).any() or np.isinf(resid16).any() or np.isinf(prev16).any():
        raise ValueError("Non-finite activation values")
    path = directory / shard_name(trajectory_id)
    digest = write_safetensors(path, {"resid": resid16, "final_prev_position": prev16})
    record = {"schema_version": 1, "trajectory_id": trajectory_id, "shard": path.name,
              "shard_sha256": digest, "n_rows": int(resid16.shape[0]),
              "n_sites": int(resid16.shape[1]), "d_model": int(resid16.shape[2]),
              "dtype": STORED_DTYPE, "activation_site": ACTIVATION_SITE,
              "bytes": path.stat().st_size, **meta}
    write_json(directory / "shard_meta.json", record)
    return record


def read_shard(directory: Path, *, expected_sites: int | None = None,
               expected_d_model: int | None = None) -> tuple[dict, dict]:
    from safetensors.numpy import load_file

    meta = read_json(directory / "shard_meta.json")
    path = directory / meta["shard"]
    if validate_artifact(path) != meta["shard_sha256"]:
        raise ValueError("Shard checksum differs from its metadata")
    tensors = load_file(str(path))
    resid, prev = tensors["resid"], tensors["final_prev_position"]
    n, sites, d = resid.shape
    if (n, sites, d) != (meta["n_rows"], meta["n_sites"], meta["d_model"]):
        raise ValueError("Shard shape differs from its metadata")
    if expected_sites is not None and sites != expected_sites:
        raise ValueError(f"Expected {expected_sites} sites, found {sites}")
    if expected_d_model is not None and d != expected_d_model:
        raise ValueError(f"Expected d_model {expected_d_model}, found {d}")
    if resid.dtype != np.float16 or prev.dtype != np.float16 or prev.shape != (n, d):
        raise ValueError("Shard dtype or final_prev_position shape mismatch")
    if not np.isfinite(resid).all():
        raise ValueError("Shard contains non-finite residual values")
    if len(meta["boundaries"]) != n:
        raise ValueError("Boundary metadata does not match the row count")
    return meta, tensors


def index_rows(meta: dict, shard_relative: str, sentences: dict[str, dict],
               labels_sha256: str, model: dict) -> list[dict]:
    rows = []
    for row_idx, b in enumerate(meta["boundaries"]):
        s = sentences[b["sentence_id"]]
        rows.append({"schema_version": 1, "sentence_id": b["sentence_id"],
                     "trajectory_id": meta["trajectory_id"], "problem_id": int(s["problem_id"]),
                     "trace_kind": s["trace_kind"], "sentence_idx": int(b["sentence_idx"]),
                     "shard_path": shard_relative, "row_idx": row_idx,
                     "n_layers": meta["n_sites"], "d_model": meta["d_model"],
                     "dtype": meta["dtype"], "activation_site": meta["activation_site"],
                     "boundary_token_idx": int(b["boundary_token_idx"]),
                     "target_token_start": int(b["target_token_start"]),
                     "prefix_token_count": int(b["prefix_token_count"]),
                     "overlapping_token": bool(b["overlapping_token"]),
                     "final_prev_position_available": bool(meta["final_prev_available"][row_idx]),
                     "model_id": model["model_id"], "model_revision": model["revision"],
                     "tokenizer_revision": model["tokenizer_revision"],
                     "trace_sha256": meta["trace_sha256"],
                     "token_ids_sha256": meta["token_ids_sha256"],
                     "eligible": bool(s["eligible"]), "eligible_k4": bool(s["eligible_k4"]),
                     "exclusion_reason": s.get("exclusion_reason"),
                     "label_source": s["label_source"], "labels_sha256": labels_sha256})
    return rows


def index_table(rows: list[dict]) -> pa.Table:
    table = pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)
    ids = table.column("sentence_id").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sentence_id in the activation index")
    return table


def _torch():
    import torch

    return torch
