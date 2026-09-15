"""Strict pre-emission observable features (SPEC §3.9, AGENTS §5.1-5.2).

Every primary column is computable from the token prefix ending at the boundary and the
stored boundary states alone. Next-token statistics apply the model's final RMSNorm and
``lm_head`` to the stored ``layer=48`` boundary state in float32; full-vocabulary logits are
never stored. Excluded quantities exist only as ``diagnostic__*`` columns.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

DIAGNOSTIC_PREFIX = "diagnostic__"

# SPEC §3.9 allow-list, in order; must equal ``configs/mvp.yaml`` ``features.observables``.
PRIMARY_FEATURES = [
    "sentence_idx", "generated_token_count", "prefix_token_count", "log1p_prefix_token_count",
    "prefix_context_fraction", "previous_sentence_tokens", "previous_emitted_token_logprob",
    "next_token_entropy", "next_token_max_probability", "next_token_probability_margin",
    "next_token_top5_mass", "next_token_top10_mass",
]
UNCERTAINTY_FEATURES = PRIMARY_FEATURES[6:]

DIAGNOSTIC_FEATURES = [
    "diagnostic__first_target_token_logprob",
    "diagnostic__relative_completed_trace_position",
]

# Columns that identify a row, locate its activations or carry the Stage 4 target. None is
# a model input; Stage 8 reads primary features through ``PRIMARY_FEATURES`` only.
IDENTITY_COLUMNS = ["schema_version", "sentence_id", "trajectory_id", "problem_id",
                    "trace_kind", "split"]
BOUNDARY_COLUMNS = ["boundary_token_idx", "target_token_start", "overlapping_token",
                    "final_prev_position_available"]
ACTIVATION_COLUMNS = ["activation_run_id", "shard_path", "row_idx", "trace_sha256",
                      "token_ids_sha256"]
TARGET_COLUMNS = ["eligible", "eligible_k4", "exclusion_reason", "label_source",
                  "importance_released", "importance_legacy", "importance_canonical",
                  "importance_primary", "importance_log1p", "anchor_threshold", "anchor"]

FEATURES_SCHEMA = pa.schema([
    pa.field("schema_version", pa.int32(), nullable=False),
    pa.field("sentence_id", pa.string(), nullable=False),
    pa.field("trajectory_id", pa.string(), nullable=False),
    pa.field("problem_id", pa.int64(), nullable=False),
    pa.field("trace_kind", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("sentence_idx", pa.int64(), nullable=False),
    pa.field("generated_token_count", pa.int64(), nullable=False),
    pa.field("prefix_token_count", pa.int64(), nullable=False),
    pa.field("log1p_prefix_token_count", pa.float64(), nullable=False),
    pa.field("prefix_context_fraction", pa.float64(), nullable=False),
    pa.field("previous_sentence_tokens", pa.int64(), nullable=False),
    pa.field("previous_emitted_token_logprob", pa.float64(), nullable=False),
    pa.field("next_token_entropy", pa.float64(), nullable=False),
    pa.field("next_token_max_probability", pa.float64(), nullable=False),
    pa.field("next_token_probability_margin", pa.float64(), nullable=False),
    pa.field("next_token_top5_mass", pa.float64(), nullable=False),
    pa.field("next_token_top10_mass", pa.float64(), nullable=False),
    pa.field("diagnostic__first_target_token_logprob", pa.float64(), nullable=False),
    pa.field("diagnostic__relative_completed_trace_position", pa.float64(), nullable=False),
    pa.field("boundary_token_idx", pa.int64(), nullable=False),
    pa.field("target_token_start", pa.int64(), nullable=False),
    pa.field("overlapping_token", pa.bool_(), nullable=False),
    pa.field("final_prev_position_available", pa.bool_(), nullable=False),
    pa.field("activation_run_id", pa.string(), nullable=False),
    pa.field("shard_path", pa.string(), nullable=False),
    pa.field("row_idx", pa.int64(), nullable=False),
    pa.field("trace_sha256", pa.string(), nullable=False),
    pa.field("token_ids_sha256", pa.string(), nullable=False),
    pa.field("eligible", pa.bool_(), nullable=False),
    pa.field("eligible_k4", pa.bool_(), nullable=False),
    pa.field("exclusion_reason", pa.string(), nullable=True),
    pa.field("label_source", pa.string(), nullable=False),
    pa.field("importance_released", pa.float64(), nullable=True),
    pa.field("importance_legacy", pa.float64(), nullable=True),
    pa.field("importance_canonical", pa.float64(), nullable=True),
    pa.field("importance_primary", pa.float64(), nullable=True),
    pa.field("importance_log1p", pa.float64(), nullable=True),
    pa.field("anchor_threshold", pa.float64(), nullable=True),
    pa.field("anchor", pa.bool_(), nullable=False),
])


def check_feature_names(config, columns: list[str]) -> dict:
    """The primary allow-list equals the frozen config; excluded names never appear
    outside ``diagnostic__`` columns; every column is accounted for."""
    observables = list(config.features["observables"])
    if observables != PRIMARY_FEATURES:
        raise ValueError("Primary feature list differs from the frozen configuration")
    excluded = set(config.features["excluded"])
    prefix = config.features.get("diagnostic_prefix", DIAGNOSTIC_PREFIX)
    if prefix != DIAGNOSTIC_PREFIX:
        raise ValueError("Diagnostic prefix differs from the frozen configuration")
    leaked = [c for c in columns if not c.startswith(prefix) and c in excluded]
    if leaked:
        raise ValueError(f"Excluded quantity in a non-diagnostic column: {leaked}")
    known = set(IDENTITY_COLUMNS + PRIMARY_FEATURES + DIAGNOSTIC_FEATURES + BOUNDARY_COLUMNS
                + ACTIVATION_COLUMNS + TARGET_COLUMNS)
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise ValueError(f"Unclassified feature-table columns: {unknown}")
    missing = [c for c in known if c not in columns]
    if missing:
        raise ValueError(f"Missing feature-table columns: {missing}")
    for name in DIAGNOSTIC_FEATURES:
        if name[len(prefix):] not in excluded:
            raise ValueError(f"Diagnostic column {name} is not a declared exclusion")
    return {"primary": PRIMARY_FEATURES, "diagnostic": DIAGNOSTIC_FEATURES,
            "excluded": sorted(excluded), "diagnostic_prefix": prefix}


@dataclass
class Head:
    """Final RMSNorm and unembedding in float32; ``torch`` tensors on ``device``."""

    norm_weight: object
    lm_head_weight: object
    eps: float
    device: str
    weights_sha256: str
    source: str

    @property
    def vocab_size(self) -> int:
        return int(self.lm_head_weight.shape[0])

    def logits(self, states) -> object:
        """``lm_head(final_norm(h))`` in float32 for states ``[n, d]`` (any float dtype)."""
        import torch

        with torch.inference_mode():
            h = torch.as_tensor(states).to(self.device, torch.float32)
            variance = h.pow(2).mean(-1, keepdim=True)
            normed = self.norm_weight * (h * torch.rsqrt(variance + self.eps))
            return normed @ self.lm_head_weight.T


def _weights_digest(norm, head) -> str:
    import torch

    digest = hashlib.sha256()
    for tensor in (norm, head):
        tensor = tensor.detach().to("cpu").contiguous()
        digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def head_from_model(model, device: str | None = None) -> Head:
    """Head of an already loaded causal LM (tests and same-process diagnostics)."""
    device = device or str(next(model.parameters()).device)
    norm = model.model.norm.weight.detach().to(device).float()
    head = model.lm_head.weight.detach().to(device).float()
    return Head(norm, head, float(model.model.norm.variance_epsilon), device,
                _weights_digest(model.model.norm.weight, model.lm_head.weight), "loaded_model")


def load_head(config, device: str = "cpu") -> Head:
    """Only ``model.norm.weight`` and ``lm_head.weight`` of the pinned revision, from the
    local cache; the decoder stack is never loaded (CPU-only stages, AGENTS §8)."""
    import json

    if device != "cpu":
        from preanchor.control.leases import require

        require(["gpu:h100"])   # the head is small, but it is still the shared GPU

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    kwargs = {"repo_id": config.model.id, "revision": config.model.revision,
              "local_files_only": True}
    with open(hf_hub_download(filename="config.json", **kwargs), "rb") as handle:
        model_config = json.load(handle)
    if model_config.get("tie_word_embeddings", False):
        raise ValueError("Unexpected tied embeddings for the pinned revision")
    if int(model_config["hidden_size"]) != config.model.d_model:
        raise ValueError("Hidden size differs from the frozen d_model")
    with open(hf_hub_download(filename="model.safetensors.index.json", **kwargs), "rb") as h:
        weight_map = json.load(h)["weight_map"]
    tensors = {}
    for name in ("model.norm.weight", "lm_head.weight"):
        shard = hf_hub_download(filename=weight_map[name], **kwargs)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensors[name] = handle.get_tensor(name)
    if tensors["lm_head.weight"].dtype != torch.bfloat16:
        raise ValueError("Pinned lm_head is not bfloat16")
    if tuple(tensors["lm_head.weight"].shape) != (int(model_config["vocab_size"]),
                                                  config.model.d_model):
        raise ValueError("lm_head shape differs from the pinned configuration")
    return Head(tensors["model.norm.weight"].to(device).float(),
                tensors["lm_head.weight"].to(device).float(),
                float(model_config["rms_norm_eps"]), device,
                _weights_digest(tensors["model.norm.weight"], tensors["lm_head.weight"]),
                f"{config.model.id}@{config.model.revision}")


def next_token_statistics(logits) -> dict[str, np.ndarray]:
    """Float32 statistics of the boundary next-token distribution; nothing per-vocab leaves."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    top = torch.topk(probs, 10, dim=-1).values
    entropy = -(probs * logp).sum(-1)
    out = {"next_token_entropy": entropy, "next_token_max_probability": top[:, 0],
           "next_token_probability_margin": top[:, 0] - top[:, 1],
           "next_token_top5_mass": top[:, :5].sum(-1), "next_token_top10_mass": top.sum(-1)}
    return {k: v.to("cpu").numpy().astype(np.float64) for k, v in out.items()}


def token_logprobs(logits, token_ids) -> np.ndarray:
    import torch

    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(1, ids[:, None])[:, 0].to("cpu").numpy().astype(np.float64)


def prompt_token_count(offsets: list[tuple[int, int]], prompt: str, full_cot: str) -> int:
    """Tokens of the exact full-trace tokenisation that end within the stored prompt.

    The prompt is a prefix of the stored trace (Stage 3 invariant); counting by offsets
    avoids re-tokenising a substring (AGENTS §5.1).
    """
    if not full_cot.startswith(prompt):
        raise ValueError("Stored trace does not start with its prompt")
    # A token straddling the prompt end counts as generated; it cannot occur here because
    # every stored prompt ends with the special <think> token followed by a newline.
    return sum(1 for _, end in offsets if end <= len(prompt))


def trajectory_features(boundaries: list[dict], *, prompt_tokens: int, n_sentences: int,
                        context_length: int, boundary_logits, prev_logits, prev_available,
                        boundary_token_ids: list[int], first_target_ids: list[int]) -> list[dict]:
    """Primary and diagnostic columns for one trajectory's stored rows (ascending idx).

    ``boundary_logits``/``prev_logits`` are float32 ``[n, vocab]`` tensors from the stored
    ``layer=48`` states at the boundary and at ``boundary - 1``; they never leave this call.
    """
    n = len(boundaries)
    stats = next_token_statistics(boundary_logits)
    prev_logprob = token_logprobs(prev_logits, boundary_token_ids)
    first_target = token_logprobs(boundary_logits, first_target_ids)
    rows = []
    for j, b in enumerate(boundaries):
        prefix = int(b["prefix_token_count"])
        if prefix != int(b["boundary_token_idx"]) + 1:
            raise ValueError("prefix_token_count must equal boundary_token_idx + 1")
        previous = boundaries[j - 1] if j else None
        rows.append({
            "sentence_idx": int(b["sentence_idx"]),
            "generated_token_count": prefix - prompt_tokens,
            "prefix_token_count": prefix,
            "log1p_prefix_token_count": float(math.log1p(prefix)),
            "prefix_context_fraction": prefix / context_length,
            # Token length of S_{i-1}: its own span [target_token_start, target_token_end),
            # entirely inside the prefix; zero for the first sentence.
            "previous_sentence_tokens": (int(previous["target_token_end"])
                                         - int(previous["target_token_start"])) if previous
            else 0,
            "previous_emitted_token_logprob": (float(prev_logprob[j]) if prev_available[j]
                                               else float("nan")),
            **{k: float(v[j]) for k, v in stats.items()},
            "diagnostic__first_target_token_logprob": float(first_target[j]),
            "diagnostic__relative_completed_trace_position": int(b["sentence_idx"]) / n_sentences,
        })
    if n and any(r["generated_token_count"] < 0 for r in rows):
        raise ValueError("A boundary lies inside the prompt")
    return rows


def features_table(rows: list[dict]) -> pa.Table:
    table = pa.Table.from_pylist(rows, schema=FEATURES_SCHEMA)
    # pyarrow does not enforce nullability on construction; a null primary feature is a
    # silent NaN in disguise (SPEC Stage 7 validation: "no silent NaNs").
    nulls = [f.name for f in FEATURES_SCHEMA if not f.nullable
             and table.column(f.name).null_count]
    if nulls:
        raise ValueError(f"Null values in non-nullable feature columns: {nulls}")
    ids = table.column("sentence_id").to_pylist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sentence_id in the feature table")
    return table


def sample_rows_seed(seed_root: int, purpose: str = "validate_features") -> int:
    from preanchor.ids import derived_seed

    return derived_seed(seed_root, purpose)


def feature_path(root: Path) -> Path:
    return root / "data/processed/stage7/features.parquet"
