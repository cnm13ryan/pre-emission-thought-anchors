"""Pinned BF16 model loading with explicit single-GPU placement (AGENTS §6)."""

from __future__ import annotations

from preanchor.config import ExperimentConfig


def load_tokenizer(config: ExperimentConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.id,
                                              revision=config.model.tokenizer_revision,
                                              local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Pinned fast tokenizer required")
    return tokenizer


def load_model(config: ExperimentConfig, device: str = "cuda:0"):
    """Exact revision, bfloat16, SDPA, one explicit device; fails on CPU/meta parameters."""
    import torch
    from transformers import AutoModelForCausalLM

    if config.model.quantization is not None or config.model.dtype != "bfloat16":
        raise ValueError("Frozen model settings changed")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id, revision=config.model.revision, torch_dtype=torch.bfloat16,
        attn_implementation=config.model.attention_implementation, local_files_only=True,
        device_map={"": device})
    model.eval()
    validate_architecture(model, config)
    return model


def validate_architecture(model, config: ExperimentConfig, *, strict: bool = True) -> dict:
    """`strict=False` only for the network-independent tiny test model (SPEC §10.2)."""
    import torch

    layers = decoder_layers(model)
    if strict:
        stray = [n for n, p in model.named_parameters() if p.device.type in {"cpu", "meta"}]
        if stray:
            raise ValueError(f"Parameters not on the GPU: {stray[:3]}")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("All parameters must be bfloat16")
        if len(layers) != config.model.decoder_layers:
            raise ValueError(f"Expected {config.model.decoder_layers} layers, got {len(layers)}")
        if model.config.hidden_size != config.model.d_model:
            raise ValueError("Hidden size differs from the frozen d_model")
    if getattr(model.config, "tie_word_embeddings", False):
        raise ValueError("Unexpected tied embeddings for the pinned revision")
    if model.config._attn_implementation != config.model.attention_implementation:
        raise ValueError("Attention implementation differs from the frozen setting")
    return {"model_id": config.model.id, "revision": config.model.revision, "strict": strict,
            "n_layers": len(layers), "d_model": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "attention_implementation": model.config._attn_implementation,
            "dtype": str(next(model.parameters()).dtype),
            "n_parameters": sum(p.numel() for p in model.parameters())}


def decoder_layers(model):
    return model.model.layers


def embedding_module(model):
    return model.model.embed_tokens


def final_norm(model):
    return model.model.norm


def gpu_snapshot() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {"available": True, "name": props.name, "total_mib": props.total_memory // 2**20,
            "allocated_mib": torch.cuda.memory_allocated() // 2**20,
            "reserved_mib": torch.cuda.memory_reserved() // 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
            "torch": torch.__version__, "cuda": torch.version.cuda}
