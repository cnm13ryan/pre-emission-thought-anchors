"""Boundary-selective forward hooks: one causal replay, no full hidden-state retention."""

from __future__ import annotations

import time
from dataclasses import dataclass

from preanchor.model.load import decoder_layers, embedding_module


@dataclass
class Capture:
    resid: object  # torch [n, n_layers + 1, d] on CPU, model dtype
    final_prev_position: object  # torch [n, d] on CPU; row is NaN where boundary - 1 < 0
    final_prev_available: list[bool]
    forward_seconds: float
    peak_allocated_bytes: int
    n_tokens: int


def replay(model, ids: list[int], boundary_idx: list[int], *, device=None) -> Capture:
    """Full causal replay of `ids` (batch one) retaining only `boundary_idx` rows per site.

    Sites: layer 0 = embedding output; layer i = decoder block i output (before the final
    norm). The final_prev row is the last site at boundary - 1.
    """
    import torch

    if not ids:
        raise ValueError("Empty token sequence")
    n_tokens = len(ids)
    for b in boundary_idx:
        if not 0 <= b < n_tokens:
            raise ValueError("Boundary outside the sequence")
    device = device or next(model.parameters()).device
    layers = decoder_layers(model)
    d_model = model.config.hidden_size
    n_sites = len(layers) + 1
    dtype = next(model.parameters()).dtype
    n = len(boundary_idx)
    resid = torch.empty((n, n_sites, d_model), dtype=dtype)
    prev = torch.full((n, d_model), float("nan"), dtype=dtype)
    available = [b - 1 >= 0 for b in boundary_idx]
    sel = torch.as_tensor(boundary_idx, dtype=torch.long, device=device)
    prev_rows = [i for i, ok in enumerate(available) if ok]
    prev_sel = torch.as_tensor([boundary_idx[i] - 1 for i in prev_rows], dtype=torch.long,
                               device=device)
    prev_rows_t = torch.as_tensor(prev_rows, dtype=torch.long)
    last = len(layers)
    handles = []

    def make(site):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() != 3 or hidden.shape[0] != 1 or hidden.shape[1] != n_tokens:
                raise ValueError("Unexpected hidden-state shape in hook")
            resid[:, site] = hidden[0].index_select(0, sel).detach().to("cpu")
            if site == last and prev_rows:
                prev[prev_rows_t] = hidden[0].index_select(0, prev_sel).detach().to("cpu")
        return hook

    handles.append(embedding_module(model).register_forward_hook(make(0)))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make(i + 1)))
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            model.model(input_ids=torch.tensor([ids], device=device), use_cache=False,
                        output_attentions=False, output_hidden_states=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    finally:
        for handle in handles:
            handle.remove()
    return Capture(resid, prev, available, elapsed, int(peak), n_tokens)
