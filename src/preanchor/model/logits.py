"""Next-token statistics from stored final-site boundary states (SPEC §3.9, AGENTS §5.2)."""

from __future__ import annotations

from preanchor.model.load import final_norm

STATISTIC_COLUMNS = ["next_token_entropy", "next_token_max_probability",
                     "next_token_probability_margin", "next_token_top5_mass",
                     "next_token_top10_mass"]


def boundary_logits(model, final_states):
    """`lm_head(final_norm(h))` in float32 for a batch of final-site states [n, d]."""
    import torch

    with torch.inference_mode():
        h = final_states.to(next(model.parameters()).device, next(model.parameters()).dtype)
        return model.lm_head(final_norm(model)(h)).float()


def next_token_statistics(logits) -> list[dict]:
    """Float32 statistics; never returns or stores full-vocabulary rows."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    top = torch.topk(probs, 10, dim=-1).values
    entropy = -(probs * logp).sum(-1)
    return [{"next_token_entropy": float(entropy[i]),
             "next_token_max_probability": float(top[i, 0]),
             "next_token_probability_margin": float(top[i, 0] - top[i, 1]),
             "next_token_top5_mass": float(top[i, :5].sum()),
             "next_token_top10_mass": float(top[i].sum())} for i in range(logits.shape[0])]


def token_logprob(logits, token_ids: list[int]) -> list[float]:
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    return [float(logp[i, t]) for i, t in enumerate(token_ids)]


def logit_identity(model, ids: list[int], boundary: int, captured_final_state) -> dict:
    """Same forward pass: model logits at `boundary` versus lm_head(norm(captured state))."""
    import torch

    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], device=device), use_cache=False)
        reference = out.logits[0, boundary].float()
    recomputed = boundary_logits(model, captured_final_state[None])[0]
    diff = (reference - recomputed).abs()
    return {"max_abs": float(diff.max()),
            "argmax_equal": bool(reference.argmax() == recomputed.argmax()),
            "reference_argmax": int(reference.argmax())}
