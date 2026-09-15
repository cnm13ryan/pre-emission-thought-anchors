"""AGENTS §5.1 / SPEC §10.2 on the pinned model (GPU, cached weights); skipped otherwise.

Primary gate (amendment 2026-09-15): same-shape future-token mutation leaves every boundary
site unchanged within the frozen bf16 tolerance. Truncated-prefix replay is a recorded
diagnostic because GPU bf16 kernels round differently for short sequence shapes.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not os.environ.get("PREANCHOR_ROOT"),
    reason="needs the GPU, the cached pinned model and the persistent Stage 3/4 tables")


@pytest.fixture(scope="module")
def loaded():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from preanchor.config import Settings, load_config
    from preanchor.features.extract import load_inputs
    from preanchor.model.load import load_model, load_tokenizer

    config = load_config(ROOT / "configs/smoke.yaml")
    inputs, _ = load_inputs(Settings().persistent_root())
    try:
        tokenizer, model = load_tokenizer(config), load_model(config)
    except OSError:
        pytest.skip("pinned model not cached locally")
    return config, inputs, tokenizer, model


def test_boundary_state_is_prefix_only_on_the_pinned_model(loaded):
    from preanchor.features.extract import invariance_checks
    from preanchor.model.boundaries import audit_boundaries, boundary_table, tokenize_trace
    from preanchor.model.hooks import replay

    config, inputs, tokenizer, model = loaded
    tid = config.execution.trajectory
    ids, offsets = tokenize_trace(tokenizer, inputs["trajectories"][tid]["full_cot"])
    table = boundary_table(offsets, inputs["sentences"][tid])[: config.execution.limit_sentences]
    audit_boundaries(ids, offsets, table)
    chosen = [table[i].boundary_token_idx for i in (0, 1, len(table) // 2, len(table) - 1)]
    capture = replay(model, ids, chosen)
    assert capture.resid.shape == (4, 49, 5120)
    report = invariance_checks(model, ids, chosen, capture.resid,
                               atol=config.features["invariance_atol_bfloat16"],
                               seed=config.seed_root, logit_positions=[0, 3])
    assert report["primary_pass"], report["primary_max_abs"]
    assert report["logit_identity_pass"]
    # Diagnostic only: record, never gate (sequence-length-dependent bf16 rounding).
    assert all(e["truncated_prefix_max_abs"] >= 0 for e in report["per_boundary"])
    with pytest.raises(ValueError):
        replay(model, ids, [-1])
