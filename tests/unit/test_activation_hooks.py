"""Boundary-selective hooks, logit identity and invariance semantics on a tiny Qwen2."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from preanchor.config import load_config
from preanchor.features.extract import invariance_checks, mutate_future
from preanchor.features.store import index_rows, index_table, read_shard, shard_name, write_shard
from preanchor.model.hooks import replay
from preanchor.model.load import validate_architecture
from preanchor.model.logits import boundary_logits, logit_identity, next_token_statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from tiny_qwen2 import tiny_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ATOL = 1e-5


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 128, (60,), generator=torch.Generator().manual_seed(1)).tolist()


def test_hooks_match_output_hidden_states_and_logits(model, ids):
    boundaries = [5, 17, 40]
    capture = replay(model, ids, boundaries)
    assert capture.resid.shape == (3, 5, 32) and capture.final_prev_position.shape == (3, 32)
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids]), use_cache=False, output_hidden_states=True)
    for site in range(4):  # embedding output and the first three block outputs
        expected = out.hidden_states[site][0, boundaries]
        assert torch.allclose(capture.resid[:, site], expected, atol=ATOL)
    # HF's last hidden_states entry is post-norm; the final site is checked via the logits.
    for j, b in enumerate(boundaries):
        check = logit_identity(model, ids, b, capture.resid[j, -1])
        assert check["max_abs"] <= ATOL and check["argmax_equal"]
        with torch.inference_mode():  # HF's last entry is post-norm: compare via lm_head
            expected_prev = model.lm_head(out.hidden_states[4][0, b - 1]).float()
        assert torch.allclose(expected_prev, boundary_logits(
            model, capture.final_prev_position[j][None])[0], atol=ATOL)


def test_boundary_zero_has_no_previous_position(model, ids):
    capture = replay(model, ids, [0, 3])
    assert capture.final_prev_available == [False, True]
    assert torch.isnan(capture.final_prev_position[0]).all()
    with pytest.raises(ValueError):
        replay(model, ids, [-1])


def test_same_shape_future_mutation_is_exact_and_prefix_replay_matches(model, ids):
    report = invariance_checks(model, ids, [5, 17, 40], replay(model, ids, [5, 17, 40]).resid,
                               atol=ATOL, seed=0, logit_positions=[0, 2])
    assert report["primary_pass"] and report["primary_max_abs"] == 0.0
    assert report["diagnostic_within_atol_count"] == 3
    assert report["logit_identity_pass"]
    assert all(len(e["mutation_max_abs_by_site"]) == 5 for e in report["per_boundary"])


def test_mutation_changes_every_future_token_and_nothing_else(ids):
    mutated = mutate_future(ids, 10, 128, seed=3)
    assert len(mutated) == len(ids) and mutated[:11] == ids[:11]
    assert all(a != b for a, b in zip(mutated[11:], ids[11:], strict=True))


def test_truncated_prefix_discrepancy_is_diagnostic_not_leakage(model, ids):
    """A prefix-only discrepancy with exact same-shape invariance never fails the gate."""
    real = replay(model, ids, [5]).resid
    with pytest.MonkeyPatch.context() as mp:
        original = replay

        def perturbed(m, seq, boundaries, **kw):
            capture = original(m, seq, boundaries, **kw)
            if len(seq) < len(ids):  # emulate sequence-length-dependent kernel rounding
                capture.resid = capture.resid + 0.5
            return capture

        mp.setattr("preanchor.features.extract.replay", perturbed)
        report = invariance_checks(model, ids, [5], real, atol=ATOL, seed=0)
    assert report["primary_pass"] and report["diagnostic_within_atol_count"] == 0
    assert report["diagnostic_max_abs"] >= 0.5


def test_future_leakage_fails_the_primary_gate(model, ids):
    with pytest.MonkeyPatch.context() as mp:
        original = replay

        def leaky(m, seq, boundaries, **kw):
            capture = original(m, seq, boundaries, **kw)
            capture.resid = capture.resid + 1e-3 * sum(seq)  # depends on future tokens
            return capture

        mp.setattr("preanchor.features.extract.replay", leaky)
        report = invariance_checks(model, ids, [5], replay(model, ids, [5]).resid, atol=ATOL,
                                   seed=0)
    assert not report["primary_pass"]


def test_next_token_statistics_are_probability_consistent(model, ids):
    logits = boundary_logits(model, replay(model, ids, [5, 9]).resid[:, -1])
    for row in next_token_statistics(logits):
        assert 0 <= row["next_token_entropy"]
        assert 0 < row["next_token_max_probability"] <= row["next_token_top5_mass"] \
            <= row["next_token_top10_mass"] <= 1 + 1e-6
        assert 0 <= row["next_token_probability_margin"] <= row["next_token_max_probability"]


def test_shard_round_trip_and_index(model, ids, tmp_path):
    capture = replay(model, ids, [5, 9])
    boundaries = [{"sentence_id": "1:correct:0", "sentence_idx": 0, "char_start": 5,
                   "char_end": 8, "target_token_start": 6, "boundary_token_idx": 5,
                   "target_token_end": 8, "overlapping_token": False, "prefix_token_count": 6},
                  {"sentence_id": "1:correct:1", "sentence_idx": 1, "char_start": 9,
                   "char_end": 12, "target_token_start": 10, "boundary_token_idx": 9,
                   "target_token_end": 12, "overlapping_token": True, "prefix_token_count": 10}]
    record = write_shard(tmp_path, "1:correct", capture.resid, capture.final_prev_position,
                         {"boundaries": boundaries, "final_prev_available": [True, True],
                          "trace_sha256": "t" * 64, "token_ids_sha256": "k" * 64})
    assert record["shard"] == shard_name("1:correct") == "1_correct.safetensors"
    meta, tensors = read_shard(tmp_path, expected_sites=5, expected_d_model=32)
    assert tensors["resid"].dtype == np.float16 and tensors["resid"].shape == (2, 5, 32)
    assert np.allclose(tensors["resid"], capture.resid.numpy(), atol=1e-3)
    sentences = {b["sentence_id"]: {"problem_id": 1, "trace_kind": "correct", "eligible": True,
                                    "eligible_k4": False, "exclusion_reason": None,
                                    "label_source": "canonical_fallback"} for b in boundaries}
    table = index_table(index_rows(meta, "units/x/1_correct.safetensors", sentences, "l" * 64,
                                   {"model_id": "m", "revision": "r", "tokenizer_revision": "r"}))
    assert table.num_rows == 2 and table.column("row_idx").to_pylist() == [0, 1]
    with pytest.raises(ValueError, match="Expected 49"):
        read_shard(tmp_path, expected_sites=49)
    (tmp_path / "1_correct.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        read_shard(tmp_path)


def test_tiny_model_only_passes_non_strict_validation(model):
    config = load_config(ROOT / "configs/smoke.yaml")
    with pytest.raises(ValueError):
        validate_architecture(model, config)
    assert validate_architecture(model, config, strict=False)["n_layers"] == 4
