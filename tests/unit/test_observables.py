"""SPEC §3.9 observable definitions, allow-list and float32 head on the tiny Qwen2."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from preanchor.config import load_config
from preanchor.features.observables import (
    DIAGNOSTIC_FEATURES,
    FEATURES_SCHEMA,
    PRIMARY_FEATURES,
    check_feature_names,
    features_table,
    head_from_model,
    next_token_statistics,
    prompt_token_count,
    token_logprobs,
    trajectory_features,
)
from preanchor.model.boundaries import boundary_table, tokenize_trace
from preanchor.model.hooks import replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from tiny_qwen2 import tiny_model, tiny_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ATOL = 1e-5


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/mvp.yaml")


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def test_allow_list_matches_frozen_config_and_rejects_leaks(config):
    names = FEATURES_SCHEMA.names
    info = check_feature_names(config, names)
    assert info["primary"] == PRIMARY_FEATURES == list(config.features["observables"])
    assert info["diagnostic"] == DIAGNOSTIC_FEATURES
    assert all(c.startswith("diagnostic__") for c in DIAGNOSTIC_FEATURES)
    for forbidden in config.features["excluded"]:
        assert forbidden not in names, forbidden
        with pytest.raises(ValueError, match="Excluded quantity"):
            check_feature_names(config, [*names, forbidden])
    with pytest.raises(ValueError, match="Unclassified"):
        check_feature_names(config, [*names, "completed_trajectory_tokens"])
    with pytest.raises(ValueError, match="Missing"):
        check_feature_names(config, names[:-1])


def test_head_reproduces_lm_head_of_final_norm_in_float32(model):
    head = head_from_model(model)
    ids = torch.randint(0, 128, (40,), generator=torch.Generator().manual_seed(2)).tolist()
    capture = replay(model, ids, [7, 21])
    with torch.inference_mode():
        expected = model.lm_head(model.model.norm(capture.resid[:, -1])).float()
    logits = head.logits(capture.resid[:, -1].numpy())
    assert logits.dtype == torch.float32 and logits.shape == (2, 128)
    assert torch.allclose(logits, expected, atol=ATOL)
    assert head.vocab_size == 128 and len(head.weights_sha256) == 64
    # float16-stored states are what Stage 7 reads.
    stored = capture.resid[:, -1].to(torch.float16).numpy()
    assert torch.allclose(head.logits(stored), expected, atol=1e-2)


def test_statistics_are_probability_consistent_and_logprobs_match(model):
    head = head_from_model(model)
    ids = torch.randint(0, 128, (30,), generator=torch.Generator().manual_seed(3)).tolist()
    logits = head.logits(replay(model, ids, [4, 12, 20]).resid[:, -1])
    stats = next_token_statistics(logits)
    assert set(stats) == set(PRIMARY_FEATURES[7:])
    for j in range(3):
        assert 0 <= stats["next_token_entropy"][j] <= math.log(128) + 1e-6
        assert 0 < stats["next_token_max_probability"][j] <= stats["next_token_top5_mass"][j] \
            <= stats["next_token_top10_mass"][j] <= 1 + 1e-6
        assert 0 <= stats["next_token_probability_margin"][j] \
            <= stats["next_token_max_probability"][j]
    logp = token_logprobs(logits, [ids[5], ids[13], ids[21]])
    expected = torch.log_softmax(logits, -1)
    assert np.allclose(logp, [expected[0, ids[5]], expected[1, ids[13]], expected[2, ids[21]]])
    assert (logp <= 0).all()


def test_prompt_token_count_uses_full_trace_offsets():
    tokenizer = tiny_tokenizer()
    prompt = "Problem 1 Solution: \n<think>\n"
    trace = prompt + "Okay, so.\nLet me see."
    ids, offsets = tokenize_trace(tokenizer, trace)
    count = prompt_token_count(offsets, prompt, trace)
    assert count == len(prompt) and tokenizer.decode(ids[:count]) == prompt
    with pytest.raises(ValueError, match="start with its prompt"):
        prompt_token_count(offsets, "other", trace)


def test_trajectory_feature_definitions(model):
    tokenizer = tiny_tokenizer()
    prompt = "P 1: \n<think>\n"
    body = ["Okay, so.", "Let me see.", "So x = 2."]
    trace, spans = prompt, []
    for i, s in enumerate(body):
        if i:
            trace += " "
        spans.append((len(trace), len(trace) + len(s)))
        trace += s
    sentences = [{"sentence_id": f"1:correct:{i}", "sentence_idx": i, "char_start": a,
                  "char_end": b} for i, (a, b) in enumerate(spans)]
    ids, offsets = tokenize_trace(tokenizer, trace)
    table = boundary_table(offsets, sentences)
    boundaries = [b.__dict__ for b in table]
    head = head_from_model(model)
    b_idx = [b.boundary_token_idx for b in table]
    capture = replay(model, ids, b_idx)
    rows = trajectory_features(
        boundaries, prompt_tokens=len(prompt), n_sentences=4, context_length=512,
        boundary_logits=head.logits(capture.resid[:, -1]),
        prev_logits=head.logits(capture.final_prev_position),
        prev_available=[True, True, False],
        boundary_token_ids=[ids[b] for b in b_idx], first_target_ids=[ids[b + 1] for b in b_idx])
    assert [r["sentence_idx"] for r in rows] == [0, 1, 2]
    assert [r["prefix_token_count"] for r in rows] == [b + 1 for b in b_idx]
    assert [r["generated_token_count"] for r in rows] == [b + 1 - len(prompt) for b in b_idx]
    # Sentence 0 starts right after the prompt: the boundary is the prompt's last token.
    assert rows[0]["generated_token_count"] == 0
    # previous_sentence_tokens is the token span of S_{i-1}; the space before S_i is not
    # part of S_{i-1} under the character-level tokenizer.
    assert rows[0]["previous_sentence_tokens"] == 0
    assert rows[1]["previous_sentence_tokens"] == len("Okay, so.")
    assert rows[2]["previous_sentence_tokens"] == len("Let me see.")
    assert rows[1]["log1p_prefix_token_count"] == math.log1p(rows[1]["prefix_token_count"])
    assert rows[1]["prefix_context_fraction"] == rows[1]["prefix_token_count"] / 512
    assert [r["diagnostic__relative_completed_trace_position"] for r in rows] == [0, .25, .5]
    assert math.isnan(rows[2]["previous_emitted_token_logprob"])
    assert rows[0]["previous_emitted_token_logprob"] <= 0
    assert set(rows[0]) == set(PRIMARY_FEATURES) | set(DIAGNOSTIC_FEATURES)
    bad = [{**b, "prefix_token_count": b["prefix_token_count"] + 1} for b in boundaries]
    with pytest.raises(ValueError, match="boundary_token_idx \\+ 1"):
        trajectory_features(bad, prompt_tokens=len(prompt), n_sentences=4, context_length=512,
                            boundary_logits=head.logits(capture.resid[:, -1]),
                            prev_logits=head.logits(capture.final_prev_position),
                            prev_available=[True] * 3, boundary_token_ids=[ids[b] for b in b_idx],
                            first_target_ids=[ids[b + 1] for b in b_idx])


def test_features_table_enforces_schema_and_unique_ids():
    row = {name: None for name in FEATURES_SCHEMA.names}
    row.update({"schema_version": 1, "sentence_id": "1:correct:0", "trajectory_id": "1:correct",
                "problem_id": 1, "trace_kind": "correct", "split": "train", "sentence_idx": 0,
                "generated_token_count": 0, "prefix_token_count": 3,
                "log1p_prefix_token_count": math.log1p(3), "prefix_context_fraction": 3 / 512,
                "previous_sentence_tokens": 0, "previous_emitted_token_logprob": -0.1,
                "next_token_entropy": 0.5, "next_token_max_probability": 0.7,
                "next_token_probability_margin": 0.5, "next_token_top5_mass": 0.9,
                "next_token_top10_mass": 0.95, "diagnostic__first_target_token_logprob": -1.0,
                "diagnostic__relative_completed_trace_position": 0.0, "boundary_token_idx": 2,
                "target_token_start": 3, "overlapping_token": False,
                "final_prev_position_available": True, "activation_run_id": "r",
                "shard_path": "units/k/1_correct.safetensors", "row_idx": 0,
                "trace_sha256": "t" * 64, "token_ids_sha256": "k" * 64, "eligible": True,
                "eligible_k4": True, "label_source": "canonical_fallback", "anchor": False})
    table = features_table([row])
    assert table.schema.equals(FEATURES_SCHEMA) and table.num_rows == 1
    with pytest.raises(ValueError, match="Duplicate"):
        features_table([row, dict(row)])
    with pytest.raises(ValueError, match="Null values"):
        features_table([{**row, "next_token_entropy": None}])
