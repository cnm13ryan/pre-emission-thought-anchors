from pathlib import Path

import pytest

from preanchor.ids import sample_seed
from preanchor.labels.generate import (
    CompletionSpec,
    CompletionStore,
    FakeBackend,
    adaptive_sample_count,
    benchmark,
    parse_response,
    prompt_hash,
    tokenizer_identity_check,
)


def trajectory():
    return {"trajectory_id": "4682:correct", "prompt": "P\n<think>\n",
            "full_cot": "P\n<think>\nA", "full_cot_sha256": "input",
            "sentence_boundaries": {str(i): 10 for i in [6, 12, 13, 14, 46]}}


def test_seed_matches_frozen_derivation():
    import hashlib
    expected = int(hashlib.sha256(b"20260914:4682:correct:6:0").hexdigest()[:8], 16)
    assert sample_seed(20260914, "4682:correct", 6, 0) == expected


def test_response_parsing_and_no_answer():
    parsed = parse_response({"choices": [{"text": "" , "finish_reason": "length"}],
                             "usage": {"prompt_tokens": 3, "completion_tokens": 7}})
    assert parsed["normalized_answer"] == "<NO_ANSWER>"
    assert parsed["finish_reason"] == "length"


def test_tokenizer_mismatch_fails_closed():
    with pytest.raises(ValueError):
        tokenizer_identity_check(host_count=3, backend_count=4, tokenizer_id="x",
                                 tokenizer_revision="r", prompt_sha256=prompt_hash("x"))


def test_adaptive_only_affected_forks_extend():
    assert adaptive_sample_count(10) == 32
    assert adaptive_sample_count(9) == 100


def test_benchmark_exact_count_and_resume_skip(tmp_path: Path):
    store = CompletionStore(tmp_path / "units")
    rows = benchmark(FakeBackend(), store, trajectory(), [6, 12, 13, 14, 46], count=8,
                     seed_root=20260914, config_hash="c", input_hash="i")
    assert len(rows) == 8
    hashes = {p: p.read_bytes() for p in (tmp_path / "units").glob("*/completion.json")}
    rows2 = benchmark(FakeBackend(), store, trajectory(), [6, 12, 13, 14, 46], count=8,
                      seed_root=20260914, config_hash="c", input_hash="i")
    assert len(rows2) == 8
    assert hashes == {p: p.read_bytes() for p in hashes}


def test_corrupt_completion_is_quarantined_and_recomputed(tmp_path: Path):
    store = CompletionStore(tmp_path / "units")
    spec = CompletionSpec("t", 6, 6, "without", 0, 1, "P", "c", "i")
    store.run_one(spec, FakeBackend(), max_tokens=10, temperature=.6, top_p=.95)
    path = next((tmp_path / "units").glob("*/completion.json"))
    path.write_text("corrupt")
    store.run_one(spec, FakeBackend(["\\boxed{43}"]), max_tokens=10, temperature=.6, top_p=.95)
    assert (tmp_path / "units" / "quarantine").exists()
