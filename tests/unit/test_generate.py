from pathlib import Path

import pytest

from preanchor.ids import sample_seed
from preanchor.labels.generate import (
    CompletionSpec,
    CompletionStore,
    FakeBackend,
    adaptive_eligible_forks,
    adaptive_sample_count,
    benchmark,
    build_prompt,
    canonical_prompt_identities,
    generate_production,
    parse_response,
    production_specs,
    prompt_hash,
    required_fork_associations,
    tokenizer_identity_check,
    unique_required_forks,
)


def trajectory():
    return {"trajectory_id": "4682:correct", "prompt": "P\n<think>\n",
            "full_cot": "P\n<think>\nA", "full_cot_sha256": "input",
            "sentence_boundaries": {str(i): 10 for i in range(48)}}


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


def frozen_targets():
    return [
        {"sentence_idx": 6, "without_fork_idx": 6, "with_fork_idx": 7},
        {"sentence_idx": 12, "without_fork_idx": 12, "with_fork_idx": 13},
        {"sentence_idx": 13, "without_fork_idx": 13, "with_fork_idx": 14},
        {"sentence_idx": 14, "without_fork_idx": 14, "with_fork_idx": 15},
        {"sentence_idx": 46, "without_fork_idx": 46, "with_fork_idx": 47},
    ]


def test_shared_forks_are_one_physical_generation_set():
    targets = frozen_targets()
    assert unique_required_forks(targets) == [6, 7, 12, 13, 14, 15, 46, 47]
    assert adaptive_eligible_forks(targets) == [6, 12, 13, 14, 46]
    associations = required_fork_associations(targets)
    assert {(x["target_sentence_idx"], x["condition"]) for x in associations[13]} == {
        (12, "with"), (13, "without")}
    assert {(x["target_sentence_idx"], x["condition"]) for x in associations[14]} == {
        (13, "with"), (14, "without")}
    identities = canonical_prompt_identities(trajectory(), targets)
    assert identities[13]["prompt_sha256"] == prompt_hash(build_prompt(trajectory(), 12, 13))
    assert identities[13]["prompt_sha256"] == prompt_hash(build_prompt(trajectory(), 13, 13))


def test_shared_fork_rollout_identity_and_seed_are_association_independent():
    targets = frozen_targets()
    specs = production_specs(trajectory(), targets, 20260914, "c", "i", samples_per_fork=1)
    by_fork = {spec.fork_idx: spec for spec in specs}
    # Same canonical fork/sample has one key and the frozen seed derivation.
    assert len(specs) == 8
    assert by_fork[13].seed == sample_seed(20260914, "4682:correct", 13, 0)
    assert set(by_fork[13].associations) == {(12, "with"), (13, "without")}
    shared_a = CompletionSpec("4682:correct", 12, 13, "with", 0,
                              by_fork[13].seed, by_fork[13].prompt, "c", "i")
    shared_b = CompletionSpec("4682:correct", 13, 13, "without", 0,
                              by_fork[13].seed, by_fork[13].prompt, "c", "i")
    assert shared_a.rollout_id == shared_b.rollout_id


def test_shared_adaptive_fork_extends_once():
    specs = production_specs(trajectory(), frozen_targets(), 20260914, "c", "i",
                             samples_per_fork=100)
    # 100 samples for each of 8 canonical forks, rather than once per target
    # association (which would duplicate shared forks 13 and 14).
    assert len(specs) == 800
    assert len({s.rollout_id for s in specs if s.fork_idx == 13}) == 100


def test_shared_fork_store_reuses_one_completed_sample(tmp_path: Path):
    targets = frozen_targets()
    spec = next(s for s in production_specs(trajectory(), targets, 20260914, "c", "i",
                                            samples_per_fork=1) if s.fork_idx == 13)
    alternate = CompletionSpec("4682:correct", 13, 13, "without", 0, spec.seed,
                               spec.prompt, "c", "i", spec.associations)
    store = CompletionStore(tmp_path / "units")
    _, first_created = store.run_one(spec, FakeBackend(), max_tokens=10,
                                     temperature=.6, top_p=.95)
    _, second_created = store.run_one(alternate, FakeBackend(), max_tokens=10,
                                      temperature=.6, top_p=.95)
    assert first_created and not second_created


def test_production_scheduler_uses_unique_forks(tmp_path: Path):
    rows = generate_production(FakeBackend(), CompletionStore(tmp_path / "units"),
                               trajectory(), frozen_targets(), seed_root=20260914,
                               config_hash="c", input_hash="i", samples_per_fork=2,
                               max_tokens=10, temperature=.6, top_p=.95)
    assert len(rows) == 16
    assert len({row["rollout_id"] for row in rows}) == 16


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
