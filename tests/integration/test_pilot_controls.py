"""No GPU or network: exercise production admission and expensive-output reuse."""
import multiprocessing
import time
from pathlib import Path

import pytest

from preanchor.ids import file_hash
from preanchor.labels.generate import (
    CompletionSpec,
    CompletionStore,
    FakeBackend,
    production_specs,
    support_counts,
)
from preanchor.labels.pilot import FORKS, assert_plan, guard_reason, tokenizer_check


def spec():
    return CompletionSpec("4682:correct", 13, 13, "without", 0, 123,
                          "prompt", "config", "input", ((13, "without"),))


class BrokenScorer:
    def score(self, reference, replacement):
        raise RuntimeError("scoring failed")


class Scorer:
    def score(self, reference, replacement):
        return .79


def test_scoring_failure_does_not_repeat_expensive_generation(tmp_path):
    backend = FakeBackend(["A valid replacement. Next sentence."])
    store = CompletionStore(tmp_path / "units")
    args = dict(max_tokens=16384, temperature=.6, top_p=.95, reference_texts={13: "Original"})
    with pytest.raises(RuntimeError, match="scoring failed"):
        store.run_one(spec(), backend, similarity_scorer=BrokenScorer(), **args)
    raw = store.raw_units.root / spec().rollout_id / "response.json"
    digest, mtime = file_hash(raw), raw.stat().st_mtime_ns
    row, created = store.run_one(spec(), backend, similarity_scorer=Scorer(), **args)
    assert created and backend.calls == 1
    assert (file_hash(raw), raw.stat().st_mtime_ns) == (digest, mtime)
    assert row["is_dissimilar"] is True


def killed_worker(root):
    class BlockingScorer:
        def score(self, reference, replacement):
            Path(root, "scoring_started").touch()
            time.sleep(60)
    CompletionStore(Path(root) / "units").run_one(
        spec(), FakeBackend(["A valid replacement. Next sentence."]),
        max_tokens=16384, temperature=.6, top_p=.95,
        reference_texts={13: "Original"}, similarity_scorer=BlockingScorer())


def test_kill_after_generation_resumes_without_regeneration(tmp_path):
    worker = multiprocessing.get_context("fork").Process(target=killed_worker, args=(tmp_path,))
    worker.start()
    deadline = time.monotonic() + 10
    while not (tmp_path / "scoring_started").exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert (tmp_path / "scoring_started").exists()
    raw = tmp_path / "units_raw" / spec().rollout_id / "response.json"
    before = (file_hash(raw), raw.stat().st_mtime_ns)
    worker.kill()
    worker.join(timeout=5)
    backend = FakeBackend()
    store = CompletionStore(tmp_path / "units")
    kwargs = dict(max_tokens=16384, temperature=.6, top_p=.95,
                  reference_texts={13: "Original"}, similarity_scorer=Scorer())
    store.run_one(spec(), backend, **kwargs)
    assert backend.calls == 0 and before == (file_hash(raw), raw.stat().st_mtime_ns)
    completed = store.units.root / spec().rollout_id / "completion.json"
    before_complete = (file_hash(completed), completed.stat().st_mtime_ns)
    _, created = store.run_one(spec(), backend, **kwargs)
    assert not created and before_complete == (file_hash(completed), completed.stat().st_mtime_ns)


def test_plan_rejects_duplicates_and_extra_adaptive_samples():
    trajectory = {"trajectory_id": "4682:correct", "full_cot": "P" * 100,
                  "sentence_boundaries": {str(f): f for f in FORKS}}
    targets = [{"sentence_idx": f, "without_fork_idx": f, "with_fork_idx": f + 1}
               for f in [6, 12, 13, 14, 46]]
    jobs = production_specs(trajectory, targets, 20260914, "c", "i")
    assert_plan(jobs)
    with pytest.raises(ValueError):
        assert_plan(jobs + [jobs[0]])
    with pytest.raises(ValueError):
        assert_plan(production_specs(trajectory, targets, 20260914, "c", "i",
                                     samples_per_fork=100))


def test_equal_token_counts_do_not_hide_different_ids():
    with pytest.raises(ValueError, match="token IDs mismatch"):
        tokenizer_check([1, 2], [2, 1])


def test_budget_guard_after_32_and_truncation_guard():
    row = {"finish_reason": "stop", "runtime_seconds": 150}
    # Wall-clock projection only once 32 units completed in this session.
    assert guard_reason([row] * 31, session_seconds=31 * 150, session_completed=31,
                        rate=1.99) is None
    assert guard_reason([row] * 32, session_seconds=32 * 150, session_completed=32,
                        rate=1.99) == "BASE_PROJECTION_EXCEEDS_AUTHORIZATION"
    assert guard_reason([row] * 32, session_seconds=32 * 110, session_completed=32,
                        rate=1.99) is None
    # Truncation is judged on a minimum sample, never on a single early row.
    assert guard_reason([{**row, "finish_reason": "length"}], rate=1.99) is None
    assert guard_reason([{**row, "finish_reason": "length"}] * 20,
                        rate=1.99) == "LENGTH_TRUNCATION_ABOVE_5_PERCENT"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_scores_never_count(value):
    result = support_counts([{"extraction_status": "success",
                              "first_replacement_sentence": "text", "similarity": value}])
    assert result["valid_similarity"] == result["dissimilar_valid"] == 0
