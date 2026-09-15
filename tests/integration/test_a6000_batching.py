"""Concurrent scheduling must preserve expensive physical sample identities."""

import threading
from dataclasses import replace

import pytest

from preanchor.ids import file_hash
from preanchor.labels.batching import admitted_concurrency, continuous, required_capacity, roofline
from preanchor.labels.generate import CompletionSpec, CompletionStore


def jobs():
    return [
        CompletionSpec(
            "4682:correct", 13, 13, "without", i, 123 + i, f"prompt-{i}", "config", "input"
        )
        for i in range(8)
    ]


def test_memory_admission():
    assert required_capacity(2) == 40060
    assert required_capacity(3) == 60089
    assert admitted_concurrency(60088) == 2
    assert admitted_concurrency(60089) == 3
    assert admitted_concurrency(40059) == 1
    assert 49 < roofline(2, 2750) < 51


def test_refill_identity_seed_durability_and_resume(tmp_path):
    released = threading.Event()
    calls = []
    mutex = threading.Lock()

    class Backend:
        def complete(self, prompt, *, seed, **kwargs):
            with mutex:
                calls.append((prompt, seed))
            if prompt == "prompt-0":
                assert released.wait(10), "Queue waited for whole batch before refilling"
            if prompt == "prompt-2":
                released.set()
            return {
                "raw_answer": prompt,
                "runtime_seconds": 0.1,
                "finish_reason": "stop",
                "generated_token_count": 1,
            }

    store = CompletionStore(tmp_path / "units")

    def work(spec):
        return store.raw_one(spec, Backend(), max_tokens=16384, temperature=0.6, top_p=0.95)

    records = list(continuous(jobs(), work, 2))
    assert len(records) == len({row["rollout_id"] for _, row, _ in records}) == 8
    assert sorted(calls) == [(s.prompt, s.seed) for s in jobs()]
    for spec, row, created in records:
        assert created and row["raw_answer"] == spec.prompt and row["seed"] == spec.seed
        store.raw_units.validate(spec.rollout_id)
    before = {
        p: (file_hash(p), p.stat().st_mtime_ns)
        for p in store.raw_units.root.glob("*/response.json")
    }
    assert all(not created for _, _, created in continuous(jobs(), work, 3))
    assert len(calls) == 8
    assert before == {p: (file_hash(p), p.stat().st_mtime_ns) for p in before}
    damaged = store.raw_units.root / jobs()[0].rollout_id / "response.json"
    damaged.write_text("corrupt")
    list(continuous(jobs(), work, 2))
    assert len(calls) == 9
    assert len(list((store.raw_units.root / "quarantine").iterdir())) == 1
    store.raw_units.validate(jobs()[0].rollout_id)


def test_duplicate_rejected_before_any_submission():
    def work(spec):
        pytest.fail("No jobs should be submitted")

    with pytest.raises(ValueError, match="Duplicate"):
        list(continuous([jobs()[0], jobs()[0]], work, 2))


def test_swapped_response_rejected():
    def work(spec):
        return {"rollout_id": replace(spec, sample_idx=99).rollout_id, "seed": spec.seed}, True

    with pytest.raises(ValueError, match="identity"):
        list(continuous(jobs(), work, 2))


def test_stream_timings_and_seed_are_request_local(monkeypatch):
    import json

    import httpx

    from preanchor.labels.batching import StreamingVLLMBackend

    recorded = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for event in [
                {"id": "one", "choices": [{"index": 0, "text": "First ", "finish_reason": None}]},
                {
                    "id": "one",
                    "choices": [{"index": 0, "text": "sentence.", "finish_reason": "stop"}],
                },
                {"id": "one", "choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 3}},
            ]:
                yield "data: " + json.dumps(event)
            yield "data: [DONE]"

    def stream(*args, **kwargs):
        recorded.append(kwargs["json"])
        return Stream()

    monkeypatch.setattr(httpx, "stream", stream)
    backend = StreamingVLLMBackend()
    monkeypatch.setattr(backend, "metrics", lambda: pytest.fail("Global timing contamination"))
    row = backend.complete("prompt", seed=123, max_tokens=16384, temperature=0.6, top_p=0.95)
    assert row["raw_answer"] == "First sentence." and row["generated_token_count"] == 2
    assert row["ttft_seconds"] >= 0 and row["decode_seconds"] >= 0
    assert recorded[0]["seed"] == 123 and recorded[0]["max_tokens"] == 16384
    assert recorded[0]["temperature"] == 0.6 and recorded[0]["top_p"] == 0.95


def concurrent_kill_worker(root):
    import time
    from pathlib import Path

    class Backend:
        def complete(self, prompt, **kwargs):
            if prompt != "prompt-0":
                time.sleep(60)
            return {
                "raw_answer": prompt,
                "generated_token_count": 1,
                "runtime_seconds": 0.1,
                "finish_reason": "stop",
            }

    store = CompletionStore(Path(root) / "units")

    def work(spec):
        return store.raw_one(spec, Backend(), max_tokens=16384, temperature=0.6, top_p=0.95)

    for _, _, _ in continuous(jobs(), work, 2):
        Path(root, "durable").touch()


def test_kill_concurrent_worker_loses_only_inflight(tmp_path):
    import multiprocessing
    import time

    process = multiprocessing.get_context("fork").Process(
        target=concurrent_kill_worker, args=(str(tmp_path),)
    )
    process.start()
    deadline = time.monotonic() + 10
    while not (tmp_path / "durable").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert (tmp_path / "durable").exists()
    finally:
        process.kill()
        process.join(5)
    store = CompletionStore(tmp_path / "units")
    store.raw_units.validate(jobs()[0].rollout_id)
    path = store.raw_units.root / jobs()[0].rollout_id / "response.json"
    before = file_hash(path), path.stat().st_mtime_ns

    class Backend:
        def complete(self, prompt, **kwargs):
            assert prompt != "prompt-0", "Regenerated a completed request after kill"
            return {
                "raw_answer": prompt,
                "generated_token_count": 1,
                "runtime_seconds": 0.1,
                "finish_reason": "stop",
            }

    def work(spec):
        return store.raw_one(spec, Backend(), max_tokens=16384, temperature=0.6, top_p=0.95)

    records = list(continuous(jobs(), work, 2))
    assert sum(created for _, _, created in records) == 7
    assert before == (file_hash(path), path.stat().st_mtime_ns)
