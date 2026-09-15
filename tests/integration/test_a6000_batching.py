"""Concurrent scheduling must preserve expensive physical sample identities."""

import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from preanchor.ids import file_hash
from preanchor.labels.batching import (
    MAX_CONCURRENCY,
    admitted_concurrency,
    choose,
    continuous,
    required_capacity,
    roofline,
)
from preanchor.labels.generate import CompletionSpec, CompletionStore

ROOT = Path(__file__).resolve().parents[2]


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
    # An 80 GB card's measured capacity admits more than the A6000's three, never more
    # than the profile ceiling, always with the 15% worst-case margin.
    assert required_capacity(8) == 160237
    assert admitted_concurrency(160236) == 7
    assert admitted_concurrency(160237) == 8
    assert admitted_concurrency(10**9) == MAX_CONCURRENCY
    with pytest.raises(ValueError, match="Insufficient"):
        admitted_concurrency(20000)
    # H100 PCIe vendor figures: the roofline scales with bandwidth, not the A6000 constants.
    assert roofline(8, 2750, bandwidth=2.0e12, dense_flops=756e12) > roofline(8, 2750)


def test_hardware_profiles_drive_admission():
    from preanchor.labels.a6000 import HARDWARE, MAX_MODEL_LEN, hardware, profile_utilization

    assert MAX_MODEL_LEN == "20480"
    assert profile_utilization("NVIDIA RTX A6000") == 0.92
    assert profile_utilization("NVIDIA H100 PCIe") == 0.90
    assert set(HARDWARE) == {"NVIDIA RTX A6000", "NVIDIA H100 PCIe"}
    with pytest.raises(ValueError, match="measured cards"):
        hardware("NVIDIA A100-PCIE-40GB")


def test_selection_chain_rule():
    def candidate(n, rate):
        return {"concurrency": n, "aggregate_tokens_per_second": rate}

    assert choose([candidate(2, 40.0)])["concurrency"] == 2
    assert choose([candidate(2, 40.0), candidate(3, 43.9)])["concurrency"] == 2
    assert choose([candidate(3, 44.0), candidate(2, 40.0)])["concurrency"] == 3
    # The chain stops at the first step that fails to improve by 10%; later candidates
    # are not compared against an earlier one they never beat step by step.
    chain = [candidate(4, 100.0), candidate(8, 150.0), candidate(10, 160.0), candidate(12, 200.0)]
    assert choose(chain)["concurrency"] == 8
    with pytest.raises(ValueError, match="Duplicate"):
        choose([candidate(4, 1.0), candidate(4, 2.0)])
    with pytest.raises(ValueError, match="No benchmark"):
        choose([])


@pytest.mark.parametrize("seqs", ["0", "17", "2x"])
def test_start_vllm_rejects_out_of_range_concurrency(tmp_path, seqs):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/start_vllm.sh"), "configs/lambda/h100_pcie_80.yaml"],
        env={"PATH": "/usr/bin:/bin", "PREANCHOR_ROOT": str(tmp_path),
             "PREANCHOR_REQUEST_CONCURRENCY": seqs},
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1 and "between 1 and 16" in result.stderr


def test_start_vllm_reads_only_utilization_from_profile():
    script = (ROOT / "scripts/start_vllm.sh").read_text()
    assert 'MAX_MODEL_LEN=20480' in script and "32768" not in script
    assert "gpu_memory_utilization" in script and '--max-num-seqs "$SEQS"' in script
    for name, expected in [("h100_pcie_80", "0.90"), ("a6000_48", "0.92")]:
        util = subprocess.run(
            ["sed", "-n", r"s/^[[:space:]]*gpu_memory_utilization:[[:space:]]*\([0-9.]*\).*/\1/p",
             str(ROOT / f"configs/lambda/{name}.yaml")],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert util == expected


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

    records = list(continuous(jobs(), work, 8))
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
    for bad in (0, MAX_CONCURRENCY + 1):
        with pytest.raises(ValueError, match="Unsupported"):
            list(continuous(jobs(), work, bad))


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
