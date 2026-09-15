"""Stage 5 controls: path-independent identity, backend failures, hangs, over-long prompts,
explicit special tokens, prompt-token assertions, checkpoint idempotence, measured projections.
No GPU or network: HTTP is mocked at the httpx boundary."""

import json
from pathlib import Path

import httpx
import pyarrow as pa
import pytest

from preanchor.config import load_config
from preanchor.io import UnitStore, write_json, write_parquet
from preanchor.labels import pilot
from preanchor.labels.batching import StreamingVLLMBackend
from preanchor.labels.generate import (
    ADD_SPECIAL_TOKENS,
    CompletionSpec,
    CompletionStore,
    FakeBackend,
    VLLMBackend,
    request_timeout,
)
from preanchor.labels.pilot import (
    benchmark_evidence,
    checkpoint,
    guard_reason,
    hourly_rate,
    inputs,
    measured_projection,
)
from preanchor.manifest import create_run, load_manifest, save_manifest

ROOT = Path(__file__).resolve().parents[2]


def spec(prompt="one two three"):
    return CompletionSpec("4682:correct", 13, 13, "without", 0, 123, prompt, "config", "input",
                          ((13, "without"),))


# ---------------------------------------------------------------------------
# Rollout identity must not depend on where the data or checkout lives
# ---------------------------------------------------------------------------
def write_stage3(root: Path) -> None:
    base = root / "data/processed/stage3"
    base.mkdir(parents=True)
    trace = "P <think>\nS0. S1. S2."
    import hashlib

    trajectories = [{"trajectory_id": "4682:correct", "problem_id": 4682,
                     "trace_kind": "correct", "prompt": "P <think>\n", "full_cot": trace,
                     "full_cot_sha256": hashlib.sha256(trace.encode()).hexdigest()}]
    sentences = []
    for i, text in enumerate(["S0.", "S1.", "S2."]):
        start = trace.index(text)
        sentences.append({"trajectory_id": "4682:correct", "sentence_idx": i, "text": text,
                          "char_start": start, "char_end": start + len(text)})
    write_parquet(base / "trajectories.parquet", pa.Table.from_pylist(trajectories))
    write_parquet(base / "sentences.parquet", pa.Table.from_pylist(sentences))


def test_input_hashes_are_location_independent(tmp_path):
    first, second = tmp_path / "root_a", tmp_path / "nested/root_b"
    write_stage3(first)
    write_stage3(second)
    _, _, _, hashes_a = inputs(first, ROOT)
    _, _, _, hashes_b = inputs(second, ROOT)
    assert hashes_a == hashes_b
    assert set(hashes_a) == {"stage3/trajectories.parquet", "stage3/sentences.parquet",
                             "configs/pilot_targets_4682.yaml"}
    assert not any(key.startswith("/") for key in hashes_a)


# ---------------------------------------------------------------------------
# Backend request contract and failure handling
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://127.0.0.1:8000/v1/completions")
            raise httpx.HTTPStatusError("error", request=request,
                                        response=httpx.Response(self.status_code,
                                                                request=request))

    def json(self):
        return self._payload


def completion_payload(prompt_tokens=3, text="A valid replacement sentence. Next."):
    return {"choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7}}


def test_requests_send_explicit_special_token_flag_and_bounded_timeout(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured[url.rsplit("/", 1)[1]] = (json, timeout)
        if url.endswith("/tokenize"):
            return FakeResponse(payload={"tokens": [1, 2, 3], "count": 3})
        return FakeResponse(payload=completion_payload())

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = VLLMBackend()
    backend.metrics = lambda: {}
    assert backend.tokenize_ids("one two three") == [1, 2, 3]
    assert captured["tokenize"][0]["add_special_tokens"] is False
    backend.complete("one two three", seed=1, max_tokens=16384, temperature=.6, top_p=.95)
    body, timeout = captured["completions"]
    assert body["add_special_tokens"] is False and ADD_SPECIAL_TOKENS is False
    assert body["seed"] == 1 and body["max_tokens"] == 16384
    assert timeout.read == request_timeout(16384) == 4096.0
    assert timeout.connect == 30
    assert request_timeout(100) == 600.0


def test_prompt_token_mismatch_fails_closed_without_accepting_a_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "post", lambda url, *, json, timeout: FakeResponse(
        payload=completion_payload(prompt_tokens=4)))
    backend = VLLMBackend()
    backend.metrics = lambda: {}
    store = CompletionStore(tmp_path / "units")
    with pytest.raises(ValueError, match="prompt token count 4 != host count 3"):
        store.raw_one(spec(), backend, max_tokens=16, temperature=.6, top_p=.95,
                      expected_prompt_tokens=3)
    with pytest.raises(ValueError):
        store.raw_units.validate(spec().rollout_id)
    # Matching count is accepted once.
    monkeypatch.setattr(httpx, "post", lambda url, *, json, timeout: FakeResponse(
        payload=completion_payload(prompt_tokens=3)))
    row, created = store.raw_one(spec(), backend, max_tokens=16, temperature=.6, top_p=.95,
                                 expected_prompt_tokens=3)
    assert created and row["prompt_token_count"] == 3


def test_fake_backend_honours_expected_prompt_tokens(tmp_path):
    store = CompletionStore(tmp_path / "units")
    with pytest.raises(ValueError, match="host count 99"):
        store.run_one(spec(), FakeBackend(), max_tokens=1, temperature=.6, top_p=.95,
                      expected_prompt_tokens=99)
    row, _ = store.run_one(spec(), FakeBackend(), max_tokens=1, temperature=.6, top_p=.95,
                           expected_prompt_tokens=3)
    assert row["prompt_token_count"] == 3


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("hung request"),
    httpx.ConnectError("server down"),
])
def test_hanging_or_failed_requests_raise_and_leave_no_unit(monkeypatch, tmp_path, failure):
    def fake_post(url, *, json, timeout):
        raise failure

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = VLLMBackend()
    backend.metrics = lambda: {}
    store = CompletionStore(tmp_path / "units")
    with pytest.raises(type(failure)):
        store.raw_one(spec(), backend, max_tokens=16, temperature=.6, top_p=.95)
    assert not (store.raw_units.root / spec().rollout_id / "SUCCESS.json").exists()
    assert not list(store.raw_units.root.glob("*/SUCCESS.json"))


def test_over_long_prompt_400_is_an_error_not_a_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "post", lambda url, *, json, timeout: FakeResponse(status=400))
    backend = VLLMBackend()
    backend.metrics = lambda: {}
    store = CompletionStore(tmp_path / "units")
    with pytest.raises(httpx.HTTPStatusError):
        store.raw_one(spec("x" * 50000), backend, max_tokens=16384, temperature=.6, top_p=.95)
    assert not list(store.raw_units.root.glob("*/SUCCESS.json"))


def test_streaming_backend_uses_bounded_timeout_and_flag(monkeypatch):
    captured = {}

    class Stream:
        def __init__(self, method, url, *, json, timeout):
            captured["json"], captured["timeout"] = json, timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"id": "r1", "choices": [{"index": 0, "text": "Hello there. Next."}]}'
            yield ('data: {"id": "r1", "choices": [{"index": 0, "text": "", '
                   '"finish_reason": "stop"}], "usage": {"prompt_tokens": 3, '
                   '"completion_tokens": 4}}')
            yield "data: [DONE]"

    monkeypatch.setattr(httpx, "stream", Stream)
    backend = StreamingVLLMBackend()
    result = backend.complete("one two three", seed=1, max_tokens=16384, temperature=.6,
                              top_p=.95, expected_prompt_tokens=3)
    assert captured["json"]["add_special_tokens"] is False
    assert captured["timeout"].read == request_timeout(16384)
    assert result["prompt_token_count"] == 3 and result["finish_reason"] == "stop"
    with pytest.raises(ValueError, match="host count 2"):
        backend.complete("one two three", seed=1, max_tokens=16384, temperature=.6,
                         top_p=.95, expected_prompt_tokens=2)


# ---------------------------------------------------------------------------
# Guards, rate and projections
# ---------------------------------------------------------------------------
def test_hourly_rate_comes_from_environment(monkeypatch):
    monkeypatch.delenv("PREANCHOR_HOURLY_RATE", raising=False)
    with pytest.raises(ValueError, match="PREANCHOR_HOURLY_RATE"):
        hourly_rate()
    monkeypatch.setenv("PREANCHOR_HOURLY_RATE", "1.09")
    assert hourly_rate() == 1.09
    monkeypatch.setenv("PREANCHOR_HOURLY_RATE", "-1")
    with pytest.raises(ValueError):
        hourly_rate()


def test_guards_use_wall_clock_and_minimum_truncation_sample():
    stop = {"finish_reason": "stop", "runtime_seconds": 1.0}
    length = {**stop, "finish_reason": "length"}
    # One early truncation among few rows must not abort the run.
    assert guard_reason([length] * 2 + [stop] * 5, rate=1.0) is None
    assert guard_reason([length] * 2 + [stop] * 18,
                        rate=1.0) == "LENGTH_TRUNCATION_ABOVE_5_PERCENT"  # 10% of 20
    assert guard_reason([length] + [stop] * 19, rate=1.0) is None  # exactly 5% is not above
    assert guard_reason([length] + [stop] * 39, rate=1.0) is None  # 2.5%
    # Projection: wall-clock seconds per new unit times 256 against 10 h and $20.
    rows = [stop] * 32
    assert guard_reason(rows, session_seconds=32 * 100, session_completed=32, rate=1.0) is None
    assert guard_reason(rows, session_seconds=32 * 150, session_completed=32,
                        rate=1.0) == "BASE_PROJECTION_EXCEEDS_AUTHORIZATION"
    assert guard_reason(rows, session_seconds=32 * 100, session_completed=32,
                        rate=5.0) == "BASE_PROJECTION_EXCEEDS_AUTHORIZATION"
    assert guard_reason(rows, session_seconds=31 * 100, session_completed=31, rate=5.0) is None


def fake_provenance(monkeypatch):
    source = {"git_sha": "a" * 40, "dirty_diff_hash": "b" * 64, "lock_hash": "c" * 64,
              "package_freeze": [], "dirty": False}
    monkeypatch.setattr("preanchor.manifest.provenance", lambda repo: (source.copy(), b""))
    return source


def test_projection_is_measured_from_a_real_benchmark_run(tmp_path, monkeypatch):
    fake_provenance(monkeypatch)
    config = load_config(ROOT / "configs/pilot.yaml")
    root = tmp_path / "root"
    write_stage3(root)
    trajectory, targets, _, hashes = inputs(root, ROOT)
    bench = create_run(root, ROOT, config, ["bench"])
    manifest = load_manifest(bench)
    manifest["stages"]["stage5"] = {"state": "benchmark_complete", "namespace": "benchmark"}
    save_manifest(bench, manifest)
    stage = bench / "stages/stage5"
    specs = [spec("P <think>\nS0. ")]
    for i in range(8):
        unit = stage / "benchmark_units" / f"{i:064x}"
        write_json(unit / "completion.json", {"runtime_seconds": 100.0 + i,
                                              "generated_token_count": 4000,
                                              "finish_reason": "stop",
                                              "prompt_sha256": pilot.prompt_hash(specs[0].prompt)})
    write_json(stage / "resource_projection.json", {"elapsed_seconds": 880.0,
                                                    "peak_vram_mib": 34875,
                                                    "peak_ram_bytes": 10**9})
    evidence = benchmark_evidence(bench, config, specs)
    assert evidence["benchmark_units"] == 8 and evidence["measured_seconds_per_unit"] == 110.0
    assert evidence["benchmark_commit"] == "a" * 40 and evidence["peak_vram_mib"] == 34875
    projection = measured_projection(evidence, "f" * 40, config, hashes, 1.09)
    assert projection["projected_seconds"] == 110.0 * 256
    assert projection["projected_cost_usd"] == pytest.approx(110.0 * 256 / 3600 * 1.09)
    assert projection["empirical_run"] == bench.name and projection["empirical_commit"] == "a" * 40
    assert projection["code_commit"] == "f" * 40 and projection["price_usd_per_hour"] == 1.09
    # A benchmark with a different config or foreign prompts is refused.
    manifest["identity"]["resolved_config_hash"] = "other"
    save_manifest(bench, manifest)
    with pytest.raises(ValueError, match="config differs"):
        benchmark_evidence(bench, config, specs)


def test_projection_requires_measured_vram(tmp_path, monkeypatch):
    fake_provenance(monkeypatch)
    config = load_config(ROOT / "configs/pilot.yaml")
    bench = create_run(tmp_path, ROOT, config, ["bench"])
    manifest = load_manifest(bench)
    manifest["stages"]["stage5"] = {"state": "benchmark_complete", "namespace": "benchmark"}
    save_manifest(bench, manifest)
    for i in range(8):
        write_json(bench / "stages/stage5/benchmark_units" / f"{i:064x}" / "completion.json",
                   {"runtime_seconds": 1.0, "generated_token_count": 1, "finish_reason": "stop"})
    with pytest.raises(ValueError, match="peak VRAM"):
        benchmark_evidence(bench, config, [spec()])


# ---------------------------------------------------------------------------
# Checkpoint idempotence
# ---------------------------------------------------------------------------
def test_checkpoint_is_idempotent_and_does_not_duplicate_unchanged_telemetry(tmp_path, monkeypatch):
    fake_provenance(monkeypatch)
    config = load_config(ROOT / "configs/pilot.yaml")
    root = tmp_path / "root"
    run = create_run(root, ROOT, config, ["run"])
    stage = run / "stages/stage5"
    store = CompletionStore(stage / "units")
    row, _ = store.run_one(spec(), FakeBackend(), max_tokens=1, temperature=.6, top_p=.95)
    (run / "logs/wandb").mkdir(parents=True)
    (run / "logs/wandb/run.wandb").write_bytes(b"telemetry-v1")
    first = checkpoint(run, [row])
    destination = root / "artifacts/backups" / ("stage5-live-" + run.name)
    snapshots = lambda: sorted(p for p in (destination / "wandb").iterdir() if p.is_dir())  # noqa: E731
    files_after_first = {p for p in destination.rglob("*") if p.is_file()}
    second = checkpoint(run, [row])
    files_after_second = {p for p in destination.rglob("*") if p.is_file()}
    assert first["files"] == second["files"]
    assert len(snapshots()) == 1  # unchanged telemetry is referenced, not copied again
    assert files_after_second - files_after_first == {Path(second["path"]),
                                                      Path(second["path"] + ".sha256")}
    (run / "logs/wandb/run.wandb").write_bytes(b"telemetry-v2")
    third = checkpoint(run, [row])
    assert len(snapshots()) == 2 and third["files"] == first["files"]
    UnitStore(store.units.root).validate(row["rollout_id"])
