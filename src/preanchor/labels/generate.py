"""Stage 5 completion generation, benchmarking, and resumable storage.

The module keeps generation backend-neutral: tests use ``FakeBackend`` while
the production path talks to the pinned localhost vLLM OpenAI-compatible API.
Each completion is an immutable work unit and is never rewritten on resume.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from preanchor.ids import sample_seed
from preanchor.io import UnitStore, read_json, write_json
from preanchor.labels.importance import normalize_answer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def build_prompt(trajectory: dict[str, Any], target_sentence_idx: int, fork_idx: int) -> str:
    """Use the exact stored Thought Anchors prompt and prefix; no chat template."""
    if fork_idx < 0 or target_sentence_idx < 0:
        raise ValueError("indices must be non-negative")
    if fork_idx == 0:
        return trajectory["prompt"]
    # sentence_boundaries contains exact character starts for each stored
    # sentence. The target mapping, rather than a reconstructed sentence, is
    # the authority for each fork boundary.
    marker = trajectory.get("sentence_boundaries", {}).get(str(fork_idx))
    if marker is None:
        raise ValueError(f"missing exact boundary for fork {fork_idx}")
    return trajectory["full_cot"][: int(marker)]


def completion_seed(seed_root: int, trajectory_id: str, fork_idx: int, sample_idx: int) -> int:
    return sample_seed(seed_root, trajectory_id, fork_idx, sample_idx)


def normalize_response(text: str | None) -> str:
    return normalize_answer(text)


def parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("backend response has no choices")
    choice = choices[0]
    message = choice.get("message", {})
    text = choice.get("text") or message.get("content") or ""
    usage = payload.get("usage") or {}
    return {
        "raw_answer": text,
        "normalized_answer": normalize_response(text),
        "finish_reason": choice.get("finish_reason"),
        "generated_token_count": int(usage.get("completion_tokens", 0)),
        "prompt_token_count": int(usage.get("prompt_tokens", 0)),
    }


class Backend(Protocol):
    name: str
    image_digest: str | None

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float, top_p: float) -> dict[str, Any]: ...


class VLLMBackend:
    name = "vllm-openai"

    def __init__(self, endpoint: str = "http://127.0.0.1:8000", model: str = MODEL_ID,
                 image_digest: str | None = None):
        self.endpoint, self.model, self.image_digest = endpoint.rstrip("/"), model, image_digest

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float, top_p: float) -> dict[str, Any]:
        started = time.perf_counter()
        response = httpx.post(f"{self.endpoint}/v1/completions", json={
            "model": self.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
        }, timeout=None)
        response.raise_for_status()
        result = parse_response(response.json())
        result["runtime_seconds"] = time.perf_counter() - started
        result["backend"] = self.name
        result["image_digest"] = self.image_digest
        return result

    def tokenize(self, prompt: str) -> int:
        response = httpx.post(f"{self.endpoint}/tokenize", json={
            "model": self.model, "prompt": prompt,
        }, timeout=30)
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("vLLM tokenize response missing tokens")
        return len(tokens)


class FakeBackend:
    name = "fake"
    image_digest = "fake"

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = outputs or ["\\boxed{42}"]
        self.calls = 0

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float, top_p: float) -> dict[str, Any]:
        value = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        return {"raw_answer": value, "normalized_answer": normalize_response(value),
                "finish_reason": "stop", "generated_token_count": 1,
                "prompt_token_count": len(prompt.split()), "runtime_seconds": 0.0,
                "backend": self.name, "image_digest": self.image_digest}


def tokenizer_identity_check(*, host_count: int, backend_count: int,
                             tokenizer_id: str, tokenizer_revision: str,
                             prompt_sha256: str,
                             special_tokens: dict[str, Any] | None = None) -> dict:
    result = {"tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision,
              "prompt_sha256": prompt_sha256, "host_token_count": host_count,
              "backend_token_count": backend_count, "special_tokens": special_tokens or {},
              "equal": host_count == backend_count}
    if not result["equal"]:
        raise ValueError("host/vLLM tokenizer mismatch")
    return result


def adaptive_sample_count(dissimilar_count: int, *, base: int = 32,
                          maximum: int = 100, minimum_dissimilar: int = 10) -> int:
    """Extend only an affected fork when its observed support is insufficient."""
    if dissimilar_count >= minimum_dissimilar:
        return base
    return maximum


@dataclass(frozen=True)
class CompletionSpec:
    trajectory_id: str
    target_sentence_idx: int
    fork_idx: int
    condition: str
    sample_idx: int
    seed: int
    prompt: str
    config_hash: str
    input_hash: str
    associations: tuple[tuple[int, str], ...] = ()

    @property
    def rollout_id(self) -> str:
        # A completion is a physical sample from one canonical fork.  Target
        # associations are label metadata and must not create a second sample
        # when adjacent targets share that fork.
        value = f"{self.trajectory_id}:{self.fork_idx}:{self.sample_idx}:" \
                f"{self.seed}:{prompt_hash(self.prompt)}:{self.config_hash}:{self.input_hash}"
        return hashlib.sha256(value.encode()).hexdigest()


class CompletionStore:
    def __init__(self, root: Path):
        self.units = UnitStore(root)

    def run_one(self, spec: CompletionSpec, backend: Backend, *, max_tokens: int,
                temperature: float, top_p: float) -> tuple[dict, bool]:
        key = spec.rollout_id
        def writer(directory: Path) -> None:
            result = backend.complete(spec.prompt, seed=spec.seed, max_tokens=max_tokens,
                                      temperature=temperature, top_p=top_p)
            row = {"schema_version": 1, "rollout_id": key,
                   "trajectory_id": spec.trajectory_id,
                   "target_sentence_idx": spec.target_sentence_idx,
                   "fork_idx": spec.fork_idx, "condition": spec.condition,
                   "target_associations": [
                       {"target_sentence_idx": target, "condition": condition}
                       for target, condition in spec.associations
                   ],
                   "sample_idx": spec.sample_idx, "seed": spec.seed,
                   "prompt_sha256": prompt_hash(spec.prompt),
                   "config_hash": spec.config_hash, "input_hash": spec.input_hash,
                   "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                   "status": "complete",
                   "first_replacement_sentence": result.get("first_replacement_sentence"),
                   "similarity": result.get("similarity"),
                   "is_dissimilar": result.get("is_dissimilar"),
                   "correctness": result.get("correctness"), **result}
            write_json(directory / "completion.json", row)
        directory, created = self.units.produce(key, writer)
        return read_json(directory / "completion.json"), created


def required_fork_associations(targets: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Return target/condition associations grouped by canonical fork."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        target_idx = int(target["sentence_idx"])
        for condition, key in (("without", "without_fork_idx"), ("with", "with_fork_idx")):
            fork_idx = int(target[key])
            grouped.setdefault(fork_idx, []).append({
                "target_sentence_idx": target_idx, "condition": condition,
            })
    return {fork: grouped[fork] for fork in sorted(grouped)}


def unique_required_forks(targets: list[dict[str, Any]]) -> list[int]:
    """Canonical generation set, independent of target count or associations."""
    return sorted(required_fork_associations(targets))


def adaptive_eligible_forks(targets: list[dict[str, Any]]) -> list[int]:
    """Forks whose *without* support is tested by the frozen protocol."""
    return sorted({int(target["without_fork_idx"]) for target in targets})


def canonical_prompt_identities(
        trajectory: dict[str, Any], targets: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Resolve and verify one prompt identity for every shared canonical fork."""
    identities: dict[int, dict[str, Any]] = {}
    for fork_idx, associations in required_fork_associations(targets).items():
        observed = []
        for association in associations:
            prompt = build_prompt(trajectory, association["target_sentence_idx"], fork_idx)
            observed.append((prompt_hash(prompt), prompt, len(prompt)))
        hashes = {item[0] for item in observed}
        if len(hashes) != 1:
            raise ValueError(f"shared fork {fork_idx} resolves to multiple prompt identities")
        digest, prompt, byte_length = observed[0]
        identities[fork_idx] = {"prompt_sha256": digest, "prompt": prompt,
                                "prompt_bytes": byte_length,
                                "associations": associations}
    return identities


def production_specs(trajectory: dict[str, Any], targets: list[dict[str, Any]],
                     seed_root: int, config_hash: str, input_hash: str,
                     *, samples_per_fork: int = 32) -> list[CompletionSpec]:
    """Emit one physical job per canonical fork/sample, with associations attached."""
    specs: list[CompletionSpec] = []
    grouped = required_fork_associations(targets)
    identities = canonical_prompt_identities(trajectory, targets)
    for fork_idx in unique_required_forks(targets):
        associations = grouped[fork_idx]
        # The first association is deterministic metadata; label assembly can
        # join the same rollout to every association in this list.
        association = associations[0]
        prompt = identities[fork_idx]["prompt"]
        association_values = tuple((item["target_sentence_idx"], item["condition"])
                                   for item in associations)
        for sample_idx in range(samples_per_fork):
            seed = completion_seed(seed_root, trajectory["trajectory_id"], fork_idx, sample_idx)
            specs.append(CompletionSpec(trajectory["trajectory_id"],
                                        association["target_sentence_idx"], fork_idx,
                                        association["condition"], sample_idx, seed,
                                        prompt, config_hash, input_hash, association_values))
    return specs


def benchmark_specs(trajectory: dict[str, Any], target_indices: list[int], seed_root: int,
                    config_hash: str, input_hash: str, count: int) -> list[CompletionSpec]:
    specs = []
    for i in range(count):
        target = target_indices[i % len(target_indices)]
        fork = target
        prompt = build_prompt(trajectory, target, fork)
        seed = completion_seed(seed_root, trajectory["trajectory_id"], fork, i)
        specs.append(CompletionSpec(trajectory["trajectory_id"], target, fork, "without", i,
                                    seed, prompt, config_hash, input_hash))
    return specs


def benchmark(backend: Backend, store: CompletionStore, trajectory: dict[str, Any],
              target_indices: list[int], *, count: int, seed_root: int,
              config_hash: str, input_hash: str, max_tokens: int = 16384,
              temperature: float = 0.6, top_p: float = 0.95) -> list[dict]:
    if count < 1:
        raise ValueError("benchmark count must be positive")
    rows = []
    specs = benchmark_specs(trajectory, target_indices, seed_root, config_hash, input_hash, count)
    for spec in specs:
        row, _ = store.run_one(spec, backend, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)
        rows.append(row)
    return rows
