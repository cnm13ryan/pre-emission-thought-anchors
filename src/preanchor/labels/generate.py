"""Stage 5 completion generation, benchmarking, and resumable storage.

The module keeps generation backend-neutral: tests use ``FakeBackend`` while
the production path talks to the pinned localhost vLLM OpenAI-compatible API.
Each completion is an immutable work unit and is never rewritten on resume.
"""
from __future__ import annotations

import hashlib
import math
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
# Stored Thought Anchors prompts already carry every token the trace had; the server
# must never prepend BOS/EOS. Sent explicitly on every request, never left to defaults.
ADD_SPECIAL_TOKENS = False


def request_timeout(max_tokens: int, *, floor_tokens_per_second: float = 4.0,
                    minimum_seconds: float = 600.0) -> float:
    """Read timeout that only a hung request can exceed at a pessimistic decode rate."""
    if max_tokens < 1 or floor_tokens_per_second <= 0:
        raise ValueError("max_tokens and the floor rate must be positive")
    return max(minimum_seconds, max_tokens / floor_tokens_per_second)


def check_prompt_tokens(result: dict[str, Any], expected: int | None) -> None:
    """Fail closed when the server's prompt token count differs from the host count."""
    if expected is None:
        return
    actual = result.get("prompt_token_count")
    if actual != expected:
        raise ValueError(f"Backend prompt token count {actual} != host count {expected}")


@dataclass(frozen=True)
class ExtractionResult:
    """Deterministic first-chunk extraction from a raw continuation."""

    text: str | None
    success: bool
    failure_reason: str | None = None


def extract_first_replacement_sentence(raw_completion: str | None) -> ExtractionResult:
    """Apply the pinned Thought Anchors ``split_solution_into_chunks`` rules."""
    if not isinstance(raw_completion, str) or not raw_completion.strip():
        return ExtractionResult(None, False, "empty_output")
    text = raw_completion
    if "<think>" in text:
        text = text.split("<think>", 1)[1].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[0].strip()
    if not text:
        return ExtractionResult(None, False, "empty_after_think_delimiters")
    chunks: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(text):
        current.append(text[i])
        paragraph = text.startswith("\n\n", i) or text.startswith("\r\n\r\n", i)
        sentence_end = text[i] in ".!?" and i + 1 < len(text) and text[i + 1] in " \n"
        if paragraph or sentence_end:
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
        i += 1
    # Preserve the pinned upstream behavior: an unterminated final fragment is
    # not promoted to a sentence.
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1
    if not chunks:
        return ExtractionResult(None, False, "no_terminated_sentence")
    return ExtractionResult(chunks[0], True)


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


def is_dissimilar(similarity: float, threshold: float = 0.8) -> bool:
    """Apply the frozen strict cosine-similarity decision rule."""
    return float(similarity) < float(threshold)


def parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("backend response has no choices")
    choice = choices[0]
    message = choice.get("message", {})
    text = choice.get("text") or message.get("content") or ""
    extraction = extract_first_replacement_sentence(text)
    backend_field = (choice.get("first_replacement_sentence")
                     or message.get("first_replacement_sentence")
                     or payload.get("first_replacement_sentence"))
    usage = payload.get("usage") or {}
    return {
        "raw_answer": text,
        "first_replacement_sentence": extraction.text,
        "extraction_status": "success" if extraction.success else "failed",
        "extraction_failure_reason": extraction.failure_reason,
        "backend_first_replacement_sentence": backend_field,
        "backend_extraction_matches": (
            None if backend_field is None else backend_field == extraction.text
        ),
        "normalized_answer": normalize_response(text),
        "finish_reason": choice.get("finish_reason"),
        "generated_token_count": int(usage.get("completion_tokens", 0)),
        "prompt_token_count": int(usage.get("prompt_tokens", 0)),
    }


class Backend(Protocol):
    name: str
    image_digest: str | None

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                temperature: float, top_p: float,
                expected_prompt_tokens: int | None = None) -> dict[str, Any]: ...


class SimilarityScorer(Protocol):
    def score(self, reference_text: str, replacement_text: str) -> float: ...


class MiniLMSimilarityScorer:
    """Pinned MiniLM scorer used after canonical extraction."""

    def __init__(self, *, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            device=device,
        )

    def score(self, reference_text: str, replacement_text: str) -> float:
        embeddings = self.model.encode(
            [reference_text, replacement_text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return float(embeddings[0] @ embeddings[1])


class VLLMBackend:
    name = "vllm-openai"

    def __init__(self, endpoint: str = "http://127.0.0.1:8000", model: str = MODEL_ID,
                 image_digest: str | None = None, read_timeout_seconds: float | None = None):
        self.endpoint, self.model, self.image_digest = endpoint.rstrip("/"), model, image_digest
        self.read_timeout_seconds = read_timeout_seconds

    def timeout(self, max_tokens: int) -> httpx.Timeout:
        read = self.read_timeout_seconds or request_timeout(max_tokens)
        return httpx.Timeout(connect=30, read=read, write=30, pool=30)

    def request(self, prompt: str, *, seed: int, max_tokens: int, temperature: float,
                top_p: float) -> dict[str, Any]:
        return {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p, "seed": seed,
                "add_special_tokens": ADD_SPECIAL_TOKENS}

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float, top_p: float,
                 expected_prompt_tokens: int | None = None) -> dict[str, Any]:
        before = self.metrics()
        started = time.perf_counter()
        response = httpx.post(f"{self.endpoint}/v1/completions",
                              json=self.request(prompt, seed=seed, max_tokens=max_tokens,
                                                temperature=temperature, top_p=top_p),
                              timeout=self.timeout(max_tokens))
        response.raise_for_status()
        result = parse_response(response.json())
        check_prompt_tokens(result, expected_prompt_tokens)
        result["runtime_seconds"] = time.perf_counter() - started
        result["backend"] = self.name
        result["image_digest"] = self.image_digest
        after = self.metrics()
        delta = {k: after[k] - before[k] for k in before.keys() & after.keys()}
        result["backend_metric_deltas"] = delta
        result["decode_seconds"] = delta.get("vllm:request_decode_time_seconds_sum")
        result["prefill_seconds"] = delta.get("vllm:request_prefill_time_seconds_sum")
        result["ttft_seconds"] = delta.get("vllm:time_to_first_token_seconds_sum")
        return result

    def metrics(self) -> dict[str, float]:
        try:
            response = httpx.get(f"{self.endpoint}/metrics", timeout=10)
            response.raise_for_status()
            result: dict[str, float] = {}
            for line in response.text.splitlines():
                if line.startswith("vllm:"):
                    name, value = line.rsplit(" ", 1)
                    name = name.split("{")[0]
                    if name.endswith(("_sum", "_total")):
                        result[name] = result.get(name, 0) + float(value)
            return result
        except (httpx.HTTPError, ValueError):
            return {}  # Missing telemetry is explicit, never fabricated timing.

    def tokenize(self, prompt: str) -> int:
        return len(self.tokenize_ids(prompt))

    def tokenize_ids(self, prompt: str) -> list[int]:
        response = httpx.post(f"{self.endpoint}/tokenize", json={
            "model": self.model, "prompt": prompt, "add_special_tokens": ADD_SPECIAL_TOKENS,
        }, timeout=30)
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("vLLM tokenize response missing tokens")
        return tokens


class FakeBackend:
    name = "fake"
    image_digest = "fake"

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = outputs or ["\\boxed{42}"]
        self.calls = 0

    def complete(self, prompt: str, *, seed: int, max_tokens: int,
                 temperature: float, top_p: float,
                 expected_prompt_tokens: int | None = None) -> dict[str, Any]:
        value = self.outputs[self.calls % len(self.outputs)]
        self.calls += 1
        result = {"raw_answer": value, "normalized_answer": normalize_response(value),
                  "finish_reason": "stop", "generated_token_count": 1,
                  "prompt_token_count": len(prompt.split()), "runtime_seconds": 0.0,
                  "backend": self.name, "image_digest": self.image_digest}
        check_prompt_tokens(result, expected_prompt_tokens)
        return result


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


def support_counts(rows: list[dict[str, Any]], threshold: float = 0.8,
                   *, target_sentence_idx: int | None = None,
                   condition: str | None = None) -> dict[str, int]:
    """Count valid samples, optionally for one target association."""
    extracted = 0
    scored = 0
    dissimilar = 0
    extraction_failed = 0
    similarity_failed = 0
    for row in rows:
        if row.get("extraction_status") != "success" or not row.get(
                "first_replacement_sentence"):
            extraction_failed += 1
            continue
        extracted += 1
        similarity = row.get("similarity")
        if target_sentence_idx is not None or condition is not None:
            matches = [x for x in row.get("association_similarity", [])
                       if (target_sentence_idx is None
                           or x.get("target_sentence_idx") == target_sentence_idx)
                       and (condition is None or x.get("condition") == condition)]
            if not matches:
                similarity_failed += 1
                continue
            similarity = matches[0].get("similarity")
        if (not isinstance(similarity, (int, float)) or isinstance(similarity, bool)
                or not math.isfinite(similarity)):
            similarity_failed += 1
            continue
        scored += 1
        dissimilar += int(is_dissimilar(float(similarity), threshold))
    return {"total_generated": len(rows), "valid_extracted": extracted,
            "valid_similarity": scored, "dissimilar_valid": dissimilar,
            "failed_extraction": extraction_failed,
            "failed_similarity": similarity_failed}


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
        # Preserve expensive model output independently of derived scoring.
        self.raw_units = UnitStore(root.parent / (root.name + "_raw"))

    def raw_one(self, spec: CompletionSpec, backend: Backend, *, max_tokens: int,
                temperature: float, top_p: float,
                expected_prompt_tokens: int | None = None) -> tuple[dict, bool]:
        """Durable generation without loading or sharing a similarity model."""
        request = {"rollout_id": spec.rollout_id, "seed": spec.seed,
                   "prompt_sha256": prompt_hash(spec.prompt), "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": top_p}
        def writer(directory):
            result = backend.complete(spec.prompt, seed=spec.seed, max_tokens=max_tokens,
                                      temperature=temperature, top_p=top_p,
                                      expected_prompt_tokens=expected_prompt_tokens)
            write_json(directory / "response.json", result)
            write_json(directory / "request.json", request)
        directory, created = self.raw_units.produce(spec.rollout_id, writer)
        if read_json(directory / "request.json") != request:
            raise ValueError("Stored request identity mismatch")
        result = read_json(directory / "response.json")
        return {**result, **request, "fork_idx": spec.fork_idx,
                "sample_idx": spec.sample_idx}, created

    def run_one(self, spec: CompletionSpec, backend: Backend, *, max_tokens: int,
                temperature: float, top_p: float,
                similarity_scorer: SimilarityScorer | None = None,
                reference_texts: dict[int, str] | None = None,
                expected_prompt_tokens: int | None = None) -> tuple[dict, bool]:
        key = spec.rollout_id
        def writer(directory: Path) -> None:
            def raw_writer(raw_directory: Path) -> None:
                result = backend.complete(spec.prompt, seed=spec.seed, max_tokens=max_tokens,
                                          temperature=temperature, top_p=top_p,
                                          expected_prompt_tokens=expected_prompt_tokens)
                write_json(raw_directory / "response.json", result)
            raw_directory, _ = self.raw_units.produce(key, raw_writer)
            result = read_json(raw_directory / "response.json")
            extraction = extract_first_replacement_sentence(result.get("raw_answer"))
            association_scores: list[dict[str, Any]] = []
            if extraction.success and similarity_scorer is not None and reference_texts is not None:
                for target_idx, condition in spec.associations or (
                        (spec.target_sentence_idx, spec.condition),):
                    reference = reference_texts.get(target_idx)
                    if reference is None:
                        raise ValueError(f"missing reference sentence {target_idx}")
                    score = similarity_scorer.score(reference, extraction.text or "")
                    if not math.isfinite(score):
                        raise ValueError("Nonfinite similarity; raw generation is retained")
                    association_scores.append({
                        "target_sentence_idx": target_idx, "condition": condition,
                        "similarity": score, "is_dissimilar": is_dissimilar(score),
                        "reference_sha256": prompt_hash(reference),
                    })
            primary_score = association_scores[0]["similarity"] if association_scores else None
            row = {**result, "schema_version": 1, "rollout_id": key,
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
                   "output_sha256": prompt_hash(result["raw_answer"]),
                   "generation_status": "success",
                   "similarity_status": "success" if association_scores else "not_scored",
                   "status": "complete",
                   "first_replacement_sentence": extraction.text,
                   "association_similarity": association_scores,
                   "similarity": primary_score,
                   "is_dissimilar": (
                       None if primary_score is None else is_dissimilar(primary_score)
                   ),
                   "extraction_status": "success" if extraction.success else "failed",
                   "extraction_failure_reason": extraction.failure_reason,
                   "scientific_sample_valid": bool(
                       extraction.success and bool(association_scores)
                   ),
                   "backend_first_replacement_sentence": result.get(
                       "backend_first_replacement_sentence"
                   ),
                   "backend_extraction_matches": (
                       None if result.get("backend_first_replacement_sentence") is None
                       else result.get("backend_first_replacement_sentence") == extraction.text
                   )}
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


def generate_production(backend: Backend, store: CompletionStore,
                        trajectory: dict[str, Any], targets: list[dict[str, Any]], *,
                        seed_root: int, config_hash: str, input_hash: str,
                        samples_per_fork: int = 32, max_tokens: int = 16384,
                        temperature: float = 0.6, top_p: float = 0.95,
                        similarity_scorer: SimilarityScorer | None = None,
                        reference_texts: dict[int, str] | None = None) -> list[dict]:
    """Generate the base pilot once per canonical fork/sample.

    Adaptive extension is a second call with ``samples_per_fork=100`` after
    support has been assessed; the caller must schedule only sample indices
    32..99 for already-complete forks.
    """
    rows = []
    for spec in production_specs(trajectory, targets, seed_root, config_hash,
                                 input_hash, samples_per_fork=samples_per_fork):
        row, _ = store.run_one(spec, backend, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p,
                               similarity_scorer=similarity_scorer,
                               reference_texts=reference_texts)
        rows.append(row)
    return rows
