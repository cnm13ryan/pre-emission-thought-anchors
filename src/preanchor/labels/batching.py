"""Continuous independent requests and A6000 admission calculations."""

from __future__ import annotations

import concurrent.futures as futures
import json
import math
import time

import httpx

from preanchor.labels.generate import (
    ADD_SPECIAL_TOKENS,
    VLLMBackend,
    check_prompt_tokens,
    parse_response,
)

KV_BYTES_PER_TOKEN = 196608
MAX_CONTEXT = 17417


def required_capacity(concurrency):
    return math.ceil(concurrency * MAX_CONTEXT * 1.15)


def admitted_concurrency(capacity):
    for n in (3, 2, 1):
        if capacity >= required_capacity(n):
            return n
    raise ValueError("Insufficient KV capacity even for one sequence with margin")


def roofline(batch, context, parameters=14.77e9, bandwidth=768e9, dense_flops=154.85e12):
    seconds = batch * context * KV_BYTES_PER_TOKEN / bandwidth + max(
        2 * batch * parameters / dense_flops, 2 * parameters / bandwidth
    )
    return batch / seconds


def continuous(specs, work, concurrency, *, tick=lambda: None, stopping=lambda: False):
    """Bounded futures, immediately refilled; identity bound to each future.

    No mutable per-request backend/scorer state is shared by this scheduler.
    A stop or error drains in-flight durable writes and launches no new jobs.
    """
    specs = list(specs)
    if concurrency not in (1, 2, 3):
        raise ValueError("Unsupported concurrency")
    if len({s.rollout_id for s in specs}) != len(specs):
        raise ValueError("Duplicate rollout IDs")
    pending = iter(specs)
    active = {}
    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:

        def refill():
            while len(active) < concurrency and not stopping():
                spec = next(pending, None)
                if spec is None:
                    break
                active[pool.submit(work, spec)] = spec

        refill()
        while active:
            done, _ = futures.wait(active, timeout=0.25, return_when=futures.FIRST_COMPLETED)
            completed = []
            for task in done:
                spec = active.pop(task)
                row, created = task.result()
                if row["rollout_id"] != spec.rollout_id or row["seed"] != spec.seed:
                    raise ValueError("Completed identity/seed mismatch")
                completed.append((spec, row, created))
            refill()
            tick()
            yield from completed


class StreamingVLLMBackend(VLLMBackend):
    """Per-request timings from its own stream, never global counter deltas."""

    def complete(self, prompt, *, seed, max_tokens, temperature, top_p,
                 expected_prompt_tokens=None):
        started = time.perf_counter()
        first = last = None
        pieces, usage, finish, response_id = [], None, None, None
        request = dict(
            model=self.model,
            prompt=prompt,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        # A streaming read timeout bounds the silence between chunks, so a hung
        # server surfaces as an error instead of an ever-healthy heartbeat.
        with httpx.stream(
            "POST",
            self.endpoint + "/v1/completions",
            json=request,
            timeout=self.timeout(max_tokens),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                value = line[6:]
                if value == "[DONE]":
                    break
                payload = json.loads(value)
                if "error" in payload:
                    raise ValueError("vLLM stream error")
                rid = payload.get("id")
                if response_id is not None and rid is not None and rid != response_id:
                    raise ValueError("Response ID changed within stream")
                response_id = rid or response_id
                if payload.get("usage"):
                    usage = payload["usage"]
                for choice in payload.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Unexpected multiple completions")
                    text = choice.get("text", "")
                    if text:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        pieces.append(text)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        if usage is None or finish not in ("stop", "length"):
            raise ValueError("Incomplete stream: usage or finish reason missing")
        result = parse_response(
            {"choices": [{"text": "".join(pieces), "finish_reason": finish}], "usage": usage}
        )
        check_prompt_tokens(result, expected_prompt_tokens)
        elapsed = time.perf_counter() - started
        decode = last - first if first is not None and last is not None else None
        result.update(
            runtime_seconds=elapsed,
            ttft_seconds=(first - started if first else None),
            decode_seconds=decode,
            backend=self.name,
            image_digest=self.image_digest,
            response_id=response_id,
            timing_source="request_stream_client_clock",
            decode_tokens_per_second=(
                (result["generated_token_count"] - 1) / decode if decode else None
            ),
        )
        return result
