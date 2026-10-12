"""Minimal durable client for the pinned vLLM OpenAI-compatible server.

Independent reimplementation (no MVP import). Every request/response pair is stored once
under ``store/<sha256(request)>.json``; a stored response is reused, never regenerated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"


def sample_seed(root: int, *parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, (root, *parts))).encode()).hexdigest()[:8], 16)


class Client:
    def __init__(self, endpoint: str, store: Path):
        self.endpoint, self.store = endpoint.rstrip("/"), store
        store.mkdir(parents=True, exist_ok=True)

    def complete(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float = 1.0,
                 seed: int | None = None) -> dict:
        req = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p, "add_special_tokens": False}
        if seed is not None:
            req["seed"] = seed
        key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
        path = self.store / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        r = requests.post(f"{self.endpoint}/v1/completions", json=req, timeout=600)
        r.raise_for_status()
        body = r.json()
        choice = body["choices"][0]
        resp = {"text": choice.get("text", ""), "finish_reason": choice.get("finish_reason"),
                "usage": body.get("usage", {})}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"request": req, "response": resp}))
        os.replace(tmp, path)
        return resp
