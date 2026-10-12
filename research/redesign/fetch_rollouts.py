"""Phase 1 (redesign): reacquire released raw forks and keep a compact per-rollout table.

Exploratory, CPU only, no generation. Every fork file in the Stage 4 raw run's retained
``source_manifest.json`` is downloaded at the pinned dataset revision into memory,
checked against its recorded size, Git blob SHA-1 and SHA-256, reduced to the fields
Phase 1 needs, and written as one Parquet per fork. Raw payloads are never written to
disk. Resumable: forks whose Parquet already exists are skipped.

Usage:
  PYTHONPATH=src python research/redesign/fetch_rollouts.py --out DIR [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_url
from tokenizers import Tokenizer

from preanchor.labels.importance import NO_ANSWER, normalize_answer

RAW_RUN = Path.home() / "preanchor-root/artifacts/runs/20260915T134433Z-d7d055d8b7-05a81618"
TOKENIZER = (Path.home() / "preanchor-root/cache/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-"
             "Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761/tokenizer.json")
TAIL_CHARS = 400


def fork_index(path: str) -> int:
    return int(Path(path).parent.name.removeprefix("chunk_"))


def verify(payload: bytes, record: dict) -> None:
    if len(payload) != record["size"]:
        raise ValueError("size mismatch")
    if record["lfs_sha256"]:
        if hashlib.sha256(payload).hexdigest() != record["lfs_sha256"]:
            raise ValueError("LFS sha256 mismatch")
    elif hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() \
            != record["git_blob_sha1"]:
        raise ValueError("git blob sha1 mismatch")
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("sha256 differs from the Stage 4 raw run record")


def rows_for(samples: list[dict], record: dict, tokenizer: Tokenizer) -> dict[str, list]:
    traj = f"{record['problem_id']}:{record['trace_kind']}"
    fork = fork_index(record["path"])
    cols: dict[str, list] = {k: [] for k in (
        "trajectory_id", "problem_id", "trace_kind", "fork_idx", "sample_idx", "has_error",
        "answer_raw", "answer_is_str", "category", "is_correct", "chunk_removed",
        "chunk_resampled", "rollout_chars", "boxed_count", "has_think_close",
        "rollout_tokens", "tail")}
    for i, s in enumerate(samples):
        rollout = s.get("rollout") if isinstance(s.get("rollout"), str) else ""
        answer = s.get("answer")
        category = normalize_answer(answer)
        missing = category == NO_ANSWER
        cols["trajectory_id"].append(traj)
        cols["problem_id"].append(record["problem_id"])
        cols["trace_kind"].append(record["trace_kind"])
        cols["fork_idx"].append(fork)
        cols["sample_idx"].append(i)
        cols["has_error"].append("error" in s)
        cols["answer_raw"].append(answer if isinstance(answer, str) else None)
        cols["answer_is_str"].append(isinstance(answer, str))
        cols["category"].append(category)
        cols["is_correct"].append(s.get("is_correct") is True)
        cols["chunk_removed"].append(s.get("chunk_removed") if isinstance(
            s.get("chunk_removed"), str) else None)
        cols["chunk_resampled"].append(s.get("chunk_resampled") if isinstance(
            s.get("chunk_resampled"), str) else None)
        cols["rollout_chars"].append(len(rollout))
        cols["boxed_count"].append(rollout.count("\\boxed{"))
        cols["has_think_close"].append("</think>" in rollout)
        # Token counts only where needed (missing answers) to keep the pass cheap.
        cols["rollout_tokens"].append(
            len(tokenizer.encode(rollout, add_special_tokens=False).ids) if missing else None)
        cols["tail"].append(rollout[-TAIL_CHARS:] if missing else None)
    return cols


def process(record: dict, out: Path, tokenizer: Tokenizer, session: requests.Session) -> str:
    traj = f"{record['problem_id']}_{record['trace_kind']}"
    target = out / traj / f"fork_{fork_index(record['path']):04d}.parquet"
    if target.exists():
        return "skipped"
    url = hf_hub_url("uzaymacar/math-rollouts", record["path"], repo_type="dataset",
                     revision="a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2")
    for attempt in range(5):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            payload = response.content
            verify(payload, record)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    value = json.loads(payload)
    if isinstance(value, dict):
        value = value.get("solutions", value.get("samples"))
    table = pa.table(rows_for(value, record, tokenizer))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, target)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads((RAW_RUN / "stages/stage4_raw/source_manifest.json").read_text())
    records = [r for r in manifest["files"] if r["role"] == "raw_resamples"][: args.limit]
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    session = requests.Session()
    failures, done, started = [], 0, time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process, r, args.out, tokenizer, session): r for r in records}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except Exception as exc:  # recorded, never silently dropped
                failures.append({"path": futures[future]["path"], "error": repr(exc)})
            if done % 200 == 0:
                print(f"{done}/{len(records)} {time.monotonic() - started:.0f}s "
                      f"failures={len(failures)}", flush=True)
    (args.out / "fetch_failures.json").write_text(json.dumps(failures, indent=1))
    print(json.dumps({"records": len(records), "failures": len(failures),
                      "seconds": time.monotonic() - started}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
