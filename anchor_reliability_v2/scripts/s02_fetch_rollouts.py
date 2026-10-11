"""Reacquire all 6,474 released fork files (pinned revision, hash-verified) and keep a rich
per-rollout table. Raw JSON is never written to disk; one Parquet per fork; resumable.

Resume is receipt-checked: each fork Parquet gets a ``.receipt.json`` naming the source
record's sha256, the pinned revision, the receipt version, and the Parquet's own sha256,
row count and columns. A fork is skipped only when its receipt validates. Anything else,
including forks written before receipts existed, is quarantined and reacquired.

Per rollout: stored upstream answer/correctness, v1 and v2 parses, every balanced box,
token count (pinned tokenizer), ``</think>`` presence, looping diagnostics, the first
replacement sentence (``chunk_resampled``) and a 600-char tail for non-answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.answers import balanced_boxes, category_v1, parse_v2  # noqa: E402
from arv2.paths import DERIVED, INPUTS  # noqa: E402

REVISION = "a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2"
URL = "https://huggingface.co/datasets/uzaymacar/math-rollouts/resolve/{rev}/{path}"
RECEIPT_VERSION = 1
COLUMNS = ("trajectory_id", "problem_id", "trace_kind", "fork_idx", "sample_idx", "has_error",
           "answer_stored", "is_correct_stored", "category_v1", "v2_status", "v2_answer",
           "boxes", "box_unterminated", "rollout_chars", "rollout_tokens", "has_think_close",
           "dup_sentence_frac", "max_repeat_run", "chunk_removed", "chunk_resampled", "tail")
TOKENIZER = Path(os.environ.get("ARV2_TOKENIZER", Path.home() / "preanchor-root/cache/hf/hub/"
                 "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/"
                 "1df8507178afcc1bef68cd8c393f61a886323761/tokenizer.json"))


def verify(payload: bytes, rec: dict) -> None:
    if len(payload) != rec["size"]:
        raise ValueError("size")
    if rec["lfs_sha256"] and hashlib.sha256(payload).hexdigest() != rec["lfs_sha256"]:
        raise ValueError("lfs sha256")
    if not rec["lfs_sha256"] and hashlib.sha1(
            b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() \
            != rec["git_blob_sha1"]:
        raise ValueError("git blob sha1")
    if hashlib.sha256(payload).hexdigest() != rec["sha256"]:
        raise ValueError("sha256 differs from the MVP raw record")


def loop_metrics(text: str) -> tuple[float, int]:
    """Duplicate-sentence fraction over the last 8,000 chars, and the longest run of one
    repeated sentence."""
    window = text[-8000:]
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", window) if len(p.strip()) > 20]
    if len(parts) < 4:
        return 0.0, 0
    dup = 1 - len(set(parts)) / len(parts)
    run = best = 1
    for a, b in zip(parts, parts[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return float(dup), int(best)


def receipt_path(target: Path) -> Path:
    return target.with_name(target.stem + ".receipt.json")


def receipt_ok(target: Path, rec: dict) -> bool:
    """True only if the fork Parquet matches a receipt tied to this source record."""
    rp = receipt_path(target)
    if not (target.is_file() and rp.is_file()):
        return False
    try:
        r = json.loads(rp.read_text())
        meta = pq.read_metadata(target)
        names = tuple(pq.read_schema(target).names)
    except (OSError, ValueError, pa.ArrowException):
        return False
    return (r.get("version") == RECEIPT_VERSION and r.get("revision") == REVISION
            and r.get("source_sha256") == rec["sha256"] and r.get("source_path") == rec["path"]
            and r.get("parquet_sha256") == hashlib.sha256(target.read_bytes()).hexdigest()
            and r.get("rows") == meta.num_rows and tuple(r.get("columns", ())) == names
            and names == COLUMNS)


def quarantine(target: Path) -> None:
    """Move an unverifiable fork (and any receipt) aside; nothing is deleted."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for path in (target, receipt_path(target)):
        if path.exists():
            os.replace(path, path.with_name(f"{path.name}.quarantined-{stamp}"))


def process(rec: dict, out: Path, tok: Tokenizer, session: requests.Session) -> str:
    fork = int(Path(rec["path"]).parent.name.removeprefix("chunk_"))
    target = out / f"{rec['problem_id']}_{rec['trace_kind']}" / f"fork_{fork:04d}.parquet"
    if receipt_ok(target, rec):
        return "skipped"
    if target.exists() or receipt_path(target).exists():
        quarantine(target)  # unverifiable on resume: reacquire from the pinned source
    for attempt in range(7):
        try:
            r = session.get(URL.format(rev=REVISION, path=rec["path"]), timeout=180)
            if r.status_code == 429:
                raise RuntimeError("429")
            r.raise_for_status()
            payload = r.content
            verify(payload, rec)
            break
        except Exception:
            if attempt == 6:
                raise
            time.sleep(min(120, 5 * 2 ** attempt))
    samples = json.loads(payload)
    if isinstance(samples, dict):
        samples = samples.get("solutions", samples.get("samples"))
    rollouts = [s.get("rollout") if isinstance(s.get("rollout"), str) else "" for s in samples]
    counts = [len(e.ids) for e in tok.encode_batch(rollouts, add_special_tokens=False)]
    cols: dict[str, list] = {k: [] for k in COLUMNS}
    for i, (s, text, n) in enumerate(zip(samples, rollouts, counts, strict=True)):
        parsed = parse_v2(text, token_count=n)
        boxes, unterminated = balanced_boxes(text)
        dup, run = loop_metrics(text)
        stored = s.get("answer")
        vals = {"trajectory_id": f"{rec['problem_id']}:{rec['trace_kind']}",
                "problem_id": rec["problem_id"], "trace_kind": rec["trace_kind"],
                "fork_idx": fork, "sample_idx": i, "has_error": "error" in s,
                "answer_stored": stored if isinstance(stored, str) else None,
                "is_correct_stored": s.get("is_correct") is True,
                "category_v1": category_v1(stored), "v2_status": parsed.status,
                "v2_answer": parsed.answer, "boxes": [b[:80] for b in boxes][-8:],
                "box_unterminated": unterminated, "rollout_chars": len(text),
                "rollout_tokens": n, "has_think_close": "</think>" in text,
                "dup_sentence_frac": dup, "max_repeat_run": run,
                "chunk_removed": s.get("chunk_removed") if isinstance(
                    s.get("chunk_removed"), str) else None,
                "chunk_resampled": s.get("chunk_resampled") if isinstance(
                    s.get("chunk_resampled"), str) else None,
                "tail": text[-600:] if parsed.status != "answered" else None}
        for k, v in vals.items():
            cols[k].append(v)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    pq.write_table(pa.table(cols), tmp)
    os.replace(tmp, target)
    receipt = {"version": RECEIPT_VERSION, "revision": REVISION, "source_path": rec["path"],
               "source_sha256": rec["sha256"],
               "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "rows": len(samples), "columns": list(COLUMNS)}
    rtmp = receipt_path(target).with_suffix(".tmp")
    rtmp.write_text(json.dumps(receipt, sort_keys=True))
    os.replace(rtmp, receipt_path(target))
    return "created"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = DERIVED / "rollouts"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUTS / "raw_source_manifest.json").read_text())
    recs = [r for r in manifest["files"] if r["role"] == "raw_resamples"][: args.limit]
    tok = Tokenizer.from_file(str(TOKENIZER))
    session = requests.Session()
    failures, t0 = [], time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, r, out, tok, session): r for r in recs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as exc:
                failures.append({"path": futs[f]["path"], "error": repr(exc)})
            if k % 250 == 0:
                print(f"{k}/{len(recs)} {time.time() - t0:.0f}s failures={len(failures)}",
                      flush=True)
    (out / "fetch_failures.json").write_text(json.dumps(failures, indent=1))
    print(json.dumps({"records": len(recs), "failures": len(failures),
                      "seconds": round(time.time() - t0)}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
