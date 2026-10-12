"""Phase 2 (redesign): two independent 100-sample reruns of 200 forks. Exploratory.

Reuses the frozen generation parts: the exact stored prompt (``full_cot[:char_start]``,
fork 0 = stored prompt), ``sample_seed``, ``StreamingVLLMBackend`` (temperature 0.6,
top_p 0.95, max_tokens 16384, no special tokens) and content-addressed
``CompletionStore.raw_one`` units (resumable; completed units are never rewritten).

Replicate seeds: seed_root_r = uint32(sha256("20260914:phase2_replicate:<r>")[:8]),
then the frozen per-sample derivation. Jobs run in sample-major order across both
replicates, so an early stop leaves every fork with the same N in both replicates.

Subcommands:
  build-inputs  (local) prompts + target sentences for the planned forks -> inputs.parquet
  check         host vs vLLM prompt token counts and the context bound
  benchmark     first --units jobs of the production order; writes resource_projection.json
  run           all jobs; requires --confirm-projection <sha256 of resource_projection.json>
  index         completion index parquet from stored units
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from preanchor.ids import sample_seed  # noqa: E402
from preanchor.io import atomic_bytes, json_bytes, write_json  # noqa: E402

TEMPERATURE, TOP_P, MAX_TOKENS = 0.6, 0.95, 16384
SAMPLES, REPLICATES = 100, (1, 2)
MAX_MODEL_LEN = 24576
IMAGE_DIGEST = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
PLAN = REPO / "research/redesign/phase2_forks.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replicate_seed_root(r: int) -> int:
    return int(sha(f"20260914:phase2_replicate:{r}".encode())[:8], 16)


def phase2_config(inputs_sha: str) -> dict:
    return {"schema_version": 1, "experiment": "redesign_phase2_label_reliability",
            "stage": "exploratory", "preregistered": False,
            "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "model_revision": "1df8507178afcc1bef68cd8c393f61a886323761",
            "vllm_image_digest": IMAGE_DIGEST, "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN, "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_new_tokens": MAX_TOKENS, "samples_per_fork": SAMPLES,
            "replicates": {str(r): replicate_seed_root(r) for r in REPLICATES},
            "seed_derivation": "sha256('{seed_root}:{trajectory}:{fork}:{sample}')[:8]",
            "plan_sha256": sha(PLAN.read_bytes()), "inputs_sha256": inputs_sha}


def config_hash(cfg: dict) -> str:
    return sha(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode())


# ---------------------------------------------------------------- build-inputs (local)
def build_inputs(args) -> None:
    import pandas as pd

    root = Path(args.preanchor_root)
    plan = json.loads(PLAN.read_text())
    trajs = pd.read_parquet(root / "data/processed/stage3/trajectories.parquet")
    sents = pd.read_parquet(root / "data/processed/stage3/sentences.parquet")
    rows = []
    for t in plan["trajectories"]:
        tr = trajs[trajs.trajectory_id == t["trajectory_id"]].iloc[0]
        ss = sents[sents.trajectory_id == t["trajectory_id"]].set_index("sentence_idx")
        for fork in t["forks"]:
            prompt = tr.prompt if fork == 0 else tr.full_cot[: int(ss.loc[fork, "char_start"])]
            rows.append({"trajectory_id": t["trajectory_id"], "problem_id": t["problem_id"],
                         "trace_kind": t["trace_kind"], "fork_idx": fork,
                         "prompt": prompt, "prompt_sha256": sha(prompt.encode()),
                         "target_text": ss.loc[fork, "text"],
                         "target_text_sha256": ss.loc[fork, "text_sha256"],
                         "is_label_fork": fork in t["sentence_labels"]})
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print({"forks": len(out), "sha256": sha(Path(args.out).read_bytes())})


# ---------------------------------------------------------------- shared helpers
def load_inputs(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if len(rows) != 200 or len({(r["trajectory_id"], r["fork_idx"]) for r in rows}) != 200:
        raise SystemExit("inputs must hold exactly 200 unique forks")
    for r in rows:
        if sha(r["prompt"].encode()) != r["prompt_sha256"]:
            raise SystemExit("prompt hash mismatch")
    return rows


def job_specs(rows: list[dict], cfg_hash: str, inputs_sha: str):
    from preanchor.labels.generate import CompletionSpec

    specs = []
    for sample in range(SAMPLES):              # sample-major: balanced on early stop
        for r in REPLICATES:
            root = replicate_seed_root(r)
            for row in rows:
                seed = sample_seed(root, row["trajectory_id"], row["fork_idx"], sample)
                specs.append((r, CompletionSpec(row["trajectory_id"], row["fork_idx"],
                                                row["fork_idx"], f"replicate_{r}", sample,
                                                seed, row["prompt"], cfg_hash, inputs_sha)))
    if len({s.rollout_id for _, s in specs}) != len(specs):
        raise SystemExit("duplicate rollout ids")
    return specs


def git_identity() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout

    diff = run("git", "diff", "HEAD")
    return {"commit": run("git", "rev-parse", "HEAD").strip(),
            "dirty_diff_sha256": sha(diff.encode()),
            "untracked_redesign": sorted(run("git", "ls-files", "--others",
                                             "--exclude-standard", "research/redesign")
                                         .split())}


def run_dir(args) -> Path:
    path = Path(args.run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_config(path: Path, cfg: dict) -> str:
    target = path / "phase2_config.json"
    if target.exists():
        if json.loads(target.read_text()) != cfg:
            raise SystemExit("run directory belongs to a different Phase 2 config")
    else:
        write_json(target, cfg)
    return config_hash(cfg)


def execute(specs, endpoints, concurrency, store, stop, on_done):
    """One bounded, immediately refilled pool per endpoint; job i -> endpoint i % E."""
    from preanchor.labels.batching import StreamingVLLMBackend

    lanes = [[s for i, s in enumerate(specs) if i % len(endpoints) == e]
             for e in range(len(endpoints))]
    errors: list[BaseException] = []

    def lane(e: int):
        backend = StreamingVLLMBackend(endpoints[e], image_digest=IMAGE_DIGEST)
        pending, active = iter(lanes[e]), {}
        with ThreadPoolExecutor(concurrency) as pool:
            def refill():
                while len(active) < concurrency and not stop.is_set() and not errors:
                    item = next(pending, None)
                    if item is None:
                        return
                    r, spec = item
                    active[pool.submit(store.raw_one, spec, backend, max_tokens=MAX_TOKENS,
                                       temperature=TEMPERATURE, top_p=TOP_P)] = (r, spec)
            refill()
            while active:
                done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                for task in done:
                    r, spec = active.pop(task)
                    try:
                        row, created = task.result()
                    except BaseException as exc:  # recorded; no new launches after this
                        errors.append(exc)
                        continue
                    if row["rollout_id"] != spec.rollout_id or row["seed"] != spec.seed:
                        errors.append(ValueError("identity mismatch"))
                        continue
                    try:
                        on_done(r, spec, row, created)
                    except BaseException as exc:  # a broken monitor must stop the run
                        errors.append(exc)
                refill()

    threads = [threading.Thread(target=lane, args=(e,)) for e in range(len(endpoints))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{len(errors)} job errors; first: {errors[0]!r}")


class Monitor:
    def __init__(self, path: Path, total: int, rate: float, max_cost: float | None,
                 max_hours: float | None, stop: threading.Event):
        self.path, self.total, self.rate = path, total, rate
        self.max_cost, self.max_hours, self.stop = max_cost, max_hours, stop
        self.started, self.lock = time.time(), threading.Lock()
        self.created = self.skipped = self.tokens = self.length = 0
        self.last_beat = 0.0
        self.reason = None

    def done(self, r, spec, row, created):
        with self.lock:
            if created:
                self.created += 1
                self.tokens += int(row.get("generated_token_count") or 0)
                self.length += row.get("finish_reason") == "length"
            else:
                self.skipped += 1
            self.beat()

    def beat(self, force=False):
        now = time.time()
        if not force and now - self.last_beat < 30:
            return
        self.last_beat = now
        hours = (now - self.started) / 3600
        cost = hours * self.rate if self.rate else None
        if self.max_cost is not None and cost is not None and cost >= self.max_cost:
            self.reason = f"cost guard {cost:.2f} >= {self.max_cost}"
            self.stop.set()
        if self.max_hours is not None and hours >= self.max_hours:
            self.reason = f"runtime guard {hours:.2f} h >= {self.max_hours}"
            self.stop.set()
        done = self.created + self.skipped
        tps = self.tokens / max(now - self.started, 1e-9)
        eta = ((self.total - done) / max(self.created, 1)) * (now - self.started) \
            if self.created else None
        atomic_bytes(self.path, json_bytes({"schema_version": 1, "at": now, "done": done,
                               "total": self.total, "created": self.created,
                               "skipped_existing": self.skipped,
                               "generated_tokens": self.tokens, "aggregate_tps": tps,
                               "length_finishes": self.length, "elapsed_hours": hours,
                               "cost_so_far": cost, "eta_seconds": eta,
                               "stop_reason": self.reason}), replace=True)


def endpoints_of(args) -> list[str]:
    return [e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]


# ---------------------------------------------------------------- check
def check(args) -> None:
    import httpx
    from transformers import AutoTokenizer

    rows = load_inputs(Path(args.inputs))
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                                        revision="1df8507178afcc1bef68cd8c393f61a886323761")
    report = {"mismatches": [], "max_prompt_tokens": 0}
    for e in endpoints_of(args):
        for r in rows:
            host = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            resp = httpx.post(f"{e}/tokenize", json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "prompt": r["prompt"],
                "add_special_tokens": False}, timeout=60)
            resp.raise_for_status()
            server = resp.json().get("count", len(resp.json().get("tokens", [])))
            report["max_prompt_tokens"] = max(report["max_prompt_tokens"], host)
            if host != server:
                report["mismatches"].append([e, r["trajectory_id"], r["fork_idx"], host, server])
    report["context_bound_ok"] = report["max_prompt_tokens"] + MAX_TOKENS <= MAX_MODEL_LEN
    report["passed"] = not report["mismatches"] and report["context_bound_ok"]
    atomic_bytes(run_dir(args) / "check.json", json_bytes(report), replace=True)
    print(json.dumps(report))
    sys.exit(0 if report["passed"] else 1)


# ---------------------------------------------------------------- benchmark / run
def prepare(args):
    from preanchor.labels.generate import CompletionStore

    inputs = Path(args.inputs)
    inputs_sha = sha(inputs.read_bytes())
    rows = load_inputs(inputs)
    cfg = phase2_config(inputs_sha)
    path = run_dir(args)
    cfg_hash = ensure_config(path, cfg)
    if not (path / "check.json").exists() or not json.loads(
            (path / "check.json").read_text())["passed"]:
        raise SystemExit("run `check` first; it must pass")
    store = CompletionStore(path / "completions")
    return path, cfg_hash, job_specs(rows, cfg_hash, inputs_sha), store


def rate_and_limits():
    def num(name):
        value = os.environ.get(name)
        return float(value) if value else None
    rate = num("PREANCHOR_HOURLY_RATE")
    if rate is None:
        raise SystemExit("PREANCHOR_HOURLY_RATE (instance $/h) must be set")
    return rate, num("PREANCHOR_MAX_COST"), num("PREANCHOR_MAX_RUNTIME_HOURS")


def benchmark(args) -> None:
    path, cfg_hash, specs, store = prepare(args)
    rate, max_cost, max_hours = rate_and_limits()
    endpoints = endpoints_of(args)
    subset = specs[: args.units]
    stop = threading.Event()
    mon = Monitor(path / "heartbeat.json", len(subset), rate, None, None, stop)
    rows = []

    def done(r, spec, row, created):
        rows.append({**row, "_created": created})
        mon.done(r, spec, row, created)

    t0 = time.time()
    execute(subset, endpoints, args.concurrency, store, stop, done)
    wall = time.time() - t0
    created = [r for r in rows if r["_created"]]
    if not created:
        raise SystemExit("benchmark created no new units; raise --units")
    tokens = sum(int(r["generated_token_count"]) for r in created)
    mean_tokens = sum(int(r["generated_token_count"]) for r in rows) / len(rows)
    tps = tokens / wall
    plan = json.loads(PLAN.read_text())["projection"]
    total = len(specs)
    remaining_tokens = mean_tokens * (total - len(rows))
    seconds = remaining_tokens / tps
    projection = {
        "schema_version": 1, "config_hash": cfg_hash, "git": git_identity(),
        "endpoints": endpoints, "concurrency_per_endpoint": args.concurrency,
        "benchmark_units": len(rows), "benchmark_units_created": len(created),
        "benchmark_wall_seconds": wall,
        "benchmark_generated_tokens": tokens, "mean_tokens_per_rollout": mean_tokens,
        "aggregate_tokens_per_second": tps,
        "length_finishes": sum(r.get("finish_reason") == "length" for r in created),
        "release_based_mean_tokens": plan["mean_tokens_per_rollout_est"],
        "units_total": total, "projected_remaining_seconds": seconds,
        "projected_remaining_hours": seconds / 3600,
        "hourly_rate": rate, "projected_cost": seconds / 3600 * rate,
        "projected_cost_release_mean": plan["mean_tokens_per_rollout_est"]
        * (total - len(rows)) / tps / 3600 * rate,
        "max_cost": max_cost, "max_runtime_hours": max_hours,
        "within_limits": bool((max_cost is None or seconds / 3600 * rate <= max_cost)
                              and (max_hours is None or seconds / 3600 <= max_hours)),
        "note": "benchmark units are the first jobs of the production order and are reused",
        "created_this_call": len(created)}
    target = path / "resource_projection.json"
    atomic_bytes(target, json_bytes(projection), replace=True)
    print(json.dumps(projection, indent=1))
    print("resource_projection sha256:", sha(target.read_bytes()))


def run(args) -> None:
    path, cfg_hash, specs, store = prepare(args)
    rate, max_cost, max_hours = rate_and_limits()
    target = path / "resource_projection.json"
    if not target.exists() or sha(target.read_bytes()) != args.confirm_projection:
        raise SystemExit("--confirm-projection must equal sha256(resource_projection.json)")
    proj = json.loads(target.read_text())
    if proj["config_hash"] != cfg_hash or proj["git"]["commit"] != git_identity()["commit"]:
        raise SystemExit("projection is stale for this config or commit; re-benchmark")
    if not proj["within_limits"]:
        raise SystemExit("projection exceeds the authorised runtime or budget")
    if max_cost is None or max_hours is None:
        raise SystemExit("PREANCHOR_MAX_COST and PREANCHOR_MAX_RUNTIME_HOURS must be set")
    stop = threading.Event()
    mon = Monitor(path / "heartbeat.json", len(specs), rate, max_cost, max_hours, stop)
    write_json(path / f"run_start_{int(time.time())}.json",
               {"git": git_identity(), "config_hash": cfg_hash,
                "projection_sha256": args.confirm_projection,
                "endpoints": endpoints_of(args), "concurrency": args.concurrency})
    try:
        execute(specs, endpoints_of(args), args.concurrency, store, stop, mon.done)
    finally:
        mon.beat(force=True)
    print(json.dumps(json.loads((path / "heartbeat.json").read_text()), indent=1))


def index(args) -> None:
    import pandas as pd

    path = Path(args.run_dir)
    cfg = json.loads((path / "phase2_config.json").read_text())
    rows = load_inputs(Path(args.inputs))
    out = []
    for r, spec in job_specs(rows, config_hash(cfg), cfg["inputs_sha256"]):
        unit = path / "completions_raw" / spec.rollout_id
        rec = {"replicate": r, "trajectory_id": spec.trajectory_id, "fork_idx": spec.fork_idx,
               "sample_idx": spec.sample_idx, "seed": spec.seed,
               "rollout_id": spec.rollout_id, "status": "MISSING"}
        if (unit / "SUCCESS.json").exists():
            resp = json.loads((unit / "response.json").read_text())
            rec.update(status="OK", raw_answer=resp["raw_answer"],
                       finish_reason=resp["finish_reason"],
                       generated_token_count=resp["generated_token_count"],
                       prompt_token_count=resp["prompt_token_count"],
                       first_replacement_sentence=resp["first_replacement_sentence"],
                       extraction_status=resp["extraction_status"])
        out.append(rec)
    frame = pd.DataFrame(out)
    frame.to_parquet(path / "completions_index.parquet", index=False)
    print(frame.groupby(["replicate", "status"]).size().to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-inputs")
    b.add_argument("--preanchor-root", default=str(Path.home() / "preanchor-root"))
    b.add_argument("--out", required=True)
    for name in ("check", "benchmark", "run", "index"):
        s = sub.add_parser(name)
        s.add_argument("--inputs", required=True)
        s.add_argument("--run-dir", required=True)
        if name != "index":
            s.add_argument("--endpoints", default="http://127.0.0.1:8000")
        if name in ("benchmark", "run"):
            s.add_argument("--concurrency", type=int, required=True)
        if name == "benchmark":
            s.add_argument("--units", type=int, default=64)
        if name == "run":
            s.add_argument("--confirm-projection", required=True)
    args = p.parse_args()
    {"build-inputs": build_inputs, "check": check, "benchmark": benchmark, "run": run,
     "index": index}[args.cmd](args)


if __name__ == "__main__":
    main()
