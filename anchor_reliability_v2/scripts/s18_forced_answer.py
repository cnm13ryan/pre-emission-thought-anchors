"""Plan v3 PILOT: forced-answer readout at each Phase 2 sentence boundary
(``plan_v3_forced_answer_pilot.md``). EXPLORATORY; prepared before any forced-answer data
exist.

At fork i (problem + trace prefix up to just before S_i, the exact stored Phase 2 prompt)
the thinking block is closed and an answer is forced:

    forced_i = prompt_i.rstrip() + "\\n</think>\\n\\n**Final Answer**\\n\\\\[\\n\\\\boxed"

The forced prefix ends at ``\\boxed``, not ``\\boxed{``: the Qwen2 pre-tokenizer merges ``{``
with a following punctuation character (``{\\``, ``{(``), but never with the letters of
``boxed``, so scoring the continuation ``{answer}`` starts on a clean token boundary.

Two readouts (plan §3):
(a) candidate scoring: log P("{" + surface + "}" | forced_i) from prompt logprobs
    (completions API, ``echo`` + ``logprobs``); surfaces of one category are summed
    (logsumexp), categories renormalised over the candidate set.
(b) sampling: K short completions of forced_i (max 30 tokens, temperature 0.6, top_p 0.95),
    first balanced box parsed with the v2 normaliser.

Subcommands:
  build-requests  (local, CPU) inputs.parquet + pooled resamples -> request JSONL, prompts,
                  per-fork targets and candidate sets under DATA/derived/plan_v3/
  run             send the requests to an OpenAI-compatible vLLM endpoint through the
                  durable store (one file per request, never regenerated), with retries,
                  a concurrency limit and a cost / runtime guard
  score           every preregistered metric from the stored responses ->
                  results/plan_v3_pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2 import answers as A  # noqa: E402
from arv2 import reliability as R  # noqa: E402
from arv2.paths import DATA, INPUTS, RESULTS, STUDY, write_json  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"     # = arv2.vllm_client.MODEL
REVISION = "1df8507178afcc1bef68cd8c393f61a886323761"
PHASE2_INPUTS = Path.home() / "preanchor-root/data/processed/redesign_phase2/inputs.parquet"
PHASE2_INDEX = (DATA / "generations/phase2_mirror/20261009T0717Z-phase2-23f5e3e6"
                / "completions_index.parquet")
BUILD = DATA / "derived/plan_v3"
STORE = DATA / "generations/plan_v3_forced"

# (strip trailing whitespace of the stored prefix?, suffix). FIXED in plan §3.
SUFFIXES = {
    "primary": (True, "\n</think>\n\n**Final Answer**\n\\[\n\\boxed"),
    "alt1": (False, "\n</think>\n\nThe final answer is \\boxed"),          # s08 style
    "alt2": (True, "\n\nI'll stop here and give my answer.\n</think>\n\nThe answer is $\\boxed"),
}
OTHER = "<OTHER>"
SPECIAL = {A.TRUNCATED, A.NON_ANSWER}
MIN_SHARE, MAX_CATS, MAX_SURFACES = 0.005, 12, 2
K_SAMPLES, SAMPLE_PASSES, SAMPLE_MAX_TOKENS = 32, ("s1", "s2"), 30
FORMAT_FORKS_PER_TRAJ = 2
SEED_MATERIAL = "20260914:arv2:plan_v3"
REPS = 2000
THRESH = {"H1a_spearman_level": 0.60, "H1b_within_level": 0.30, "S2_sampled_level": 0.60,
          "S3_entropy": 0.50, "S4_signed_change": 0.30, "S5_tv_debiased": 0.20,
          "S6_partial_fork_tv_given_floor": 0.10, "S7_ap_real_effects": 0.25}
GATES = {"span_mismatch_max": 0.01, "median_mass_min": 0.50, "sample_other_max": 0.20,
         "completeness_min": 0.95, "replay_spearman_min": 0.99, "replay_maxabs_max": 0.02,
         "format_spearman_min": 0.80}
LABEL_SETS = {"r12": (1, 2), "r012": (0, 1, 2)}      # primary first
PRIMARY_LABELS = "r12"
BASELINES = ["position_None", "text+position_None", "pre_8", "pre+post_32",
             "valueprobe_dphat_32"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------- prompts


def forced_prompt(prompt: str, suffix: str = "primary") -> str:
    """Stored fork prompt -> forced-answer prefix (ends at ``\\boxed``). Fails closed if
    the prompt is not an open think block, so the suffix can never be appended twice."""
    if prompt.count("<think>") != 1 or "</think>" in prompt:
        raise ValueError("prompt must contain one open <think> block and no </think>")
    strip, text = SUFFIXES[suffix]
    return (prompt.rstrip() if strip else prompt) + text


def continuation(surface: str) -> str:
    return "{" + surface + "}"


def score_payload(forced: str, surface: str) -> dict:
    return {"model": MODEL, "prompt": forced + continuation(surface), "max_tokens": 1,
            "temperature": 0.0, "echo": True, "logprobs": 1, "add_special_tokens": False}


def sample_payload(forced: str, seed: int) -> dict:
    # identical field set to arv2.vllm_client.Client.complete, so the store layouts match
    return {"model": MODEL, "prompt": forced, "max_tokens": SAMPLE_MAX_TOKENS,
            "temperature": 0.6, "top_p": 0.95, "add_special_tokens": False, "seed": seed}


def store_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- candidates


def surface_of(boxes, v2_answer) -> str | None:
    """Raw last box whose v2 normalisation equals the stored category."""
    if v2_answer is None or boxes is None or not len(boxes):
        return None
    raw = str(boxes[-1])
    return raw if raw.strip() and A.normalize_v2(raw) == v2_answer else None


def surface_from_text(text) -> tuple[str | None, str | None]:
    """(raw surface, category) of a full continuation, by the parse_v2 region rule."""
    if not isinstance(text, str):
        return None, None
    region = text.rsplit("</think>", 1)[1] if "</think>" in text else text
    boxes = [b for b in A.balanced_boxes(region)[0] if b.strip()]
    if not boxes:
        return None, None
    cat = A.normalize_v2(boxes[-1])
    return (boxes[-1], cat) if cat else (None, None)


def candidate_set(counts: dict, surfaces: dict, gt_cat: str, gt_raw: str) -> list[dict]:
    """Answer categories with pooled share >= MIN_SHARE (at most MAX_CATS, by count), plus
    the ground truth always. Non-answers are never candidates. Each category keeps up to
    MAX_SURFACES raw strings that normalise to it (ground truth: its reference string
    first)."""
    answers = {c: n for c, n in counts.items() if c not in SPECIAL}
    total = sum(counts.values())
    ranked = sorted(answers, key=lambda c: (-answers[c], c))
    keep = [c for c in ranked if total and answers[c] / total >= MIN_SHARE][:MAX_CATS]
    if gt_cat not in keep:
        keep.append(gt_cat)
    out = []
    for c in keep:
        seen = Counter({s: n for s, n in surfaces.get(c, {}).items()
                        if A.normalize_v2(s) == c})
        forms = [gt_raw] if c == gt_cat and A.normalize_v2(gt_raw) == c else []
        for s, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(forms) >= MAX_SURFACES:
                break
            if s not in forms:
                forms.append(s)
        if not forms:
            forms = [gt_raw] if c == gt_cat else [c]
        out.append({"category": c, "share": answers.get(c, 0) / total if total else 0.0,
                    "is_gt": c == gt_cat, "surfaces": forms[:MAX_SURFACES]})
    return out


# ----------------------------------------------------------------------------- logprobs


def span_logprob(tokens: list[str], logprobs: list, offsets: list[int], start: int,
                 end: int) -> dict:
    """Sum of prompt-token logprobs over the character span [start, end). Tokens at or past
    ``end`` (the generated token) are excluded; a token that begins before ``start``
    straddles the boundary and is flagged."""
    total, text, straddle, missing = 0.0, [], False, False
    for j, (tok, lp, off) in enumerate(zip(tokens, logprobs, offsets, strict=True)):
        stop = offsets[j + 1] if j + 1 < len(offsets) else off + len(tok)
        if off >= end or stop <= start:
            continue
        straddle |= off < start
        if lp is None:
            missing = True
        else:
            total += float(lp)
        text.append(tok)
    return {"logprob": float("nan") if missing else total, "text": "".join(text),
            "straddle": bool(straddle)}


def logsumexp(values) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return float("-inf")
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def category_distribution(surface_logprobs: dict[str, list[float]]) -> tuple[dict, float]:
    """{category: [log P(surface)]} -> (normalised distribution over categories, raw mass).
    Surfaces of one category are alternative strings for the same answer, so they add."""
    cat_lp = {c: logsumexp(v) for c, v in surface_logprobs.items()}
    finite = {c: v for c, v in cat_lp.items() if np.isfinite(v)}
    if not finite:
        return {c: float("nan") for c in cat_lp}, 0.0
    z = logsumexp(list(finite.values()))
    dist = {c: float(math.exp(v - z)) if np.isfinite(v) else 0.0 for c, v in cat_lp.items()}
    return dist, float(math.exp(z))


def entropy_bits(p) -> float:
    p = np.asarray([x for x in p if x > 0], float)
    if not len(p):
        return float("nan")
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def tv_dist(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def parse_forced_sample(text: str | None) -> str:
    """Category of a forced completion of a prompt ending at ``\\boxed``."""
    if not isinstance(text, str) or not text.lstrip(" ").startswith("{"):
        return A.NON_ANSWER
    boxes, _ = A.balanced_boxes("\\boxed" + text.lstrip(" "))
    if not boxes or not boxes[0].strip():
        return A.NON_ANSWER
    return A.normalize_v2(boxes[0]) or A.NON_ANSWER


# ----------------------------------------------------------------------------- build


def _load_s13():
    spec = importlib.util.spec_from_file_location(
        "s13", STUDY / "scripts/s13_phase2_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_subset(forks: pd.DataFrame) -> set[tuple[str, int]]:
    """FIXED 20-fork subset for the format check: per trajectory, the FORMAT_FORKS_PER_TRAJ
    forks with the smallest sha256(material:trajectory:fork)."""
    out = set()
    for tid, g in forks.groupby("trajectory_id"):
        ranked = sorted(g.fork_idx, key=lambda f: sha(f"{SEED_MATERIAL}:format:{tid}:{f}"))
        out |= {(tid, int(f)) for f in ranked[:FORMAT_FORKS_PER_TRAJ]}
    return out


def build_rows(forks: pd.DataFrame, cands: dict, k: int = K_SAMPLES) -> tuple[list, list]:
    """forks: trajectory_id, problem_id, fork_idx, prompt. cands[tid] = {"out": [...],
    "in": [...]} candidate sets. Returns (request rows, prompt rows)."""
    from arv2.vllm_client import sample_seed  # lazy: the client module imports requests

    fmt = format_subset(forks)
    rows, prompts = [], []
    for f in forks.sort_values(["trajectory_id", "fork_idx"]).itertuples():
        tid, fork = f.trajectory_id, int(f.fork_idx)
        surf = []
        for which in ("out", "in"):
            for c in cands[tid][which]:
                surf += [(c["category"], s) for s in c["surfaces"]]
        surf = list(dict.fromkeys(surf))
        for suffix in ("primary", "alt1", "alt2"):
            if suffix != "primary" and (tid, fork) not in fmt:
                continue
            forced = forced_prompt(f.prompt, suffix)
            prompts.append({"trajectory_id": tid, "fork_idx": fork, "suffix": suffix,
                            "forced_prompt": forced, "forced_sha256": sha(forced)})
            stores = ("main", "replay") if suffix == "primary" else ("main",)
            for store in stores:
                for cat, s in surf:
                    rows.append({"request_id": f"score|{suffix}|{store}|{tid}|{fork}|{sha(s)[:12]}",
                                 "kind": "score", "suffix": suffix, "store": store,
                                 "trajectory_id": tid, "problem_id": int(f.problem_id),
                                 "fork_idx": fork, "category": cat, "surface": s,
                                 "forced_sha256": sha(forced), "span_start": len(forced),
                                 "span_end": len(forced) + len(continuation(s))})
            if suffix == "primary":
                for p in SAMPLE_PASSES:
                    root = R.seed_from(f"{SEED_MATERIAL}:sample:{p}")
                    for j in range(k):
                        rows.append({"request_id": f"sample|primary|main|{tid}|{fork}|{p}:{j}",
                                     "kind": "sample", "suffix": suffix, "store": "main",
                                     "trajectory_id": tid, "problem_id": int(f.problem_id),
                                     "fork_idx": fork, "pass": p, "sample_idx": j,
                                     "seed": sample_seed(root, tid, fork, j),
                                     "forced_sha256": sha(forced)})
    return rows, prompts


def payload_of(row: dict, forced: str) -> dict:
    if sha(forced) != row["forced_sha256"]:
        raise ValueError(f"forced prompt hash mismatch: {row['request_id']}")
    if row["kind"] == "score":
        return score_payload(forced, row["surface"])
    return sample_payload(forced, int(row["seed"]))


def build_requests(args) -> None:
    s13 = _load_s13()
    inputs = pd.read_parquet(args.inputs)
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    gt_raw = dict(zip(trajs.trajectory_id, trajs.ground_truth_answer, strict=True))
    samples = s13.load_samples(args.index)            # replicates 0/1/2, ~300 per fork
    window = {(t, int(f)) for t, f in zip(inputs.trajectory_id, inputs.fork_idx, strict=True)}
    samples = samples[[(t, int(f)) in window for t, f in
                       zip(samples.trajectory_id, samples.fork_idx, strict=True)]]
    # per-fork targets. PRIMARY label set "r12": replicates 1 and 2 only (200 per fork,
    # same image, revision and sampling settings as this pilot). Sensitivity "r012": the
    # 300-sample pool including the released replicate 0 (plan §4).
    targets = []
    for label_set, reps in LABEL_SETS.items():
        sub = samples[samples.replicate.isin(reps)]
        for (tid, fork), g in sub.groupby(["trajectory_id", "fork_idx"]):
            counts = g.cat.value_counts().to_dict()
            targets.append({"label_set": label_set, "trajectory_id": tid,
                            "problem_id": int(tid.split(":")[0]), "fork_idx": int(fork),
                            "n": len(g), "acc_level": float(g.correct.mean()),
                            "entropy": entropy_bits(np.array(list(counts.values())) / len(g)),
                            "counts_json": json.dumps(counts, sort_keys=True)})
    targets = pd.DataFrame(targets)
    # surfaces inside the window: replicates 1/2 (full text) and released window forks
    idx = pd.read_parquet(args.index, columns=["trajectory_id", "fork_idx", "status",
                                               "raw_answer"])
    in_surf: dict[str, dict[str, Counter]] = {}
    for t, text in zip(idx.trajectory_id, idx.raw_answer, strict=True):
        s, c = surface_from_text(text)
        if s is not None:
            in_surf.setdefault(t, {}).setdefault(c, Counter())[s] += 1
    cands, coverage = {}, {}
    rel_dir = DATA / "derived/rollouts"
    for tid in sorted(inputs.trajectory_id.unique()):
        problem = tid.split(":")[0]
        gt_cat = A.normalize_v2(gt_raw[tid])
        # out-of-window: released resamples of the same problem, excluding this
        # trajectory's Phase 2 window forks (the validation forks)
        out_counts, out_surf = Counter(), {}
        for path in sorted(rel_dir.glob(f"{problem}_*/fork_*.parquet")):
            r = pd.read_parquet(path, columns=["trajectory_id", "fork_idx", "v2_status",
                                               "v2_answer", "boxes"])
            r = r[[t != tid or (t, int(f)) not in window
                   for t, f in zip(r.trajectory_id, r.fork_idx, strict=True)]]
            for st, ans, bx in zip(r.v2_status, r.v2_answer, r.boxes, strict=True):
                cat = ans if st in ("answered", "unboxed_stated") and ans else A.NON_ANSWER
                out_counts[cat] += 1
                s = surface_of(bx, ans)
                if s is not None:
                    out_surf.setdefault(cat, Counter())[s] += 1
        in_counts = Counter()
        # in-window candidates use the full 300 pool (a candidate list, not a label)
        for c in targets[(targets.trajectory_id == tid)
                         & (targets.label_set == "r012")].counts_json:
            in_counts.update(json.loads(c))
        surf_all = {c: in_surf.get(tid, {}).get(c, Counter()) + out_surf.get(c, Counter())
                    for c in set(in_surf.get(tid, {})) | set(out_surf)}
        cands[tid] = {"gt_category": gt_cat, "gt_raw": gt_raw[tid],
                      "out": candidate_set(dict(out_counts), surf_all, gt_cat, gt_raw[tid]),
                      "in": candidate_set(dict(in_counts), surf_all, gt_cat, gt_raw[tid])}
        tot = sum(in_counts.values())
        coverage[tid] = {w: sum(in_counts.get(c["category"], 0) for c in cands[tid][w]) / tot
                         for w in ("out", "in")}
    rows, prompts = build_rows(inputs, cands, args.k)
    BUILD.mkdir(parents=True, exist_ok=True)
    with open(BUILD / "requests.jsonl.tmp", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(BUILD / "requests.jsonl.tmp", BUILD / "requests.jsonl")
    pd.DataFrame(prompts).to_parquet(BUILD / "prompts.parquet")
    targets.to_parquet(BUILD / "targets.parquet")
    write_json(BUILD / "candidates.json", {"rule": {"min_share": MIN_SHARE,
               "max_categories": MAX_CATS, "max_surfaces": MAX_SURFACES},
               "candidates": cands, "window_coverage": coverage})
    kinds = Counter((r["kind"], r["suffix"], r["store"]) for r in rows)
    report = {"requests": len(rows), "by_kind": {"|".join(k): v for k, v in kinds.items()},
              "forks": int(len(inputs)), "window_coverage": coverage,
              "inputs_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
              "requests_sha256": hashlib.sha256((BUILD / "requests.jsonl").read_bytes())
              .hexdigest()}
    if args.tokenizer_check:
        report["token_boundary"] = tokenizer_check(rows, prompts)
    write_json(BUILD / "build_report.json", report)
    print(json.dumps(report, indent=1))


def tokenizer_check(rows: list, prompts: list) -> dict:
    """Optional CPU check with the local tokenizer: does every scored span start on a token
    boundary? (The plan's quality gate Q1 re-checks this from vLLM's own offsets.)"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    forced = {p["forced_sha256"]: p["forced_prompt"] for p in prompts}
    bad = n = 0
    for r in rows:
        if r["kind"] != "score":
            continue
        text = forced[r["forced_sha256"]] + continuation(r["surface"])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = {a for a, _ in enc["offset_mapping"]}
        n += 1
        bad += r["span_start"] not in starts
    return {"checked": n, "not_on_boundary": bad}


# ----------------------------------------------------------------------------- run


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    import requests  # lazy: keeps build/score importable without network packages

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def span_logprob_suffix(tokens: list[str], logprobs: list, n_prompt: int, cont: str) -> dict:
    """Sum of logprobs of the shortest run of final prompt tokens whose text covers the
    continuation ``cont`` (which ends the prompt). Uses ``usage.prompt_tokens`` to drop the
    generated token, so it does not depend on ``text_offset`` (vLLM builds offsets from
    detokenised strings, which drift after any split multi-byte character earlier in the
    prompt)."""
    total, text, missing, j = 0.0, "", False, n_prompt - 1
    while j >= 0 and len(text) < len(cont):
        text = tokens[j] + text
        if logprobs[j] is None:
            missing = True
        else:
            total += float(logprobs[j])
        j -= 1
    return {"logprob": float("nan") if missing or not text.endswith(cont) else total,
            "text": text, "straddle": text != cont}


def response_record(row: dict, body: dict) -> dict:
    """Compact stored response. Score requests keep only the scored tail of the echoed
    prompt logprobs (a full echo is ~1,500 tokens per request)."""
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    rec = {"text": choice.get("text", ""), "finish_reason": choice.get("finish_reason"),
           "usage": usage}
    if row["kind"] == "score":
        lp = choice.get("logprobs") or {}
        toks, lps, offs = lp.get("tokens", []), lp.get("token_logprobs", []), \
            lp.get("text_offset", [])
        cont = continuation(row["surface"])
        n_prompt = usage.get("prompt_tokens")
        if isinstance(n_prompt, int) and 0 < n_prompt <= len(toks):
            span, method = span_logprob_suffix(toks, lps, n_prompt, cont), "suffix"
            keep = list(range(max(0, n_prompt - 48), len(toks)))
        else:                                   # fallback: character offsets
            span, method = span_logprob(toks, lps, offs, row["span_start"],
                                        row["span_end"]), "offset"
            keep = [j for j, o in enumerate(offs) if o >= row["span_start"] - 32]
        rec.update(span_logprob=span["logprob"], span_text=span["text"],
                   straddle=span["straddle"], span_method=method,
                   span_ok=bool(span["text"] == cont and np.isfinite(span["logprob"])),
                   tail={"tokens": [toks[j] for j in keep],
                         "token_logprobs": [lps[j] for j in keep],
                         "text_offset": [offs[j] for j in keep] if offs else []})
    return rec


def store_path(store_root: Path, row: dict, payload: dict) -> Path:
    return store_root / row["store"] / f"{store_key(payload)}.json"


def send_one(row: dict, payload: dict, store_root: Path, endpoint: str, post: Callable,
             retries: int, backoff: float, timeout: float) -> tuple[bool, str | None]:
    path = store_path(store_root, row, payload)
    if path.exists():
        return False, None
    err = None
    for attempt in range(retries + 1):
        try:
            body = post(f"{endpoint.rstrip('/')}/v1/completions", payload, timeout)
            rec = response_record(row, body)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": payload, "response": rec,
                                       "request_id": row["request_id"]}))
            os.replace(tmp, path)
            return True, None
        except Exception as e:  # noqa: BLE001 - every failure is retried, then recorded
            err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(backoff * 2 ** attempt)
    return False, err


def run_requests(rows: list, forced: dict, store_root: Path, endpoint: str, *,
                 concurrency: int, rate: float, max_cost: float, max_hours: float,
                 spent_hours: float = 0.0, post: Callable | None = None, retries: int = 4,
                 backoff: float = 2.0, timeout: float = 600.0,
                 clock: Callable[[], float] = time.monotonic, heartbeat: Path | None = None
                 ) -> dict:
    """Send every row not yet in the store. Stops launching when (spent + elapsed) hours
    reach ``max_hours`` or that time × ``rate`` reaches ``max_cost``."""
    post = post or _http_post
    t0, stop, lock = clock(), threading.Event(), threading.Lock()
    stats = {"sent": 0, "reused": 0, "failed": 0, "not_launched": 0, "stop_reason": None,
             "failures": []}

    def guard() -> None:
        hours = spent_hours + (clock() - t0) / 3600
        if hours >= max_hours:
            stats["stop_reason"] = f"runtime guard {hours:.2f} h >= {max_hours}"
            stop.set()
        elif hours * rate >= max_cost:
            stats["stop_reason"] = f"cost guard ${hours * rate:.2f} >= ${max_cost}"
            stop.set()

    pending = []
    for r in rows:
        p = payload_of(r, forced[r["forced_sha256"]])
        if store_path(store_root, r, p).exists():
            stats["reused"] += 1
        else:
            pending.append((r, p))
    it = iter(pending)
    with ThreadPoolExecutor(concurrency) as pool:
        active = set()

        def refill() -> None:
            while len(active) < concurrency and not stop.is_set():
                guard()
                if stop.is_set():
                    return
                nxt = next(it, None)
                if nxt is None:
                    return
                active.add(pool.submit(send_one, *nxt, store_root, endpoint, post, retries,
                                       backoff, timeout))

        refill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for fut in done:
                active.discard(fut)
                created, err = fut.result()
                with lock:
                    if err:
                        stats["failed"] += 1
                        stats["failures"].append(err)
                    elif created:
                        stats["sent"] += 1
            if heartbeat is not None:
                hours = spent_hours + (clock() - t0) / 3600
                write_json(heartbeat, {**{k: v for k, v in stats.items() if k != "failures"},
                                       "total": len(rows), "hours": hours,
                                       "cost_so_far": hours * rate})
            refill()
    stats["not_launched"] = sum(1 for _ in it)
    stats["hours"] = spent_hours + (clock() - t0) / 3600
    stats["cost"] = stats["hours"] * rate
    stats["failures"] = stats["failures"][:50]
    return stats


def load_build(build: Path) -> tuple[list, dict]:
    rows = [json.loads(line) for line in open(build / "requests.jsonl")]
    pr = pd.read_parquet(build / "prompts.parquet")
    forced = dict(zip(pr.forced_sha256, pr.forced_prompt, strict=True))
    return rows, forced


def run(args) -> None:
    rows, forced = load_build(args.build)
    if args.only:
        kinds = set(args.only.split(","))
        rows = [r for r in rows if f"{r['kind']}:{r['suffix']}:{r['store']}" in kinds
                or r["kind"] in kinds]
    if args.limit:
        rows = rows[:args.limit]                # smoke test: the first N requests
    stats = run_requests(rows, forced, args.store, args.endpoint,
                         concurrency=args.concurrency, rate=args.rate,
                         max_cost=args.max_cost, max_hours=args.max_hours,
                         spent_hours=args.spent_hours, retries=args.retries,
                         heartbeat=args.store / "heartbeat.json")
    write_json(args.store / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
               {**stats, "endpoint": args.endpoint, "concurrency": args.concurrency,
                "rate": args.rate, "max_cost": args.max_cost, "max_hours": args.max_hours})
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=1))


# ----------------------------------------------------------------------------- score


def load_responses(rows: list, forced: dict, store_root: Path) -> pd.DataFrame:
    out = []
    for r in rows:
        p = payload_of(r, forced[r["forced_sha256"]])
        path = store_path(store_root, r, p)
        rec = {k: r.get(k) for k in ("request_id", "kind", "suffix", "store", "trajectory_id",
                                     "problem_id", "fork_idx", "category", "surface", "pass",
                                     "sample_idx")}
        if path.exists():
            resp = json.loads(path.read_text())["response"]
            rec.update(present=True, text=resp.get("text"),
                       span_logprob=resp.get("span_logprob"), span_ok=resp.get("span_ok"))
        else:
            rec["present"] = False
        out.append(rec)
    return pd.DataFrame(out)


def fork_readouts(resp: pd.DataFrame, cands: dict, suffix: str = "primary",
                  store: str = "main", which: str = "out") -> pd.DataFrame:
    """Option (a): per-fork distribution over one candidate set."""
    sc = resp[(resp.kind == "score") & (resp.suffix == suffix) & (resp.store == store)
              & resp.present]
    rows = []
    for (tid, fork), g in sc.groupby(["trajectory_id", "fork_idx"]):
        c = cands[tid]
        keys = zip(g.category, g.surface, strict=True)
        lp = dict(zip(keys, g.span_logprob.astype(float), strict=True))
        surf = {x["category"]: [lp.get((x["category"], s), float("nan")) for s in x["surfaces"]]
                for x in c[which]}
        if any(not np.isfinite(v).any() for v in map(np.asarray, surf.values())):
            continue                                # incomplete fork: excluded and counted
        dist, mass = category_distribution(surf)
        rows.append({"trajectory_id": tid, "problem_id": int(g.problem_id.iloc[0]),
                     "fork_idx": int(fork), "p_forced": dist.get(c["gt_category"], 0.0),
                     "mass": mass, "entropy_forced": entropy_bits(list(dist.values())),
                     "dist": dist})
    return pd.DataFrame(rows)


def sample_readouts(resp: pd.DataFrame, cands: dict, which: str = "out") -> pd.DataFrame:
    """Option (b): per fork and pass, sampled answer distribution."""
    sm = resp[(resp.kind == "sample") & resp.present]
    rows = []
    for (tid, fork, p), g in sm.groupby(["trajectory_id", "fork_idx", "pass"]):
        cats = [parse_forced_sample(t) for t in g.text]
        allowed = {x["category"] for x in cands[tid][which]}
        mapped = [c if c in allowed or c == A.NON_ANSWER else OTHER for c in cats]
        cnt = Counter(mapped)
        rows.append({"trajectory_id": tid, "problem_id": int(g.problem_id.iloc[0]),
                     "fork_idx": int(fork), "pass": p, "k": len(cats),
                     "p_sampled": float(np.mean([c == cands[tid]["gt_category"] for c in cats])),
                     "other_rate": cnt[OTHER] / len(cats),
                     "nonanswer_rate": cnt[A.NON_ANSWER] / len(cats),
                     "dist": {c: n / len(cats) for c, n in cnt.items()}})
    return pd.DataFrame(rows)


def tv_floor_analytic(ca: dict, cb: dict) -> float:
    """Normal approximation to E[TV] under no effect (as s13 ``tv_null_analytic``)."""
    na, nb = sum(ca.values()), sum(cb.values())
    keys = set(ca) | set(cb)
    p = np.array([(ca.get(k, 0) + cb.get(k, 0)) / (na + nb) for k in keys])
    sd = np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return 0.5 * float((sd * np.sqrt(2 / np.pi)).sum())


def sentence_table(forks: pd.DataFrame, targets: pd.DataFrame, units: list,
                   effects: pd.DataFrame | None) -> pd.DataFrame:
    """One row per labelled sentence i (forks i and i+1): forced changes and targets."""
    fk = forks.set_index(["trajectory_id", "fork_idx"])
    tg = targets.set_index(["trajectory_id", "fork_idx"])
    eff = (effects.set_index(["trajectory_id", "sentence_idx"]) if effects is not None
           else None)
    rows = []
    for tid, prob, i in units:
        if (tid, i) not in tg.index or (tid, i + 1) not in tg.index:
            continue
        ca, cb = json.loads(tg.loc[(tid, i), "counts_json"]), json.loads(tg.loc[(tid, i + 1),
                                                                               "counts_json"])
        na, nb = sum(ca.values()), sum(cb.values())
        ftv = 0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in set(ca) | set(cb))
        floor = tv_floor_analytic(ca, cb)
        rec = {"trajectory_id": tid, "problem_id": prob, "sentence_idx": i,
               "acc_before": tg.loc[(tid, i), "acc_level"],
               "acc_after": tg.loc[(tid, i + 1), "acc_level"],
               "pooled_fork_tv": ftv, "floor": floor, "tv_debiased": ftv - floor}
        rec["signed_dacc"] = rec["acc_after"] - rec["acc_before"]
        if (tid, i) in fk.index and (tid, i + 1) in fk.index:
            a, b = fk.loc[(tid, i)], fk.loc[(tid, i + 1)]
            rec["dp_forced"] = b.p_forced - a.p_forced
            rec["forced_tv"] = tv_dist(a.dist, b.dist)
        if eff is not None and (tid, i) in eff.index:
            e = eff.loc[(tid, i)]
            rec["real_effect"] = bool(e.real_tv_bh10 or e.real_dacc_bh10)
        rows.append(rec)
    return pd.DataFrame(rows)


def partial_spearman(x, y, z) -> float:
    """Spearman of x and y after regressing the ranks of the columns of z out of both."""
    from scipy.stats import rankdata

    x, y = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in np.atleast_2d(z)])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def average_precision(y, s) -> float:
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def within_values(x, y, traj) -> np.ndarray:
    return np.array([R.spearman(x[traj == t], y[traj == t]) for t in np.unique(traj)])


def decide(name: str, fn: Callable, groups: np.ndarray, n: int, threshold: float | None,
           reps: int) -> dict:
    """Point estimate, 95% problem-cluster interval and (if a threshold is given) the
    three-way rule of plan §7 / phase2_plan.md §7."""
    est = fn(np.arange(n))
    lo, hi, bad = R.interval(R.cluster_bootstrap(groups, fn, reps,
                                                 R.seed_from(f"{SEED_MATERIAL}:boot:{name}")))
    rec = {"estimate": est, "lo": lo, "hi": hi, "nonfinite": bad}
    if threshold is not None:
        rec["threshold"] = threshold
        rec["decision"] = R.three_way(lo, hi, threshold)
    return rec


def compute_metrics(forks: pd.DataFrame, sents: pd.DataFrame, *,
                    sampled: pd.DataFrame | None = None, baselines: pd.DataFrame | None = None,
                    reps: int = REPS) -> dict:
    """forks: trajectory_id, problem_id, fork_idx, p_forced, entropy_forced, acc_level,
    entropy (pooled). sents: sentence_table output (+ baseline columns joined by caller)."""
    f = forks.dropna(subset=["p_forced", "acc_level"]).reset_index(drop=True)
    g, traj = f.problem_id.to_numpy(), f.trajectory_id.to_numpy()
    p, acc = f.p_forced.to_numpy(float), f.acc_level.to_numpy(float)
    pos = f.fork_idx.to_numpy(float)
    out: dict = {"forks": len(f), "problems": int(len(set(g)))}
    out["H1a_spearman_level"] = decide("H1a", lambda r: R.spearman(p[r], acc[r]), g, len(f),
                                       THRESH["H1a_spearman_level"], reps)
    out["H1_descriptive"] = {"pearson": float(np.corrcoef(p, acc)[0, 1]),
                             "icc_2_1": R.icc_2_1(p, acc),
                             "mean_abs_error": float(np.abs(p - acc).mean())}
    w = within_values(p, acc, traj)
    tg = np.array([f.problem_id[traj == t].iloc[0] for t in np.unique(traj)])
    out["H1b_within_level"] = decide("H1b", lambda r: float(np.nanmedian(w[r]))
                                     if np.isfinite(w[r]).any() else float("nan"),
                                     tg, len(w), THRESH["H1b_within_level"], reps)
    out["H1b_within_level"]["per_trajectory"] = dict(zip(np.unique(traj).tolist(),
                                                         map(float, w), strict=True))
    parts = [out["H1a_spearman_level"]["decision"], out["H1b_within_level"]["decision"]]
    out["H1_primary"] = ("FAIL" if "FAIL" in parts else
                         "PASS" if all(x == "PASS" for x in parts) else "INCONCLUSIVE")
    if "entropy_forced" in f and "entropy" in f:
        hf, hp = f.entropy_forced.to_numpy(float), f.entropy.to_numpy(float)
        out["S3_entropy"] = decide("S3", lambda r: R.spearman(hf[r], hp[r]), g, len(f),
                                   THRESH["S3_entropy"], reps)
    # position baseline for the level (sign-free, generous to the baseline)
    out["H4_level_vs_position"] = decide(
        "H4lvl", lambda r: R.spearman(p[r], acc[r]) - abs(R.spearman(pos[r], acc[r])),
        g, len(f), 0.0, reps)
    if "problem_baseline" in f and np.isfinite(f.problem_baseline.to_numpy(float)).all():
        # a per-problem constant (s08 no-reasoning accuracy): how far problem identity alone
        # goes on the across-fork statistic
        pb = f.problem_baseline.to_numpy(float)
        out["level_problem_constant"] = {
            "spearman": R.spearman(pb, acc),
            "diff_forced_minus_constant": decide(
                "H4const", lambda r: R.spearman(p[r], acc[r]) - R.spearman(pb[r], acc[r]),
                g, len(f), 0.0, reps)}
    if sampled is not None and len(sampled):
        sp = sampled.merge(f[["trajectory_id", "fork_idx", "acc_level", "p_forced"]],
                           on=["trajectory_id", "fork_idx"])
        s1 = sp[sp["pass"] == SAMPLE_PASSES[0]].reset_index(drop=True)
        if len(s1) >= 3:
            ps, a1, gs = s1.p_sampled.to_numpy(), s1.acc_level.to_numpy(), s1.problem_id.to_numpy()
            out["S2_sampled_level"] = decide("S2", lambda r: R.spearman(ps[r], a1[r]), gs,
                                             len(s1), THRESH["S2_sampled_level"], reps)
            out["sampled_vs_scored_spearman"] = R.spearman(ps, s1.p_forced.to_numpy())
            out["sample_other_rate_mean"] = float(s1.other_rate.mean())
            out["sample_nonanswer_rate_mean"] = float(s1.nonanswer_rate.mean())
        wide = sp.pivot_table(index=["trajectory_id", "fork_idx"], columns="pass",
                              values="p_sampled")
        if set(SAMPLE_PASSES) <= set(wide.columns):
            wide = wide.dropna()
            out["R2_sampled_seed_agreement"] = {
                "icc_2_1": R.icc_2_1(wide[SAMPLE_PASSES[0]], wide[SAMPLE_PASSES[1]]),
                "spearman": R.spearman(wide[SAMPLE_PASSES[0]], wide[SAMPLE_PASSES[1]]),
                "forks": len(wide)}
    # sentence-level change metrics
    s = sents.dropna(subset=["dp_forced"]).reset_index(drop=True) if "dp_forced" in sents \
        else sents.iloc[0:0]
    out["sentences"] = len(s)
    if len(s) >= 3:
        gs = s.problem_id.to_numpy()
        dp, ftv = s.dp_forced.to_numpy(float), s.forced_tv.to_numpy(float)
        sd, tvd = s.signed_dacc.to_numpy(float), s.tv_debiased.to_numpy(float)
        fork_tv, floor = s.pooled_fork_tv.to_numpy(float), s.floor.to_numpy(float)
        out["S4_signed_change"] = decide("S4", lambda r: R.spearman(dp[r], sd[r]), gs, len(s),
                                         THRESH["S4_signed_change"], reps)
        out["S5_tv_debiased"] = decide("S5", lambda r: R.spearman(ftv[r], tvd[r]), gs, len(s),
                                       THRESH["S5_tv_debiased"], reps)
        out["S6_partial_fork_tv_given_floor"] = decide(
            "S6", lambda r: partial_spearman(ftv[r], fork_tv[r], floor[r]), gs, len(s),
            THRESH["S6_partial_fork_tv_given_floor"], reps)
        out["change_descriptive"] = {
            "forced_tv_vs_fork_tv": R.spearman(ftv, fork_tv),
            "abs_dp_vs_abs_dacc": R.spearman(np.abs(dp), np.abs(sd)),
            "forced_tv_vs_floor": R.spearman(ftv, floor),
            "within_dp_vs_signed_dacc_median": float(np.nanmedian(
                within_values(dp, sd, s.trajectory_id.to_numpy())))}
        if "real_effect" in s and s.real_effect.notna().all():
            y = s.real_effect.to_numpy(bool)
            out["real_effects"] = int(y.sum())
            out["ap_prevalence"] = float(y.mean())
            out["S7_ap_real_effects"] = decide("S7", lambda r: average_precision(y[r], ftv[r]),
                                               gs, len(s), THRESH["S7_ap_real_effects"], reps)
            out["ap_abs_dp_forced"] = average_precision(y, np.abs(dp))
            base = {"position_sentence_idx": s.sentence_idx.to_numpy(float)}
            if baselines is not None:
                for b in BASELINES:
                    if b in s:
                        v = s[b].to_numpy(float)
                        base[b] = np.abs(v) if b.startswith("valueprobe") else v
            comp = {}
            for name, v in base.items():
                if not np.isfinite(v).all():
                    continue
                comp[name] = {
                    "ap": average_precision(y, v),
                    "spearman_vs_pooled_fork_tv": R.spearman(v, fork_tv),
                    "ap_diff_forced_minus_baseline": decide(
                        f"H4ap:{name}", lambda r, v=v: average_precision(y[r], ftv[r])
                        - average_precision(y[r], v), gs, len(s), 0.0, reps)}
            out["H4_change_vs_baselines"] = comp
    return out


def classify(m: dict, gates: dict) -> dict:
    """The plan §8 conclusion table, applied mechanically."""
    h1 = m["H1_primary"]
    change = [m[k]["decision"] for k in ("S4_signed_change", "S7_ap_real_effects")
              if k in m]
    if not all(gates.values()):
        return {"conclusion": "INCONCLUSIVE_QUALITY_GATE", "failed_gates":
                [k for k, v in gates.items() if not v]}
    if h1 == "PASS" and "PASS" in change:
        c = "READOUT_VALID_AND_CARRIES_CHANGE"
    elif h1 == "PASS" and change and all(x == "FAIL" for x in change):
        c = "READOUT_TRACKS_LEVEL_ONLY"
    elif h1 == "PASS":
        c = "READOUT_TRACKS_LEVEL_CHANGE_INCONCLUSIVE"
    elif h1 == "FAIL":
        c = "READOUT_NOT_REPRESENTATIVE"
    else:
        c = "INCONCLUSIVE"
    return {"conclusion": c, "H1": h1, "change_decisions": change}


def score(args) -> None:
    rows, forced = load_build(args.build)
    cands = json.loads((args.build / "candidates.json").read_text())["candidates"]
    targets = pd.read_parquet(args.build / "targets.parquet")
    resp = load_responses(rows, forced, args.store)
    plan = json.loads((STUDY / "configs/phase2_forks.json").read_text())
    units = [(t["trajectory_id"], t["problem_id"], i) for t in plan["trajectories"]
             for i in t["sentence_labels"]]
    effects = pd.read_csv(RESULTS / "s13_sentence_effects.csv")
    oof = pd.read_parquet(RESULTS / "s14_oof_predictions.parquet")
    mem = pd.read_csv(RESULTS / "s08_memorisation.csv")
    nr_acc = dict(zip(mem.problem_id, mem.no_reasoning_accuracy, strict=True))
    sc = resp[resp.kind == "score"]
    completeness = {f"{k}|{s}|{st}": float(g.present.mean()) for (k, s, st), g in
                    resp.groupby(["kind", "suffix", "store"])}
    span_mismatch = float((~sc[sc.present].span_ok.astype(bool)).mean()) if sc.present.any() \
        else float("nan")
    result: dict = {"stage": "exploratory_pilot", "preregistration":
                    "plan_v3_forced_answer_pilot.md", "completeness": completeness,
                    "span_mismatch_rate": span_mismatch}

    def analyse(which: str, store: str, labels: str, drop: int | None = None
                ) -> tuple[dict, pd.DataFrame]:
        tg = targets[(targets.label_set == labels) & (targets.problem_id != drop)
                     ].drop(columns="label_set")
        fr = fork_readouts(resp, cands, "primary", store, which).merge(
            tg[["trajectory_id", "fork_idx", "acc_level", "entropy"]],
            on=["trajectory_id", "fork_idx"])
        fr["problem_baseline"] = fr.problem_id.map(nr_acc)
        sents = sentence_table(fr, tg, units, effects)
        sents["sentence_id"] = sents.trajectory_id + ":" + sents.sentence_idx.astype(str)
        sents = sents.merge(oof[["sentence_id"] + [b for b in BASELINES if b in oof]],
                            on="sentence_id", how="left")
        sampled = sample_readouts(resp, cands, which)
        return compute_metrics(fr, sents, sampled=sampled, baselines=oof, reps=args.reps), fr

    primary, fr = analyse("out", "main", PRIMARY_LABELS)
    result["primary_labels"] = PRIMARY_LABELS
    result["primary_candidates_out_of_window"] = primary
    result["sensitivity_candidates_in_window"] = analyse("in", "main", PRIMARY_LABELS)[0]
    result["sensitivity_labels_r012_pool"] = analyse("out", "main", "r012")[0]
    result["sensitivity_without_1591"] = analyse("out", "main", PRIMARY_LABELS, 1591)[0]
    result["real_effects_caveat"] = ("real-effect flags come from s13 on the 300-sample "
                                     "pool including replicate 0; secondary target only")
    # R1: determinism of option (a)
    rp = fork_readouts(resp, cands, "primary", "replay", "out")
    j = fr.merge(rp, on=["trajectory_id", "fork_idx"], suffixes=("", "_replay"))
    r1 = {"forks": len(j)}
    if len(j) >= 3:
        r1.update(spearman=R.spearman(j.p_forced, j.p_forced_replay),
                  max_abs_diff=float(np.abs(j.p_forced - j.p_forced_replay).max()))
    result["R1_scoring_determinism"] = r1
    # format sensitivity on the fixed 20-fork subset
    fmt = {}
    for alt in ("alt1", "alt2"):
        a = fork_readouts(resp, cands, alt, "main", "out")
        jj = fr.merge(a, on=["trajectory_id", "fork_idx"], suffixes=("", "_alt"))
        fmt[alt] = {"forks": len(jj), "spearman": R.spearman(jj.p_forced, jj.p_forced_alt)
                    if len(jj) >= 3 else float("nan"),
                    "mean_abs_diff": float(np.abs(jj.p_forced - jj.p_forced_alt).mean())
                    if len(jj) else float("nan")}
    result["format_sensitivity"] = fmt
    result["candidate_mass_median"] = float(fr.mass.median()) if len(fr) else float("nan")
    gates = {
        "Q1_span_mismatch": bool(np.isfinite(span_mismatch)
                                 and span_mismatch <= GATES["span_mismatch_max"]),
        "Q2_candidate_mass": bool(result["candidate_mass_median"] >= GATES["median_mass_min"]),
        "Q3_completeness": bool(completeness.get("score|primary|main", 0)
                                >= GATES["completeness_min"]),
    }
    result["quality_gates"] = gates
    result["advisory"] = {
        "R1_deterministic": bool(r1.get("spearman", 0) >= GATES["replay_spearman_min"]
                                 and r1.get("max_abs_diff", 1) <= GATES["replay_maxabs_max"]),
        "format_robust": all(v["spearman"] >= GATES["format_spearman_min"]
                             for v in fmt.values() if np.isfinite(v["spearman"])),
        "sample_other_ok": primary.get("sample_other_rate_mean", 1) <= GATES["sample_other_max"],
    }
    result["conclusion"] = classify(primary, gates)
    write_json(args.out, result)
    print(json.dumps({"conclusion": result["conclusion"], "gates": gates,
                      "advisory": result["advisory"]}, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-requests")
    b.add_argument("--inputs", type=Path, default=PHASE2_INPUTS)
    b.add_argument("--index", type=Path, default=PHASE2_INDEX)
    b.add_argument("--k", type=int, default=K_SAMPLES)
    b.add_argument("--tokenizer-check", action="store_true",
                   help="verify token boundaries with the local tokenizer (CPU)")
    r = sub.add_parser("run")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--build", type=Path, default=BUILD)
    r.add_argument("--store", type=Path, default=STORE)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--rate", type=float, default=2.49, help="$/h")
    r.add_argument("--max-cost", type=float, required=True, help="hard $ guard (plan: 25)")
    r.add_argument("--max-hours", type=float, required=True, help="hard runtime guard")
    r.add_argument("--spent-hours", type=float, default=0.0,
                   help="instance hours already used (setup), counted against the guard")
    r.add_argument("--retries", type=int, default=4)
    r.add_argument("--only", default="", help="comma list: score, sample, or kind:suffix:store")
    r.add_argument("--limit", type=int, default=0, help="send only the first N (smoke)")
    s = sub.add_parser("score")
    s.add_argument("--build", type=Path, default=BUILD)
    s.add_argument("--store", type=Path, default=STORE)
    s.add_argument("--reps", type=int, default=REPS)
    s.add_argument("--out", type=Path, default=RESULTS / "plan_v3_pilot.json")
    args = ap.parse_args()
    {"build-requests": build_requests, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
