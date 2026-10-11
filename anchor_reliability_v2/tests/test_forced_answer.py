"""Plan v3 pilot (s18): prompt construction, candidates, logprob aggregation, the durable
run loop against a MOCK endpoint, and the scorer on synthetic data with a known truth."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts/s18_forced_answer.py")
s18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s18)

PROMPT = ("Solve this math problem step by step. You MUST put your final answer in \\boxed{}. "
          "Problem: 2+2? \n Solution: \n<think>\nOkay, two plus two. ")

# ------------------------------------------------------------------ prompt construction


def test_suffix_appended_once_and_think_closed_once():
    forced = s18.forced_prompt(PROMPT)
    assert forced.count("</think>") == 1 and forced.count("<think>") == 1
    assert forced.endswith(s18.SUFFIXES["primary"][1])
    assert forced.startswith(PROMPT.rstrip())
    assert "two. \n</think>" not in forced and "two.\n</think>" in forced   # stripped junction
    with pytest.raises(ValueError):                 # a forced prompt cannot be forced again
        s18.forced_prompt(forced)
    with pytest.raises(ValueError):
        s18.forced_prompt("no think block")
    alt1 = s18.forced_prompt(PROMPT, "alt1")
    assert alt1 == PROMPT + "\n</think>\n\nThe final answer is \\boxed"      # s08 style, raw
    for name in s18.SUFFIXES:
        assert s18.forced_prompt(PROMPT, name).count("</think>") == 1


def test_forced_prefix_ends_at_boxed_and_continuation_carries_braces():
    forced = s18.forced_prompt(PROMPT)
    assert forced.endswith("\\boxed") and not forced.endswith("{")
    p = s18.score_payload(forced, "\\frac{1}{2}")
    assert p["prompt"] == forced + "{\\frac{1}{2}}"
    assert p["echo"] is True and p["max_tokens"] == 1 and p["temperature"] == 0.0
    q = s18.sample_payload(forced, 7)
    assert q["prompt"] == forced and q["seed"] == 7 and q["max_tokens"] == 30
    assert (q["temperature"], q["top_p"]) == (0.6, 0.95)


# ------------------------------------------------------------------ candidates


def test_candidate_set_rules():
    counts = {"3": 600, "4": 300, "<TRUNCATED>": 50, "<NON_ANSWER>": 40, "7": 2, "9": 8}
    surfaces = {"3": {"3": 500, "3.0": 90, "03": 10}, "4": {"4": 300}, "9": {"9": 8},
                "x": {"x": 5}}
    c = s18.candidate_set(counts, surfaces, gt_cat="5", gt_raw="5")
    cats = [x["category"] for x in c]
    assert cats == ["3", "4", "9", "5"]           # by count, share >= 0.5%, then GT appended
    assert "<TRUNCATED>" not in cats and "7" not in cats   # 2/1000 < 0.5%
    by = {x["category"]: x for x in c}
    assert by["3"]["surfaces"] == ["3", "3.0"]    # top MAX_SURFACES that normalise to "3"
    assert by["5"]["is_gt"] and by["5"]["surfaces"] == ["5"] and by["5"]["share"] == 0
    g = s18.candidate_set({"\\frac12": 10}, {"\\frac12": {"\\dfrac{1}{2}": 9}},
                          gt_cat="\\frac12", gt_raw="\\frac{1}{2}")
    assert g[0]["surfaces"] == ["\\frac{1}{2}", "\\dfrac{1}{2}"]   # reference string first
    many = {str(i): 100 - i for i in range(30)}
    assert len(s18.candidate_set(many, {}, "0", "0")) == s18.MAX_CATS


def test_surface_extraction_matches_category():
    assert s18.surface_of(np.array(["2", "\\dfrac{1}{2}"]), "\\frac12") == "\\dfrac{1}{2}"
    assert s18.surface_of(np.array(["5"]), "6") is None
    assert s18.surface_from_text("x \\boxed{3} </think> so \\boxed{1,000}") == ("1,000", "1000")
    assert s18.surface_from_text("thinking only") == (None, None)


# ------------------------------------------------------------------ logprob aggregation


def test_span_logprob_excludes_generated_token_and_flags_straddle():
    prefix = "abc\\boxed"
    text = prefix + "{12}"
    toks = ["abc", "\\", "boxed", "{", "1", "2", "}", "\n"]          # last = generated
    offs = [0, 3, 4, 9, 10, 11, 12, 13]
    lps = [None, -1.0, -0.5, -0.1, -0.2, -0.3, -0.4, -9.0]
    sp = s18.span_logprob(toks, lps, offs, len(prefix), len(text))
    assert math.isclose(sp["logprob"], -1.0) and sp["text"] == "{12}" and not sp["straddle"]
    merged = ["abc", "\\", "box", "ed{", "1", "2", "}"]
    sp2 = s18.span_logprob(merged, [None, -1, -1, -0.3, -0.2, -0.3, -0.4],
                           [0, 3, 4, 7, 10, 11, 12], len(prefix), len(text))
    assert sp2["straddle"] and sp2["text"] == "ed{12}"


def test_category_distribution_sums_surfaces_and_normalises():
    lp = {"a": [math.log(0.2), math.log(0.1)], "b": [math.log(0.1)],
          "c": [float("-inf")]}
    dist, mass = s18.category_distribution(lp)
    assert math.isclose(sum(dist.values()), 1.0)
    assert math.isclose(dist["a"], 0.75) and math.isclose(dist["b"], 0.25)
    assert dist["c"] == 0.0 and math.isclose(mass, 0.4)
    assert math.isclose(s18.entropy_bits([0.5, 0.5]), 1.0)
    assert math.isclose(s18.tv_dist({"a": 1.0}, {"b": 1.0}), 1.0)


def test_parse_forced_sample():
    assert s18.parse_forced_sample("{\\dfrac{1}{2}}\n\\]") == "\\frac12"
    assert s18.parse_forced_sample("{1,000}") == "1000"
    assert s18.parse_forced_sample("{12") == s18.A.NON_ANSWER          # unterminated at cap
    assert s18.parse_forced_sample("Wait, let me") == s18.A.NON_ANSWER
    assert s18.parse_forced_sample(None) == s18.A.NON_ANSWER


# ------------------------------------------------------------------ mock endpoint


TRUE = {"4": 0.7, "5": 0.2}                      # planted forced-answer probabilities


class FakeVLLM:
    """OpenAI-compatible /v1/completions stand-in. Echo requests return per-character
    prompt tokens with offsets; the answer span carries log TRUE[answer]. Sampling returns
    a boxed answer chosen by the seed."""

    def __init__(self, fail_first: int = 0):
        self.calls, self.fail_first = 0, fail_first

    def __call__(self, url, payload, timeout):
        self.calls += 1
        assert url.endswith("/v1/completions") and payload["add_special_tokens"] is False
        if self.calls <= self.fail_first:
            raise ConnectionError("transient")
        prompt = payload["prompt"]
        if payload.get("echo"):
            cut = prompt.rindex("\\boxed") + len("\\boxed")
            ans = prompt[cut + 1:-1]
            body = [prompt[:cut]] + list(prompt[cut:])
            offs = [0] + list(range(cut, len(prompt)))
            share = math.log(TRUE.get(ans, 0.1)) / (len(body) - 1)
            lps = [None] + [share] * (len(body) - 1)
            return {"choices": [{"text": prompt + "\n", "finish_reason": "length",
                                 "logprobs": {"tokens": body + ["\n"],
                                              "token_logprobs": lps + [-0.1],
                                              "text_offset": offs + [len(prompt)]}}],
                    "usage": {"prompt_tokens": len(body)}}
        ans = "4" if payload["seed"] % 10 < 7 else "5"
        return {"choices": [{"text": "{" + ans + "}\n\\]", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}}


def _tiny_build():
    forks = pd.DataFrame({"trajectory_id": ["9:correct"] * 2, "problem_id": [9, 9],
                          "fork_idx": [3, 4], "prompt": [PROMPT, PROMPT + "Hmm. "]})
    c = [{"category": "4", "share": 0.8, "is_gt": True, "surfaces": ["4"]},
         {"category": "5", "share": 0.2, "is_gt": False, "surfaces": ["5"]}]
    cands = {"9:correct": {"gt_category": "4", "gt_raw": "4", "out": c, "in": c}}
    rows, prompts = s18.build_rows(forks, cands, k=4)
    forced = {p["forced_sha256"]: p["forced_prompt"] for p in prompts}
    return rows, forced, cands


def test_build_rows_deterministic_ids_and_seeds():
    rows, forced, _ = _tiny_build()
    again, _, _ = _tiny_build()
    assert [r["request_id"] for r in rows] == [r["request_id"] for r in again]
    assert len({r["request_id"] for r in rows}) == len(rows)
    seeds = [r["seed"] for r in rows if r["kind"] == "sample"]
    assert len(set(seeds)) == len(seeds) == 2 * 2 * 4          # forks × passes × k
    stores = {(r["kind"], r["store"]) for r in rows}
    assert ("score", "replay") in stores                         # determinism control
    for r in rows:
        assert s18.payload_of(r, forced[r["forced_sha256"]])["model"] == s18.MODEL


def test_run_with_mock_endpoint_store_reuse_retry_and_guard(tmp_path):
    rows, forced, cands = _tiny_build()
    fake = FakeVLLM(fail_first=1)
    st = s18.run_requests(rows, forced, tmp_path, "http://mock:8000", concurrency=1,
                          rate=2.49, max_cost=25, max_hours=10, post=fake, backoff=0.0)
    assert st["failed"] == 0 and st["sent"] == len(rows) and st["stop_reason"] is None
    again = s18.run_requests(rows, forced, tmp_path, "http://mock:8000", concurrency=4,
                             rate=2.49, max_cost=25, max_hours=10, post=fake)
    assert again["sent"] == 0 and again["reused"] == len(rows)  # never regenerated
    resp = s18.load_responses(rows, forced, tmp_path)
    assert resp.present.all() and resp[resp.kind == "score"].span_ok.all()
    fr = s18.fork_readouts(resp, cands)
    assert np.allclose(fr.p_forced, 0.7 / 0.9) and np.allclose(fr.mass, 0.9)
    rp = s18.fork_readouts(resp, cands, store="replay")
    assert np.allclose(rp.p_forced, fr.p_forced)               # deterministic scoring
    sm = s18.sample_readouts(resp, cands)
    assert set(sm["pass"]) == {"s1", "s2"} and (sm.k == 4).all()
    # cost guard: a clock already past the budget launches nothing
    clock = iter([0.0] + [3600.0 * 20] * 1000).__next__
    st2 = s18.run_requests(rows, forced, tmp_path / "other", "http://mock:8000",
                           concurrency=2, rate=2.49, max_cost=25, max_hours=100,
                           post=fake, clock=clock)
    assert st2["sent"] == 0 and st2["not_launched"] == len(rows)
    assert st2["stop_reason"].startswith("cost guard")
    dead = s18.run_requests(rows[:1], forced, tmp_path / "dead", "http://mock:8000",
                            concurrency=1, rate=2.49, max_cost=25, max_hours=10,
                            post=FakeVLLM(fail_first=99), retries=2, backoff=0.0)
    assert dead["failed"] == 1 and dead["sent"] == 0


# ------------------------------------------------------------------ scorer, known truth


def _synthetic(signal: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    forks, targets, units, effects = [], [], [], []
    for prob in range(10):
        tid = f"{prob}:correct"
        acc = np.clip(rng.uniform(0.2, 0.8) + np.cumsum(rng.normal(0, 0.08, 20)), 0.02, 0.98)
        for i in range(20):
            p = (np.clip(acc[i] + rng.normal(0, 0.03), 0, 1) if signal
                 else rng.uniform(0, 1))
            forks.append({"trajectory_id": tid, "problem_id": prob, "fork_idx": i,
                          "p_forced": p, "entropy_forced": s18.entropy_bits([p, 1 - p]),
                          "dist": {"gt": p, "other": 1 - p}, "mass": 0.9})
            n_c = int(round(acc[i] * 300))
            counts = {"gt": n_c, "other": 300 - n_c}
            targets.append({"trajectory_id": tid, "problem_id": prob, "fork_idx": i,
                            "n": 300, "acc_level": n_c / 300,
                            "entropy": s18.entropy_bits([n_c / 300, 1 - n_c / 300]),
                            "counts_json": json.dumps(counts)})
        for i in range(19):
            units.append((tid, prob, i))
            effects.append({"trajectory_id": tid, "sentence_idx": i,
                            "real_tv_bh10": abs(acc[i + 1] - acc[i]) > 0.15,
                            "real_dacc_bh10": False})
    f, t = pd.DataFrame(forks), pd.DataFrame(targets)
    f = f.merge(t[["trajectory_id", "fork_idx", "acc_level", "entropy"]],
                on=["trajectory_id", "fork_idx"])
    sents = s18.sentence_table(f, t, units, pd.DataFrame(effects))
    return f, sents


def test_scorer_recovers_planted_signal():
    f, sents = _synthetic(signal=True)
    m = s18.compute_metrics(f, sents, reps=300)
    assert m["forks"] == 200 and m["sentences"] == 190
    assert m["H1a_spearman_level"]["decision"] == "PASS"
    assert m["H1b_within_level"]["decision"] == "PASS" and m["H1_primary"] == "PASS"
    assert m["S4_signed_change"]["decision"] == "PASS"
    assert m["S3_entropy"]["estimate"] > 0.5
    assert m["S7_ap_real_effects"]["estimate"] > m["ap_prevalence"]
    gates = {"Q1": True, "Q2": True, "Q3": True}
    assert s18.classify(m, gates)["conclusion"].startswith("READOUT_")


def test_scorer_fails_on_noise():
    f, sents = _synthetic(signal=False, seed=1)
    m = s18.compute_metrics(f, sents, reps=300)
    assert m["H1a_spearman_level"]["decision"] == "FAIL" and m["H1_primary"] == "FAIL"
    assert m["S4_signed_change"]["decision"] == "FAIL"
    assert s18.classify(m, {"Q1": True})["conclusion"] == "READOUT_NOT_REPRESENTATIVE"
    assert s18.classify(m, {"Q1": False})["conclusion"] == "INCONCLUSIVE_QUALITY_GATE"


def test_partial_spearman_removes_shared_driver():
    rng = np.random.default_rng(4)
    z = rng.normal(size=400)
    x, y = z + 0.3 * rng.normal(size=400), z + 0.3 * rng.normal(size=400)
    assert s18.R.spearman(x, y) > 0.8
    assert abs(s18.partial_spearman(x, y, z)) < 0.15


def test_suffix_span_is_robust_to_offset_drift():
    # a split multi-byte character earlier in the prompt makes vLLM's offsets drift;
    # the suffix method uses usage.prompt_tokens and the token strings only
    toks = ["Let ", "�", "�", " x", "\\", "boxed", "{(", "1", ",", "2", ")}", "\n"]
    lps = [None, -0.1, -0.1, -0.2, -0.1, -0.1, -0.5, -0.25, -0.25, -0.25, -0.25, -3.0]
    sp = s18.span_logprob_suffix(toks, lps, n_prompt=11, cont="{(1,2)}")
    assert sp["text"] == "{(1,2)}" and not sp["straddle"]
    assert math.isclose(sp["logprob"], -1.5)
    row = {"kind": "score", "surface": "(1,2)", "span_start": 999, "span_end": 1006}
    body = {"choices": [{"text": "", "logprobs": {"tokens": toks, "token_logprobs": lps,
                                                  "text_offset": list(range(12))}}],
            "usage": {"prompt_tokens": 11}}
    rec = s18.response_record(row, body)
    assert rec["span_ok"] and rec["span_method"] == "suffix"
    merged = s18.span_logprob_suffix(["a", "\\boxed{", "7", "}"], [None, -1, -1, -1], 4, "{7}")
    assert merged["straddle"]                    # boundary inside a token is flagged
