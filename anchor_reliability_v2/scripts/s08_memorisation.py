"""Memorisation / contamination check for every MVP problem (plan §10).

1. No-reasoning accuracy: the stored Thought Anchors prompt, with the think block closed
   immediately and ``\\boxed{`` forced. 20 samples at temperature 0.6 (top_p 0.95),
   max 32 tokens; the answer is the text up to the first closing brace, scored with the v2
   normaliser.
2. Verbatim continuation: raw problem text (no instruction) cut at 50% of its words; greedy
   continuation of 64 tokens. Reports exact match of the next 10 words and the normalised
   edit similarity of the continuation to the true remainder (same length in characters).

Flag rule (FIXED): no-reasoning accuracy >= 0.5, or next-10-words exact match.
Run on a machine that can reach the vLLM endpoint (e.g. the GPU instance, or via an SSH
tunnel ``-L 8000:127.0.0.1:8000``).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.answers import normalize_v2  # noqa: E402
from arv2.paths import DATA, INPUTS, RESULTS, SEED_ROOT, write_json  # noqa: E402
from arv2.vllm_client import Client, sample_seed  # noqa: E402

SAMPLES = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    client = Client(args.endpoint, DATA / "generations/memorisation")
    trajs = pd.read_parquet(INPUTS / "trajectories.parquet")
    probs = trajs.drop_duplicates("problem_id")[["problem_id", "problem_text",
                                                 "ground_truth_answer"]]
    correct_prompt = trajs[trajs.trace_kind == "correct"].set_index("problem_id").prompt
    rows = []
    for p in probs.itertuples():
        prompt = correct_prompt[p.problem_id]
        forced = prompt + "\n</think>\n\nThe final answer is \\boxed{"
        gt = normalize_v2(p.ground_truth_answer)
        answers = []
        for s in range(SAMPLES):
            out = client.complete(forced, max_tokens=32, temperature=0.6, top_p=0.95,
                                  seed=sample_seed(SEED_ROOT, "memorisation", p.problem_id, s))
            text = out["text"]
            depth, buf = 1, []
            for ch in text:
                depth += (ch == "{") - (ch == "}")
                if depth == 0:
                    break
                buf.append(ch)
            answers.append(normalize_v2("".join(buf)) if depth == 0 else None)
        acc = sum(a == gt for a in answers) / SAMPLES
        words = p.problem_text.split()
        cut = max(1, len(words) // 2)
        head, rest = " ".join(words[:cut]), " ".join(words[cut:])
        cont = client.complete(head, max_tokens=64, temperature=0.0)["text"]
        cw, rw = cont.split(), rest.split()
        ten = min(10, len(rw))
        exact10 = ten > 0 and cw[:ten] == rw[:ten]
        ref = rest[: len(cont.strip())]
        edit_sim = difflib.SequenceMatcher(None, cont.strip(), ref).ratio() if ref else 0.0
        rows.append({"problem_id": int(p.problem_id), "ground_truth": gt,
                     "no_reasoning_accuracy": acc,
                     "no_reasoning_unparsed": sum(a is None for a in answers),
                     "no_reasoning_modal_answer": pd.Series(answers).mode().iloc[0]
                     if any(a is not None for a in answers) else None,
                     "words_in_problem": len(words), "next10_exact": bool(exact10),
                     "next_words_compared": ten, "continuation_edit_similarity": edit_sim,
                     "continuation_preview": cont[:120],
                     "flagged": bool(acc >= 0.5 or exact10)})
        print(rows[-1]["problem_id"], acc, exact10, round(edit_sim, 2), flush=True)
    table = pd.DataFrame(rows).sort_values("problem_id")
    table.to_csv(RESULTS / "s08_memorisation.csv", index=False)
    write_json(RESULTS / "s08_memorisation.json", {
        "stage": "exploratory", "samples_no_reasoning": SAMPLES,
        "flag_rule": "no_reasoning_accuracy >= 0.5 or next10_exact",
        "flagged": table[table.flagged].problem_id.tolist(), "rows": rows})
    print(table.drop(columns="continuation_preview").to_string())


if __name__ == "__main__":
    main()
