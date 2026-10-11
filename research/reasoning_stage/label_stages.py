# ruff: noqa: E501
"""Assign rule-based reasoning-stage labels to every sentence.

Exploratory post-hoc analysis (not part of the frozen MVP pipeline).

Taxonomy (7 stages, ordered rule priority):
  conclude  - states the final answer
  backtrack - doubt, error detection, reconsideration ("Wait...", "that can't be right")
  verify    - deliberate checking/confirmation of a prior step
  plan      - strategy selection / intent ("Let me try...", "Alternatively...", formulas)
  orient    - restating / understanding the problem
  compute   - default for sentences carrying mathematical content
  other     - fragments and connective tissue with no signal

Rule order resolves overlaps: conclude wins over everything; a sentence with an
explicit check verb is `verify` even if it opens with "Wait"; otherwise doubt
markers win over plan/orient/compute. Marker booleans are stored so analyses can
inspect overlaps. Deterministic, no randomness, no external services.

Usage: uv run python research/reasoning_stage/label_stages.py
Writes: research/reasoning_stage/outputs/stage_labels.parquet
        research/reasoning_stage/outputs/stage_label_audit.md (30 examples/stage)
"""
import re
from pathlib import Path

import pandas as pd

SENTENCES = "/Users/cnm13ryan/preanchor-root/data/processed/stage3/sentences.parquet"
OUT = Path(__file__).parent / "outputs"

RX = {
    "conclude": r"\bfinal answer\b|\\boxed|\banswer is\b|\banswer should be\b|\banswer:\s|\bso the answer\b",
    "verify": r"\bcheck\b|\bchecking\b|\bverify\b|\bconfirm|\bdouble-?check|\bmake sure\b|\bsanity\b"
    r"|\bthat seems (right|correct|consistent|reasonable|to work|fine|okay)\b|\bthat(’|')?s correct\b"
    r"|\bseems correct\b|\bmakes sense\b|\bconsistent with\b|\bmatches\b|\bagrees with\b|^correct\b"
    r"|\bplug (it |that |them |back )?(in|into)\b|\btest (it|this|that|with)\b|^yes\b",
    "backtrack": r"^(wait|hmm|hold on|oh wait|but wait|no wait|oops|ah)\b|\bwait\b|\bhold on\b|\bmistake\b"
    r"|\bmiscalculat|\bwrong\b|\bcan(’|')?t be right\b|\bthat can(’|')?t be\b|\berror\b"
    r"|\bi think i might have\b|\blet me re(do|calculate|compute|examine|consider|think|read|evaluate)\b"
    r"|\bactually,\b|\bon second thought\b|\bre-?examine\b|\bnot right\b|\bdoesn(’|')?t seem right\b"
    r"|\bdiscrepanc|\bcontradict",
    "plan": r"\blet me try\b|\blet(’|')?s try\b|\bi(’|')?ll\b|\bi will\b|\bi need to\b|\bi should\b"
    r"|\bmaybe i can\b|\bperhaps i can\b|\bi can use\b|\bwe can use\b|\bapproach\b|\bstrategy\b|\bmethod\b"
    r"|\bformula\b|\bi remember\b|\brecall\b|\balternatively\b|\banother way\b|\binstead\b|\bthe idea is\b"
    r"|\bplan\b|\blet me think\b|\bhow (do|can|should) (i|we)\b|\bmodel this\b|\bto solve\b"
    r"|\blet me (figure|start|begin|denote|define|set|assign|consider|see)\b"
    r"|\blet(’|')?s (denote|define|set|assign|consider|see|figure|start|begin)\b",
    "orient": r"\bthe problem (says|states|is|asks|gives|mentions)\b|\bis asking\b|\bwe(’|')?re looking for\b"
    r"|\bthe question\b|\bgiven that\b|\bwe are given\b|\bi(’|')?ve got this problem\b|\bthey want\b"
    r"|\bwe have to find\b|\bneed to find\b|\bso, the problem\b|\bthis problem\b|\bthe task\b",
    "compute": r"[=×÷±≈]|\\frac|\\times|\bcompute\b|\bcalculat|\bmultipl|\bsubtract"
    r"|\badd(ing)?\b|\bdivide|\bsimplif|\bsolve\b|\bsubstitut|\bexpand|\bfactor\b|\bsum\b|\bproduct\b|\d",
}
RXC = {k: re.compile(v) for k, v in RX.items()}


def classify(text: str) -> str:
    x = text.strip().lower()
    m = {k: bool(r.search(x)) for k, r in RXC.items()}
    if m["conclude"]:
        return "conclude"
    if m["backtrack"] and not m["verify"]:
        return "backtrack"
    if m["verify"]:
        return "verify"
    if m["plan"]:
        return "plan"
    if m["orient"]:
        return "orient"
    if m["compute"]:
        return "compute"
    return "other"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    s = pd.read_parquet(SENTENCES)
    s["stage"] = s["text"].map(classify)
    for k, r in RXC.items():
        s[f"marker_{k}"] = s["text"].str.strip().str.lower().map(lambda t, r=r: bool(r.search(t)))
    cols = ["sentence_id", "trajectory_id", "problem_id", "trace_kind", "sentence_idx", "stage"] + [
        f"marker_{k}" for k in RXC
    ]
    s[cols].to_parquet(OUT / "stage_labels.parquet", index=False)
    with open(OUT / "stage_label_audit.md", "w") as f:
        f.write("# Stage label audit examples (30 random per stage, seed 7)\n\n")
        f.write(s["stage"].value_counts().to_string() + "\n\n")
        for st in ["orient", "plan", "compute", "verify", "backtrack", "conclude", "other"]:
            f.write(f"## {st}\n\n")
            sub = s[s["stage"] == st]
            for t in sub["text"].sample(min(30, len(sub)), random_state=7):
                f.write("- " + t[:160].replace("\n", " ") + "\n")
            f.write("\n")
    print(s["stage"].value_counts())
    print("written:", OUT / "stage_labels.parquet")


if __name__ == "__main__":
    main()
