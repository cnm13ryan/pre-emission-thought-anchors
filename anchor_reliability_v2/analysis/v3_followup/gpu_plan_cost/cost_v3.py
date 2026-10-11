"""Cost model for docs/v3_followup_gpu_plan.md v3 (EXPLORATORY planning aid; nothing is run).

Differences from cost.py (v1/v2): the new problems are post-cutoff competition problems whose
full traces are 2-3x longer than MATH (MathArena: 10.9k-15.6k mean output tokens for this
model), so every arm has its own tokens-per-request assumption, and the run is split into
Run 1 (screening + dry run + pilots) and Run 2 (main). Throughput and lengths are
assumptions the dry run replaces.

Usage: python analysis/v3_followup/gpu_plan_cost/cost_v3.py [--K 4] [--M 200] [--math-nested]
"""

from __future__ import annotations

import argparse
import json

L_SCREEN = 11_000     # full trace from the problem start, post-cutoff problems (MathArena ~11-16k)
L_CONT = 6_000        # continuation from a mid-trace fork of a <=12k-median trace (about half)
L_MATH = 4_242        # MATH continuation (Phase 2 measured mean)
L_PUZ = 3_000         # puzzle continuation (assumed; no published 14B trace lengths)
NODE = {"low": 8 * 1000 * 0.85, "mid": 8 * 1600 * 0.85, "high": 8 * 2200 * 0.85}  # tok/s, 8xA100
RATES = (14.32, 10.32)


def arms(a) -> tuple[dict, dict]:
    anchors = round(a.N * a.F * a.anchor_rate)
    run1 = {
        "S1_screen_16": a.pool * 16 * L_SCREEN,
        "S2_screen_32": a.survivors * 32 * L_SCREEN,
        "S3_puzzle_screen": a.puzzles * 16 * L_PUZ,
        "S4_contamination_probes": a.pool * 8 * 300,           # no-reasoning answers, short
        "P1_short_budget_pilot": 5 * 20 * 4 * 32 * 300,         # K sweep {0,64,256,1024}, mean ~300
        "D_dry_run": 2 * (20 * 8 + 2 * 5 * 200) * L_CONT,
        "C_calibration_a100": 20 * 100 * L_MATH,
    }
    run2 = {
        "B1_fork_screen": a.N * a.screen_forks * 8 * L_CONT,
        "A1_nested_core": a.core * a.F * (a.K + 1) * a.M * L_CONT,
        "A1b_labels_reliability": (a.N - a.core) * a.F * a.label_n * L_CONT,
        "A1m_math_nested_optional": (5 * a.F * (4 + 1) * a.M * L_MATH) if a.math_nested else 0,
        "A2_token_level": anchors * 2 * 100 * L_CONT,
        "A3_counterfactual": anchors * 3 * 5 * 100 * L_CONT,
        "A4_short_budget": a.N * a.F * 32 * 300,
        "A5_prompt_format_2x2": 80 * 100 * 2 * L_MATH,
        "A6_puzzles": 10 * 10 * 5 * 100 * L_PUZ + 10 * 20 * 100 * L_PUZ,
    }
    return run1, run2


def hours(tokens: int, fwd_gpu_h: float, overhead_h: float) -> dict:
    return {s: tokens / tps / 3600 + fwd_gpu_h / 8 + overhead_h for s, tps in NODE.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--core", type=int, default=20)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--label-n", type=int, default=100)     # 2 x 50 halves: split-half + BH
    ap.add_argument("--screen-forks", type=int, default=50)  # every ~4th pre-convergence boundary
    ap.add_argument("--anchor-rate", type=float, default=0.036)
    ap.add_argument("--pool", type=int, default=250)
    ap.add_argument("--survivors", type=int, default=100)
    ap.add_argument("--puzzles", type=int, default=200)
    ap.add_argument("--math-nested", action="store_true")
    ap.add_argument("--fwd-gpu-h", type=float, default=12.0)
    a = ap.parse_args()
    r1, r2 = arms(a)
    t1, t2 = sum(r1.values()), sum(r2.values())
    h1, h2 = hours(t1, 1.0, 2.0), hours(t2, a.fwd_gpu_h, 1.5)
    out = {"params": vars(a), "assumptions": {"L_screen": L_SCREEN, "L_cont": L_CONT,
           "L_math": L_MATH, "L_puzzle": L_PUZ, "node_tokens_per_s": NODE},
           "run1_tokens": r1, "run2_tokens": r2, "run1_total": t1, "run2_total": t2,
           "run1_hours": h1, "run2_hours": h2,
           "run1_usd": {f"{s}@{r}": h * r for s, h in h1.items() for r in RATES},
           "run2_usd": {f"{s}@{r}": h * r for s, h in h2.items() for r in RATES}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
