"""Cost model for docs/v3_followup_gpu_plan.md (EXPLORATORY planning aid; nothing is run).

Tokens per arm = requests x generated tokens per request. GPU-hours = tokens / throughput.
Two throughput scenarios: the measured Phase 2 rate (1,075 tok/s realised over 168.4M tokens,
43.5 h, prefix caching on, 16 concurrent per endpoint) and a target rate that the dry run must
confirm (higher --max-num-seqs and n=K requests). Pilot per-request cost is NOT used: pilot
requests were short forced answers.

Usage: python analysis/v3_followup/gpu_plan_cost/cost.py [--K 10 --M 20 ...] > cost.json
"""

from __future__ import annotations

import argparse
import json

RATE = 2.49                      # $/h, 1x H100 PCIe on-demand list price (docs/compute_budget.md)
TPS = {"measured_phase2": 1075.0, "target_dry_run": 2500.0}
L_FULL = 4242                    # mean generated tokens per full continuation (Phase 2, 40k rows)
L_SENT = 40                      # tokens per sampled sentence (stop at sentence boundary)


def arms(a) -> dict:
    pre_conv_forks = a.forks_screened
    nested_forks = a.N * a.F
    anchors = round(nested_forks * a.anchor_rate)
    out = {
        "B0_problem_screen": {"requests": a.candidates * a.base_samples,
                              "tokens": a.candidates * a.base_samples * L_FULL},
        "B1_fork_screen": {"requests": a.N * pre_conv_forks * a.screen_n,
                           "tokens": a.N * pre_conv_forks * a.screen_n * L_FULL},
        "A1_nested_sentences": {"requests": nested_forks * a.K,
                                "tokens": nested_forks * a.K * L_SENT},
        "A1_nested_continuations": {"requests": nested_forks * (a.K + 1) * a.M,
                                    "tokens": nested_forks * (a.K + 1) * a.M * L_FULL},
        "A2_token_level": {"requests": anchors * a.a2_positions * a.a2_n,
                           "tokens": anchors * a.a2_positions * a.a2_n * L_FULL},
        "A3_sensitivity": {"requests": anchors * (1 + a.a3_controls) * a.a3_conditions * a.a3_n,
                           "tokens": anchors * (1 + a.a3_controls) * a.a3_conditions * a.a3_n
                           * L_FULL},
        "A4_short_budget": {"requests": nested_forks * a.a4_n,
                            "tokens": nested_forks * a.a4_n * (a.a4_budget + 30)},
        "A5_released_settings": {"requests": a.a5_forks * a.a5_n,
                                 "tokens": a.a5_forks * a.a5_n * L_FULL},
        "A6_second_source": {"requests": a.a6_problems * (pre_conv_forks * a.screen_n
                                                          + a.a6_forks * (a.a6_K + 1) * a.a6_M),
                             "tokens": a.a6_problems * (pre_conv_forks * a.screen_n
                                                        + a.a6_forks * (a.a6_K + 1) * a.a6_M)
                             * L_FULL},
    }
    for v in out.values():
        v["gpu_h"] = {k: v["tokens"] / tps / 3600 for k, tps in TPS.items()}
        v["usd"] = {k: h * RATE for k, h in v["gpu_h"].items()}
    out["F_forward_passes"] = {"requests": None, "tokens": None,
                               "gpu_h": {k: a.forward_h for k in TPS},
                               "usd": {k: a.forward_h * RATE for k in TPS},
                               "note": "teacher-forced HF passes; estimate, measured in dry run"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)            # T7
    ap.add_argument("--F", type=int, default=20)            # forks per trajectory (T7)
    ap.add_argument("--K", type=int, default=10)            # PENDING T6
    ap.add_argument("--M", type=int, default=20)            # PENDING T6
    ap.add_argument("--candidates", type=int, default=120)
    ap.add_argument("--base-samples", type=int, default=8)
    ap.add_argument("--forks-screened", type=int, default=110)   # pre-convergence forks / trace
    ap.add_argument("--screen-n", type=int, default=8)
    ap.add_argument("--anchor-rate", type=float, default=0.036)  # T3 deployable criterion
    ap.add_argument("--a2-positions", type=int, default=2)
    ap.add_argument("--a2-n", type=int, default=100)
    ap.add_argument("--a3-controls", type=int, default=2)        # matched non-anchors per anchor
    ap.add_argument("--a3-conditions", type=int, default=5)      # 3x2 minus resample@0.6 (=A1)
    ap.add_argument("--a3-n", type=int, default=100)
    ap.add_argument("--a4-n", type=int, default=32)
    ap.add_argument("--a4-budget", type=int, default=256)
    ap.add_argument("--a5-forks", type=int, default=80)
    ap.add_argument("--a5-n", type=int, default=100)
    ap.add_argument("--a6-problems", type=int, default=10)
    ap.add_argument("--a6-forks", type=int, default=10)
    ap.add_argument("--a6-K", type=int, default=5)
    ap.add_argument("--a6-M", type=int, default=20)
    ap.add_argument("--forward-h", type=float, default=6.0)
    a = ap.parse_args()
    out = arms(a)
    tot = {k: sum(v["gpu_h"][k] for v in out.values()) for k in TPS}
    print(json.dumps({"params": vars(a), "rate_usd_per_h": RATE, "tokens_per_s": TPS,
                      "mean_tokens_full_continuation": L_FULL, "arms": out,
                      "total_gpu_h": tot, "total_usd": {k: h * RATE for k, h in tot.items()},
                      "total_tokens": sum(v["tokens"] or 0 for v in out.values())},
                     indent=1))


if __name__ == "__main__":
    main()
