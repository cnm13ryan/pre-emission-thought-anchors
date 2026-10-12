# Plan v3 forced-answer pilot: results

**Status:** PILOT, preregistered in `plan_v3_forced_answer_pilot.md` v1.1 (commit `9ad6095`,
pushed before any request was sent). Source: `results/plan_v3_pilot.json`, produced by
`scripts/s18_forced_answer.py score` (code unchanged since `9ad6095`).

## Preregistered conclusion: `INCONCLUSIVE_QUALITY_GATE`

| Gate | Result |
|---|---|
| Q1 spans clean | pass (mismatch rate 0.0) |
| Q2 median candidate mass ≥ 0.50 | **FAIL (median 0.0)** |
| Q3 completeness ≥ 95% | pass (100% of every stage) |

Under §7 every decision is reported, but the conclusion is set by the failed gate. The
rule says to fix the readout mechanics and then rerun. Nothing below overrides that.

## Run

| Item | Value |
|---|---|
| Instance | 1× H100 PCIe 80 GB; pinned vLLM image `sha256:61fc8a89…`; model rev `1df8507…` |
| Requests | 1,160 main + 1,160 replay + 232 alternative-suffix scoring requests, 12,800 sampled completions; 0 failures |
| Wall time (requests) | 05:08Z to 05:23Z |
| Instance cost | about $2.3 (setup included); guard $25 |
| Teardown | vLLM stopped, `service:vllm` and `gpu:h100` released (receipts 20261011T0523Z) |

Bootstrap deviation: the full bootstrap failed on one unrelated last-digit float-equality
report test, the same failure as on the previous instance. The owner approved `--quick`.

## Decisions as reported

All rows use r12 labels and out-of-window candidates (primary).

| Statistic | Estimate [95% problem-cluster CI] | Threshold | Decision |
|---|---|---|---|
| H1a p_forced vs accuracy, across forks | 0.096 [−0.468, 0.641] | 0.60 | INCONCLUSIVE |
| H1b within-trajectory median | 0.210 [−0.052, 0.567] | 0.30 | INCONCLUSIVE |
| S2 sampled readout, level | 0.097 [−0.253, 0.387] | 0.60 | FAIL |
| S3 forced entropy vs resampled entropy | 0.035 [−0.405, 0.416] | 0.50 | FAIL |
| S4 Δp_forced vs signed Δacc | −0.088 [−0.211, 0.028] | 0.30 | FAIL |
| S5 forced TV vs debiased TV | 0.066 [−0.076, 0.229] | 0.20 | INCONCLUSIVE |
| S6 partial correlation with fork_tv given the floor | 0.054 [−0.071, 0.220] | 0.10 | INCONCLUSIVE |
| S7 AP for the 7 real effects | 0.210 [0.008, 0.330] | 0.25 | INCONCLUSIVE |

Per-trajectory H1b Spearman (primary) runs from −0.54 (6596) to +0.65 (330).

| Sensitivity | H1a | H1b |
|---|---|---|
| In-window candidates (circular; does not count) | 0.653 [0.265, 0.792] | 0.283 |
| r012 labels | 0.074 | 0.151 |
| Without problem 1591 | 0.175 | 0.215 |

### Controls

| Control | Result | Flag |
|---|---|---|
| R1 scoring determinism, main vs replay | Spearman 1.00, max \|Δ\| 0.051 | fails the 0.02 rule; batch non-determinism in bf16 |
| Format robustness | Spearman 0.95 and 0.91 | ok |
| Sampled readout across seeds | ICC 0.95, Spearman 0.87 | stable |
| Sampled answers outside the candidate set (`<OTHER>`) | **78%** on average | non-answers 0.5% |

## Diagnosis of the gate failure (EXPLORATORY, after the data)

The failure is not a mechanics bug in span scoring:

- **Spans are clean.** Q1 is 0.0, and every span is on a token boundary.
- **Candidates get almost no probability.** The median span log-probability is −16, with
  a 5–95% range of −33 to −7.
- **The model is answering something else.** The *candidate-free* sampled readout
  produces well-formed boxed answers 99.5% of the time, but 78% of them are not among the
  answers that continued reasoning reaches. When forced to answer mid-trace, the model
  mostly states a different, premature guess, for example small integers or partial
  results, sometimes followed by a fresh "Step-by-Step Explanation".
- **The candidate-free readout fails too.** It is stable across seeds (ICC 0.95) and does
  not depend on Q2, yet it still FAILs to track the resampled accuracy level (S2) and the
  answer entropy (S3).

Interpretation (exploratory): the forced "answer now" distribution of this distilled
reasoning model is mostly *not* the distribution of answers it reaches by continuing to
reason. In this form the forced-answer readout does not represent the outcome. The gate
did its job: it flagged that the scored candidates do not cover the model's forced
distribution. The candidate-free secondary readout supports the substantive reading,
which is close to the plan's `READOUT_NOT_REPRESENTATIVE`. The formal conclusion still
stays `INCONCLUSIVE_QUALITY_GATE`.

## What this does and does not show

- **It does not show** that no outcome readout exists. Only this suffix and this
  temperature-1 scoring were tested, on 10 trajectories.
- **It does show** that a naive forced-answer probe is a poor proxy for the resampled
  outcome here. Any future readout needs its own coverage check, like Q2, and a
  candidate-free control.
