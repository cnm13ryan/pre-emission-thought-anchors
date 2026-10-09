# Reliability diagnostics (existing released rollouts)

**Status:** EXPLORATORY. These are the 20 MVP problems, all previously seen. Sources:
`results/s05_reliability.json`, `results/s05_within_problem.csv`, `results/s06_power.csv`,
`results/s07_validity.json`. Scripts: `s04_scores.py`, `s05_reliability.py`,
`s06_simulate.py`.

## 1. KL smoothing instability (confirmed)

- The MVP score is KL(without ‖ with) with α = 1e-9. A category absent from the "with"
  side gets probability ≈ α/N, so an answer seen in 6% of "without" samples contributes
  ≈ 0.06 · ln(0.06 · 10¹¹) ≈ 1.35 nats on its own. That is the MVP anchor threshold
  (`tests/test_core.py::test_kl_reproduces_mvp_rare_answer_blowup`).
- On the MVP anchors, 96% draw at least half their KL from such absent answers
  (Phase 1 audit, reproduced exactly here).
- Bounded scores remove this. With TV, an absent answer contributes at most its own mass,
  0.06 in the example (test `test_tv_jsd_bounds_and_identities`).

## 2. Split-half reliability

Each fork's 100 released samples were split in half with a seed. The 4,750 eligible
sentences have a defined score in both halves for 3,989 sentences (4,739 for `fork_tv`).
Intervals are 95% problem-cluster bootstrap, 2,000 replicates.

| Score | Spearman | ICC(2,1) | ICC → full N (Spearman–Brown) | κ at q90 | Within-problem Spearman, median [min, max] |
|---|---|---|---:|---|---|
| `fork_tv` | 0.557 [0.435, 0.659] | **0.615** [0.515, 0.676] | 0.76 | 0.366 [0.275, 0.420] | 0.44 [0.21, 0.78] |
| `tv` | 0.587 [0.490, 0.667] | 0.557 [0.477, 0.621] | 0.72 | 0.389 [0.319, 0.457] | 0.53 [0.33, 0.76] |
| `jsd` | 0.592 [0.455, 0.685] | 0.526 [0.418, 0.607] | 0.69 | 0.395 [0.326, 0.445] | 0.50 [0.24, 0.75] |
| `dacc` | 0.352 [0.307, 0.404] | 0.394 [0.348, 0.438] | 0.57 | 0.282 [0.236, 0.325] | 0.36 [0.19, 0.51] |
| `kl_1e-9` (MVP, log1p) | 0.468 [0.355, 0.554] | 0.428 [0.302, 0.510] | 0.60 | 0.235 [0.175, 0.269] | 0.32 [0.12, 0.55] |

- Every statistic is **FAIL** under the three-way rule at half N: each interval's upper
  bound is below its threshold.
- The Spearman–Brown projections to N = 100 are approximate. They are not decisions.

## 3. Threshold sensitivity

κ of the flags by training-quantile threshold. Each half uses its own training-split
quantile.

| Score | q80 | q90 | q95 |
|---|---:|---:|---:|
| `fork_tv` | 0.439 | 0.366 | 0.320 |
| `tv` | 0.394 | 0.389 | 0.318 |
| `jsd` | 0.453 | 0.395 | 0.417 |
| `dacc` | 0.272 | 0.282 | 0.262 |
| `kl_1e-9` | 0.331 | 0.235 | 0.177 |

Agreement falls as the threshold rises. The rarer the flag, the less reproducible it is.
At q90, prevalence is 0.088–0.100 in each half.

## 4. Finite-sample noise

- For each sentence, the TV expected when there is **no** true effect was computed: two
  independent samples of the observed sizes from the pooled distribution, 300 draws. Its
  median is **0.065**. The median observed TV is 0.071.
- **Only 54.5% of eligible sentences have TV above their own noise floor.** Most measured
  importance is resampling noise.

## 5. Phase 2 power and the minimum usable number of resamples

Two independent replicates of n resamples per fork were simulated for the 190 planned
sentences, with truth taken as the released rows. This is optimistic. 200 simulations per
cell, 300-replicate cluster intervals. A form PASSes only if both its ICC and Spearman
PASS.

| n | `fork_tv` ICC / Spearman | P(PASS / FAIL / INC) | `jsd` ICC / Spearman | P(PASS / FAIL / INC) |
|---:|---|---|---|---|
| 25 | 0.48 / 0.44 | 0 / 1.00 / 0 | 0.40 / 0.48 | 0 / 1.00 / 0 |
| 50 | 0.60 / 0.52 | 0 / 1.00 / 0 | 0.50 / 0.59 | 0 / 0.995 / 0.005 |
| 100 | 0.73 / 0.62 | 0 / 0.98 / 0.02 | 0.64 / 0.72 | 0 / 0.655 / 0.345 |
| 200 | 0.83 / 0.71 | 0 / 0.25 / 0.75 | 0.79 / 0.83 | 0.025 / 0.005 / 0.97 |
| 400 | 0.90 / 0.79 | 0 / 0 / 1.00 | 0.88 / 0.91 | **0.74** / 0 / 0.26 |

**Minimum usable resamples**, meaning expected ICC ≥ 0.75:
- ≈ 200 per fork for `fork_tv`, `tv` and `jsd`;
- ≈ 200–400 for `dacc`.

Meeting **both** preregistered criteria with high probability needs ≈ 400, and then only
for `jsd`. Spearman lags ICC for TV-type scores because many sentences tie near zero.

## 6. What this establishes, and what it does not

- **Confirmed:** at the released budget (100 resamples, about 50 dissimilar), no candidate
  score, MVP KL included, reproduces across independent halves at the preregistered
  levels. The MVP KL is the least reliable of all of them.
- **Assumption:** half-sample results extrapolate to full N by Spearman–Brown, and the
  simulation's truth is the empirical distribution. Phase 2's replicate data test both.
- **Proposed change:** when the score is used as a target, use ≥ 200 resamples per fork,
  and prefer `jsd` or `fork_tv` to KL.
