# T2 (item 1): split-half reliability of resampling importance

**Status:** EXPLORATORY, POST HOC. Nothing pre-registered depends on this. Numbers are in `results.json`; the run is `python analysis/v3_followup/t2_split_half/run.py` (about 21 min on 3 threads). Cached half-scores are in `DATA/derived/v3_followup/t2_split_half/`.

## Verdict

- **The anchoring-specific signal is not reliable enough to use as a label,** pooled or per problem.
  - `tv_debiased` reliability at n = 100 per fork (dataset A) is Spearman–Brown (SB) Spearman **0.16 [0.13, 0.19]** and ICC **0.49 [0.40, 0.58]**. Both FAIL the 0.80 / 0.75 benchmarks.
  - Within problems it is 0.16 [0.12, 0.19], and no single problem reaches 0.80.
  - At n = 300 (dataset B) it is SB Spearman **0.18 [−0.03, 0.36]** (FAIL).
- **The raw `fork_tv` looks borderline reliable,** at 0.81 [0.75, 0.86] (A, INCONCLUSIVE vs 0.80), but that reliability belongs to the noise floor.
  - The floor alone is 0.97 [0.96, 0.98] and `acc_level` is 0.99. Both PASS.
  - Most of `fork_tv`'s reliable variance is answer entropy, not anchoring.
- **In B, the only problems where `tv_debiased` has any reliability are the three that hold the 7 BH-10% real effects** (2238: 0.59, 3935: 0.55, 330: 0.36; per-problem SB Spearman).
  - In every other problem it is about 0 or negative.
  - The pooled ICC of 0.78 in B (versus Spearman 0.18) comes from those few large effects. It is not broad rank reliability.

## Method (fixed in code before the run)

- **Splits.** 1,000 random splits per dataset. Each fork's samples are partitioned into halves independently: 50/50 in A and 150/150 in B. Answer grouping is canonical (`arv2.canon`) unless a score is marked raw. Scores are rounded to 10 decimals before ranking.
- **Datasets.**
  - **A** is the released MVP set: 6,434 sentences, 20 problems, 40 traces, 100 per fork.
  - **B** is Phase 2 pooled (r0 window forks plus r1 and r2): 190 units, 10 problems with one trace each, 300 per fork.
- **Scores.** `fork_tv`, `tv_debiased`, `floor`, `tv_without` (similarity < 0.8), `abs_dacc`, `signed_dacc`, `kl_log1p` (s12 KL, α = 1e-9, without vs fork i+1), `acc_level` and `fork_tv_raw`.
- **Spearman–Brown.** Statistics are averaged over splits, then SB-corrected: factor 2 to full n, and n/n_half for projections. SB is applied only when r > 0. Projection beyond full n is flagged valid only for scores whose true value does not depend on n (`tv_debiased`, the dacc scores, `acc_level`). `fork_tv`, `floor`, `tv_without` and KL carry an n-dependent floor, so their projections overstate.
- **Intervals.** These use a problem-cluster bootstrap, 2,000 draws, seeds `20260914:arv2:v3f:t2_split_half:<A|B>:<score>`. The draw sequence is the same as `R.cluster_bootstrap`, and the statistic is recomputed inside each draw:
  - ICC and the covariance decomposition are exact over all 1,000 splits, from per-(split, problem) sufficient statistics.
  - Pooled Spearman is computed exactly as a multiplicity-weighted Spearman, identical to re-ranking the resample. Inside the bootstrap this uses the first 200 splits only, for runtime; the split SD is ≤ 0.012, so the effect is about 0.001. The point estimate uses all 1,000 splits.
  - Within-problem means are bootstrapped by resampling the split-averaged per-problem values.
  - Both shortcuts were checked against `R.cluster_bootstrap` and matched to machine precision.

## A — released MVP, n = 100 per fork (half = 50)

| score | split-half Spearman | SB full-n Spearman | SB full-n ICC | within-problem SB Spearman (mean) | between-problem share of reliable var. | n/fork for Spearman 0.80 | problems ≥ 0.80 |
|---|---|---|---|---|---|---|---|
| fork_tv | 0.686 [0.602, 0.753] | **0.814 [0.752, 0.859]** INC | 0.806 [0.738, 0.847] INC | 0.767 [0.720, 0.810] | 0.13 [0.03, 0.17] | 91* | 9/20 |
| fork_tv_raw | 0.686 | 0.814 [0.753, 0.860] INC | 0.806 INC | 0.767 | 0.13 | 91* | 9/20 |
| tv_debiased | 0.087 [0.071, 0.107] | **0.161 [0.132, 0.193]** FAIL | **0.488 [0.400, 0.581]** FAIL | 0.156 [0.124, 0.188] | 0.004 [0.000, 0.007] | ~2,090 | 0/20 |
| floor | 0.941 | 0.969 [0.958, 0.975] PASS | 0.974 PASS | 0.941 | 0.18 | 13* | 20/20 |
| tv_without | 0.714 | 0.833 [0.783, 0.868] INC | 0.770 [0.712, 0.812] INC | 0.802 | 0.09 | 80* | 11/20 |
| abs_dacc | 0.463 | 0.633 [0.577, 0.683] FAIL | 0.640 [0.575, 0.698] FAIL | 0.598 | 0.07 | 232 | 0/20 |
| signed_dacc | 0.157 | 0.272 [0.242, 0.299] FAIL | 0.458 [0.393, 0.523] FAIL | 0.276 | 0.001 | ~1,070 | 0/20 |
| kl_log1p | 0.563 | 0.720 [0.650, 0.773] FAIL | 0.608 [0.478, 0.670] FAIL | 0.655 | 0.31 | 155* | 0/20 |
| acc_level | 0.977 | 0.988 [0.985, 0.989] PASS | 0.992 PASS | 0.969 | 0.09 | 5 | 20/20 |

\* Projection invalid: the true score shrinks with n, because the floor ∝ 1/√n.

- **Decision labels.** INC = INCONCLUSIVE; decisions come from `R.three_way`, with benchmarks Spearman 0.80 and ICC 0.75.
- **NaN rates.** The `tv_without` and KL halves are NaN for 16% of sentence pairs, where no fork-i sample had similarity < 0.8.
- **SB-projected `tv_debiased` Spearman at n = 200 / 300 / 500:** 0.28 / 0.37 / 0.49.
- **Between- vs within-problem decomposition.** In A, most reliable variance lies *within* problems for every score: the between share is ≤ 0.31.
  - The between-problem part is itself near-perfectly reliable (r ≈ 0.97–1.0), except for `tv_debiased` (0.48).
  - The within-problem reliability of `tv_debiased` is r = 0.49 on the covariance (ICC-like) scale, but only 0.16 on ranks. A minority of large values carries the linear agreement.
- **Per problem (SB Spearman).**
  - `fork_tv` ranges from 0.40 (2870) to 0.89 (3550, 4164).
  - `tv_debiased` ranges from 0.06 (3916) to 0.29 (2238). Its per-problem ICC ranges from 0.15 to 0.80, with one problem (2236) ≥ 0.75.
  - The full table is `datasets.A.scores.<score>.per_problem` in `results.json`.

## B — Phase 2 pooled, n = 300 per fork (half = 150)

| score | split-half Spearman | SB full-n Spearman | SB full-n ICC | within-problem SB Spearman (mean / median split-half) | between share |
|---|---|---|---|---|---|
| fork_tv | 0.497 [0.256, 0.654] | 0.664 [0.408, 0.791] FAIL | 0.827 [0.544, 0.887] INC | 0.348 [0.059, 0.594] / 0.065 | 0.28 |
| fork_tv_raw | 0.499 | 0.665 [0.418, 0.799] FAIL | 0.827 INC | 0.351 | 0.28 |
| tv_debiased | 0.100 [−0.033, 0.220] | **0.182 [−0.033, 0.361]** FAIL | 0.779 [0.011, 0.879] INC | 0.119 [−0.071, 0.334] / −0.027 | 0.13 |
| floor | 0.917 | 0.956 [0.874, 0.973] PASS | 0.961 PASS | 0.691 | 0.51 |
| tv_without | 0.508 | 0.674 [0.495, 0.785] FAIL | 0.674 INC | 0.533 | 0.18 |
| abs_dacc | 0.219 | 0.359 [0.109, 0.524] FAIL | 0.778 [0.240, 0.873] INC | 0.198 | 0.25 |
| signed_dacc | 0.180 | 0.305 [0.106, 0.435] FAIL | 0.683 INC | 0.275 | 0.03 |
| kl_log1p | 0.525 | 0.688 [0.539, 0.745] FAIL | 0.563 FAIL | 0.370 | 0.67 |
| acc_level | 0.940 | 0.969 [0.901, 0.980] PASS | 0.993 PASS | 0.669 | 0.96 |

**Per-problem SB Spearman, `fork_tv` / `tv_debiased` (19 sentences each, one trace per problem, so there are no "two traces" to pool in B):**

| problem | fork_tv | tv_debiased |
|---|---|---|
| 330 | 0.91 | 0.36 |
| 3935 | 0.83 | 0.55 |
| 2238 | 0.58 | 0.59 |
| 4605 | 0.38 | 0.39 |
| 4019 | 0.13 | 0.06 |
| 6596 | 0.11 | −0.14 |
| 2050 | 0.05 | −0.10 |
| 4164 | 0.05 | −0.14 |
| 2189 | −0.12 | −0.09 |
| 1591 | −0.13 | −0.19 |

- **How B differs from A.** Within a 19-sentence window even the floor is only moderately rank-reliable (0.69), because entropy varies little locally.
- **Where real effects sit.** The 7 real effects are 2238 ×4, 3935 ×2 and 330 ×1.

## Cross-checks

- **s14 (one even/odd split, raw categories, within = mean over trajectories).**
  - Reproduced exactly with unrounded scores: `fork_tv` split-half 0.6698 (s14: 0.6698).
  - `tv_debiased`, `floor` and `acc_level` also match s14 to ≥ 4 decimals.
  - Two discrepancies explain the brief's figures:
    1. The brief's "fork_tv 0.90, tv_debiased 0.39" are s14's **`ceiling_corr` = √(SB reliability)**, not reliabilities. The SB reliabilities are **0.80 and 0.15** (s14) and 0.81 and 0.16 here.
    2. Rounding to 10 decimals collapses 139 floating-point-distinct `fork_tv` values to 30 true values. This raises the even/odd Spearman from 0.670 to 0.685. The 1,000-split mean is 0.686, so the float-tie artefact, not the split, accounts for the s14 vs T2 gap of 0.016.
- **s13 (measured curve at n = 150, raw categories, 40 draws).**
  - This task's B split-half values are `fork_tv` Spearman 0.499 / ICC 0.705, `tv_debiased` 0.100 / 0.638 and `abs_dacc` 0.219 / 0.637.
  - s13 reported 0.492 / 0.702, 0.088 / 0.633 and 0.207 / 0.632. All are within one s13 draw-SD (about 0.04–0.07).
  - s13's `within_spearman` uses a different estimator: a per-draw median, versus the mean of split-averaged per-problem values here.

## What "reliability" means here

- Split-half reliability measures whether a score is reproducible from independent resamples of the *same* forks.
- `fork_tv`, `tv_without` and KL include a finite-sample floor that tracks answer entropy. Entropy is a stable property of a fork, with floor reliability 0.97. So these scores look reliable even when the true fork-i vs fork-i+1 difference is zero.
- `tv_debiased` removes the floor and is the anchoring-specific quantity. It has almost no rank reliability:
  - 0.16 at n = 100;
  - 0.18 at n = 300;
  - about 2,000 samples per fork are needed for 0.80, by SB extrapolation.
- That extrapolation assumes a spread-out true signal. The B pattern instead points to a sparse signal: a handful of real effects and near-zero elsewhere. On a sparse signal, extra samples improve detection of the few effects, not rank reliability over all sentences.

## Resolved / remaining

- **Resolved.**
  - Split-half reliability is now quantified for every importance score, with problem-cluster intervals, and the s14 and s13 discrepancies are explained.
  - fork_tv's apparent reliability is floor or entropy reliability.
  - Anchoring-specific reliability is low, pooled (A and B) and per problem.
  - Canonical vs raw grouping makes no material difference (0.814 vs 0.814).
- **Remaining.**
  - Whether a *detection* label (sparse real-effect flags at higher n) is reliable is a different question. Only 3 of 10 B problems carry signal, so it needs targeted sampling (Step 3 or T-sampling), not more split-half analysis.
  - The B intervals are wide (10 problems).
  - SB projections for floor-bearing scores are not valid beyond full n.
