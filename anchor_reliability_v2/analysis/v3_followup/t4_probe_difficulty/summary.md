# T4 (item 8): does problem difficulty account for the probe null?

**Status: EXPLORATORY, POST HOC.** Not preregistered. Source: `run.py`, which writes
`results.json`. Intermediate predictions are in
`~/preanchor-root/anchor_reliability_v2/derived/v3_followup/t4_probe_difficulty/`.

## Verdict

**No. Difficulty does not account for the null, and it does not account for the positive
signals either.**

- **Across problems.** The across-problem signal that exists is not difficulty.
  - Partialling difficulty out of both the label and the prediction leaves every probe's
    overall Spearman unchanged (all changes are 0.03 or less).
  - The probe's problem-mean predictions do not follow difficulty.
  - Most of what the probes get right is variation *within* a problem.
- **Within problems.** No signal on the effect targets appears once difficulty is
  controlled or stratified.
  - `tv_debiased` and `signed_dacc` stay at ≤ 0.07 in every stratum, every layer and every
    demeaning variant. Their pairwise-ranking accuracy is 0.50–0.52.
  - For `p_own` and `entropy`, the activation probe stays **below** the position baseline in
    every variant and every stratum.
- **Full width.** Ridge on all 5,120 dimensions changes nothing material. Paired with
  PCA-128 under the same alpha-selection rule, the full-width minus PCA-128 difference in
  within-trace Spearman is between +0.00 and +0.03 at every layer and target.

## Data, checks and protocol

**Data:**
- 6,434 sentences, 40 traces and 20 problems, as in s14 and s16.
- Targets come from the shared `released_canon` table. The primary grouping is canonical.
- The raw-grouping targets reproduce s16 exactly (maximum absolute difference 0).
- Canonical and raw results differ by at most 0.0034 on any metric reported here, so only
  canonical numbers are shown.

**Activation cache.** The s16 cache `states_24_32_40.npy` has shape (6474, 3, 5120) in
`mvp_features` row order. It matches the raw shards exactly, checked on two full shards.

**s16 replica (ridge PCA-128, layer 32, raw targets).** These match s16:
- within-trace: `acc_level` 0.04, `entropy` 0.54, `p_own` 0.65;
- overall `acc_level` 0.16.

**Protocol:**
- Leave-one-problem-out, with the scaler and PCA fit on the training problems only (as in
  s16).
- Bootstraps resample problems: 2,000 replicates with shared draws, so paired
  differences are valid.
- Scores are rounded to 10 decimals before every rank statistic.

**Difficulty definitions (fixed in code before the run):**
- `d1` is the mean canonical resample accuracy over all forks of both traces. It ranges
  from 0.44 to 0.84 across the 20 problems.
- `d2` is the no-reasoning accuracy from s08.
  - **`d2` is degenerate:** 18 of 20 problems score 0. Only 1591 (0.80) and 4019 (0.15)
    are non-zero.
  - Its terciles are therefore undefined: the strata are 18 / 0 / 2 problems. The median
    split is 18 vs 2.
  - Any `d2` result is effectively a test of "problems 1591 and 4019 vs the rest". The
    Spearman correlation between `d1` and `d2` is −0.06.

**Strata sizes are small, so the CIs are wide.**
- The `d1` median split gives 10 + 10 problems. The `d1` terciles give 7 / 6 / 7.
- With 6 or 7 problems per tercile, per-stratum CIs are wide and stratum-level
  differences have little power.
- No formal test of heterogeneity between strata was pre-specified, so none is reported.

## 1. Is the across-problem signal mostly difficulty? No

**Structure of the labels** (share of label variance):

| Target | Between problems | Between traces |
|---|---|---|
| acc_level | 0.09 | 0.60 |
| entropy | 0.16 | 0.27 |
| p_own | 0.07 | 0.31 |
| tv_debiased | 0.003 | 0.013 |
| signed_dacc | 0.0004 | 0.003 |

- Accuracy varies mostly between the correct and incorrect trace of a problem, not between
  problems.
- `d1` *is* the problem-mean accuracy: it correlates 1.0 with the problem-mean `acc_level`.

**Layer 32, ridge PCA-128 (s16 probe).** Values are Spearman with 95% problem-bootstrap
CIs.

| Target | Overall | Partial \| d1 | Partial \| d2 | After problem demeaning | Problem-mean prediction vs d1 (n=20) |
|---|---|---|---|---|---|
| acc_level | 0.16 [0.02, 0.30] | 0.19 [0.05, 0.32] | 0.15 [0.02, 0.30] | 0.22 [0.07, 0.36] | −0.19 |
| entropy | 0.48 [0.36, 0.58] | 0.48 [0.36, 0.58] | 0.48 [0.35, 0.58] | 0.51 [0.38, 0.61] | −0.16 |
| p_own | 0.46 [0.36, 0.57] | 0.47 [0.37, 0.58] | 0.46 [0.36, 0.57] | 0.53 [0.43, 0.61] | 0.34 |
| tv_debiased | 0.04 [0.01, 0.08] | 0.04 [0.01, 0.07] | 0.04 [0.01, 0.08] | 0.04 [0.01, 0.08] | −0.19 |
| signed_dacc | 0.00 [−0.01, 0.02] | 0.00 [−0.01, 0.02] | 0.00 [−0.01, 0.02] | 0.01 [−0.01, 0.03] | 0.10 |

**What the table shows:**
- Removing difficulty, or all problem-level variation (problem fixed effects), never lowers
  the overall Spearman. It slightly *raises* it, so the "overall" signal is within-problem
  signal.
- The pattern is the same at layers 24 and 40, for full-width ridge, and for every
  demeaned variant.

**Baselines (overall Spearman):**
- **Difficulty alone** (LOPO ridge on `d1`) reaches 0.28 [0.19, 0.34] for `acc_level`.
  This is a label-derived oracle, because `d1` is the problem-mean label.
  - The activation probe is below it: the paired difference is −0.12 [−0.30, 0.08].
  - Difficulty alone gives −0.25 for `p_own` and −0.06 for `entropy`.
- **Position alone** matches the activation probe on `acc_level`:
  - overall 0.21 [0.03, 0.38];
  - paired difference (activation − position) −0.05 [−0.14, 0.03].
- The modest overall `acc_level` signal is therefore not specific to activations, and it
  is not difficulty.

## 1b. Difficulty strata (d1)

The table shows the global probe at layer 32, evaluated inside each stratum. Each cell is
the within-trace Spearman.

| Stratum (problems) | acc_level | entropy | p_own | tv_debiased | signed_dacc | Position-only p_own |
|---|---|---|---|---|---|---|
| low d1, harder (10) | 0.13 [0.06, 0.20] | 0.58 [0.48, 0.67] | 0.69 [0.61, 0.75] | 0.06 [0.03, 0.09] | −0.00 [−0.04, 0.03] | 0.83 [0.78, 0.89] |
| high d1, easier (10) | −0.05 [−0.14, 0.03] | 0.50 [0.39, 0.59] | 0.62 [0.55, 0.69] | 0.03 [−0.02, 0.07] | 0.03 [−0.01, 0.06] | 0.75 [0.68, 0.82] |
| tercile t1, hardest (7) | 0.10 [0.02, 0.19] | 0.62 [0.53, 0.71] | 0.69 [0.58, 0.79] | 0.06 [0.01, 0.10] | −0.00 [−0.05, 0.03] | 0.84 [0.77, 0.92] |
| tercile t2 (6) | 0.00 [−0.17, 0.16] | 0.56 [0.43, 0.66] | 0.67 [0.57, 0.75] | 0.04 [−0.02, 0.10] | 0.03 [−0.03, 0.08] | 0.79 [0.71, 0.85] |
| tercile t3, easiest (7) | 0.00 [−0.08, 0.09] | 0.43 [0.31, 0.53] | 0.60 [0.54, 0.66] | 0.03 [−0.02, 0.07] | 0.01 [−0.03, 0.05] | 0.74 [0.66, 0.82] |

**Probes trained only inside each stratum** (leave-one-problem-out inside the stratum)
are equal or worse:
- low d1, `acc_level` within-trace: 0.14 [0.06, 0.21];
- t1, `acc_level`: 0.02 [−0.08, 0.13];
- t3, `p_own`: 0.49.

All layers and the overall Spearman are in `results.json` → `strata`.

**Reading:**
- **`acc_level` in harder problems.** There is a small within-trace `acc_level` signal in
  the harder half (0.13) and none in the easier half.
  - The position-only baseline shows the same pattern: 0.09 [0.04, 0.13] in the low half.
  - This is one cell out of many, and it was not tested for heterogeneity. It fits the
    s16 sign-cancellation account, under which the balance of correct and incorrect trace
    behaviour shifts with difficulty. It is **not** evidence that the state encodes
    correctness.
  - At around 0.13, against a within-trace ceiling of about 0.96, it is far from decoding.
- **The effect targets** (`tv_debiased`, `signed_dacc`) are at or below 0.07 in every
  stratum, at every layer, whether the probe is global or trained within the stratum.
- **`p_own` and `entropy`** are decodable in every stratum. The activation probe is below
  position in every stratum.
- **`d2` strata.** The 18-problem stratum reproduces the global numbers. The 2-problem
  stratum (1591 and 4019) gives `acc_level` within-trace 0.18. With 2 problems, no CI is
  meaningful.

## 2. Labels that vary within a problem

**Probes trained on demeaned features and targets (layer 32).** Columns are the pooled
Spearman of the demeaned label against the demeaned prediction, and pairwise accuracy
within a trace.

| Target | Trace-demeaned, pooled | Trace-demeaned, pairwise | Problem-demeaned, pooled | Problem-demeaned, pairwise (same problem) |
|---|---|---|---|---|
| acc_level | 0.21 [0.09, 0.31] | 0.50 [0.47, 0.52] | 0.25 [0.08, 0.40] | 0.55 [0.51, 0.60] |
| entropy | 0.57 [0.48, 0.65] | 0.73 [0.70, 0.76] | 0.53 [0.40, 0.63] | 0.69 [0.65, 0.72] |
| p_own | 0.68 [0.64, 0.72] | 0.78 [0.76, 0.80] | 0.56 [0.45, 0.65] | 0.71 [0.68, 0.74] |
| tv_debiased | 0.05 [0.02, 0.09] | 0.52 [0.51, 0.53] | 0.05 [0.01, 0.08] | 0.52 [0.51, 0.53] |
| signed_dacc | 0.01 [−0.01, 0.03] | 0.50 [0.50, 0.51] | 0.01 [−0.01, 0.04] | 0.51 [0.50, 0.51] |

**How noisy the labels are.** Split-half agreement within a trace, at 50 samples per half:

| Target | Within-trace Spearman | Pairwise accuracy |
|---|---|---|
| tv_debiased | 0.08 | 0.53 |
| signed_dacc | 0.16 | 0.56 |
| acc_level | 0.84 | 0.85 |
| p_own | 0.85 | 0.86 |

The effect labels leave very little room for any pairwise metric, so a pairwise accuracy
of 0.50–0.52 cannot exclude small effects.

**Activations against baselines (within-trace Spearman, paired).**

| Comparison | Variant | p_own | entropy | acc_level | tv_debiased | signed_dacc |
|---|---|---|---|---|---|---|
| activation − position | raw | −0.14 [−0.17, −0.11] | −0.16 [−0.19, −0.13] | +0.01 [−0.05, 0.06] | −0.01 [−0.05, 0.04] | +0.01 [−0.03, 0.04] |
| activation − position | trace-demeaned | −0.09 [−0.11, −0.07] | −0.12 | −0.01 | −0.01 | 0.00 |
| activation − position | problem-demeaned | −0.11 [−0.14, −0.09] | −0.14 | +0.01 | 0.00 | 0.00 |
| (activation + position) − position | raw | −0.09 [−0.11, −0.07] | −0.13 | +0.04 [−0.02, 0.09] | −0.01 | +0.01 |
| (activation + position + entropy) − (position + entropy) | raw | −0.06 [−0.07, −0.04] | n/a | −0.03 [−0.08, 0.03] | +0.00 | −0.10 [−0.14, −0.07] |

- **Activations never beat position.** Adding activations to position *lowers*
  out-of-problem accuracy for `p_own` and `entropy`, which points to overfitting across
  problems. This holds in every demeaning variant.
- **Activations add nothing beyond position + entropy.**
  - Position + entropy reaches `p_own` within-trace 0.83 [0.77, 0.89] and pairwise 0.88.
  - It reaches `acc_level` 0.17 [0.05, 0.29].
- **Caution on the position + entropy baseline.** It uses the *observed* entropy, computed
  from the same fork-i resamples as the labels.
  - It reaches `signed_dacc` within-trace 0.11 [0.09, 0.15]. Most likely this is shared
    sampling noise: `entropy_i` and `acc_i` come from the same 100 samples, and
    `signed_dacc` contains −`acc_i`.
  - It is not a decodable effect, and it was not investigated further. The activation
    probe does not reproduce it (−0.10 [−0.14, −0.07]).

## 3. Full width (5,120 dimensions) vs PCA-128

**Method:**
- Exact ridge in dual form, with one n × n eigendecomposition per outer fold. The kept rank
  is 5,120 in every fold.
- Alpha is chosen per target by exact inner leave-one-problem-out CV over the 19 training
  problems, using closed-form group-deletion residuals.
- PCA-128 ridge uses the identical rule ("PCA-128 LOGO").
- A brute-force check on synthetic data reproduced the inner-CV error and the predictions
  to 1e-14.

**Paired difference: full width − PCA-128 LOGO.**

| Target | L24, within-trace | L32, within-trace | L40, within-trace | L32, overall |
|---|---|---|---|---|
| acc_level | +0.02 [0.00, 0.04] | +0.00 [−0.01, 0.01] | +0.00 [−0.01, 0.01] | −0.02 [−0.05, 0.01] |
| entropy | +0.02 [−0.00, 0.04] | +0.03 [0.01, 0.04] | +0.03 [0.00, 0.05] | −0.01 [−0.04, 0.02] |
| p_own | +0.02 [0.00, 0.03] | +0.01 [0.00, 0.02] | +0.02 [0.00, 0.03] | +0.02 [−0.00, 0.04] |
| tv_debiased | +0.00 [−0.01, 0.01] | +0.00 [−0.01, 0.01] | +0.00 [−0.00, 0.01] | +0.00 [−0.00, 0.01] |
| signed_dacc | +0.00 [−0.00, 0.01] | +0.00 [−0.00, 0.01] | +0.00 [−0.00, 0.01] | −0.01 [−0.01, 0.00] |

**Absolute values at layer 32 (full width).** Within-trace Spearman:

| Target | Within-trace |
|---|---|
| acc_level | 0.01 [−0.06, 0.08] |
| entropy | 0.58 [0.50, 0.64] |
| p_own | 0.67 [0.62, 0.73] |
| tv_debiased | 0.05 [0.02, 0.08] |
| signed_dacc | 0.01 [−0.01, 0.04] |

**Against the s16-style RidgeCV PCA-128** (row-level LOO alpha):
- Full width is +0.02 to +0.04 better on within-trace `entropy` and `p_own`.
- It is −0.02 within-trace and −0.07 overall [−0.10, −0.03] on `acc_level`.
- Grouped CV picks a larger alpha for `acc_level`.

**Alphas.** For `signed_dacc`, inner leave-one-problem-out CV chose the top of the grid
(1e9) in most folds. Maximal shrinkage means the CV finds nothing to fit. This accounts for
most of the 14–20% of choices at the grid edge.

## What is resolved and what remains

**Ruled out** (at 20 problems, the stored boundary token, and layers 24/32/40):
- **Problem difficulty as a confound behind the null.** Neither `d1` nor `d2`, whether used
  as a covariate, as problem fixed effects, or as strata, uncovers within-problem decoding
  of `tv_debiased`, `signed_dacc` or within-trace `acc_level`.
- **Difficulty as the source of the positive signals** (`entropy`, `p_own`, overall
  `acc_level`). These survive partialling unchanged and are within-problem.
- **PCA-128 as the bottleneck for a linear readout.** Exact full-width ridge moves no
  target by more than 0.03.
- **Activations beyond position for `p_own` and `entropy`.** In every demeaning variant
  and every stratum, activations are below position, and below position + entropy.

**Not ruled out:**
- **Small effects.** Effects of about 0.05–0.1 on the effect targets cannot be excluded.
  Their label reliability is low: within-trace ceilings are about 0.4–0.5 from s16, and
  split-half pairwise agreement at n = 50 is only 0.53–0.56.
- **Difficulty heterogeneity.** Strata of 6–7 problems give wide CIs. The harder-half
  `acc_level` within-trace value of 0.13 vs −0.05 was not tested formally, and position
  shows the same pattern.
- **Difficulty measured independently of the resamples.** `d2` is degenerate (18 of 20
  problems score 0) and `d1` is label-derived. An external difficulty measure, such as the
  MATH level, was not available in the inputs used.
- **Other representations.** Nonlinear full-width models, other tokens (for example, the
  sentence-end token or mean-pooled sentence states), and other layers were not tested.

## Deviations from the brief

- **`d2` strata.** The `d2` terciles are undefined because of ties, and the strata are
  reported as 18 / 0 / 2. Stratum-trained probes were fit only for strata with at least
  4 problems, so not for the 2-problem `d2` stratum.
- **CI coverage.**
  - Row-level CIs (overall, partial and pooled-demeaned Spearman) are computed for the
    headline layer 32 and the baselines.
  - Layers 24 and 40 have point estimates for these metrics, plus CIs for the per-unit
    metrics (within-trace, within-problem and pairwise).
  - This was done for runtime. Within strata, CIs are computed at every layer.
- **Raw grouping.** It is reported as point estimates only, in the JSON. It differs from
  canonical by at most 0.0034.
- **Inner-CV preprocessing.** For the full-width and PCA-128 LOGO comparison, centring
  and scaling are fit on the outer training fold and not refit inside the inner folds.
  This is equivalent to fitting them as preprocessing on the training problems only, so no
  test data leaks.
- **Runtime.** Fitting took 49 min and evaluation 22 min, about 71 min in total. Nothing
  was cut.
