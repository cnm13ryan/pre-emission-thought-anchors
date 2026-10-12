# T5 (item 12): headline numbers without problem 1591

**Status: EXPLORATORY, POST HOC.** This analysis was not preregistered. It recomputes every
headline number from s12, s13, s14, s16, s17 and s18 twice: once on all problems ("with") and
once with problem 1591 removed everywhere ("without"). Removing 1591 drops its rows, its
bootstrap cluster and its leave-one-problem-out (LOPO) fold. For the probes it also drops its
activations from the PCA training sets.

- **Source:** `run.py` produces `results.json`.
  - `results.json → headline_table` holds 462 rows. Each row gives the value with 1591, the
    value without it, the difference and both decisions.
  - `results.json → parts` holds the full outputs.
- **Groupings:** canonical grouping is the primary. Raw grouping, the original basis, is also
  reported.
- **Intervals:** problem-cluster bootstrap with 2,000 replicates and seeds
  `20260914:arv2:v3f:t5_without_1591:*`. "With" has 10 Phase 2 problems (20 MVP problems);
  "without" has 9 (19).

## Verdict

**No conclusion depends on problem 1591.** Every preregistered decision and every qualitative
conclusion is the same without it.

- Only one decision cell changes, and it is a secondary cell: κ of `tv` for replicate 2 vs
  replicate 0 moves from FAIL to INCONCLUSIVE. Its upper bound goes from 0.565 to 0.608,
  against a threshold of 0.60.
- The primary `fork_tv` decisions do not change, and no form passes in either condition.

**The floor dominance does not come from 1591.** With 1591 removed:
- the floor still reproduces (floor r1 vs floor r2, 0.82);
- fork_tv still mostly tracks the floor;
- 71% of units still fall below 1.5× the floor.

## Reproduction checks

- **s12 raw without 1591** reproduces the committed `sensitivity_without_1591` point estimates
  to within about 0.006. The remaining gap is the 10-decimal rounding before ranks.
- **s14 OOF predictions** (raw grouping, with 1591) reproduce the committed
  `s14_oof_predictions.parquet`.
  - Maximum absolute difference: 1e-13 or less for position, pre_8, pre+post_32 and the value
    probe.
  - For text+position it is 1e-6, because MiniLM ran on CPU.
- **s18 canon without 1591** gives H1a = 0.175, the same as Step 1's `pilot_without_1591`.

## Headline table (canonical grouping; raw is given only where it differs materially)

### Phase 2 confirmatory (s12, replicate 1 vs 2)

| Number | With 1591 | Without 1591 | Δ | Decision change |
|---|---|---|---|---|
| fork_tv Spearman | 0.444 [0.18, 0.63] FAIL | 0.441 [0.16, 0.65] FAIL | −0.003 | no |
| fork_tv ICC | 0.559 [0.23, 0.68] FAIL | 0.579 [0.21, 0.70] FAIL | +0.020 | no |
| fork_tv κ (q90) | 0.659 [0.00, 0.85] INCONCL. | 0.659 [0.00, 0.85] INCONCL. | 0 | no |
| tv / jsd / dacc / KL Spearman | 0.48 / 0.61 / 0.33 / 0.51 | 0.48 / 0.60 / 0.31 / 0.51 | ≤0.02 | no (all FAIL) |
| fork_tv r1 vs r0: Spearman / ICC | 0.521 / 0.614 FAIL | 0.490 / 0.631 FAIL | −0.03 / +0.02 | no |
| fork_tv r2 vs r0: Spearman / ICC | 0.522 / 0.620 FAIL | 0.491 / 0.637 FAIL | −0.03 / +0.02 | no |
| tv κ, r2 vs r0 (secondary) | 0.305 [−0.03, 0.565] FAIL | 0.376 [−0.03, 0.608] INCONCL. | +0.07 | **yes** (secondary) |

### Phase 2 diagnostics (s13)

| Number | With 1591 | Without 1591 | Δ |
|---|---|---|---|
| Categories in the pool | 170 | 145 | −25 |
| fork_tv r1v2 within-problem Spearman | 0.28 | 0.37 | +0.09 |
| tv_debiased r1v2 Spearman | 0.151 [−0.03, 0.31] | 0.195 [0.03, 0.34] | +0.044 |
| acc_level r1v2 Spearman / ICC | 0.913 / 0.980 | 0.917 / 0.981 | +0.004 |
| floor r1 vs floor r2 | 0.863 [0.62, 0.93] | 0.824 [0.53, 0.93] | −0.04 |
| fork_tv r1 vs floor r2 | 0.527 [0.22, 0.69] | 0.484 [0.13, 0.70] | −0.04 |
| Share of fork_tv < 1.5× floor | 0.742 [0.66, 0.84] | 0.713 [0.64, 0.80] | −0.03 |
| Projected fork_tv at n = 300 (40 draws, ICC / Spearman / within) | 0.84 / 0.76 / 0.41 | 0.85 / 0.72 / 0.44 | small |
| Projected tv_debiased at n = 300 (ICC / Spearman / within) | 0.75 / 0.26 / 0.17 | 0.77 / 0.29 / 0.23 | small |
| Real effects, BH 10% (canon) | 7 of 190 | 7 of 171 (same 7 units) | 0 |
| Real effects, BH 10% (raw) | 7 of 190 | **8** of 171 (adds 3935:correct:13, tv) | +1 |

**Effects of removing 1591:**
- tv_debiased's lower bound moves just above 0, which makes it slightly more reproducible.
- Within-problem fork_tv agreement rises.
- Neither change comes close to a threshold.
- None of the real effects is in 1591.
  - Under canonical grouping, BH over 171 tests flags the same 7 units.
  - Under raw grouping, BH admits one extra borderline unit.

### Root cause (s14; 19-problem LOPO for "without")

Coverage:
- all six layers (8–48);
- representations: position, text+position, pre, post, pre+post and pre+position;
- `text_S_i` and `delta` were not run.

| Number | With 1591 | Without 1591 |
|---|---|---|
| Ceilings: fork_tv / tv_debiased / signed_dacc / acc_level | 0.90 / 0.39 / 0.51 / 0.99 | 0.90 / 0.39 / 0.51 / 0.99 |
| Best pre+post (L32) fork_tv: overall / within | 0.450 / 0.434 | 0.441 / 0.424 |
| Best pre+post (L32) floor: overall / within | 0.546 / 0.591 | 0.534 / 0.583 |
| Best post (L32) tv_debiased: overall / within | 0.080 / 0.078 | 0.080 / 0.079 |
| Best pre+post (L40) signed_dacc: overall / within | 0.026 / 0.019 | 0.023 / 0.015 |
| Best pre/pre+post acc_level: overall / within | 0.156 / 0.036 (pre, L32); 0.169 / 0.053 (pre+post) | 0.132 / 0.038 (pre, L8); 0.140 / 0.052 (pre+post) |
| Position fork_tv: overall / within | 0.276 / 0.572 | 0.259 / 0.563 |
| Pre L32 fork_tv probe: partial given floor | −0.009 [−0.06, 0.04] | −0.007 [−0.06, 0.04] |
| That probe: correlation with floor / with fork_tv | 0.52 / 0.43 | 0.51 / 0.42 |

Only one best layer changes: pre for acc_level moves from L32 to L8, and the difference is
within noise. The probe still reads the floor or uncertainty and carries nothing beyond the
floor.

### Probe capacity (s16; layer 32, ridge on PCA-128, pre)

| Within-trace number | With 1591 | Without 1591 |
|---|---|---|
| acc_level | 0.036 [−0.03, 0.10] | 0.039 [−0.04, 0.11] |
| signed_dacc | 0.013 [−0.02, 0.04] | 0.020 [−0.01, 0.04] |
| tv_debiased | 0.044 [0.01, 0.07] | 0.036 [0.00, 0.07] |
| entropy | 0.542 [0.47, 0.61] | 0.529 [0.45, 0.60] |
| p_own | 0.654 [0.60, 0.71] | 0.645 [0.58, 0.70] |
| Position-only p_own | 0.793 | 0.793 |
| p_own partial given position | 0.071 [0.02, 0.12] | 0.064 [0.02, 0.11] |
| p_own partial given position and entropy | 0.017 [−0.03, 0.06] | 0.013 [−0.03, 0.05] |
| Trace-correctness row AUC | 0.49 | 0.47 |

### Targeted sampling (s17)

Coverage:
- the entropy AP analysis was run in full;
- the simulation used 300 reps, restricted to uniform N = 300 and the two-stage m20 N300 cells,
  which include the recommended design.

| Number (canon) | With 1591 | Without 1591 |
|---|---|---|
| Positives / base rate | 7 / 0.037 | 7 / 0.041 |
| AP of entropy_i, full depth (permutation p) | 0.207 (p = 0.018) | 0.237 (p = 0.008) |
| AP of entropy_pair, full depth (p) | 0.083 (p = 0.16) | 0.115 (p = 0.10) |
| Recommended design (m20 N300 top50 entropy_pair), $ at λ = 1 / 0.75 / 0.5 | 1,905 / 1,839 / 1,718 | 1,641 / 1,628 / 1,384 |
| Uniform N = 300, $ | 2,323 | 2,091 |
| Recommended design FDP | 0.075–0.085 | 0.078–0.081 |

**Effect of removing 1591 on the design:**
- Costs fall by about 10–20%, because prevalence rises from 7/190 to 7/171.
- The recommended design stays cheaper than uniform N = 300 at every λ.
- Under raw grouping the recommended design costs $1,866 / 1,844 / 1,670 with 1591 and
  $1,457 / 1,403 / 1,237 without (8 positives).

**The trajectory-cluster AP intervals still reach about 1.0,** so one caveat stands: with 7
positives in 3 problems, the effect of the entropy prior is barely identified.

### plan_v3 pilot (s18; primary analysis, out-of-window candidates, labels from r1 and r2)

Real-effect flags were recomputed with BH over 171 tests. The H4 baselines come from this
task's s14 rerun under the same condition.

| Statistic | With 1591 | Without 1591 | Decision change |
|---|---|---|---|
| H1a level Spearman | 0.096 [−0.46, 0.66] INCONCL. | 0.175 [−0.38, 0.72] INCONCL. | no |
| H1b within-trajectory | 0.210 [−0.05, 0.57] INCONCL. | 0.215 [−0.09, 0.62] INCONCL. | no |
| S2 sampled level | 0.072 FAIL | 0.171 FAIL | no |
| S3 entropy | 0.040 FAIL | 0.120 FAIL | no |
| S4 signed change | −0.090 FAIL | −0.074 FAIL | no |
| S5 tv_debiased | 0.065 INCONCL. | 0.061 INCONCL. | no |
| S6 partial fork_tv given floor | 0.051 INCONCL. | 0.051 INCONCL. | no |
| S7 AP of real effects | 0.210 [0.01, 0.33] INCONCL. | 0.212 [0.01, 0.34] INCONCL. | no |
| H4 level vs position; H4 AP vs 6 baselines | all INCONCL. | all INCONCL. | no |
| Gates Q1 / Q2 / Q3 | pass / **fail** (mass median 2.3e-6) / pass | pass / **fail** (1.5e-6) / pass | no |
| Conclusion | INCONCLUSIVE_QUALITY_GATE | INCONCLUSIVE_QUALITY_GATE | no |

Under raw grouping, S7 without 1591 uses 8 real effects and gives 0.192 [0.01, 0.33], still
INCONCLUSIVE. The controls are also essentially unchanged:
- R1 replay Spearman is 1.000;
- format Spearman is 0.95 / 0.91 with 1591 and 0.94 / 0.88 without.

## Resolved, and what remains

**Resolved:**
- Problem 1591 does not drive the Phase 2 reliability failure, the floor dominance, the
  real-effect count, the probe conclusions, the targeted-sampling recommendation or the pilot
  verdict.
- Its main effect is to shrink the pool's category count, from 170 to 145.

**Remaining:**
- With 9–10 clusters, every interval is wide. Removing any single problem shifts point
  estimates by a few hundredths. This task tested only 1591. A full leave-one-problem-out
  sweep, for example removing 2238, which holds 4 of the 7 real effects, was not run.
- The single extra raw-grouping real effect (3935:correct:13) is borderline under BH.

## Deviations and implementation notes

- **Real-effect p-values:**
  - Per-unit permutation p-values were reused from the shared
    `sentence_effects_{raw,canon}.csv`. They are per-unit tests that do not depend on other
    units.
  - Only BH was recomputed over 171 tests.
- **Seeds inside imported code:**
  - s16 and s17 hardcode their seed material, so it was redirected through a proxy for
    `arv2.reliability`.
  - s18 seeds were redirected by setting `SEED_MATERIAL`.
  - s16 section E ran with 2,000 bootstrap replicates, not 1,000.
  - No existing file was edited.
- **s14:**
  - The scaler and PCA blocks were cached per fold. The maths is identical to
    `s14.lopo_predict`, as the OOF reproduction shows.
  - `text_S_i`, `delta`, section E and the MVP permutation count were not rerun.
- **s16 baseline:** the baseline is the s16 main-protocol row (leading 128 components of
  PCA-512). The partials come from section E (PCA-128).
- **s17:** the simulation was restricted to 5 configurations. The committed s17 numbers came
  from raw grouping with different seeds, so compare "with" against "without" here, not
  against the committed file.
- **Activation states:** layers 8–48 were reloaded from the hash-verified shards and cached at
  `derived/v3_followup/t5_without_1591/states_*.npy`, outside git. No s16 layer cache existed.
