# Step 1 (item 6): raw vs canonical answer grouping

**Status:** EXPLORATORY, POST HOC (follow-up to the plan_v3 pilot, `dd94808`). No pilot file
was edited. `run.py` imports the s12, s13 and s18 scoring functions and feeds them
re-grouped inputs.

| | |
|---|---|
| Module | `src/arv2/canon.py`, tested in `tests/test_canon.py` (35 pass) |
| Shared tables, for Step 2 (not in git) | `DATA/derived/v3_followup/released_canon.parquet` (647,400 released MVP resamples) and `phase2_canon.parquet` (40,000 Phase 2 r1/r2) |
| Recomputed real-effect tables | `sentence_effects_{raw,canon}.csv` |

## The canonicaliser

- **Input.** It works from the **raw** answer surface, either the boxed string or the
  stated answer. It does not use the normalised category, because the v1 step strips
  braces: `\frac{5}{12}` becomes `\frac512`, which is ambiguous.
- **Presentation stripping.** It removes `$`, `\left`/`\right`, spacing, `\text{}`
  wrappers, `%`, degree marks, trailing units, `x =` and trailing periods.
- **Exact rationals.** Integers, decimals, thousands separators, `\frac`, `a/b` and
  signs are reduced to an exact rational.
- **Closed forms.** `\sqrt`, `\pi`, powers and factorials go through sympy.
- **Fallback.** Anything else is grouped by string.
- **Not merged.** Rounding variants such as 6.17 vs 6.18 stay separate.

## Category counts

| Pool | Raw | Canonical |
|---|---|---|
| Phase 2 pool (r0 + r1 + r2, 200 window forks) | **174** | **170** |
| Released MVP, all forks, global | 1,033 | 999 |
| Released MVP, sum of per-trajectory counts | 1,538 | 1,501 |

- **The four Phase 2 merges** are all decimal vs fraction forms of the same value:
  - 429.624 = `\frac{53703}{125}` (176 samples);
  - 4.8 = `\frac{24}{5}`;
  - 425.25 = `\frac{1701}{4}`;
  - 427.437 = `\frac{427437}{1000}`.
- **Correctness.** It changes for 0 Phase 2 samples and 3 released samples.
- **String fallbacks.** 10 of the 647,400 released samples remain, all prose-like answers.
- **Pilot candidates.** No candidate set merged in any trajectory.

## Recomputed numbers (raw → canonical)

| Number | Raw | Canonical |
|---|---|---|
| Phase 2 fork_tv Spearman r1 vs r2 | 0.444 [0.198, 0.613] FAIL | 0.442 [0.195, 0.615] FAIL |
| Phase 2 fork_tv ICC | 0.561 [0.228, 0.676] FAIL | 0.559 [0.226, 0.675] FAIL |
| Phase 2 fork_tv κ (q90) | 0.659 [0.000, 0.855] INCONCL. | 0.659 [0.000, 0.855] INCONCL. |
| Phase 2 tv (without set) Spearman | 0.478 | 0.476 |
| Without 1591: fork_tv Spearman / ICC | 0.438 / 0.581 | 0.435 / 0.579 |
| r2 vs released, fork_tv ICC | 0.624 | 0.620 |
| Mean noise floor r1 / r2 | 0.0486 / 0.0479 | 0.0485 / 0.0478 |
| Floor r1 vs floor r2 Spearman | 0.867 [0.619, 0.934] | 0.863 [0.591, 0.934] |
| fork_tv r1 vs floor r2 | 0.532 | 0.530 |
| Share of fork_tv below 1.5× floor | 0.742 | 0.742 |
| tv_debiased Spearman r1 vs r2 | 0.158 | 0.151 |
| Real effects, BH 10% (TV / Δacc) | 7 / 7 | 7 / 7, the same 7 sentences |
| Pilot H1a | 0.096 [−0.468, 0.641] | identical |
| Pilot H1b | 0.210 [−0.052, 0.567] | identical |
| Pilot S2 sampled level | 0.097 FAIL | 0.072 FAIL |
| Pilot S3 entropy | 0.035 FAIL | 0.040 FAIL |
| Pilot S4 | −0.088 FAIL | identical |
| Pilot S5 | 0.066 [−0.076, 0.229] | 0.065 [−0.079, 0.229] |
| Pilot S6 | 0.055 | 0.053 |
| Pilot S7 AP | 0.210 [0.008, 0.330] | identical |
| Pilot candidate mass median | 2.3e-6 | identical (Q2 gate still fails) |
| Pilot `<OTHER>` rate in sampled readout | 77.8% | 77.7% |

**Reproduction.** The raw pass reproduces the committed numbers: s13 categories (174),
real effects (7), pilot H1a, S5 and S7, the Phase 2 ICC, and the build's targets, which
match exactly.

**The Phase 2 Spearman differs slightly.** It is 0.4439 here against the committed 0.4453,
although the score values agree to 1e-16. The cause is **floating-point tie-breaking**:
many fork_tv values are exact ties (multiples of 0.005), and summation order decides how
they rank. Rounding to 10 decimals before ranking gives 0.4463. Reloading the committed
CSV gives 0.459. So the committed Spearman carries roughly ±0.01 of
numerical noise. No decision changes, because the interval's upper end is far below
0.80. **Step 2 tasks round scores to 10 decimals before any rank statistic.**

## What is resolved and what remains

- **Resolved: item 6 is not a cause.** Answer grouping explains none of the pilot or Phase 2
  results.
  - Almost all category spread is genuine: different values, not different spellings.
  - The 78% `<OTHER>` rate of the forced readout is not a grouping artefact, since
    canonical grouping leaves it unchanged. The forced answers really are different
    values.
  - The noise-floor diagnosis stands: the floor reflects real answer dispersion, such as
    the many distinct wrong numeric answers for problems 1591, 330 and 3935.
- **Remains.**
  - The high category counts for 330 (34) and 3935 (51) are real distinct wrong answers.
    Whether TV over a long tail of singletons is the right outcome space, rather than
    correct/incorrect or top-k plus other, is a target-definition question for T2 and T6.
  - Canonical grouping is the default for Step 2. Raw is reported where it matters.
