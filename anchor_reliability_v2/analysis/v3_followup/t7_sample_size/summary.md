# T7 (item 10): Phase 3 sample size

**Status: EXPLORATORY, POST HOC.** This is a planning calculation. It uses variances from the pilot and Phase 2: 10 problems, about 20 forks or 19 sentences per problem trajectory. Answers use canonical grouping (`arv2.canon`). Raw-grouping widths come from the committed results and give essentially the same N (see `committed_widths_raw` in `results.json`).

## Question and methods

**Question.** How many problems give a 95% problem-cluster bootstrap interval of width hi − lo ≤ 0.30?

**Setup.** The contrast list, the N grid {10, 20, 30, 50, 80, 120, 200}, the units-per-problem grid {5, 10, 20, 40, 80} and the cutoff for calling a contrast within-dominated (within share ≥ 0.5) were all fixed in `run.py` before the run. Every bootstrap uses 2,000 replicates with seeds `20260914:arv2:v3f:t7_sample_size:*`. Scores are rounded to 10 decimals.

**The two methods:**
- **Analytic.** N = 10·(w10/0.30)². Here w10 is recomputed by importing `s18.compute_metrics` and `s12.agreement`, which use their own seeds. The column "N an." uses this; the own-seed bootstrap agrees within about ±10%, except for the AP contrasts (see below).
- **Bootstrap extrapolation.** Resample N problems from the 10 and take the smallest grid N with width ≤ 0.30. The table also gives a log-log interpolation between grid points.

**Within share.** This is V(within-problem row bootstrap) / V(problem bootstrap), both at N = 10. A share above 1 occurs for the AP statistics, because resampling rows reshuffles the 7 rare positives. Read it as "within-dominated".

## Results (all 10 problems; without problem 1591 in brackets)

| contrast | est. | w10 | N an. | N boot. (interp.) | within share | N at 40 units/problem |
|---|---|---|---|---|---|---|
| **H4_level_vs_position** | 0.09 | 1.34 | **201** [165] | **221** (>200) [112] | 0.02 | 195 |
| H1a Spearman level | 0.10 | 1.11 | 137 [131] | 158 [142] | 0.02 | 136 |
| Phase 2 fork_tv kappa(q90) | 0.66 | 0.85 | 81 [73] | 97 [83] | 0.37 | 65 |
| S3 entropy | 0.04 | 0.83 | 77 [56] | 73 [51] | 0.09 | 72 |
| S2 sampled level (= c: level readout) | 0.07 | 0.63 | 44 [31] | 44 [28] | 0.09 | 46 |
| H1b within-trajectory median | 0.21 | 0.62 | 43 [50] | 83* [565*] | 0.38 | 34 |
| Phase 2 fork_tv ICC | 0.56 | 0.45 | 22 [25] | 20 [20] | 0.50 | 17 |
| Phase 2 fork_tv Spearman | 0.44 | 0.42 | 20 [26] | 20 [25] | 0.26 | 19 |
| H4 AP diff vs pre+post_32 | -0.01 | 0.41 (1.03 own seed) | 19-117 | 49 [45] | 0.54 | 84 |
| c: split-half tv_debiased Spearman | 0.15 | 0.34 | 13 [10] | 13 [10] | 0.77 | 8 |
| S7 AP of real effects | 0.21 | 0.32 | 12 [11] | 13 [12] | >1 | 5 |
| S5 (forced TV vs tv_debiased) | 0.07 | 0.31 | 11 [12] | 11 [11] | 0.71 | 7 |
| S6 (partial, given floor) | 0.05 | 0.29 | 10 [10] | 11 [11] | 0.78 | 6 |
| S4 signed change (= c: change correlation) | -0.09 | 0.24 | 6 [7] | 7 [7] | >1 | 3 |
| other H4 AP diffs (position x2, text+position, pre_8, value probe) | 0.10-0.18 | 0.28-0.40 | 9-18 | 5-14 | >1 | 3-6 |

\* H1b is a median over trajectories. Its bootstrap distribution is discrete, so the grid width sticks at about 0.50 between two observed per-trajectory values (for example, 0.50 at N = 30-200 without 1591). It then collapses once the median pins to one value. The extrapolation is therefore an artefact for H1b; use the analytic value of about 43-50.

## What is resolved

- **The binding contrast is H4_level_vs_position, at about 200 problems (165-221).** H1a comes next at about 140-160. Both are pooled across-fork Spearmans, and both are almost entirely between-problem: within share is 0.02. Accuracy level varies mostly between problems, so these statistics behave like correlations over about 10 points. Adding forks per problem does not help them; at 40 forks per problem they still need about 136 and 195 problems.
- **Phase 2 kappa(q90) needs about 80-100 problems.** The flag rate is about 3%, so the lower bound sits at 0 until about N = 50. Its within share of 0.37 means more forks help only modestly (about 65 at 40 units per problem).
- **The likely GPU-plan primaries (c) are cheap:**
  - level-readout validity (S2 proxy): about 44 problems (28-31 without 1591);
  - within-trace change correlation (S4 proxy): about 7;
  - split-half tv_debiased reliability: about 13.
- **Phase 2 fork_tv Spearman and ICC need about 20-25 problems** for width 0.30.
- **Within-dominated contrasts** (share ≥ 0.5): S4, S5, S6, S7, the H4 AP differences, the tv_debiased split-half and the fork_tv ICC. For these, more sentences per problem trade almost one for one against problems. For example, the tv_debiased split-half needs 22 problems at 10 units per problem but 8 at 40. H1b sits between the two groups: 59 problems at 10 forks, 43 at 20 and 34 at 40.
- **Without problem 1591**, the ranking does not change. S2 drops to about 30 and H4_level to 112-165. No contrast becomes binding that was not already.

## Answer to the deliverable question

- **If H1a and H4_level_vs_position stay primary:** N ≈ 200 new problems, and H4_level_vs_position binds. Forks per problem cannot substitute for problems here.
- **If the GPU plan's primaries are family (c) plus the Phase 2 reliability statistics (recommended):** N = 50 new problems, at about 20 forks per trajectory. The level-readout validity Spearman (S2 proxy, about 44) binds. At N = 50, H1b (about 43) and fork_tv Spearman and ICC (about 20) are also covered. kappa(q90), H1a and H4_level would then be secondary, with expected widths of about 0.4-0.5 at N = 50 (bootstrap grid at N=50: kappa 0.46, H1a 0.53, H4_level 0.66).

## What remains and caveats

- **The bootstrap extrapolation only reuses the 10 observed problems.** It therefore understates between-problem heterogeneity in new problems, and with it the tails, nonfinite draws (about 2.5% for the AP contrasts) and the real-effect prevalence (7/190). Treat every N as a lower bound.
- **The AP contrasts are unstable at N = 10.** Different seeds change w10 by up to 2.5x; for example, pre+post_32 gives 0.41 with the s18 seed and 1.03 with ours. Their N is uncertain, but the grid width falls below 0.30 by N = 20-50 for every AP contrast.
- **The variance decomposition is crude.** The within-problem row bootstrap treats adjacent forks as independent and ignores sampling noise from the roughly 100 samples per fork label. The forks-versus-problems trade-off therefore leans optimistic for adding forks.
- **Proxies.** S2 and S4 stand in for the GPU plan's readouts, which do not exist yet. If the new readout has higher validity, width shrinks somewhat and N falls.

Files: `run.py`, `results.json`. The log is in `~/preanchor-root/anchor_reliability_v2/derived/v3_followup/t7_sample_size/run.log`.
