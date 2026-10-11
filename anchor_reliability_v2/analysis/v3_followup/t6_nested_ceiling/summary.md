# T6 (item 4): nested-variance ceiling on pre-emission predictability

**Status: EXPLORATORY, POST HOC.** Nothing here was preregistered. Reproduce with
`python analysis/v3_followup/t6_nested_ceiling/run.py` (about 18 min; the MiniLM vectors and
cluster labels are cached under `DATA/derived/v3_followup/t6_nested_ceiling/`). All numbers
are in `results.json`.

## Verdict

1. **Signed importance has a pre-emission ceiling of about 0. This is settled by theory and
   confirmed by the data.**
   - The tower property gives E_S[acc(h,S)] = acc(h). So any signed change caused by the
     realised sentence, such as `signed_dacc`, has conditional mean 0 given the history h.
     This holds whenever S_i is a draw from P(S|h).
   - Measured R²_max for signed Δacc, released data, primary cell: **0.03 [0.01, 0.05]**.
     B: 0.02 [−0.01, 0.04] (r1/r2) and 0.00 [−0.01, 0.02] (with r0).
   - The small nonzero value is what dropping small clusters produces on its own. On the
     synthetic truth at min size 5, the true value is 0.02 and the estimate is 0.02.
   - Consequence: the probe nulls on `signed_dacc` (and, by the same martingale argument, on
     `dp_own` and `dp_modal`) were guaranteed. They say nothing about what the representation
     contains.
   - Caveat: the traces were selected as correct or incorrect, which adds a small drift
     toward the final outcome. That drift is at most about 1/L per sentence.
2. **Unsigned importance (TV, |Δacc|) has a ceiling that is not small, but it is only a
   PARTIAL estimate.**
   - Released data, primary cell (similarity 0.8, min cluster size 5), canonical TV:
     **R²_max = 0.57 [0.38, 0.77]**. Within problem: 0.57 [0.38, 0.77]. Within trajectory:
     0.56 [0.37, 0.77].
   - Same cell, |Δacc|: 0.50 [0.37, 0.66].
   - Phase 2 windows, TV: 0.29 [0.13, 0.88] with r1/r2 only, and 0.48 [0.35, 0.98] with r0
     added.
   - Across all 9 (threshold × min size) cells for the released data, TV R²_max ranges from
     0.50 to 0.75.
   - The estimator recovers R²_max (TV bias −0.02 to −0.06; |Δacc| bias −0.1 to −0.25) only
     when the cluster effects are large relative to the sampling noise. In weak-effect
     synthetics whose debiased mean m(h) is about 0.002, which matches the real data, it
     breaks down.
   - The real signal is tiny. The debiased mean m(h) is 0.0022 [0.0016, 0.0029]. The naive
     m(h) is 0.041, against 0.039 for the same data with answers permuted.
   - The noise-corrected variance is 4% of the naive variance: 1.9e-4 [1.1e-4, 3.1e-4]
     against 4.7e-3, so the SD of the true I is about 0.014.
   - The real-data signal variance is about 10 times what the estimator returns on the weak
     synthetics. This fits a reading in which what gets measured is a sparse set of
     larger effects, but that is not established.
3. **Comparison with the probes.** Within-trace Spearman on `tv_debiased` was at most 0.08
   (s14/s16), against a noise ceiling of 0.40.
   - Disattenuated, that is about 0.2, so the probes explain about 4% of the true-score
     variance.
   - An oracle function of h would reach roughly √0.57 × 0.40 ≈ 0.30 on the same noisy target.
     This is a rough conversion that treats Spearman as Pearson.
   - So for unsigned magnitude, the probe null sits **far below** the ceiling, not near it. The
     irreducible within-history randomness of S does not explain it.
   - Note that fork_tv for sentence i equals TV(P(A|h, S_i = original), P(A|h)), which is
     exactly I at the realised original sentence. R²_max therefore bounds the
     h-predictable share of the true-score fork_tv, up to the coarsening by clusters.
   - Two caveats:
     - R²_max is a variance share. If it is carried by a few forks with large effects, it
       need not turn into rank correlation.
     - Clustering coarsens S, which removes within-cluster variance and so pushes R²_max
       upward.

## Estimator (exact)

- **Data units.** Per fork (history h), sampled sentences are embedded with MiniLM
  (all-MiniLM-L6-v2 @ 1110a243, normalised). They are clustered with average-linkage
  agglomerative clustering on cosine distance, cut at 1 − threshold.
  - Thresholds: similarity {0.8 primary, 0.7, 0.9}.
  - Minimum cluster size: {5 primary, 3, 10}.
  - Smaller clusters are left out of the (h, c) population. Weights w_c = n_c / Σ n over the
    estimable clusters. P(A|h) always uses all of the fork's samples.
  - Forks average 3.0 estimable clusters, covering 92% of samples (primary cell, released).
- **Definitions.** I_c = TV(P̂(A|h,c), P̂(A|h)) over canonical categories. Two variants:
  `tv_raw` (raw categories) and `acc` (TV on correct/incorrect, which equals |Δacc|).
  - `acc_signed` = acc(c) − acc(h). It is linear and needs no correction.
- **Mean bias.** E0_c is the exact permutation-null expectation: cluster sizes are fixed and
  answers exchangeable within the fork, so each category count is hypergeometric.
  E0_c = ½ Σ_a E|X_a/n_c − K_a/N|. It is computed from the pmf and checked against Monte
  Carlo to within 0.001.
  - Debiased I~_c = I_c − E0_c, and m~(h) = Σ w_c I~_c.
- **Noise variance.** 16 random half-splits, stratified within each cluster. In each half,
  x_c (half A) and y_c (half B) are the debiased I, using that half's own fork and cluster
  distributions and a half-sample null. Noise in x and y is independent, so cross-half
  products estimate signal moments:
  - Total = mean_h Σ w x y − x̄ȳ.
  - Between = mean_h (Σ w x)(Σ w y) − x̄ȳ.
  - Within = Total − Between. This identity is exact.
  - Across-problem (or across-trajectory) = the fork-weighted covariance of the group means.
  - R²_max = Between / Total. The within-problem version is (B − A_p) / (T − A_p).
  - Components are averaged over splits before forming ratios.
- **Intervals.** 2,000-replicate problem-cluster bootstrap (`R.cluster_bootstrap`) per cell.
  Seeds are `seed_from("20260914:arv2:v3f:t6_nested_ceiling:<dataset>:<thr>:<min>:<metric>")`.
  - When Total is close to 0, the ratio is unstable. Several Phase 2 intervals run outside
    [0, 1] for this reason, and they are reported as they came out.
  - B_r12 at similarity 0.9 / min 5 has Total < 0 (R²_max NaN).
- **Diagnostics.**
  - Across-problem variance is 0.4% of Total; across-trajectory is 1.7% (released, TV). So
    the overall, within-problem and within-trajectory ceilings coincide.
  - Fork answer entropy (H, H²) explains only **2% [1%, 6%]** of the between-history
    variance (cross-half projection). On B it is 11% (r1/r2) and 6% (with r0), with wide
    intervals. What makes a history important is not its uncertainty, which is the one
    thing the probes do read.
  - No rank statistics are computed in this task, so the 10-decimal rounding rule is not
    triggered.

## Validation

- **Synthetic null.** Answers were permuted within each fork, on the real released
  clusterings.
  - The debiased m is 0.0003 (naive 0.039).
  - The variance components are about −1e-5 (naive Total 4.1e-3).
  - So the correction removes about 99.8% of the spurious variance. R²_max is undefined
    there, as it should be.
- **Synthetic known truth.** Six answer models on the real cluster structure, n = 100:
  three "strong" settings (pre-planned) and three "weak" settings (added after the first
  run). The full table follows.
  - **Strong settings, TV, min size 5:** true / estimated R²_max 0.78/0.75, 0.73/0.68 and
    0.66/0.63.
  - The debiased variance components and m are attenuated by a factor of 2 to 5, because
    the null correction over-subtracts when effects are nonzero. Most of this cancels in the
    ratio.
  - |Δacc| is biased down by 0.1 to 0.25.
  - Signed Δacc is recovered almost exactly.
  - **Weak settings (effect scale × 0.15):** the variance estimates collapse toward 0 and
    R²_max is meaningless (−0.3 to 0.7, with true values 0.67 to 0.80). The naive R² is not
    a usable substitute either.


| setting | min | metric | true R2 | est R2 | naive R2 | true / est var_total | true / est mean m |
|---|---|---|---|---|---|---|---|
| high_between | 10 | acc | 0.767 | 0.638 | 0.667 | 1.8e-03 / 5.7e-04 | 0.0216 / 0.0102 |
| high_between | 10 | acc_signed | 0.110 | 0.107 | 0.131 | 2.2e-03 / 2.1e-03 | 0.0002 / -0.0002 |
| high_between | 10 | tv | 0.842 | 0.819 | 0.743 | 5.0e-03 / 1.1e-03 | 0.0666 / 0.0325 |
| high_between | 3 | acc | 0.632 | 0.383 | 0.462 | 2.1e-03 / 7.2e-04 | 0.0234 / 0.0101 |
| high_between | 3 | acc_signed | 0.006 | 0.006 | 0.012 | 2.7e-03 / 2.7e-03 | 0.0000 / -0.0000 |
| high_between | 3 | tv | 0.762 | 0.698 | 0.559 | 6.1e-03 / 1.2e-03 | 0.0725 / 0.0322 |
| high_between | 5 | acc | 0.672 | 0.482 | 0.530 | 2.0e-03 / 6.4e-04 | 0.0228 / 0.0103 |
| high_between | 5 | acc_signed | 0.021 | 0.020 | 0.032 | 2.5e-03 / 2.5e-03 | 0.0000 / 0.0000 |
| high_between | 5 | tv | 0.784 | 0.747 | 0.623 | 5.7e-03 / 1.2e-03 | 0.0706 / 0.0327 |
| low_between | 10 | acc | 0.729 | 0.624 | 0.667 | 2.3e-03 / 9.0e-04 | 0.0269 / 0.0131 |
| low_between | 10 | acc_signed | 0.090 | 0.089 | 0.116 | 3.1e-03 / 3.1e-03 | 0.0001 / 0.0004 |
| low_between | 10 | tv | 0.751 | 0.728 | 0.733 | 6.8e-03 / 1.5e-03 | 0.0804 / 0.0396 |
| low_between | 3 | acc | 0.582 | 0.405 | 0.482 | 2.8e-03 / 1.0e-03 | 0.0294 / 0.0129 |
| low_between | 3 | acc_signed | 0.006 | 0.004 | 0.011 | 3.7e-03 / 3.6e-03 | 0.0000 / 0.0000 |
| low_between | 3 | tv | 0.631 | 0.580 | 0.559 | 8.2e-03 / 1.6e-03 | 0.0878 / 0.0393 |
| low_between | 5 | acc | 0.625 | 0.483 | 0.543 | 2.7e-03 / 9.8e-04 | 0.0286 / 0.0130 |
| low_between | 5 | acc_signed | 0.020 | 0.018 | 0.030 | 3.5e-03 / 3.4e-03 | 0.0000 / 0.0001 |
| low_between | 5 | tv | 0.664 | 0.633 | 0.620 | 7.7e-03 / 1.5e-03 | 0.0855 / 0.0396 |
| mid | 10 | acc | 0.739 | 0.629 | 0.669 | 2.4e-03 / 9.5e-04 | 0.0251 / 0.0124 |
| mid | 10 | acc_signed | 0.106 | 0.122 | 0.135 | 3.0e-03 / 3.0e-03 | 0.0003 / 0.0005 |
| mid | 10 | tv | 0.801 | 0.768 | 0.742 | 7.4e-03 / 1.9e-03 | 0.0745 / 0.0366 |
| mid | 3 | acc | 0.608 | 0.399 | 0.474 | 2.9e-03 / 1.1e-03 | 0.0273 / 0.0122 |
| mid | 3 | acc_signed | 0.006 | 0.004 | 0.011 | 3.6e-03 / 3.6e-03 | 0.0001 / 0.0000 |
| mid | 3 | tv | 0.705 | 0.641 | 0.563 | 8.8e-03 / 2.0e-03 | 0.0810 / 0.0365 |
| mid | 5 | acc | 0.652 | 0.487 | 0.540 | 2.7e-03 / 1.0e-03 | 0.0265 / 0.0123 |
| mid | 5 | acc_signed | 0.021 | 0.022 | 0.032 | 3.4e-03 / 3.3e-03 | 0.0000 / 0.0001 |
| mid | 5 | tv | 0.732 | 0.677 | 0.623 | 8.3e-03 / 2.0e-03 | 0.0789 / 0.0370 |
| weak_high_between | 10 | acc | 0.770 | nan | 0.613 | 4.1e-05 / -3.8e-06 | 0.0034 / 0.0005 |
| weak_high_between | 10 | acc_signed | 0.103 | -0.144 | 0.146 | 5.3e-05 / 7.7e-05 | 0.0000 / -0.0003 |
| weak_high_between | 10 | tv | 0.851 | 0.043 | 0.688 | 1.2e-04 / 1.5e-05 | 0.0101 / 0.0017 |
| weak_high_between | 3 | acc | 0.634 | 0.007 | 0.401 | 5.0e-05 / 3.0e-05 | 0.0036 / 0.0005 |
| weak_high_between | 3 | acc_signed | 0.006 | -0.014 | 0.015 | 6.3e-05 / 1.3e-04 | 0.0000 / -0.0001 |
| weak_high_between | 3 | tv | 0.775 | 0.019 | 0.478 | 1.4e-04 / 2.3e-05 | 0.0109 / 0.0016 |
| weak_high_between | 5 | acc | 0.674 | -29.306 | 0.468 | 4.7e-05 / 1.1e-07 | 0.0036 / 0.0006 |
| weak_high_between | 5 | acc_signed | 0.020 | -0.032 | 0.039 | 6.0e-05 / 1.5e-04 | 0.0000 / -0.0001 |
| weak_high_between | 5 | tv | 0.796 | -0.285 | 0.547 | 1.3e-04 / 6.2e-06 | 0.0107 / 0.0017 |
| weak_low_between | 10 | acc | 0.735 | nan | 0.607 | 5.1e-05 / -3.7e-06 | 0.0040 / 0.0007 |
| weak_low_between | 10 | acc_signed | 0.101 | -0.003 | 0.143 | 6.6e-05 / 9.2e-05 | 0.0000 / 0.0005 |
| weak_low_between | 10 | tv | 0.752 | 0.373 | 0.687 | 1.5e-04 / 2.0e-05 | 0.0121 / 0.0017 |
| weak_low_between | 3 | acc | 0.590 | nan | 0.411 | 6.2e-05 / -4.9e-07 | 0.0044 / 0.0007 |
| weak_low_between | 3 | acc_signed | 0.005 | 0.014 | 0.015 | 8.1e-05 / 1.2e-04 | 0.0000 / 0.0002 |
| weak_low_between | 3 | tv | 0.633 | 0.372 | 0.477 | 1.8e-04 / 2.1e-05 | 0.0132 / 0.0016 |
| weak_low_between | 5 | acc | 0.628 | -8.383 | 0.479 | 5.8e-05 / 3.0e-07 | 0.0042 / 0.0006 |
| weak_low_between | 5 | acc_signed | 0.018 | 0.050 | 0.042 | 7.6e-05 / 8.8e-05 | 0.0000 / 0.0004 |
| weak_low_between | 5 | tv | 0.666 | 0.730 | 0.549 | 1.7e-04 / 1.2e-05 | 0.0129 / 0.0016 |
| weak_mid | 10 | acc | 0.730 | 0.600 | 0.617 | 5.4e-05 / 1.4e-05 | 0.0037 / 0.0008 |
| weak_mid | 10 | acc_signed | 0.102 | 0.292 | 0.165 | 6.8e-05 / 1.5e-04 | 0.0000 / -0.0000 |
| weak_mid | 10 | tv | 0.796 | nan | 0.696 | 1.7e-04 / -3.8e-06 | 0.0111 / 0.0017 |
| weak_mid | 3 | acc | 0.600 | 0.050 | 0.403 | 6.4e-05 / 3.9e-05 | 0.0040 / 0.0007 |
| weak_mid | 3 | acc_signed | 0.005 | 0.009 | 0.015 | 8.0e-05 / 1.6e-04 | -0.0000 / 0.0001 |
| weak_mid | 3 | tv | 0.702 | -0.125 | 0.479 | 2.0e-04 / 2.2e-05 | 0.0120 / 0.0016 |
| weak_mid | 5 | acc | 0.636 | 0.129 | 0.468 | 6.1e-05 / 2.9e-05 | 0.0039 / 0.0006 |
| weak_mid | 5 | acc_signed | 0.018 | 0.016 | 0.039 | 7.6e-05 / 1.1e-04 | -0.0000 / 0.0000 |
| weak_mid | 5 | tv | 0.728 | -0.129 | 0.550 | 1.9e-04 / 9.8e-06 | 0.0117 / 0.0016 |


## Full grid (every cell; point [95% problem-bootstrap CI])

R²_max is the cross-half, null-corrected estimate. "SD between / within" is √ of the
corrected components for TV. "mean m~" is the debiased mean m(h) for TV. "naive R2" is the
uncorrected full-sample ratio.
#### A_released

| sim | min | coverage | R2_max tv | R2_max tv_raw | R2_max \|acc\| | R2_max signed acc | SD between / within (tv) | mean m~ tv | naive R2 tv |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 5 | 0.92 | 0.57 [0.38, 0.77] | 0.57 [0.38, 0.76] | 0.50 [0.37, 0.66] | 0.03 [0.01, 0.05] | 0.0105 / 0.0091 | 0.0022 | 0.65 |
| 0.8 | 3 | 0.96 | 0.56 [0.35, 0.76] | 0.56 [0.34, 0.76] | 0.47 [0.31, 0.64] | 0.01 [0.00, 0.02] | 0.0103 / 0.0092 | 0.0021 | 0.58 |
| 0.8 | 10 | 0.86 | 0.68 [0.51, 0.83] | 0.68 [0.50, 0.84] | 0.60 [0.42, 0.81] | 0.15 [0.08, 0.25] | 0.0112 / 0.0077 | 0.0021 | 0.76 |
| 0.7 | 5 | 0.96 | 0.52 [0.38, 0.64] | 0.52 [0.40, 0.63] | 0.50 [0.31, 0.63] | 0.02 [0.00, 0.04] | 0.0075 / 0.0071 | 0.0017 | 0.64 |
| 0.7 | 3 | 0.98 | 0.54 [0.39, 0.71] | 0.54 [0.38, 0.72] | 0.52 [0.33, 0.79] | 0.00 [-0.00, 0.01] | 0.0074 / 0.0068 | 0.0017 | 0.57 |
| 0.7 | 10 | 0.91 | 0.64 [0.51, 0.79] | 0.64 [0.51, 0.81] | 0.63 [0.48, 0.86] | 0.09 [0.04, 0.15] | 0.0077 / 0.0058 | 0.0017 | 0.75 |
| 0.9 | 5 | 0.85 | 0.55 [0.40, 0.68] | 0.55 [0.38, 0.69] | 0.55 [0.39, 0.73] | 0.06 [0.02, 0.11] | 0.0126 / 0.0115 | 0.0031 | 0.66 |
| 0.9 | 3 | 0.90 | 0.50 [0.35, 0.65] | 0.50 [0.34, 0.64] | 0.47 [0.33, 0.64] | 0.02 [0.00, 0.03] | 0.0120 / 0.0120 | 0.0031 | 0.59 |
| 0.9 | 10 | 0.76 | 0.75 [0.67, 0.82] | 0.75 [0.67, 0.82] | 0.79 [0.70, 0.92] | 0.19 [-0.01, 0.37] | 0.0149 / 0.0085 | 0.0032 | 0.79 |

#### B_r12

| sim | min | coverage | R2_max tv | R2_max tv_raw | R2_max \|acc\| | R2_max signed acc | SD between / within (tv) | mean m~ tv | naive R2 tv |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 5 | 0.96 | 0.29 [0.13, 0.88] | 0.29 [0.12, 0.82] | 0.28 [0.02, 1.56] | 0.02 [-0.01, 0.04] | 0.0044 / 0.0069 | 0.0025 | 0.49 |
| 0.8 | 3 | 0.98 | 0.19 [0.08, 1.71] | 0.19 [0.08, 1.50] | 0.16 [0.08, 1.50] | -0.00 [-0.02, 0.00] | 0.0043 / 0.0088 | 0.0026 | 0.44 |
| 0.8 | 10 | 0.92 | 0.53 [0.26, 2.00] | 0.54 [0.28, 1.82] | 0.47 [0.05, 0.98] | 0.02 [-0.05, 0.07] | 0.0051 / 0.0047 | 0.0025 | 0.57 |
| 0.7 | 5 | 0.98 | 0.66 [0.14, 2.09] | 0.67 [0.16, 2.25] | 0.81 [0.08, 2.84] | -0.01 [-0.04, -0.00] | 0.0066 / 0.0048 | 0.0025 | 0.57 |
| 0.7 | 3 | 0.99 | 0.58 [0.11, 1.69] | 0.60 [0.12, 1.80] | 0.81 [0.05, 3.32] | -0.01 [-0.05, 0.00] | 0.0063 / 0.0053 | 0.0024 | 0.53 |
| 0.7 | 10 | 0.96 | 0.50 [0.11, 0.86] | 0.52 [0.13, 0.89] | 0.57 [0.00, 0.90] | -0.01 [-0.07, 0.00] | 0.0066 / 0.0066 | 0.0026 | 0.64 |
| 0.9 | 5 | 0.90 | nan [-1.40, 4.08] | nan [-1.71, 3.62] | 4.99 [-0.71, 4.08] | 0.08 [-0.01, 0.58] | 0.0029 / 0.0000 | 0.0024 | 0.51 |
| 0.9 | 3 | 0.94 | 0.38 [-0.92, 3.31] | 0.39 [-1.10, 4.41] | 0.11 [-0.92, 2.92] | 0.03 [-0.03, 0.05] | 0.0020 / 0.0026 | 0.0024 | 0.45 |
| 0.9 | 10 | 0.83 | 0.31 [-1.12, 0.61] | 0.31 [-0.95, 0.56] | 0.24 [-1.00, 0.36] | -0.24 [-0.63, -0.02] | 0.0047 / 0.0071 | 0.0020 | 0.60 |

#### B_r012

| sim | min | coverage | R2_max tv | R2_max tv_raw | R2_max \|acc\| | R2_max signed acc | SD between / within (tv) | mean m~ tv | naive R2 tv |
|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 5 | 0.96 | 0.48 [0.35, 0.98] | 0.48 [0.35, 0.93] | 0.75 [0.44, 2.65] | 0.00 [-0.01, 0.02] | 0.0064 / 0.0066 | 0.0036 | 0.46 |
| 0.8 | 3 | 0.98 | 0.36 [0.23, 1.20] | 0.36 [0.23, 1.32] | 0.54 [0.26, 3.21] | -0.00 [-0.01, 0.01] | 0.0061 / 0.0081 | 0.0034 | 0.43 |
| 0.8 | 10 | 0.93 | 0.52 [0.35, 0.89] | 0.52 [0.35, 0.90] | 0.63 [0.36, 2.24] | 0.03 [-0.01, 0.04] | 0.0073 / 0.0070 | 0.0037 | 0.53 |
| 0.7 | 5 | 0.98 | 0.68 [0.33, 2.79] | 0.69 [0.34, 2.38] | 0.73 [0.25, 3.25] | 0.00 [-0.02, 0.01] | 0.0073 / 0.0050 | 0.0032 | 0.54 |
| 0.7 | 3 | 0.99 | 0.65 [0.37, 2.90] | 0.65 [0.41, 2.92] | 0.70 [0.32, 3.54] | -0.00 [-0.01, 0.00] | 0.0071 / 0.0053 | 0.0031 | 0.49 |
| 0.7 | 10 | 0.96 | 0.70 [0.36, 2.44] | 0.70 [0.36, 2.43] | 0.73 [0.21, 3.45] | 0.03 [-0.02, 0.06] | 0.0075 / 0.0049 | 0.0033 | 0.60 |
| 0.9 | 5 | 0.91 | 0.39 [0.22, 0.67] | 0.39 [0.20, 0.64] | 0.54 [0.26, 1.77] | 0.03 [-0.02, 0.05] | 0.0075 / 0.0095 | 0.0048 | 0.43 |
| 0.9 | 3 | 0.94 | 0.52 [0.36, 2.44] | 0.53 [0.37, 2.12] | 0.74 [0.41, 6.52] | 0.01 [-0.02, 0.02] | 0.0073 / 0.0071 | 0.0045 | 0.40 |
| 0.9 | 10 | 0.85 | 0.42 [0.24, 0.55] | 0.42 [0.23, 0.55] | 0.58 [0.27, 2.09] | 0.07 [-0.01, 0.11] | 0.0084 / 0.0098 | 0.0048 | 0.51 |


## Recommended K and M for the GPU nested arm

- **Inputs.** Per-sentence noise: the variance of a debiased half estimate is about
  a / n_half, with a = 0.016 (TV, released) to 0.020 (TV, B r0+r1+r2).
- **Simulation.** A Gaussian nested design (problem + position + sentence), with variance
  components from the primary cell.
  - 20 positions per problem; P ∈ {20, 40}; K ∈ {2, 4, 8, 16, 32}; M ∈ {10, 20, 50, 100, 200}.
  - Cross-half nested estimators, unbiased for finite K and F; 400 simulations per cell.
  - The grid was fixed in code; all cells are in `results.json` → `design`.
- **Cheapest cell with a 95% width of at most 0.2 for the within-problem R²_max, at P = 20:**
  - TV, released components: **K = 4, M = 200** (width 0.14; 320k generations). K = 8,
    M = 100 also reaches 0.13.
  - TV, Phase 2 components: **K = 8, M = 200** (width 0.15; 640k generations).
  - |Δacc|: K = 4, M = 200 on released components; K = 32, M = 200 on Phase 2 components.
- **Recommendation:** K = 8 sentences per position and M = 200 continuations per sentence,
  over 20 problems × 20 positions, about 640k generations. M must not drop below about 200:
  - the simulation is linear and Gaussian, so it ignores the TV nonlinearity that breaks
    the estimator in the weak regime;
  - the per-sentence noise SD √(2a/M) ≈ 0.013 at M = 200 is already about the size of the
    estimated true SD of I (≈ 0.01).
- **Choice of estimand.** The nested arm should use an importance with an unbiased
  estimator rather than TV with a null correction: for example, the squared-L2 distance or
  (Δacc)², each with cross-half U-statistic products. Signed targets need not be measured
  for predictability, because their ceiling is 0.

## What is resolved, and what remains

- **Resolved.**
  - Signed importance targets are unpredictable from h by construction. The earlier probe
    nulls on signed targets carry no information.
  - Answer-entropy structure explains almost none of the between-history variation in
    importance.
  - The across-problem component is negligible, so the overall and within-trace ceilings
    are the same.
  - Raw versus canonical grouping makes no difference (tv vs tv_raw R²_max agree within 0.02 in every
    cell).
- **Remains.**
  - The unsigned ceiling (point 0.57 on released data) rests on an estimator that is valid
    only for effects that are large relative to the per-cluster noise. The real data sit
    near the weak regime where it breaks down.
  - Clustering coarsens S, which biases R²_max upward relative to the sentence-level
    ceiling.
  - 100 to 300 samples per fork leave most clusters with 5 to 30 members.
  - The proper estimator is the nested arm (K sentences, then M continuations per sentence)
    with an unbiased importance.

## Deviations from the brief

- **Added after the first run, with the pre-set grid unchanged:**
  - the `acc_signed` metric;
  - the entropy diagnostic;
  - the three weak-effect synthetic settings.
- **Corrected after the first run:** the design simulation. The first version left a
  σ²_w/K term in Between, which biased R² upward for small K.
- **Choices fixed in code before the first run:**
  - average linkage;
  - small clusters excluded with weights renormalised;
  - P(A|h) from all of the fork's samples;
  - 16 splits.
- **Hardware:** embeddings (251,269 unique sentences) were computed with torch MPS, the
  machine's local Apple GPU. No CUDA, remote or cloud GPU was used.
