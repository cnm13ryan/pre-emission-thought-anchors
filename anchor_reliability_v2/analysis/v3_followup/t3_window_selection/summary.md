# T3 (item 7): window selection and the anchor rate

**Status: EXPLORATORY, POST HOC.** This is a follow-up to the plan_v3 pilot. No pilot file, script, src file or shared table was changed. `run.py` reproduces every number here, in about 5 minutes on CPU. The full numbers are in `results.json`. Intermediates are under `~/preanchor-root/anchor_reliability_v2/derived/v3_followup/t3_window_selection/`: the per-sentence table, the per-sentence tests, the rate for each rule seed and the Phase 2 cross-check.

**Verdict.** Window selection does **not** explain the low Phase 2 anchor rate.
- **Anchors are rare everywhere.** On the released data (100 vs 100 samples, BH 10% over all 6,434 sentences), only **2.0%** of all sentences are anchors, with 95% CI [1.65, 2.47].
- **The Phase 2 rule already selects an enriched region.** The rule is confined to the pre-convergence region, where the rate is **2.8%**, with CI [2.2, 3.4].
- **The realised window is typical.** It has **7/190 = 3.7%** anchors, which is the 67th percentile of the rule's distribution over 1,000 seeds (median 3.2%, 95% range 0.5–6.8%).
- **The best fixed criterion only about doubles the rate.** That criterion is pre-convergence ∩ top entropy tercile, which gives **4.3%** [3.1, 5.7]. It covers 30% of sentences and keeps 63% of anchors.

## 1. The Phase 2 window-selection rule

The rule is in `research/redesign/phase2_select.py` (parent repo) and its output is `configs/phase2_forks.json`. `run.py` re-implements it, and with the realised seed it reproduces all 10 (trajectory, start) pairs exactly.

1. **Candidate problems.** The candidates are the 16 MVP problems in `labels.parquet`, minus the 4 test problems {6998, 2236, 2870, 3916}. They are sorted ascending.
2. **The generator.** `rng = numpy.random.default_rng(int(sha256(b"20260914:phase2_selection").hexdigest()[:8], 16))`.
3. **The draw.** First `order = rng.permutation(problems)`, then `kinds = rng.choice(["correct","incorrect"], size=16)`. Trajectories are then visited in that order until 10 are chosen.
4. **For trajectory `"{problem}:{kind}"`:**
   - `n` is the number of sentences in `~/preanchor-root/data/processed/stage4/labels.parquet`.
   - `conv = convergence_idx` is the frozen K=5 convergence index, a per-trajectory constant in labels.parquet.
   - `final = final_answer_idx` is the final-answer sentence, also from labels.parquet; it is NaN when there is none.
   - `limit = min(x for x in (n-1, conv, final) if x is not NaN)`.
5. **Admissible starts.** `starts = {s in 0..n-1 : s+18 < limit and s+19 <= n-1}`. Both sentences s..s+18 must come before `limit`, and fork s+19 must exist.
   - If `starts` is empty, the trajectory is skipped and the next draw is used. No draw was skipped for the realised seed or for any of the 1,000 alternative seeds.
   - Otherwise `s = rng.choice(starts)`, drawn uniformly. The window is forks s..s+19, with sentence labels s..s+18 (19 units).
6. **Importance plays no part in the rule.** It depends only on the seed, n, conv and final.

The admissible region of a trajectory is `sentence_idx < limit`; criterion (b) below uses this region. In labels.parquet, `final_answer_idx` is either NaN or n−1. Because a sentence needs fork i+1 to be scored, every scored sentence (i ≤ n−2) already lies before the final answer. So criterion (g) drops 0 sentences and equals (a).

## 2. Anchor definitions (fixed in code before any rate was computed)

The data are `released_canon.parquet`, replicate 0: 6,434 sentences, 40 traces and 20 problems, each fork with exactly 100 samples.

- **Primary: permutation test on canonical categories.** It compares the fork-i and fork-(i+1) distributions over canonical categories.
  - The statistic is the integer D = Σ|cA−cB| = 200·TV, which avoids floating-point ties.
  - Shuffling the labels of the 100+100 pooled samples gives group-A counts that follow a multivariate hypergeometric distribution exactly. So each permutation is drawn as a hypergeometric count vector, vectorised in batches of 20,000.
  - The test is a sequential Monte Carlo (Besag–Clifford) test. It stops once there are at least 100 exceedances, with p = exceedances/draws, checked after each batch. Otherwise it runs to 1,000,000 draws, with p = (e+1)/(N+1).
  - **Resolution.** The minimum attainable p is 1.0e-6, below the BH rank-1 threshold of 0.1/6,434 = 1.55e-5.
  - **Draw counts.** The median sentence needed 20,000 draws. 69 sentences ran to the 1,000,000 cap, all with p ≈ 1e-6. 866 sentences are trivially p = 1, because one category covers both forks or the two forks are identical.
  - **BH.** BH is applied at 10% over all 6,434 sentences. It flags **131** anchors, and its cut-off is p* = 0.00185.
- **Secondary: |Δaccuracy| on canonical correctness.** With a single binary variable the permutation null is hypergeometric, so the p-value is computed **exactly**. BH 10% over all sentences flags **91** anchors (p* = 0.0014). Agreement with the primary is κ = 0.80.
- **Reference: the MVP flag.** A sentence is flagged when `mvp_importance` ≥ 1.3549810893213545; this flags **358** sentences. 874 sentences have no MVP score (MVP-ineligible) and count as non-anchors.
  - The flag is dominated by the noise floor. Its Spearman correlation with fork-i answer entropy is 0.51 (0.62 among eligible sentences), and 78% of its flags fall in the top entropy tercile.
  - It agrees poorly with the primary: only 7.8% of its flags are primary anchors, and κ = 0.09.
- **Sensitivity: the primary test on raw categories.** It flags 133 anchors, with κ = 0.99 against canonical. Raw grouping changes nothing.

## 3. Anchor rate by selection criterion

The table uses the primary definition. Each CI is a problem-cluster bootstrap with 2,000 replicates. Enrichment is the criterion's rate divided by the all-sentence rate, from a paired bootstrap. Recall is the share of all 131 anchors that the criterion keeps. `results.json` reports every cell for all four definitions.

| Criterion | n | Coverage | Anchors | Rate % [95% CI] | Enrichment [CI] | Recall |
|---|---|---|---|---|---|---|
| (a) all sentences | 6434 | 1.00 | 131 | 2.04 [1.65, 2.47] | 1 | 1.00 |
| (b) pre-convergence (rule's admissible region) | 4627 | 0.72 | 128 | 2.77 [2.22, 3.38] | 1.36 [1.22, 1.56] | 0.98 |
| (b′) post-convergence | 1807 | 0.28 | 3 | 0.17 [0.00, 0.38] | 0.08 [0.00, 0.21] | 0.02 |
| (c) realised Phase 2 window | 190 | 0.03 | 7 | 3.68 [0.75, 7.24] | 1.81 [0.34, 3.82] | 0.05 |
| (d) position early (i/(n−1) < 1/3) | 2157 | 0.34 | 50 | 2.32 [1.63, 3.10] | 1.14 [0.82, 1.51] | 0.38 |
| (d) position mid | 2147 | 0.33 | 55 | 2.56 [1.75, 3.46] | 1.26 [0.97, 1.51] | 0.42 |
| (d) position late | 2130 | 0.33 | 26 | 1.22 [0.68, 1.80] | 0.60 [0.36, 0.82] | 0.20 |
| (e) entropy low tercile | 2145 | 0.33 | 6 | 0.28 [0.08, 0.53] | 0.14 [0.04, 0.28] | 0.05 |
| (e) entropy mid tercile | 2145 | 0.33 | 43 | 2.00 [1.41, 2.67] | 0.98 [0.76, 1.23] | 0.33 |
| (e) entropy top tercile | 2144 | 0.33 | 82 | 3.82 [2.74, 5.42] | 1.88 [1.54, 2.32] | 0.63 |
| (f) correct trace | 2932 | 0.46 | 46 | 1.57 [1.07, 2.14] | 0.77 [0.57, 0.99] | 0.35 |
| (f) incorrect trace | 3502 | 0.54 | 85 | 2.43 [1.82, 3.13] | 1.19 [1.02, 1.40] | 0.65 |
| (g) excluding sentences after the final answer | 6434 | 1.00 | 131 | 2.04 [1.66, 2.47] | 1 | 1.00 |
| (h) s17 top 50% by 20-sample entropy | 3217 | 0.50 | 107 | 3.33 [2.36, 4.57] | 1.63 [1.40, 1.95] | 0.82 |
| non-test problems | 5361 | 0.83 | 113 | 2.11 [1.64, 2.65] | 1.04 [0.99, 1.11] | 0.86 |
| pre-convergence ∩ non-test | 3862 | 0.60 | 110 | 2.85 [2.21, 3.60] | 1.40 [1.24, 1.63] | 0.84 |
| pre-convergence ∩ entropy top tercile | 1928 | 0.30 | 82 | **4.25 [3.11, 5.74]** | **2.09 [1.74, 2.56]** | 0.63 |
| pre-convergence ∩ s17 top 50% | 2926 | 0.45 | 106 | 3.62 [2.69, 4.87] | 1.78 [1.52, 2.15] | 0.81 |
| entropy = 0 at fork i | 1007 | 0.16 | 2 | 0.20 [0.00, 0.55] | 0.10 [0.00, 0.28] | 0.02 |

**How the bins and cut-offs were set.**
- **Entropy terciles** are global. Ties are broken randomly with a seed, and the cut-offs are 0.199 and 0.641 nats.
- **The s17 top-50% criterion** uses entropy from the first 20 samples of fork i, cut at 0.394 nats, with ties also broken randomly. 30% of sentences have zero 20-sample entropy.
- **Secondary and reference definitions** follow the same pattern. Under the secondary definition, the top entropy tercile gives 2.66% [1.79, 3.77] and pre-convergence gives 1.95% [1.46, 2.46].

**Caveat on (e) and (h).** Fork-i entropy comes from the same samples the test uses: all of them for (e), and 20 of the 100 for (h). Part of the entropy enrichment may therefore be test power rather than prevalence, since a low-entropy fork rarely differs detectably. Criterion (h) is the version that could actually be deployed, because its pilot would be a separate draw.

## 4. Phase 2 rule under 1,000 alternative seeds, and the cross-check

**The rule's distribution.** Under the primary definition, the anchor rate over the 190 units the rule would select has mean 3.40% and SD 1.67%. Its quantiles are 2.5% → 0.53%, 50% → 3.16% and 97.5% → 6.84%. 0.3% of seeds give zero anchors.
- **The realised seed sits in the middle.** It gives 7/190 = 3.68%. 67% of seeds give a rate at or below it, and 45% give a rate at or above it. **So the realised window is neither anchor-poor nor anchor-rich.**
- **Secondary definition.** The realised rate is 2.11%, against a seed median of 2.11%.
- **Reference flag.** The realised rate is 3.68%, against a seed median of 6.84%; only 14% of seeds give a rate at or below it. The realised window has low entropy: its mean fork-i entropy is 0.449, against a seed median of 0.576 and a 2.5% quantile of 0.406. That makes it look poor on the floor-dominated MVP flag, but not on the real-effect tests.

**Cross-check with Phase 2, on the actual 190 units.**
- **r0 alone matches the pooled result.** The r0-only primary flags (global BH) give **7/190**, the same count as the pooled 300-vs-300 real-effect test (`sentence_effects_canon.csv`, 7/190).
  - 6 of the 7 overlap, κ = 0.85.
  - Applying BH within the 190 units to the r0 p-values also gives 7, with the same 6 overlapping.
  - The pooled real units are concentrated: 4 in 2238:incorrect, 2 in 3935:correct and 1 in 330:correct.
- **Our own replicates find fewer.** r1 and r2, each tested alone at 100 vs 100 and flagged at r0's global cut-off p*, give 4/190 and 3/190 respectively.
- **The pooled test has coarse p-values.** Its permutation p-values sit at the 1/2001 floor (2,000 permutations), so it cannot tell strong effects apart.

## Resolved

- **The low Phase 2 anchor rate does not come from the window.**
  - Large detectable effects are rare in whole traces: 2.0% at 100 vs 100.
  - The rule's pre-convergence restriction already enriches by 1.36× [1.22, 1.56], because 98% of anchors lie before convergence; post-convergence has 0.17%.
  - The realised seed is a typical draw from the rule.
- **The best fixed enrichment is about 2×.** Pre-convergence ∩ top entropy tercile gives 4.25% at 30% coverage.
  - The deployable version is pre-convergence ∩ s17 top 50% by 20-sample entropy. It gives 3.62% (1.78×) at 45% coverage and keeps 81% of anchors.
  - Position does little: early and mid are about 1.2×, and late is 0.6×.
  - Incorrect traces give 1.19× over all sentences. That is only about 1.5× the correct-trace rate (2.43% vs 1.57%).
- **The MVP q90 flag is not a usable anchor definition here.** It tracks entropy (ρ = 0.51), and κ with the primary is 0.09.

## Remaining

- **The rates are detection rates at 100 vs 100 samples, not true prevalence.** Smaller effects are invisible at this depth, and the entropy criteria partly select for test power.
- **The concentration in 3 problems needs a check across the problems a future run would use.** The CIs already treat problems as clusters, with 20 clusters.
- **Expected yield.** No criterion pushes the detectable rate past about 4–6% at this depth. A future run should budget for about 1 detectable anchor per 25 targeted sentences at 100 vs 100, and a deeper N should raise this.
