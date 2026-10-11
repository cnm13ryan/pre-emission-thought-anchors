# Targeted resampling: uncertainty as an anchor-opportunity prior, and Phase 3 sizing

**Status:** EXPLORATORY. Not preregistered, written after the confirmatory FAIL
(`docs/phase2_results.md`) and the s13/s14 diagnostics. Source:
`scripts/s17_targeted_sampling.py`, which writes `results/s17_targeted_sampling.json`
(CPU only, about 7 minutes). Every number below comes from that JSON.

**The main limitation:** everything here rests on **7 positive sentences in 3 of 10
problems**, and 2 of the 7 (2238 s25/s26) may be one spike. Read the evaluation as a
plausibility check and the sizing as order-of-magnitude.

## Setup

- **Data:** Phase 2 pooled samples, 300 resamples per fork (replicates 0, 1 and 2), 200
  forks and 190 labelled sentences ("units") in 10 trajectories.
- **Positives:** the 7 units with `real_tv_bh10` in `results/s13_sentence_effects.csv`
  (permutation test on TV, 300 vs 300 samples, BH at 10%).
- **Cost model:** $108.42 / 40,000 generations, or **$2.71 per 1,000 generations**, at about
  920 generations per H100-hour.
  - The JSON also gives a token-weighted cost, using each fork's mean generated tokens. Its
    weights range from 0.48 to 1.74.
  - The token-weighted cost differs from the flat cost by under 2% for every design, because
    uncertain forks are not systematically longer. The tables use the flat cost.

## 1. Does fork uncertainty rank the 7 real effects?

### Scores

All scores are computed at fork i (`_i`) or on the pooled samples of forks i and i+1
(`_pair`). A pilot of n resamples per fork is a random subsample of the 300; there are 400
pilot draws per n.

| Score | Definition |
|---|---|
| `entropy` | Plug-in answer entropy |
| `floor` | Analytic E[TV] with no effect |
| `accunc` | 1 − \|2p − 1\| |

### How to read the table

- **Random baseline:** AP = 0.037, recall@⅓ = 0.33 and recall@½ = 0.50.
- **Each cell** shows the mean over pilot draws, then two 95% bootstrap intervals: one over
  sentences and one over trajectory clusters. Both bootstraps also resample the pilot draw.

| Pilot n | Score | AP | Recall, top ⅓ of units | Recall, top ½ of units |
|---|---|---|---|---|
| 10 | entropy_i | 0.12 [0.03, 0.44] / [0.01, 0.52] | 0.62 [0.17, 1.00] / [0.00, 1.00] | 0.73 [0.25, 1.00] / [0.20, 1.00] |
| 10 | entropy_pair | 0.09 [0.02, 0.30] / [0.02, 0.33] | 0.60 [0.00, 1.00] / [0.00, 1.00] | 0.79 [0.30, 1.00] / [0.14, 1.00] |
| 10 | accunc_i | 0.14 [0.02, 0.47] / [0.01, 0.48] | 0.61 [0.17, 1.00] / [0.00, 1.00] | 0.73 [0.25, 1.00] / [0.25, 1.00] |
| 10 | floor_i | 0.10 [0.02, 0.39] / [0.02, 0.45] | 0.62 [0.14, 1.00] / [0.00, 1.00] | 0.73 [0.25, 1.00] / [0.20, 1.00] |
| 20 | entropy_i | 0.13 [0.03, 0.45] / [0.02, 0.56] | 0.65 [0.22, 1.00] / [0.20, 1.00] | 0.73 [0.33, 1.00] / [0.40, 1.00] |
| 20 | entropy_pair | 0.09 [0.03, 0.31] / [0.02, 0.33] | 0.61 [0.12, 1.00] / [0.00, 1.00] | 0.80 [0.33, 1.00] / [0.20, 1.00] |
| 20 | accunc_i | 0.17 [0.03, 0.54] / [0.02, 0.53] | 0.66 [0.25, 1.00] / [0.08, 1.00] | 0.73 [0.33, 1.00] / [0.44, 1.00] |
| 20 | floor_i | 0.12 [0.02, 0.41] / [0.02, 0.49] | 0.64 [0.17, 1.00] / [0.09, 1.00] | 0.73 [0.25, 1.00] / [0.38, 1.00] |
| 50 | entropy_i | 0.15 [0.03, 0.46] / [0.03, 0.68] | 0.69 [0.27, 1.00] / [0.33, 1.00] | 0.73 [0.33, 1.00] / [0.50, 1.00] |
| 50 | entropy_pair | 0.09 [0.03, 0.28] / [0.02, 0.26] | 0.59 [0.17, 1.00] / [0.00, 1.00] | 0.85 [0.43, 1.00] / [0.33, 1.00] |
| 50 | accunc_i | 0.19 [0.03, 0.53] / [0.03, 0.71] | 0.69 [0.25, 1.00] / [0.38, 1.00] | 0.73 [0.33, 1.00] / [0.50, 1.00] |
| 50 | floor_i | 0.13 [0.02, 0.44] / [0.03, 0.54] | 0.68 [0.25, 1.00] / [0.22, 1.00] | 0.74 [0.33, 1.00] / [0.50, 1.00] |
| 300 | entropy_i | 0.21 [0.02, 0.53] / [0.04, 1.00] | 0.71 [0.33, 1.00] / [0.50, 1.00] | 0.71 [0.33, 1.00] / [0.50, 1.00] |
| 300 | entropy_pair | 0.08 [0.03, 0.22] / [0.02, 0.17] | 0.43 [0.11, 1.00] / [0.00, 1.00] | 1.00 [0.67, 1.00] / [0.33, 1.00] |
| 300 | accunc_i | 0.26 [0.04, 0.60] / [0.05, 1.00] | 0.71 [0.33, 1.00] / [0.50, 1.00] | 0.71 [0.33, 1.00] / [0.50, 1.00] |
| 300 | floor_i | 0.13 [0.02, 0.43] / [0.03, 0.50] | 0.57 [0.25, 1.00] / [0.14, 1.00] | 0.71 [0.33, 1.00] / [0.50, 1.00] |

### Full-depth label-permutation p-values (5,000 shuffles)

| Score | AP | Recall@⅓ | Recall@½ |
|---|---|---|---|
| entropy_i | p = 0.009 | 5/7, p = 0.042 | p = 0.22 |
| accunc_i | p = 0.005 | 5/7, p = 0.043 | p = 0.22 |
| entropy_pair | p = 0.16 | 3/7, p = 0.43 | 7/7, p = 0.007 |

### Reading

- **Uncertainty is a weak but real enrichment signal.**
  - A 10–20-resample pilot keeps about 60–66% of positives in the top third of units, and
    73–80% in the top half.
  - That is roughly 2× and 1.5× enrichment over random.
  - The pilot recovers most of what full depth gives. Ranking by uncertainty is cheap
    because entropy is reliable even at small n.
- **The intervals are very wide.**
  - The trajectory-cluster intervals for recall@⅓ typically run from 0 to 1.
  - Nothing here would survive a preregistered test at this sample size. The label-permutation
    p-values above also ignore clustering and the choice among six scores.
- **The two fork-i misses are informative.**
  - 2238 s26 and 3935 s21 sit in the bottom tercile of fork-i entropy, with accuracy 0.93
    and 0.94 at fork i.
  - In both, the answer distribution becomes uncertain *after* the sentence, so the effect
    shows at fork i+1.
  - `entropy_pair` (both forks pooled) catches them: all 7 fall in its top half. But it
    ranks the top of the list less sharply.
  - This motivated using `entropy_pair` for the design. It was chosen before the simulation
    was run, on validity grounds: it is symmetric in the two forks and covers transitions in
    either direction.
- **Held-out replicate check.**
  - Positives were re-derived from two replicates (200 vs 200, BH 10%). That gives 6, 7 and
    8 positives, of which 6, 7 and 7 overlap the s13 seven.
  - Uncertainty was then estimated from the third replicate only.
  - Pilot recall@½ was 0.66–0.98, which is consistent with the table, so the result is not
    an artefact of re-using the same samples.

## 2. Simulated two-stage design

### Generative model

- **Truth for each unit:** the two forks' answer distributions are the pooled 300-sample
  empirical distributions.
- **Real units (the 7):** keep their observed fork-i and fork-(i+1) distributions, shrunk
  toward their mixture by λ.
  - λ ∈ {1, 0.75, 0.5}. λ < 1 is a winner's-curse sensitivity: effects selected for being
    significant are biased upward.
- **Null units (183):** both forks equal the pooled mixture.
- **Stage 1:**
  - Draw m samples per fork on all 200 forks.
  - Rank units by `entropy_pair` and keep the top 50% or 33%.
  - `entropy_i` results are in the JSON. They are similar, and slightly worse at top 50%.
- **Stage 2:**
  - Top up both forks of each selected unit to N, reusing the pilot samples. Forks shared
    by adjacent selected units are paid for once.
  - Run an exact permutation TV test (multivariate-hypergeometric null, 2,000 permutations).
  - Apply BH at 10% over the selected units only.
- **Repetitions:** 300 per configuration, with common random numbers across configurations.
- **Cost unit:** one Phase 2-sized study (190 units, 200 forks). Uniform sampling at N = 200
  is the same 40,000 generations as Phase 2.

### λ = 1 (observed effect sizes taken at face value)

| Design | m | N | Cutoff | E[TP] of 7 | E[FP] | Mean FDP | Positives selected | Generations | $ | TP per $1k |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | – | 100 | all | 4.18 | 0.26 | 0.05 | 7.0 | 20,000 | 54 | 77.0 |
| uniform | – | 200 | all | 6.38 | 0.45 | 0.06 | 7.0 | 40,000 | 108 | 58.8 |
| uniform | – | 300 | all | 6.77 | 0.56 | 0.07 | 7.0 | 60,000 | 163 | 41.6 |
| uniform | – | 500 | all | 6.98 | 0.58 | 0.07 | 7.0 | 100,000 | 271 | 25.8 |
| two-stage | 10 | 200 | top50 | 4.97 | 0.44 | 0.07 | 5.5 | 26,191 | 71 | 70.0 |
| two-stage | 10 | 200 | top33 | 3.81 | 0.38 | 0.07 | 4.2 | 19,383 | 53 | 72.6 |
| two-stage | 10 | 300 | top50 | 5.32 | 0.46 | 0.07 | 5.5 | 38,939 | 106 | 50.4 |
| two-stage | 10 | 300 | top33 | 4.07 | 0.38 | 0.07 | 4.2 | 28,543 | 77 | 52.7 |
| two-stage | 10 | 500 | top50 | 5.47 | 0.51 | 0.07 | 5.5 | 64,361 | 174 | 31.4 |
| two-stage | 10 | 500 | top33 | 4.21 | 0.42 | 0.07 | 4.2 | 46,855 | 127 | 33.2 |
| two-stage | 20 | 200 | top50 | 5.18 | 0.45 | 0.07 | 5.7 | 26,119 | 71 | 73.1 |
| two-stage | 20 | 200 | top33 | 3.84 | 0.38 | 0.07 | 4.3 | 19,677 | 53 | 72.0 |
| two-stage | 20 | 300 | top50 | 5.50 | 0.45 | 0.06 | 5.7 | 38,405 | 104 | 52.8 |
| two-stage | 20 | 300 | top33 | 4.16 | 0.38 | 0.07 | 4.3 | 28,391 | 77 | 54.1 |
| two-stage | 20 | 500 | top50 | 5.65 | 0.52 | 0.07 | 5.7 | 63,003 | 171 | 33.1 |
| two-stage | 20 | 500 | top33 | 4.29 | 0.45 | 0.08 | 4.3 | 45,806 | 124 | 34.5 |
| two-stage | 30 | 200 | top50 | 5.22 | 0.44 | 0.06 | 5.7 | 26,368 | 71 | 73.1 |
| two-stage | 30 | 200 | top33 | 3.81 | 0.37 | 0.07 | 4.3 | 20,321 | 55 | 69.2 |
| two-stage | 30 | 300 | top50 | 5.56 | 0.49 | 0.07 | 5.7 | 38,381 | 104 | 53.4 |
| two-stage | 30 | 300 | top33 | 4.11 | 0.41 | 0.07 | 4.3 | 28,750 | 78 | 52.8 |
| two-stage | 30 | 500 | top50 | 5.71 | 0.51 | 0.07 | 5.7 | 62,362 | 169 | 33.8 |
| two-stage | 30 | 500 | top33 | 4.29 | 0.43 | 0.07 | 4.3 | 45,601 | 124 | 34.7 |

The uniform N = 1,000 design (7.00 TP, $542) is in the JSON.

### Shrunk effects: E[TP] / TP per $1k

| Design | λ = 0.75 | λ = 0.5 |
|---|---|---|
| uniform N = 100 | 1.91 / 35.3 | 0.35 / 6.4 |
| uniform N = 200 | 4.70 / 43.4 | 1.62 / 14.9 |
| uniform N = 300 | 6.04 / 37.2 | 3.19 / 19.6 |
| uniform N = 500 | 6.82 / 25.2 | 5.04 / 18.6 |
| two-stage m = 20, N = 200, top50 | 4.03 / 56.9 | 1.69 / 23.8 |
| two-stage m = 30, N = 200, top50 | 4.19 / 58.6 | 1.76 / 24.6 |
| **two-stage m = 20, N = 300, top50** | **4.93 / 47.4** | **2.77 / 26.6** |
| two-stage m = 30, N = 300, top50 | 5.06 / 48.7 | 2.93 / 28.1 |
| two-stage m = 20, N = 300, top33 | 3.68 / 47.8 | 1.97 / 25.5 |
| two-stage m = 20, N = 500, top50 | 5.50 / 32.2 | 4.15 / 24.3 |

### Reading the simulation

- **FDR is controlled.**
  - Mean FDP is 0.04–0.08 in every configuration. That is below the nominal 0.10, because
    π0 ≈ 0.96 and permutation p-values are discrete.
  - Reusing pilot samples in the stage-2 test did not visibly inflate it.
  - At the target scale, expect about 8–9 false discoveries per 100 true ones.
- **Selection costs positives, roughly as Task 1 predicts.**
  - Top 50% keeps about 5.7 of 7 positives (81%); top 33% keeps about 4.3 (61%).
  - Top 33% saves less money than its share suggests. Selected units in adjacent positions
    share forks, so the fork union is larger than the unit share, and the pilot is a fixed
    overhead.
  - As a result, top 33% is never clearly better than top 50%.
- **Whether targeting pays depends on the true effect size.**
  - At λ = 1, effects are large enough that cheap uniform sampling (N = 100–200) is as
    efficient as targeting.
  - At λ ≤ 0.75, two-stage top-50 designs beat the best uniform design by about 1.3–1.4×
    in true positives per dollar.
  - λ = 1 is the least plausible value, since the observed effects are selected maxima.
- **The pilot size m hardly matters.**
  - The difference between m = 10, 20 and 30 is within Monte Carlo error. m = 20–30 is
    slightly better at no real cost.
- **On the MVP's 0 of 6,434:** in the simulation, N = 100 still detects some effects, while
  the MVP found none at 100 vs 100.
  - With BH over 6,434 tests, the rank-1 threshold is 0.1/6,434 ≈ 1.6 × 10⁻⁵. That is below
    the smallest attainable permutation p-value at 2,000 permutations (5 × 10⁻⁴).
  - So the MVP null is at least partly a resolution artefact, not evidence that effects are
    absent.
  - Phase 3 must use enough permutations (or an exact or analytic test) for its number of
    tests.

## 3. Phase 3 sizing: about 100 true anchor detections

### Prevalence

- **Point estimate:** 7/190 = 3.7%.
- **Wilson 95%:** [1.8%, 7.4%].
- **Trajectory-cluster bootstrap 95%:** [0%, 8.4%]. 7 of 10 problems have no effect, so
  new problems could have none.

### Yield model

- Prevalence is defined as "detectable at uniform N = 300, BH 10%". It is calibrated to the
  observed detection rate.
- Yield per unit for a design = prevalence × E[TP](design) / E[TP](uniform N = 300), from the
  simulation at each λ.
- Under λ < 1 this implies more true effects than the 7 that were observed, of smaller
  size. That is the only way 7 detections at N = 300 are consistent with smaller effects.
- The direct-simulation yield, which treats the 7 as the complete truth, is in the JSON
  (`units_direct_sim`, `usd_direct_sim`).
  - It agrees at λ = 1.
  - At λ = 0.5 it is about 2× more expensive. For the recommended design that is $3,755,
    against $1,713 from the calibrated model.
- **Trace size:** 19 units per trace, the Phase 2 window of 20 forks.

### Sizing table

- Ranges are the Wilson CI propagated. The cluster-bootstrap lower bound gives an unbounded
  cost and is omitted.
- **Upper-prevalence costs:** the cluster-bootstrap upper prevalence (8.4%) gives costs about
  12% below the Wilson-upper (lower-cost) end.

| Design | λ | Units (range) | Traces of 19 | Generations | GPU-h | $ point [range] | False discoveries at 100 TP |
|---|---|---|---|---|---|---|---|
| uniform N = 300 | any | 2,714 [1,350, 5,570] | 143 [71, 293] | 0.86 M | 932 | **2,323** [1,155, 4,766] | 8 |
| uniform N = 200 | 1 / 0.75 / 0.5 | 2,880 / 3,488 / 5,361 | 152 / 184 / 282 | 0.61–1.13 M | 659–1,228 | 1,644 / 1,990 / 3,059 | 7–9 |
| uniform N = 500 | 1 / 0.75 / 0.5 | 2,631 / 2,406 / 1,720 | 139 / 127 / 91 | 0.9–1.4 M | 984–1,506 | 3,754 / 3,433 / 2,453 | 8–9 |
| two-stage m20 N200 top50 | 1 / 0.75 / 0.5 | 3,548 / 4,074 / 5,139 | 187 / 214 / 271 | 0.49–0.71 M | 530–769 | 1,322 / 1,519 / 1,916 | 9 / 9 / 13 |
| **two-stage m20 N300 top50** | 1 / 0.75 / 0.5 | 3,341 / 3,325 / 3,125 | 176 / 175 / 165 | 0.63–0.68 M | 687–735 | **1,831 / 1,823 / 1,713** | 8 / 9 / 9 |
| two-stage m20 N300 top33 | 1 / 0.75 / 0.5 | 4,412 / 4,457 / 4,407 | 232 / 235 / 232 | 0.66 M | 717–725 | 1,787 / 1,806 / 1,786 | 9–11 |
| two-stage m20 N500 top50 | 1 / 0.75 / 0.5 | 3,253 / 2,982 / 2,089 | 171 / 157 / 110 | 0.69–1.08 M | 753–1,173 | 2,924 / 2,683 / 1,878 | 9–10 |

**Wilson range for the recommended design:** about $850–3,760, with about 1,550–6,850 units
(82–360 traces).

## Recommended design (exploratory)

### The design

1. **Traces:** about 175 trajectories (point estimate; range 85–360). Use one 20-fork window
   per trajectory with the Phase 2 window rule (pre-convergence), which gives about 3,300
   labelled units.
2. **Stage 1:** pilot m = 20 resamples on every fork. That is about 70k generations, or
   about $190.
3. **Selection:** rank units by pooled pilot answer entropy of forks i and i+1
   (`entropy_pair`), and keep the top 50%.
4. **Stage 2:** top up both forks of each selected unit to N = 300.
5. **Test:**
   - Exact permutation TV test with at least 20,000 permutations, or an exact
     multinomial/analytic p-value, so that p-values resolve below BH thresholds over
     thousands of tests.
   - One BH at 10% over all selected units, with the selection rule fixed in advance.
6. **Expected yield:** about 108 discoveries, of which about 100 are true and 8–9 false.

### Budget

- **Generations:** about 0.65 M, or **about $1,800** at Phase 2 rates (Wilson range
  $850–3,760).
- **Compute:** about 700 H100-hours, which is about 3.7 days on 8 GPUs.
- **Contingency:** budget $4,000 to cover the Wilson upper end.

### Why this design

- **It is the most robust option.**
  - Its calibrated cost is almost flat across λ ($1,710–1,830). Uniform N = 300 costs $2,323.
  - N = 200 designs are cheaper only if effects really are as large as observed (λ = 1). At
    λ = 0.5 they fall to $1,900–3,060.
- **Uniform designs cannot be stopped early.** A uniform design pays full depth on the
  low-entropy half of units, which in this sample held about 20% of the positives.

### Cheaper variant worth piloting

- Sequential stage 2: deep-sample to N = 200, then top up to 300–500 only for units with
  0.001 < p < 0.1.
- This was not simulated. It should recover most of N = 300's power at close to N = 200's
  cost.

## Assumptions and caveats

1. **Seven positives, three problems.**
   - All enrichment numbers have trajectory-cluster CIs spanning nearly 0 to 1.
   - Prevalence could be near 0 in new problems. Cost scales as 1/prevalence, so
     problem-level heterogeneity dominates the budget uncertainty.
   - 2238 s25/s26 is likely one event, which would make the effective count 6.
2. **Circularity.**
   - The positives, the truth distributions and the effect sizes all come from the same
     pooled samples.
   - The held-out replicate check reduces, but does not remove, this concern.
3. **Effect sizes.**
   - The 7 observed effects are selected maxima. λ < 1 brackets the winner's curse, but the
     true effect-size distribution, including the many small effects plausibly hiding in
     the 183 "nulls", is unknown.
   - The simulation sets every null exactly to zero.
4. **Per-unit simulation.**
   - Adjacent units share a fork in reality, so their samples are correlated. The simulation
     draws each unit independently.
   - The cost accounting does use the true fork union.
5. **Window rule.**
   - Prevalence comes from pre-convergence 20-fork windows.
   - Whole MVP traces have a median of 151 forks. 41% of their forks exceed Phase 2's
     top-tercile entropy threshold (at n = 100), and 16% have zero entropy (against 1.6% in
     Phase 2 windows).
   - Several windows per trace might work, but prevalence outside the window rule is
     unmeasured.
6. **Traces are a real constraint.**
   - About 175 traces is far more than the MVP's 40 (20 problems). Phase 3 needs new
     problems and base traces.
   - Their generation and segmentation cost is not counted.
7. **Cost model.**
   - Costs use Phase 2 throughput (1,075 tok/s aggregate, about 4,211 tokens per generation)
     and $2.49/H100-hour.
   - Longer or shorter continuations in other windows scale cost linearly. The token-weighted
     cost was within 2% here.
8. **"Confirmed" means a BH-10% discovery.**
   - Each discovery is a single-study detection, and about 8% are false.
   - To make each event replicated, add an independent confirmation replicate on the
     discoveries only (about 108 units × 2 forks × 300, roughly 65k generations, about
     $175). Its power is roughly 0.8–0.9 per true event, so plan on about 115–125 TP before
     confirmation.
9. **Multiple comparisons in this analysis.**
   - Six uncertainty scores, four pilot sizes and 18 two-stage configurations were examined.
   - The recommendation was chosen on robustness across λ, not on the best cell, but it is
     still exploratory.
