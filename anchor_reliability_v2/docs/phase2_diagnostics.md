# Phase 2 diagnostics: why the anchor score does not reproduce

**Status:** EXPLORATORY. This analysis comes after the confirmatory FAIL in
`docs/phase2_results.md` and was not preregistered. Source:
`scripts/s13_phase2_diagnostics.py`, which writes `results/s13_phase2_diagnostics.json`
and `results/s13_sentence_effects.csv`.

## Data

- **Pooled samples:** replicate 0 (released), replicate 1 and replicate 2 give 300
  resamples for each of the 200 forks.
- **Units:** 190 labelled sentences in 10 trajectories.
- **Reliability vs depth:** draw two disjoint halves of n resamples per fork from the pool,
  score both halves, compare, and repeat 40 times.
- **Real effects:** a permutation test per sentence on all 300 + 300 samples (2,000
  permutations), with Benjamini–Hochberg control at 10%.

## Finding 1: more resamples will not fix `fork_tv` ranking

Measured agreement between two independent halves (mean of 40 draws):

| n per half | 20 | 40 | 60 | 80 | 100 | 120 | 150 |
|---|---|---|---|---|---|---|---|
| ICC | 0.35 | 0.43 | 0.50 | 0.57 | 0.61 | 0.66 | 0.70 |
| Spearman | 0.31 | 0.38 | 0.43 | 0.45 | 0.44 | 0.47 | 0.49 |
| Within-problem Spearman | 0.14 | 0.15 | 0.16 | 0.18 | 0.15 | 0.16 | 0.15 |

- **ICC keeps rising.** It is driven by a handful of large effects and is projected to
  reach 0.75 near n ≈ 170.
- **Spearman flattens.** It sits around 0.45–0.49 from n = 80 onward, so the 1/n
  projection (n ≈ 410) is not credible.
- **Within-problem ranking does not improve at all.** It stays at about 0.15 across the
  whole range.
- The plan v2 forecast (200–400 resamples) assumed noise that averages away. This curve
  shows that assumption fails.

## Finding 2: most of `fork_tv`'s reproducible ranking is noise-floor bias

- **The bias.** TV between two finite samples is positive even when nothing changes, and
  its size depends on how spread out the answer distribution is. More distinct wrong
  answers give a higher floor.
- **It is stable across replicates.** Spearman between replicate 1's floor and replicate
  2's floor is 0.867.
- **It predicts the score better than the score predicts itself.** Spearman between
  replicate 1's `fork_tv` and the floor computed from replicate 2 is 0.532. That is
  higher than `fork_tv`'s agreement with replicate 2 itself, 0.445.
- **Most sentences sit at the floor.** 74% have `fork_tv` below 1.5× their own floor.
- **Even "null" sentences agree.** Among the 183 sentences with no detectable effect,
  replicate 1 vs replicate 2 Spearman is still 0.40. That agreement is the shared bias,
  not shared counterfactual importance.
- **Removing the bias removes the ranking.** Subtracting the analytic floor
  (`tv_debiased`) drops Spearman to 0.08–0.16 at every n.

## Finding 3: real sentence-level effects are rare and clustered

At BH 10%, with 300 samples per fork, **7 of 190 sentences (3.7%)** show a real change in
the answer distribution. They sit in 3 of the 10 problems: 2238 has 4, 3935 has 2 and 330
has 1. The other 7 problems have none.

| Trajectory | Sentence | Accuracy before → after | Pooled `fork_tv` | Rank in r0 / r1 / r2 |
|---|---|---|---|---|
| 2238:incorrect | 26 | 0.930 → 0.623 | 0.307 | 1 / 1 / 2 |
| 2238:incorrect | 25 | 0.667 → 0.930 | 0.263 | 2 / 2 / 1 |
| 2238:incorrect | 32 | 0.523 → 0.313 | 0.210 | 4 / 6 / 3 |
| 3935:correct | 26 | 0.783 → 0.960 | 0.183 | 3 / 9 / 10 |
| 330:correct | 6 | 0.707 → 0.857 | 0.177 | 5 / 3 / 5 |
| 2238:incorrect | 24 | 0.513 → 0.667 | 0.160 | 5 / 12 / 5 |
| 3935:correct | 21 | 0.943 → 0.857 | 0.137 | 8 / 11 / 26 |

- **The strongest effects do reproduce.** The top two sentences rank 1–2 in all three
  replicates.
- **The pair 2238 s25/s26 is one spike.** A single fork's accuracy jumps from 0.67 to 0.93
  and falls back to 0.62. It needs reading before it is counted as two anchors.
- **Top-10 sets agree only partly.** Across replicates the top-10 sets share 5 of 10.

## Finding 4: the accuracy curve itself is highly reliable

Accuracy of the resamples at fork i (the level, not a difference) gives:

| Measure | n = 100 | n = 20 |
|---|---|---|
| ICC | 0.98 | 0.91 |
| Spearman | 0.91 | 0.73 |
| Within-problem Spearman | 0.43 (0.51 at n = 150) | — |

The per-trajectory curves show smooth drifts plus occasional jumps, all saved in the JSON.
For example, 330 climbs from 0.71 to 1.00 between forks 5 and 15, and 3935 dips to 0.78
and then jumps to 0.96.

Other targets at n = 150, as ICC / Spearman:

| Target | ICC | Spearman |
|---|---|---|
| Smoothed accuracy change (2-fork windows) | 0.73 | 0.40 |
| Absolute Δaccuracy | 0.63 | 0.21 |
| Signed Δaccuracy | 0.52 | 0.18 |

None of these is a good continuous target.

## Interpretation

### Confirmed in these data

- In these 10 traces, a sentence-level counterfactual effect exists for only about 4% of
  sentences.
- For the rest, every continuous anchor score is noise, plus a bias that tracks
  answer-distribution spread.
- That bias is a property of the problem and position, so it is easy to predict from
  activations or surface features for the wrong reason.

### Implications for the existing prediction result (`docs/evaluation_results.md`)

- **The noise ceiling was too optimistic.** It assumed the label was valid signal.
- **The activation result needs re-checking.** Activations beat baselines (Spearman 0.23
  vs 0.15), but some of that may be predicting the noise-floor bias, that is, answer
  entropy.
- **Status:** unverified; see next steps.
