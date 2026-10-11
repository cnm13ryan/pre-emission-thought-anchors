# Phase 2 results: independent reliability replication

**Status:** CONFIRMATORY for the reliability question in `phase2_plan.md` v2.1, which was
committed before any Phase 2 data were read. Nothing here tests prediction. Source:
`results/phase2_reliability.json` and `results/phase2_scores_replicate{0,1,2}.csv`,
produced by `scripts/s12_phase2_score.py` at commit `9215e9a`. The scorer and plan were
unchanged since commit `7a1a2e5`.

## Decision

| Form | Rule (plan §7) | Result |
|---|---|---|
| Continuous `fork_tv` (primary) | Spearman ≥ 0.80 and ICC ≥ 0.75 | **FAIL** |
| q90 flag of `fork_tv` | κ ≥ 0.60 | **INCONCLUSIVE** |

No score passes in any form (`no_form_passes: true`).

The anchor score, as measured with 100 resamples per fork, does not reproduce well enough
across independent resampling to serve as a prediction target. The flag result cannot
rescue it: its interval runs from 0 to 0.86.

## Run

| Item | Value |
|---|---|
| Run ID | `20261009T0717Z-phase2-23f5e3e6` |
| Generations | 40,000 (2 replicates × 200 forks × 100), all status OK, none missing |
| Common N per fork | 100 (min = median = 100); no early stop |
| Reused | 320 from the benchmark runs |
| Hit the 16,384-token cap | 269 |
| Tokens generated | 168.4 M |
| Throughput | 1,075 tok/s aggregate |
| Wall time | 43.5 h (07:56Z on 9 Oct to 03:29Z on 11 Oct) |
| Cost | $108.42 at $2.49/h; budget guard $250 / 100 h |
| Index | `completions_index.parquet`, sha256 `b186b38e…c2569` |
| Labelled sentences | 190 in 10 trajectories (10 problems) |
| MiniLM similarity | none missing |

### v2 parse statuses

| Replicate | answered | length_truncated | no_answer | unboxed_stated |
|---|---|---|---|---|
| 1 | 19,866 | 131 | 2 | 1 |
| 2 | 19,855 | 137 | 1 | 7 |

### Label status per replicate

| Replicate | OK | NO_SUPPORT |
|---|---|---|
| 0 (released) | 164 | 26 |
| 1 | 147 | 43 |
| 2 | 150 | 40 |

NO_SUPPORT means no resample was dissimilar to S_i, so the similarity-filtered scores (tv,
jsd, dacc, KL) are undefined. `fork_tv` has no filter, so it covers all 190 sentences.

## Replicate 1 vs replicate 2 (primary)

95% problem-cluster bootstrap, 2,000 replicates.

| Score | Rows | Spearman | ICC(2,1) | κ (flag) | Flag rate r1 / r2 | Within-problem Spearman (median) |
|---|---|---|---|---|---|---|
| **fork_tv** | 190 | 0.445 [0.198, 0.617] FAIL | 0.561 [0.228, 0.676] FAIL | 0.659 [0.000, 0.855] INCONCL. | 0.032 / 0.016 | 0.277 |
| tv | 144 | 0.478 [0.254, 0.656] FAIL | 0.409 [0.168, 0.540] FAIL | 0.100 [−0.049, 0.231] FAIL | 0.042 / 0.056 | 0.230 |
| jsd | 144 | 0.606 [0.394, 0.724] FAIL | 0.471 [0.286, 0.597] FAIL | −0.022 [−0.044, 0.000] FAIL | 0.014 / 0.049 | 0.288 |
| dacc | 144 | 0.333 [0.179, 0.458] FAIL | 0.388 [0.151, 0.494] FAIL | 0.289 [−0.035, 0.452] FAIL | 0.062 / 0.062 | 0.235 |
| KL (MVP, reference) | 144 | 0.513 [0.303, 0.590] FAIL | 0.397 [0.200, 0.532] FAIL | — | — | 0.204 |

## Sensitivity: without problem 1591 (memorisation flag)

9 problems, 171 rows for `fork_tv`.

| Score | Spearman | ICC | κ |
|---|---|---|---|
| fork_tv | 0.439 [0.151, 0.650] FAIL | 0.581 [0.208, 0.701] FAIL | 0.659 [0.000, 0.854] INCONCL. |
| tv | 0.480 [0.252, 0.661] FAIL | 0.409 [0.138, 0.544] FAIL | 0.127 FAIL |
| jsd | 0.597 [0.322, 0.733] FAIL | 0.468 [0.261, 0.606] FAIL | −0.013 FAIL |
| dacc | 0.310 [0.143, 0.444] FAIL | 0.378 [0.104, 0.491] FAIL | 0.336 FAIL |
| KL | 0.512 [0.265, 0.600] FAIL | 0.428 [0.211, 0.551] FAIL | — |

Leaving out 1591 changes nothing. The memorised problem does not drive the result.

## Each new replicate vs the released samples (replicate 0), `fork_tv`

| Comparison | Spearman | ICC | κ | Continuous / flag |
|---|---|---|---|---|
| 1 vs 0 | 0.518 [0.342, 0.655] | 0.614 [0.332, 0.707] | 0.515 [0.000, 0.693] | FAIL / INCONCL. |
| 2 vs 0 | 0.525 [0.318, 0.661] | 0.624 [0.342, 0.726] | 0.488 [0.000, 0.769] | FAIL / INCONCL. |

Agreement with the released data is about the same as between the two new replicates. So
the fresh generations behave like the original ones; they are not a different
distribution.

## Interpretation

### Confirmed

- The continuous anchor score fails reliability under all five definitions (`fork_tv`,
  tv, jsd, dacc, KL), both with and without problem 1591. All upper bounds fall below the
  thresholds: Spearman ≤ 0.72 against 0.80, and ICC ≤ 0.70 against 0.75.
- Agreement within a trajectory is weaker still: within-problem Spearman medians are 0.20
  to 0.29. Much of the across-sentence agreement that exists comes from differences
  between problems, not from ranking sentences within a trace.

### Compared with the forecast (`s06_power.json`, plan §13)

- The forecast was FAIL with probability 0.98 for `fork_tv` at n = 100, and it fails.
- The point estimates are lower than forecast:

| `fork_tv` at n = 100 | Forecast | Observed |
|---|---|---|
| ICC | 0.73 | 0.56 |
| Spearman | 0.62 | 0.45 |

- The released data's split halves gave ICC 0.615. The Spearman–Brown projection to full
  depth (0.76) did not materialise across independent replicates at n = 100.

### Flag-form caveat (reported, not acted on)

- The fixed q90 threshold comes from the MVP training data (`s06_power.json`). On these
  forks it flags only 3.2% and 1.6% of sentences (6 and 3 of 190), against about 10% on
  the MVP data.
- With so few positives, κ is unstable. 218 of the 2,000 bootstrap replicates had no
  flagged sentence in one replicate, which leaves κ undefined. The interval
  [0.000, 0.855] is therefore uninformative, and the rule correctly returns INCONCLUSIVE.
- The threshold was not re-tuned after seeing the data (AGENTS.md: never widen
  thresholds after seeing results).

### Implication for Phase 3

- Predicting this label from activations is not supported at the current resampling
  depth.
- Options for owner decision, none adopted:
  1. more resamples per fork (the forecast suggested at least 200 for ICC and about 400
     for both criteria, but the forecast was optimistic here);
  2. a coarser target, such as a smoothed or problem-relative score;
  3. treating the label as noisy and evaluating predictors against the noise ceiling,
     not as a reliable ground truth.

## Procedural notes

- **Index built locally, not on the instance.** The completion index was built from the
  local mirror, not on the instance, using the same driver
  (`research/redesign/phase2_generate.py`, sha256 `0207285f…`, identical on both
  machines) and the same inputs (sha256 `f6e6d667…`).
- **Mirror verified before indexing.** The mirror matched the instance file for file
  (240,011 files), and all 120,000 completion files matched their `.sha256` sidecars.
- **No exclusions or re-runs.** No completions were excluded or re-run, and none failed.
- **Launch deviation.** The run's launch before plan v2 is recorded as a deviation in plan
  §13.
