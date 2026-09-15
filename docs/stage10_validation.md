# Stage 10 validation — controls, sensitivities and lead time (2026-09-15)

Branch `agent/stage10/controls-20260915-0ede0550` (worktree `/home/ubuntu/preanchor-stage10`),
created from the Stage 9 head `9e71b1f`, merged with the reviewed control plane `a0ebdc3`
(`a8ac6ac`), implementation `c1e303c`. Control-plane session
`20260915T204240Z-implementation-95c288b3`. **Canonical Stage 10 run:
`20260915T211903Z-b230473011-815ca2b7`, commit `c1e303c`, clean tree, CPU only
(`CUDA_VISIBLE_DEVICES=""`, 8 workers, 969 s), no GPU, no model load.** Every analysis is a
prespecified sensitivity, control, subgroup, lead-time or permutation analysis of the frozen
Stage 9 result (SPEC §3.12 items 5–11, `configs/mvp.yaml` `evaluation`); none of them is,
or replaces, the primary analysis. Intervals are the 2,000-replicate paired bootstrap over
`problem_id` clusters and are descriptive only (four test problems).

## Stage 9 input verified before Stage 10 (read-only)

| Item | Value |
|---|---|
| Stage 9 run | `20260915T201107Z-484989d2fe-19f1ab0d`, commit `67f824a`, clean tree |
| Freeze hash (`frozen_selection.json`) | `e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf` |
| Test evaluation | one attempt (`fea23ca6…`), receipt `a6ef2ba8…`, problems {2236, 2870, 3916, 6998}, 689 rows, 25 anchors |
| Selected primary | `activation_plus_observables`, site 44, C 0.1 (train CV only); Ridge site 28, alpha 1e4 |
| Frozen anchor threshold | q90 of eligible training importance = 1.3549810893213545 |
| Calibration | Platt on validation only (680 rows, 54 positives) |
| Inputs | features `f3227c38…9a064d`, index `1c6c9c2f…ebe5a`, labels `fab9886e…a61b31`, Stage 8 run `20260915T193326Z-57bcb41c84-7ebd0db6` |
| Stage 9 outputs | validation predictions `779d1a20…`, validation metrics `09376ca6…`, test predictions `86e90d79…`, test metrics `8a16c03c…`, registry `9620e996…`, calibrator `d1fc20c4…` |

`validate_stage9(write_record=False)` passed from the Stage 10 worktree before any work.
SHA-256 snapshots of all 44 Stage 9 run files, 36 Stage 8 run files and the three inputs were
taken before the run and re-checked after it: **all unchanged** (`sha256sum -c`). The Stage 10
run's `input_hashes` carry the freeze hash and every Stage 9 output hash;
`validate-sensitivities` re-verifies them (`stage9.unchanged: true`).

## Implementation

`src/preanchor/probes/stage10.py` (analyses, immutable units, resume, validation),
`src/preanchor/viz/stage10.py` (figures from the stored tables only), CLI
`preanchor run-sensitivities --config configs/mvp.yaml [--stage9-run-id] [--resume --run-id]`
and `preanchor validate-sensitivities --run-id`. Two model sources appear in every table:
`frozen_stage9` (the Stage 9 predictions re-scored on a subset or against relabelled anchors)
and `refit_frozen_hyperparameters` (a sensitivity model refitted on training rows at the
Stage 8-frozen site and hyperparameter of its family: primary site 44 / C 0.1,
`activation_only` site 44 / C 0.1, `observable_combined` C 1e-4). No grid, no layer search,
no selection anywhere in Stage 10 (`tests/scientific/test_stage10_controls.py::
test_stage10_has_no_selection_or_test_driven_choice_path`). The q90 refit reproduces the
frozen Stage 9 numbers exactly (same rows, same labels, same fit).

Tests added: `tests/scientific/test_stage10_controls.py` (12: train-only thresholds, numeric
application, K=4/K=5 semantics, the `>= 0.10 OR >= 10` rule with explicit failure on missing
counts, deterministic quintiles, subgroup partition and explicit empty groups, single-class
groups, no forbidden field in a refit design, lead-one mapping including cross-trajectory and
late-boundary refusals, stratified permutations with single-class strata preserved,
deterministic seeds, absence of a selection path) and
`tests/integration/test_stage10_pipeline.py` (4: end to end on the synthetic lineage with the
frozen-primary protection, immutable resumable units with quarantine, byte-determinism,
refusal of an untested Stage 9 run).

## Outputs (`artifacts/runs/20260915T211903Z-b230473011-815ca2b7/`)

| Path | SHA-256 | Rows |
|---|---|---|
| `metrics/sensitivities.parquet` | `f4b6e48883aeb396ebf6a219b549238f4526872efeecc9fe3fb043f01d029cea` | 3,668 |
| `metrics/permutations.parquet` | `1adf0eb45794d1bcf88ad7b70a5a0218a4ff054936d966938bd5dd48d7cde5ac` | 202 |
| `predictions/lead_time.parquet` | `4e6055304b8edf688abdb19ab15eb4a089757ddc13ce75cc5f9e5e5771f8d203` | 5,476 |
| `figures/threshold_sensitivity.{html,json}` | `bcbb52d3…`, `a69a3f74…` | |
| `figures/lead_time.{html,json}` | `5064da5b…`, `8cc5da12…` | |
| `figures/permutation_null.{html,json}` | `731b0190…`, `95eea6e7…` | |
| `figures/position_entropy_controls.{html,json}` | `0b7802d2…`, `20a97904…` | |
| `stages/stage10/summary.json`, `stages/stage10/validation.json` | per-attempt records | |

No `.png` was written: the static Plotly renderer (kaleido/Chrome) is not available on the
instance; the HTML (self-contained, fixed div ids, byte-deterministic) and figure JSON are
the record. 22 immutable units under `stages/stage10/units/` (`SUCCESS.json` + hashes).

## Validation gates

| Gate | Status | Evidence |
|---|---|---|
| Stage 9 freeze/test integrity before and after | PASS | `validate_stage9` ok; 44 + 36 + 3 hashes unchanged |
| All alternate thresholds from eligible training importance | PASS | `threshold_source = train_eligible_importance_quantile`; problems ⊂ train asserted |
| q90 reproduces the frozen numeric threshold | PASS | 1.3549810893213545 == freeze (exact) |
| No new layer selection; primary at site 44 everywhere; each family at its frozen site | PASS | `validate-sensitivities` |
| Lead-one source is S_(i-1)'s own boundary; sentence zero unavailable, never shifted | PASS | 6 test / 8 validation / 19 train rows unavailable, reason `first_sentence_has_no_predecessor` |
| Permutation count, seeds, strata recorded | PASS | 100 requested = 100 effective per split; seeds `derived_seed(20260914, "permutation", k)`; strata `problem_id,trace_kind,position_quintile` |
| Empty / undefined groups explicit | PASS | `position_quintile:5:test` (53 rows, 0 anchors, 1 problem) reported as `single_class`, not pooled |
| No Stage 9 primary artifact inside the Stage 10 run; no Stage 11 marker | PASS | `validate-sensitivities` |
| Paired differences equal the Stage 9 point values | PASS | asserted inside the `paired_baseline_differences` unit (1e-9) |
| Resume / quarantine / determinism | PASS on the synthetic lineage (`test_stage10_pipeline.py`); the production run had no interruption |
| CPU suite at `c1e303c` | see `docs/continuation.md` (Stage 10 entry) |

## Results (AUCPR of the primary; brackets are 2.5–97.5 % clustered bootstrap percentiles)

Baseline = `observable_combined`; Δ = primary − baseline (paired within replicate).

### Threshold sensitivity (q from eligible training importance; anchors relabelled everywhere)

| q | threshold | train anchors | split | source | n / anchors | primary | baseline | Δ |
|---|---|---|---|---|---|---|---|---|
| q80 | 0.8167 | 530 / 2650 | test | frozen | 689 / 65 | 0.153 [0.073, 0.259] | 0.113 | 0.040 [−0.035, 0.110] |
| q80 | | | test | refit | 689 / 65 | 0.172 [0.066, 0.381] | 0.127 | 0.045 [−0.039, 0.199] |
| q80 | | | validation | frozen | 680 / 115 | 0.281 [0.134, 0.392] | 0.230 | 0.051 [−0.024, 0.098] |
| q80 | | | validation | refit | 680 / 115 | 0.258 [0.106, 0.381] | 0.266 | −0.008 [−0.110, 0.042] |
| **q90 (primary)** | 1.3550 | 265 / 2650 | test | frozen = refit | 689 / 25 | 0.094 [0.036, 0.162] | 0.057 | 0.037 [−0.051, 0.081] |
| q90 | | | validation | frozen = refit | 680 / 54 | 0.165 [0.061, 0.253] | 0.127 | 0.038 [−0.020, 0.059] |
| q95 | 1.8273 | 133 / 2650 | test | frozen | 689 / 10 | 0.033 [0.022, 0.060] | 0.050 | −0.017 [−0.161, 0.021] |
| q95 | | | test | refit | 689 / 10 | 0.040 [0.016, 0.292] | 0.051 | −0.011 [−0.137, 0.263] |
| q95 | | | validation | frozen | 680 / 26 | 0.079 [0.030, 0.127] | 0.082 | −0.003 [−0.064, 0.033] |
| q95 | | | validation | refit | 680 / 26 | 0.067 [0.032, 0.102] | 0.070 | −0.003 [−0.077, 0.017] |

### Convergence K=4 versus K=5 (frozen threshold; eligibility only)

Rows lost under K=4: train 64, validation 37, test 16 (none gained; K=4 ⊂ K=5).

| rule | split | source | n / anchors | primary | baseline | Δ |
|---|---|---|---|---|---|---|
| K=5 (primary) | test | frozen = refit | 689 / 25 | 0.094 [0.036, 0.162] | 0.057 | 0.037 [−0.051, 0.081] |
| K=4 | test | frozen | 673 / 24 | 0.096 [0.036, 0.162] | 0.059 | 0.037 [−0.060, 0.081] |
| K=4 | test | refit | 673 / 24 | 0.090 [0.037, 0.150] | 0.058 | 0.032 [−0.058, 0.073] |
| K=5 | validation | frozen = refit | 680 / 54 | 0.165 [0.061, 0.253] | 0.127 | 0.038 [−0.020, 0.059] |
| K=4 | validation | frozen | 643 / 51 | 0.172 [0.061, 0.254] | 0.130 | 0.042 [−0.017, 0.059] |
| K=4 | validation | refit | 643 / 51 | 0.163 [0.057, 0.240] | 0.129 | 0.034 [−0.030, 0.049] |

### Replacement quality (`dissimilar_count / sample_count >= 0.10 OR dissimilar_count >= 10`)

Every row has `sample_count = 100`, so the two clauses coincide; excluded rows fail both.
Retained / excluded: train 2380 / 270, validation 612 / 68, test 610 / 79.

| subset | split | source | n / anchors | primary | baseline | Δ |
|---|---|---|---|---|---|---|
| supported | test | frozen | 610 / 20 | 0.097 [0.032, 0.164] | 0.043 | 0.054 [0.005, 0.085] |
| supported | test | refit (trained on supported rows) | 610 / 20 | 0.085 [0.038, 0.139] | 0.045 | 0.040 [0.009, 0.075] |
| supported | validation | frozen | 612 / 52 | 0.179 [0.069, 0.275] | 0.132 | 0.047 [−0.009, 0.069] |
| supported | validation | refit | 612 / 52 | 0.176 [0.049, 0.253] | 0.136 | 0.040 [−0.012, 0.049] |
| unsupported (complement) | test | frozen | 79 / 5 | 0.146 [0.115, 0.500] | 0.260 | −0.115 [−0.226, 0.409] |
| unsupported | validation | frozen | 68 / 2 | 0.054 [0.047, 0.114] | 0.048 | 0.006 [−0.085, 0.014] |

### Correct versus incorrect trajectories (frozen scores; correctness is a subgroup variable only)

| subgroup | split | n / anchors (problems) | primary | AUROC | baseline | Δ |
|---|---|---|---|---|---|---|
| correct | test | 283 / 13 (4) | 0.287 [0.070, 0.469] | 0.829 | 0.101 | 0.185 [0.031, 0.249] |
| incorrect | test | 406 / 12 (4) | 0.036 [0.028, 0.069] | 0.477 | 0.053 | −0.017 [−0.184, 0.035] |
| correct | validation | 254 / 15 (4) | 0.130 [0.085, 0.228] | 0.666 | 0.078 | 0.052 [−0.034, 0.138] |
| incorrect | validation | 426 / 39 (4) | 0.198 [0.056, 0.313] | 0.655 | 0.170 | 0.028 [−0.066, 0.061] |

### Position quintiles (`sentence_idx`; train cut points 25, 53, 86, 129; bin k = (cut_{k−1}, cut_k])

| bin | split | n / anchors (problems) | primary | baseline | Δ |
|---|---|---|---|---|---|
| 1 (≤ 25) | test | 188 / 15 (4) | 0.183 [0.069, 0.260] | 0.119 | 0.064 [−0.020, 0.106] |
| 2 | test | 173 / 5 (4) | 0.038 [0.035, 0.075] | 0.023 | 0.015 [0.003, 0.057] |
| 3 | test | 174 / 3 (4) | 0.016 [0.007, 0.056] | 0.184 | −0.169 [−0.245, −0.010] |
| 4 | test | 101 / 2 (2) | 0.091 [0.091, 0.250] | 0.018 | 0.074 [0.069, 0.230] |
| 5 (> 129) | test | 53 / 0 (1) | undefined (single class) | — | — |
| 1 | validation | 184 / 26 (4) | 0.165 [0.044, 0.271] | 0.168 | −0.003 [−0.088, 0.065] |
| 2 | validation | 170 / 11 (4) | 0.351 [0.013, 0.450] | 0.172 | 0.180 [0.002, 0.212] |
| 3 | validation | 165 / 14 (4) | 0.203 [0.100, 0.284] | 0.145 | 0.058 [−0.123, 0.210] |
| 4 | validation | 112 / 2 (3) | 0.017 [0.008, 0.056] | 0.028 | −0.011 [−0.026, −0.007] |
| 5 | validation | 49 / 1 (1) | 1.000 (one anchor, one problem) | 0.030 | 0.970 |

### Entropy quintiles (`next_token_entropy`; train cut points 0.3517, 0.7524, 1.0888, 1.4328)

| bin | split | n / anchors | primary | baseline | Δ |
|---|---|---|---|---|---|
| 1 | test | 135 / 8 | 0.063 [0.038, 0.108] | 0.058 | 0.006 [−0.019, 0.029] |
| 2 | test | 135 / 5 | 0.144 [0.059, 0.333] | 0.055 | 0.089 [0.030, 0.284] |
| 3 | test | 134 / 3 | 0.109 [0.053, 0.373] | 0.101 | 0.008 [−0.208, 0.079] |
| 4 | test | 139 / 4 | 0.045 [0.008, 0.084] | 0.059 | −0.014 [−0.033, −0.004] |
| 5 | test | 146 / 5 | 0.276 [0.062, 0.500] | 0.135 | 0.141 [−0.438, 0.333] |
| 1 | validation | 147 / 6 | 0.121 [0.106, 0.334] | 0.094 | 0.027 [−0.103, 0.302] |
| 2 | validation | 131 / 9 | 0.269 [0.022, 0.499] | 0.085 | 0.184 [−0.006, 0.349] |
| 3 | validation | 145 / 16 | 0.252 [0.082, 0.379] | 0.158 | 0.094 [0.009, 0.173] |
| 4 | validation | 125 / 8 | 0.166 [0.042, 0.268] | 0.139 | 0.027 [−0.003, 0.081] |
| 5 | validation | 132 / 15 | 0.167 [0.063, 0.286] | 0.192 | −0.025 [−0.152, 0.112] |

### Lead time one (site 44; features from the boundary immediately before S_(i−1); label of S_i)

Mappable rows: train 2631 (19 unavailable), validation 672 (8), test 683 (6); all unavailable
rows are first sentences and are stored with `available = false` and null scores. Lead zero is
the frozen Stage 9 model restricted to the same rows; lead one is a refit at site 44 / C 0.1
(observables at lead one are those of the earlier boundary).

| lead | split | n / anchors | primary | AUROC | baseline (lead-matched) | Δ |
|---|---|---|---|---|---|---|
| 0 (frozen) | test | 683 / 22 | 0.077 [0.024, 0.146] | 0.611 | 0.050 | 0.027 [−0.050, 0.075] |
| 1 (refit) | test | 683 / 22 | 0.032 [0.017, 0.093] | 0.451 | 0.046 | −0.014 [−0.070, 0.032] |
| 0 (frozen) | validation | 672 / 53 | 0.166 [0.062, 0.255] | 0.653 | 0.127 | 0.039 [−0.019, 0.060] |
| 1 (refit) | validation | 672 / 53 | 0.173 [0.031, 0.329] | 0.608 | 0.120 | 0.052 [−0.031, 0.102] |

### Permutation null (100 label permutations within `(problem_id, trace_kind, position_quintile)`, site 44, C 0.1)

Training strata: 95; 50 hold both classes (1,548 rows permuted), 45 hold one class (1,102
rows kept fixed, as prespecified; nothing crosses a stratum). Seeds
`derived_seed(20260914, "permutation", k)`, digest `b550dea6…`. Each permuted-label refit is
scored on the fixed validation and test labels; the observed statistic is the frozen model's.

| split | observed AUCPR | null mean ± sd | null p95 | null max | fraction of null ≥ observed | (r+1)/(n+1) |
|---|---|---|---|---|---|---|
| validation | 0.165 | 0.118 ± 0.029 | 0.178 | 0.215 | 0.09 | 0.099 |
| test | 0.094 | 0.062 ± 0.027 | 0.107 | 0.158 | 0.16 | 0.168 |

### Paired baseline differences (frozen scores, all eligible rows; AUCPR, primary minus …)

| against | test | validation |
|---|---|---|
| observable_combined | 0.037 [−0.051, 0.081] | 0.038 [−0.020, 0.059] |
| prevalence | 0.058 [0.009, 0.112] | 0.085 [0.024, 0.134] |
| position | −0.045 [−0.236, 0.026] | 0.032 [−0.070, 0.086] |
| context | 0.047 [−0.058, 0.101] | 0.082 [0.026, 0.100] |
| uncertainty | 0.043 [−0.042, 0.099] | 0.047 [−0.007, 0.062] |
| activation_only | 0.002 [0.001, 0.004] | −0.007 [−0.012, 0.002] |
| residualized_activation | −0.033 [−0.148, 0.009] | −0.023 [−0.046, 0.002] |

## Interpretation (descriptive; four test problems; no significance claim)

- **Thresholds.** The primary (q90) gain over the combined observables (+0.037 test,
  +0.038 validation, intervals crossing zero) persists in direction at q80 (+0.040 / +0.051
  frozen; refit +0.045 / −0.008) and disappears at q95 (−0.017 / −0.003), where only 10 test
  anchors remain. The result is not robust to the stricter threshold.
- **Convergence.** K=4 removes 16 test / 37 validation rows and leaves every number within
  0.01 of K=5. Immaterial.
- **Replacement support.** Restricting to supported rows removes 79 test / 68 validation rows
  and slightly widens the gain (+0.054 [0.005, 0.085] test frozen; +0.040 [0.009, 0.075]
  refit); the excluded complement is tiny (5 and 2 anchors) and uninformative.
- **Correct vs incorrect.** On test the primary's signal sits in correct trajectories (AUCPR
  0.287, AUROC 0.83, 13 anchors) and is absent in incorrect ones (0.036, AUROC 0.48); on
  validation the pattern is not reproduced (0.130 vs 0.198). Heterogeneous and unstable
  across the two small splits.
- **Position.** Anchors concentrate early (test: 15 of 25 in bin 1); the primary beats the
  baseline in bins 1, 2 and 4 on test, loses in bin 3 and is undefined in bin 5 (no test
  anchor after sentence 129). Validation bins disagree bin by bin. Signal is not uniformly
  distributed over reasoning position and bins 4–5 carry too few anchors to say anything.
- **Entropy.** No monotone pattern; the primary exceeds the baseline in the lowest-to-middle
  and highest test bins and not in bin 4. Activation signal is not simply low or high
  next-token uncertainty, but the bins hold 3–8 test anchors each.
- **Lead one.** On test, AUCPR falls from 0.077 (lead zero, same rows) to 0.032 (below the
  lead-matched baseline, AUROC 0.45); on validation it does not fall (0.166 → 0.173). One
  sentence earlier, the test signal is gone; the validation split does not show the drop.
- **Permutation null.** The observed statistic lies at the 91st (validation) and 84th (test)
  percentile of the stratified null: above the null mean by 1.6 and 1.2 null standard
  deviations, inside the null range on both splits.
- **Incremental value over observables.** +0.037 AUCPR on test (interval −0.051 to 0.081),
  essentially identical to `activation_only` (Δ 0.002) and below the residualised control
  (−0.033). Whatever the activations add is not separable from what the observables
  already carry at this sample size.

The prespecified controls were executed faithfully; the primary result survives the
lenient threshold and the eligibility/support variants in direction, does not survive the
strict threshold or a one-sentence lead on test, and is not separable from the permutation
null with 100 permutations on four test problems. None of this changes the frozen primary
analysis.

## Portable backup

`src/preanchor/backup/classify.py::discover_stage10` / `stage10_run_summary` and
`verify.py::verify_stage10` extend the portable bundle to Stage 10 (`docs/backup_restore.md`);
`tests/integration/test_backup_stage10.py` covers classification, the historical status of an
interrupted run, the round trip into a new root, read-only verification and tamper
detection. `backup plan --for-stage 10` on the canonical root: kind canonical, highest
protected stage 10, 5,001 MUST_BACK_UP items (3.5 GB logical), 217 new objects
(18.5 MB incremental over the store), no blockers.

Bundle **`20260915T221329Z-19f7aef31d0529f4c08e8fde166351fec0cf95953135ce20199509e80e094872`**
(`--for-stage 10`, canonical, highest protected stage 10, required stages 3–10; 6,666
entries, 3,785,689,185 B logical, 6,324 unique objects, 217 created / 6,107 reused over the
MacBook-verified Stage 9 bundle `e6b1a272…`) was prepared at 22:13 UTC from the clean
checkout `4ff29e0` under `portable-store:writer` held by the CLI's short-lived
`backup-writer` session (acquired 22:12:11, released 22:13:29, receipts under
`control/receipts/`); `latest.json` points at it. `backup verify`: 6,324 objects,
3,541,052,919 B, 0 missing, 0 corrupt. Scratch restore into
`/lambda/nfs/preanchor/restore-test-stage10-cc4c216b`: 6,666 restored, `RESTORE VERIFIED`,
scientifically equivalent PASS, every Stage 7/8/9/10 gate PASS including read-only
`validate-evaluation` and `validate-sensitivities` on the restored runs. Every bundled
Stage 10 (234), Stage 9 (44) and Stage 8 (1,288) file and the Stage 3/4/7 tables hash
identically in the scratch root and in the source; the canonical source trees are
byte-identical before and after the prepare. A model-free readiness probe on the scratch
root loads the Stage 9 handoff and every SPEC Stage 11 input (Stage 9 predictions/metrics,
Stage 8 CV tables, Stage 10 tables and figures, Stage 3/4/7 tables) without fitting,
evaluation or activations. "Ready to continue Stage 10: NO" only because the checkout is
`4ff29e0`, not the producer `c1e303c`. **Primary-only: the MacBook `pull_verification`
receipt for `19f7aef3…` is pending and no secondary copy is claimed.** The scratch root
(3.6 GB on NFS) awaits the user's deletion approval.

## Limitations

- Four test problems and 25 test anchors: every subgroup interval is wide and several bins
  hold ≤ 5 anchors; one (`position_quintile 5 / test`) is undefined.
- Lead-zero comparator is the frozen model restricted to mappable rows (19 first-sentence
  training rows were used by the frozen fit but are unavailable at lead one); lead one is a refit on the 2,631 mappable training rows.
- The K=4 sensitivity keeps the frozen q90 threshold (it does not recompute a quantile on the
  K=4-eligible set): it isolates the eligibility change (`docs/decisions.md`).
- The permutation scheme permutes *training* labels within strata and refits; it does not
  permute evaluation labels. 45 of 95 training strata are single-class and stay fixed.
- PNG figures are absent (no static renderer); HTML/JSON are complete.
- `ruff check .` fails on pre-existing Stage 4 files (`labels/threshold.py`,
  `labels/convergence.py`, `tests/unit/test_stage4_labels.py`,
  `tests/unit/test_support_resume.py`); Stage 10 files are clean.
