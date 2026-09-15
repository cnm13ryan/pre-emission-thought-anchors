# Stage 9 validation — validation calibration, freeze and the single test evaluation (2026-09-15)

Branch `agent/stage9/validation-freeze-20260915-46c93dfe` (worktree `~/preanchor-stage9`),
from `codex/stage8` `0d07354` with the `codex/stage7` closure `f6ce263` and the Stage 8
record `358945f` merged in. Implementation commit **`67f824a`** (clean tree; every Stage 9
command ran at it). CPU only (`CUDA_VISIBLE_DEVICES=""`, GPU untouched). Stage 3–8 artifacts
read only; their hashes were re-verified unchanged after the run.

## Stage 8 handoff (verified by `probes.stage9.load_probe_run` before any Stage 9 write)

| Input | Value |
|---|---|
| Stage 8 run | `20260915T193326Z-57bcb41c84-7ebd0db6` (commit `0d07354`, clean; fit, select, `validate-probes` ok) |
| `probe_models/frozen_selection.pending.json` | `e1161a5a759fc41eb1ab19acc82f0e054b14c13d8810fe3cd1cd92ee16759393` |
| `probe_models/selected_model.joblib` | `af63687f10720e358dc94b4dae65ce0eecd28cbd37b52e6d427d083f1aae1c81` |
| `probe_models/cv_results.parquet` | `d3aeae7092a878f552c46c8c89ade869e57b1ae4da34a23d56ed7acac56fcac1` |
| `probe_models/fold_predictions.parquet` | `e755638dd40e136df67e5fe1d9c6b67981ff61f91c21bd3debb015a5a3220a77` |
| Stage 7 run / `activations/index.parquet` | `20260915T182306Z-3ab408d80a-7452f1c2` / `1c6c9c2fab7d9e89d76cbc2d5d0990f7391672743d3c52f9613622440ebe6e5a` |
| `data/processed/stage7/features.parquet` | `f3227c386383701741aa284cc3bbf6b7c3205f2de9593616575bf1187e9a064d` |
| `data/processed/stage4/labels.parquet` | `fab9886e478922ac132d0a98f3320c3047cb87cd6252c0d06117b99861a61b31` |
| Resolved / scientific config hash (`configs/mvp.yaml`) | `c841423878148203a22a9492dc8d8c5242ea4d7ec71d7db0538849fcdeb6ef4f` / `82e38884ec8a2eead0fc32675579685b0f819113098b8cfc96efbac6fb01a948` |
| Split file SHA-256 | `b85dc4f59e1f595099008656cfac1dc005955b617c60b3c235c635ca90617f47` |
| Selected binary | `activation_plus_observables`, site 44, C 0.1 (mean train-CV AUCPR 0.1264) |
| Selected continuous | `activation_plus_observables`, site 28, alpha 1e4 (mean train-CV Spearman 0.1835, RMSE 0.3668) |
| Training q90 threshold | `1.3549810893213545` (Stage 4 train-eligible quantile; constant across all 6,474 rows) |

Partition check on the feature table: train 2,650 / validation 680 / test 689 eligible rows;
problems {12 / 4 / 4} exactly the frozen split, both trajectory kinds per problem, all 49
sites present. Prior test evaluation: none (no `stage9_test` state, no receipt, no outputs
in any run).

## Pre-test gates

| Gate | Evidence |
|---|---|
| Tests at `67f824a` | 309 passed, 2 skipped (`tests/unit`, `tests/integration`, `tests/scientific`; 11 initially failed only because the fresh worktree lacked the vendor submodules, all passed after `git submodule update --init`) |
| Working tree | clean; `git.patch` empty; `source.dirty == false` in the run manifest |
| Freeze | `frozen_selection.json` SHA-256 `e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf` == manifest `stages.stage9_freeze.freeze_sha256` |
| Analysis registry | `metrics/analysis_registry.json` `9620e9960a9c680c9f48e70897c984773a1e17fbef9085dac29ecd176c2eb754` (named in the freeze) |
| Test lock tests | `tests/scientific/test_test_lock.py` 10 passed; `tests/integration/test_stage9_pipeline.py` 2 passed |

## Run `20260915T201107Z-484989d2fe-19f1ab0d`

| Step | Time | Result |
|---|---|---|
| `calibrate --probe-run-id 20260915T193326Z-57bcb41c84-7ebd0db6` | 60 s | 680 validation rows, 54 anchors (prevalence 0.0794), 4 problems {330, 2050, 4164, 6481}, 8 trajectories; Platt **fitted** (A −0.11335, B 1.88096, 14 iterations); `predictions/validation.parquet` `779d1a20…`, `metrics/validation.json` `09376ca6…`, `probe_models/calibrator.json` `d1fc20c4…` |
| `freeze-selection` | < 5 s | freeze `e68f1d82…` |
| `evaluate-test --confirm-frozen e68f1d82…` | 46 s | 689 test rows, 25 anchors (prevalence 0.0363), problems {2236, 2870, 3916, 6998}, 8 trajectories, 1 attempt; `predictions/test.parquet` `86e90d79b4a8855d1306372d5333f3c670198c6cdbf7c1defff6d0dd8d352c93`, `metrics/test.json` `8a16c03cf446c38372ba6b7f1aa7ba935c43daa6d8cec2f90b21a05129b1ec74`, receipt `a6ef2ba89b227f2d610ad226836afadddefa8928cf0dce813a44561f93432b00` |
| second `evaluate-test` with the same hash | — | refused: "The held-out test was already evaluated for this run" |
| `validate-evaluation` | — | `ok: true`, no problems, attempts 1 |

### Validation diagnostics (descriptive; selection unchanged)

| Model (binary) | AUCPR | AUROC | Brier | log loss | ECE | P@10 % | NDCG@10 % |
|---|---|---|---|---|---|---|---|
| prevalence | 0.0794 | 0.5000 | 0.0735 | 0.2798 | 0.061 | 0.172 | 0.187 |
| position | 0.1330 | 0.6620 | 0.2513 | 0.6958 | 0.422 | 0.172 | 0.187 |
| context | 0.0829 | 0.5259 | 0.2601 | 0.7134 | 0.433 | 0.037 | 0.050 |
| uncertainty | 0.1178 | 0.5729 | 0.2476 | 0.6883 | 0.418 | 0.148 | 0.148 |
| observable_combined | 0.1266 | 0.6078 | 0.2494 | 0.6920 | 0.421 | 0.148 | 0.146 |
| activation_only (site 44) | 0.1711 | 0.6522 | 0.1072 | 0.5372 | 0.105 | 0.151 | 0.196 |
| **activation_plus_observables (site 44, C 0.1)** | **0.1645** | 0.6520 | 0.1063 (calibrated 0.0714) | 0.5365 | 0.102 (calibrated 0.037) | 0.166 | 0.222 |
| residualized_activation (site 44) | 0.1880 | 0.6661 | 0.1534 | 2.3949 | 0.150 | 0.129 | 0.196 |

Continuous (site 28, alpha 1e4): Spearman 0.2173, RMSE 0.3241. AUCPR gain over
`observable_combined` 0.0379 (2,000-replicate clustered 95 % percentile interval
−0.020 … 0.059). Primary AUCPR interval 0.061 … 0.253. The residualised control and
`activation_only` score above the primary on validation; this is reported and changes
nothing (selection is train-CV only).

### Test result (evaluated once; exploratory, four problems, descriptive intervals)

Primary endpoint: **test AUCPR of `activation_plus_observables` = 0.0941** (2,000-replicate
clustered bootstrap, sampling unit `problem_id` with replacement, seed 401117827, 95 %
percentile interval 0.036 … 0.162, 2,000 valid replicates). Test prevalence 0.0363.

| Model (binary) | AUCPR | AUROC | Brier | log loss | ECE | P@10 % | NDCG@10 % |
|---|---|---|---|---|---|---|---|
| prevalence | 0.0363 | 0.5000 | 0.0390 | 0.1851 | 0.064 | 0.135 | 0.316 |
| position | 0.1394 | 0.7188 | 0.2516 | 0.6964 | 0.466 | 0.135 | 0.316 |
| context | 0.0470 | 0.5179 | 0.2602 | 0.7136 | 0.475 | 0.066 | 0.168 |
| uncertainty | 0.0510 | 0.4293 | 0.2488 | 0.6907 | 0.462 | 0.139 | 0.212 |
| observable_combined | 0.0573 | 0.4866 | 0.2509 | 0.6950 | 0.465 | 0.121 | 0.199 |
| activation_only (site 44) | 0.0925 | 0.6450 | 0.0674 | 0.3305 | 0.075 | 0.135 | 0.187 |
| **activation_plus_observables** | **0.0941** | 0.6498 | 0.0707 (calibrated 0.0361) | 0.3407 (calibrated 0.1639) | 0.080 (calibrated 0.038) | 0.135 | 0.196 |
| residualized_activation (site 44) | 0.1267 | 0.6816 | 0.0968 | 1.3543 | 0.101 | 0.149 | 0.328 |

Continuous (site 28): Spearman 0.2370 (0.121 … 0.333), RMSE 0.2902 (0.263 … 0.324).
AUCPR gain over `observable_combined` 0.0368 (paired interval −0.051 … 0.081). Paired
primary-minus-baseline AUCPR intervals: prevalence 0.009 … 0.112; position −0.236 … 0.026;
context −0.058 … 0.101; uncertainty −0.042 … 0.099; activation_only 0.001 … 0.004;
residualized −0.148 … 0.009. Paired AUROC intervals exclude zero against
`observable_combined` (0.130 … 0.298), `uncertainty` (0.154 … 0.369) and `prevalence`
(0.028 … 0.339), and include zero against `position`.

Reading, at the tier the evidence supports: on four held-out problems the selected probe
ranks anchors above the combined observable baseline and above prevalence, the paired
AUCPR interval against the combined baseline includes zero, and the position-only baseline
has a higher point AUCPR than the probe. Twenty-five test anchors from four problems admit no
strong claim in either direction. Nothing was retuned after the result; the MVP completeness
flag `mvp_incomplete_insufficient_test_anchors` is false (25 ≥ 20).

## Integrity checks after the run

- `sha256sum` of every Stage 9 output equals the sidecars, the manifest and the receipt
  (table above); `preanchor validate-evaluation` ok.
- Stage 8 outputs, Stage 7 index/features and Stage 4 labels re-hashed after the run: all
  equal their manifests; newest mtime under the Stage 8 run 19:50:03 UTC (before Stage 9).
- Split membership: validation predictions contain only validation problems, test only the
  four test problems; no sentence appears in both.
- Calibrator: `fitted_on: validation`, 680 rows == validation partition, hash frozen.
- Bootstrap: 2,000 requested, 2,000 valid for every primary statistic, unit `problem_id`,
  paired; sampled-problem record hash `d4a20caf7326…`.
- No Stage 10 output exists (`metrics/sensitivities.parquet`, `metrics/permutations.parquet`,
  `predictions/lead_time.parquet` absent).

## Backup

`preanchor backup prepare --for-stage 9` (commit `9e71b1f`, which adds Stage 8/9
classification and verification): bundle
`20260915T203439Z-8605ff94cce0c7a1f51da847deda5183d0a2c7a97a73827b803fa1fe654e0bf0`
(canonical, highest protected stage 9, 6,431 entries, 3.5 GB logical, 1,316 new objects /
146.1 MB incremental), local `backup verify`: 6,107 objects rehashed, 0 missing, 0 corrupt.
**Primary-only** until the MacBook runs
`./scripts/preanchor_portable.sh backup <host> --for-stage 9` and prints `BACKUP VERIFIED`.

## Limitations

- Four test problems, 25 anchors: intervals are descriptive; problem 2236/2870/3916/6998 are
  public Thought Anchors problems (exploratory held-out engineering test).
- Calibrated validation metrics are in-sample (the calibrator was fitted on those rows);
  calibrated test metrics are out-of-sample.
- Stage 10 controls (lead one, threshold/convergence sensitivities, subgroups, quintiles,
  permutation null) are prespecified in the registry and not started.
