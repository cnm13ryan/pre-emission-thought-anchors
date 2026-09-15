# Stage 8 validation — baselines and probe selection on training problems (2026-09-15)

Branch `codex/stage8`, implementation commit `0d07354` (from the `codex/stage7` merge
`480031e`). **Canonical run: `20260915T193326Z-57bcb41c84-7ebd0db6`, commit `0d07354`,
clean tree (`dirty_diff_hash 8794212a…`).** CPU only (`CUDA_VISIBLE_DEVICES=""`, 8 workers,
GPU 0 MiB throughout). Stage 3–7 artifacts read only. Everything below is training-fold
cross-validation; **no validation or test row was read and no held-out metric exists.**

Inputs: Stage 7 run `20260915T182306Z-3ab408d80a-7452f1c2`; `features.parquet`
`f3227c386383…9a064d`; `activations/index.parquet` `1c6c9c2fab7d…ebe5a`; labels
`fab9886e…a61b31`. Training partition: 2,650 eligible rows, 265 anchors (prevalence 0.100),
threshold 1.354981 (Stage 4 train-only q90, not recomputed).

## Admission at launch

| Gate | Status | Evidence |
|---|---|---|
| Stage 7 finalized and validated | PASS | `docs/stage7_validation.md`; `validate-activations` and `validate-features` re-run before launch |
| CPU tests at the launch commit | PASS | 276 passed, 2 skipped (`tests/unit`, `tests/integration`, `tests/scientific`) |
| Scientific gates §10.5/§10.6 | PASS | `test_split_isolation.py`, `test_pipeline_fit_scope.py`, `test_probe_sanity.py` |
| Real-data smoke | PASS | `20260915T191139Z-30a5e56f6f-5c6fdbe6` (smoke config, 2 sites): fit, select, validate-probes ok |
| Projection | PASS | `stages/stage8/resource_projection.json`: 1,706 s fit, $1.26; no GPU |
| Stage 7 MacBook secondary copy | **FAIL at launch** | no `pull_verification` receipt for a stage-7 bundle existed; the latest verified bundle protected stage 6. The user explicitly authorised production regardless; Stage 8 reads canonical inputs without modifying them |

## Runs

| Run | Commit / tree | Result |
|---|---|---|
| `20260915T191905Z-a2bcc389c7-fc030488` | `0d07354`, **dirty** (an untracked, initially empty `stage8_fit.log` inside the worktree) | fit complete, 103 units, 694 s; `validate-probes` ok; `select` refused because the log had grown and the identity no longer matched. Superseded, retained as evidence |
| `20260915T193326Z-57bcb41c84-7ebd0db6` | `0d07354`, clean | **canonical**: fit 839 s (peak RAM 4.58 GB), select 54 s, `validate-probes` ok |

Determinism: both runs produced `cv_results.parquet` with SHA-256
`68ea55b0fd273263b8f925e959bbe1e03863bec2da7fc58deef26896931a91c7` (byte-identical) before
the residualised control was appended.

## Canonical outputs (`probe_models/`)

- 104 units (5 baselines, 49 × `activation_only`, 49 × `activation_plus_observables`,
  1 `residualized_activation` at the selected site), each with `SUCCESS.json`.
- `cv_results.parquet` `d3aeae7092a8…` (6,596 rows; 0 non-converged fits);
  `fold_predictions.parquet` `e755638dd40e…` (4,369,850 rows, all `split == train`);
  `selected_model.joblib` `af63687f1072…`; `frozen_selection.pending.json` `e1161a5a759f…`.
- Folds (seed `derived_seed(20260914, "cv_folds")`), validation problems and anchors:
  {1591, 2137} 17 / 478 rows; {2189, 2238, 4682, 6596} 48 / 781; {3448, 3935, 4019}
  40 / 498; {3360, 3550, 4605} 160 / 893.

## Validation gates

| Gate | Status |
|---|---|
| Input identity (features, index, labels, activation run) | PASS |
| Every training row predicted exactly once per (probe, site, task, hyperparameter); no duplicate; no non-train row | PASS |
| Finite scores, probabilities in [0, 1] | PASS |
| Independent recomputation of every fold AUCPR from stored predictions | PASS (max diff 0.0) |
| Prevalence AUCPR equals fold prevalence | PASS |
| All 49 sites have train-CV results for both activation families | PASS |
| Selection reproducible from `cv_results.parquet`; joblib round trip; recorded C matches | PASS |
| Run tables equal the concatenation of validated units | PASS |
| Interruption / resume / corruption quarantine | PASS on the synthetic run (`tests/integration/test_probe_fit.py`); not exercised on production (no interruption occurred) |

## Pending selection (train CV only; Stage 9 freezes it)

Binary: `activation_plus_observables`, site 44, C 0.1, mean CV AUCPR 0.1264.
Continuous: site 28, alpha 1e4, mean CV Spearman 0.184, RMSE 0.367.
Per-family best mean CV AUCPR: prevalence 0.089, context 0.081, observable_combined 0.099,
position 0.114, uncertainty 0.117, activation_only 0.126 (site 44),
activation_plus_observables 0.126 (site 44), residualised control 0.119 (site 44).

Interpretation limits: these are training-fold diagnostics used only for selection; fold 3
holds 160 of the 265 validation anchors (problem 3550), so mean AUCPR is dominated by one
fold; differences between families are small and carry no held-out evidence. The
`overlapping_token` stratification (`docs/alternative_explanations.md`) is a Stage 10 item.

## Next

Stage 9 (`calibrate`, `freeze-selection`, `evaluate-test`) is not implemented. Before it:
Stage 7 and Stage 8 MacBook `pull_verification` receipts.
