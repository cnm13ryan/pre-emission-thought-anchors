# Stage 9 provenance audit and acceptance (2026-09-15)

Independent audit of the already-produced Stage 9 run, performed from a separate session
(`agent/ops/stage9-audit-20260915-16702032`, worktree `~/preanchor-s9-audit`, control-plane
session `20260915T204953Z-audit-b7380977`) before that run's branch was integrated. Nothing
of Stage 8 or Stage 9 was recomputed; the held-out test was not rerun. Raw evidence:
`$PREANCHOR_ROOT/control/validation/20260915T204852Z-stage9-audit-snapshot-8f4b20a/`
(byte copy and SHA-256 list of the run before any branch operation, read-only) and
`$PREANCHOR_ROOT/control/validation/20260915T210500Z-stage9-audit-evidence-63e2e55/`
(audit scripts, their JSON results, test and ruff logs).

## Decision: ACCEPT

| Item | Value |
|---|---|
| Scientific producing commit | `67f824a` (clean tree, `dirty_diff_hash 8794212a…`) |
| Run | `20260915T201107Z-484989d2fe-19f1ab0d` on Stage 8 run `20260915T193326Z-57bcb41c84-7ebd0db6` (`0d07354`) |
| Freeze SHA-256 | `e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf` |
| Test evaluations | invoked twice, **evaluated once** (second invocation refused before the claim) |
| Stage 9 branch HEAD at preservation | `8f4b20a` (`67f824a` + backup coverage `9e71b1f` + records) |
| Integration commit | `63e2e55` = merge of `a0ebdc3` (reviewed control plane + Stage 8) and `8f4b20a` |

Later integration commits are not the producer of any Stage 9 artifact; Stage 10 consumes
the artifacts produced at `67f824a` and must not regenerate them.

## Preservation before integration

Captured at 20:48:52 UTC with `codex/stage8` still at `358945f` and the Stage 9 worktree
clean at `8f4b20a`: the whole run directory (45 files) copied and hashed, the Stage 8 run
(1,399 files) hashed, the canonical Stage 3/4/7 tables hashed, snapshot made read-only,
`SNAPSHOT.sha256` `07458858…`. Key hashes:

| Artifact | SHA-256 |
|---|---|
| `frozen_selection.json` | `e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf` |
| `predictions/validation.parquet` | `779d1a2099823a26093e9d367c37e08ef55fdec49a50f83e2096b8999252b93e` |
| `predictions/test.parquet` | `86e90d79b4a8855d1306372d5333f3c670198c6cdbf7c1defff6d0dd8d352c93` |
| `metrics/validation.json` | `09376ca69b44b271aa7376c23615cee4b3edc4df90b316e73c049f0ec12e55ac` |
| `metrics/test.json` | `8a16c03cf446c38372ba6b7f1aa7ba935c43daa6d8cec2f90b21a05129b1ec74` |
| `metrics/test_receipt.json` | `a6ef2ba89b227f2d610ad226836afadddefa8928cf0dce813a44561f93432b00` |
| `metrics/analysis_registry.json` | `9620e9960a9c680c9f48e70897c984773a1e17fbef9085dac29ecd176c2eb754` |
| `probe_models/calibrator.json` | `d1fc20c4dd8c91c2f8b1d2c3c37582883a44324b8e6c288b21eecacf12802e3a` |
| `manifest.json` | `dcbb26be29a7055a9417d7d7a86774d78d1389451d2c835ba8adce99193a66e9` |
| Stage 8 `selected_model.joblib` | `af63687f10720e358dc94b4dae65ce0eecd28cbd37b52e6d427d083f1aae1c81` |
| Stage 8 `frozen_selection.pending.json` | `e1161a5a759fc41eb1ab19acc82f0e054b14c13d8810fe3cd1cd92ee16759393` |
| Stage 8 `cv_results.parquet` / `fold_predictions.parquet` | `d3aeae70…` / `e755638d…` |

All of them hashed identically after the `codex/stage8` fast-forward and after the
integration merge.

## Execution timeline (from immutable manifest revisions, not mtimes)

The run keeps every manifest revision under `manifests/<sha>.json`; the receipt, the
freeze file and the worker's transcript agree with them.

| UTC | Evidence | Event |
|---|---|---|
| 19:50:03 | Stage 8 run `probe_validation.json`, newest Stage 8 mtime | Stage 8 fit/select/validate-probes complete (`0d07354`) |
| 20:11:08 | revisions `9e602146`, `d3bbc680` | run created at `67f824a`, `input_hashes` = the Stage 8 run; `stage9_calibrate: prepared` |
| 20:11:09 | revision `62ba6bbf` | `calibrate` attempt `f9eeb6b9` running (validation partition) |
| 20:12:04 | revision `9373cd03`; outputs written | validation predictions/metrics/calibrator hashed; `calibrate: complete`, 680 rows, 54 positives, Platt fitted |
| 20:12:20 | revision `d74af3cd`; `frozen_at` | `analysis_registry.json` `9620e996…`, `frozen_selection.json` `e68f1d82…`; `stage9_freeze: complete` |
| 20:12:41 | worker transcript | pre-test checkpoint: freeze hash equals the manifest, `prior_test_state` empty |
| 20:12:46 | revision `15b39aa3` | `evaluate-test --confirm-frozen e68f1d82…` claimed (attempt `fea23ca6`, `resumed: false`) **before any test row was read** |
| 20:13:32 | revision `dcbb26be` = `manifest.json`; receipt | test predictions/metrics written, receipt `a6ef2ba8…`, `stage9_test: complete`, 689 rows, 25 positives, problems `[2236, 2870, 3916, 6998]` |
| 20:13:48 | worker transcript | second `evaluate-test` with the same hash refused ("already evaluated for this run"); `validate-evaluation` ok |
| 20:13:51 | `stages/stage9/validation.json` | read-only validation record |

`commands` in the manifest lists exactly one `calibrate`, one `freeze-selection` and one
`evaluate-test` (with the freeze hash). A refused invocation never reaches `save_manifest`,
so the single recorded attempt is the successful-evaluation count.

## Dependency closure (Stage 8 → Stage 9)

`load_probe_run` (called by every Stage 9 command) verified the Stage 8 run before any
write; the audit re-derived the same facts from the artifacts: the four Stage 8 output
hashes equal the freeze's `input_hashes` and the Stage 8 manifest; the pending selection
equals the frozen binary (site 44, C 0.1, mean train-CV AUCPR 0.12637) and Ridge (site 28,
alpha 1e4, Spearman 0.18350, RMSE 0.36677); recomputing the argmax over
`cv_results.parquet` (mean over the 4 grouped folds, whose held-out problems are all 12
training problems) reproduces both choices; the serialized models carry the same layer and
hyperparameter and the pending `train_rows_sha256`; the training q90 threshold
`1.3549810893213545` equals `numpy.quantile(importance_primary, 0.9)` over the 2,650
eligible training rows of `stage4/labels.parquet` (validation and test quantiles are 1.071
and 0.756, so the threshold is not derived from them); the split file hashes to the frozen
`b85dc4f5…` and the frozen problem sets equal `configs/splits/mvp_20_v1.yaml`.

## Calibration scope

`calibrate` loads the partition through `partition_rows(table, config, "validation")`,
which keeps only eligible rows whose `split == "validation"` and whose problem is in the
frozen validation set. The calibrator records `fitted_on: validation`, 680 rows / 54
positives, and `validation_rows_sha256 d85f13d7…` which the audit reproduced from the
validation predictions in partition order. Refitting Platt (prior-corrected targets, BFGS)
on the saved validation `score_raw` gives A −0.1133523, B 1.8809594 (saved −0.1133523,
1.8809593). Fitting on validation + test would give A −0.1210, B 2.1571, so the saved
calibrator cannot have seen test rows. Applying the saved parameters reproduces
`probability_calibrated` in both partitions to 1e-12. The skip rule (≥ 20 positives, two
classes) was correctly not triggered. Calibration changes no layer, C, alpha, threshold,
feature or split (`frozen_selection.json` copies the pending selection unchanged).

## Freeze

The file hashes to `e68f1d82…`, equal to the sidecar, the manifest (`stages.stage9_freeze`
and `output_hashes`) and the receipt; re-serialising the parsed record with `json_bytes`
reproduces the identical hash. It records code commit/dirty-diff/lock/resolved/scientific
hashes, every input hash, the split hash and problem sets, the primary/excluded feature
lists and the forbidden-feature check, the numeric q90, binary/Ridge selections, the probe
family settings, Stage 8 artifact hashes, the calibrator parameters and hash, validation
output hashes, metric definitions, the bootstrap specification (2,000 replicates, unit
`problem_id`, paired, seeds) and the registry hash. No control-plane field is present.
`verify_freeze` compares every one of these before a test row is read.

## Test lock

Path audited at `67f824a`: `check_resume` (same commit, lock, config) → `--confirm-frozen`
must be a 64-hex digest → freeze exists, passes its sidecar and equals the digest → manifest
records that digest → run/Stage 8 identity → code/lock/config identity → input hashes →
split → Stage 8 still verifies with identical selection/threshold → calibrator, registry and
validation outputs hash as frozen and calibrator fitted on validation → bootstrap count →
no receipt/complete state → claim saved as an immutable revision → first test row read.
Exact command (worker transcript, 20:12:41 UTC):

```
uv run --frozen preanchor evaluate-test --config configs/mvp.yaml \
  --run-id 20260915T201107Z-484989d2fe-19f1ab0d \
  --confirm-frozen e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf
```

Invocations: 2 (20:12:41 evaluated; 20:13:48 refused). Successful evaluations: 1. Attempts
recorded: 1, `resumed: false`, no `failures/`, no `failure.json`. Test predictions contain
exactly the problems 2236, 2870, 3916, 6998 with 2 trajectories each, 689 rows, 25 anchors;
no sentence id appears in both partitions; every scored row's label and importance equal
`stage4/labels.parquet`. No other run in the store has a `stage9_*` state or a test output.

## Post-test history

Commits after the test on the Stage 9 branch: `9e71b1f` (backup classification/verification
for Stages 8–9; `validate_stage9` gains a `write_record=False` read-only mode; a synthetic
test fixture constant) and `8f4b20a` (documentation). No change to layer, C, alpha,
features, threshold, calibration, eligibility, split, metric definitions, model family or
preprocessing. No evidence of tuning.

## Independent metric reproduction

From the saved predictions and the freeze only, with sklearn/scipy and an independent
bootstrap loop (`derived_seed(20260914, "bootstrap", split)`; problems drawn with
replacement, both trajectories and all sentences of a drawn problem, one draw shared by
every model): 810 comparisons on validation and test (AUCPR, AUROC, Brier, log loss, ECE,
per-trajectory Precision/NDCG@10 %, calibrated Brier/log loss/ECE, Spearman, RMSE, AUCPR
gain, all bootstrap interval bounds and `n_valid`, all paired-difference intervals, the
sampled-problem hash `d4a20caf…`) agree with the saved values, worst absolute difference
5.6e-17. Test primary AUCPR 0.09412 (0.036 … 0.162); validation 0.16450 (0.061 … 0.253).

## Integration and backup

`codex/stage8` fast-forwarded `358945f` → `a0ebdc3` at 20:49 UTC (clean tree, no process in
that checkout). Integration `63e2e55` merges `8f4b20a` into `a0ebdc3`; the only conflicts
were the append-only sections of `docs/continuation.md` and `docs/decisions.md`, resolved
by keeping the integration order and appending the Stage 9 sections. Test suite on the
integrated tree: three merge-interaction failures (Stage 9 backup tests predating the
writer-lease gate; a closure test that assumed stage 9 unknown) fixed in `af9f431` without
weakening any assertion; see the continuation entry for the suite result. Backup: bundle
`20260915T212836Z-99e76aa8ec5fd13b4ce82b2307b4357e044297e24b99d054c626aa2a067fb15b`
prepared under the `portable-store:writer` lease of session `20260915T204953Z-audit-b7380977`
(0 new objects; the Stage 9 files were already in the verified store), `backup verify` ok,
scratch-restored and `restore verify` passed with `scientifically_equivalent: true`.
Primary-only until the MacBook `pull_verification` receipt for this bundle exists.
