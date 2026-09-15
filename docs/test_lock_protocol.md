# Test lock protocol

Training-only selection, validation-only calibration and a single frozen-hash test
evaluation (SPEC §3.4, §3.10, §10.5, Stage 8-9; AGENTS §5.4).

## Partitions

The fixed problem split `configs/splits/mvp_20_v1.yaml` (12 train / 4 validation / 4 test
problems, both trajectories of a problem together; SHA-256
`b85dc4f59e1f595099008656cfac1dc005955b617c60b3c235c635ca90617f47`) is the only partition.
`config.py` asserts the sets are disjoint; `tests/scientific/test_split_isolation.py` asserts
no row-wise splitter exists in `src/`. The four test problems are an exploratory held-out
engineering test, never a confirmatory test.

## Stage 8 (`preanchor fit`, `preanchor select`) — training rows only

- `probes/dataset.py` selects `eligible` rows with `split == "train"` and refuses any other
  partition on the command line. Validation and test rows are never loaded into a fit,
  scaler, residualiser or selection. `validate-probes` rejects any non-training row in
  `fold_predictions.parquet`.
- Folds: `StratifiedGroupKFold(4, shuffle=True)` grouped by `problem_id`, seeded by
  `derived_seed(seed_root, "cv_folds")`; the seed and validation problems of every fold are
  recorded in the run manifest (`fold_seeds`). A fold with a single class fails the run.
- Every `StandardScaler`, residualisation projection and hyperparameter is fitted on the
  training rows of a fold only (`tests/scientific/test_pipeline_fit_scope.py`).
- The anchor threshold is the Stage 4 train-only q90 (`anchor_threshold` column); Stage 8
  reads it and never recomputes a quantile.
- Selection is by mean train-CV average precision (binary) and Spearman with RMSE tie-break
  (Ridge). The result is `probe_models/frozen_selection.pending.json`; nothing in Stage 8
  reads a validation or test metric.

## Stage 9 (`probes/stage9.py`) — one run, three commands, one code identity

A Stage 9 run is its own `run_id` whose `input_hashes` name the finalized Stage 8 run
(`probe_run_id`, pending selection, `selected_model.joblib`, both CV tables) and the Stage 7
inputs behind it. `load_probe_run` refuses a Stage 8 run unless fit and select are complete,
`validate-probes` passed after selection, every recorded hash resolves on disk, the
selection basis is `train_cv_only`, the pending record's scientific-config hash equals the
current one and the Stage 7 inputs are unchanged. All three commands go through
`check_resume`, so they run at one commit, one lock and one resolved configuration.

1. `preanchor calibrate --probe-run-id <S8>` scores the **validation** partition with every
   refitted family at its selected configuration (`score_split`, `split == "validation"`
   only), asserts the partition's `anchor_threshold` equals the training q90 and that
   `anchor == importance_primary >= threshold`, applies the prespecified skip rule
   (`calibration_decision`: fewer than 20 positives or one class → skipped with the reason
   recorded), otherwise fits the Platt sigmoid on the selected binary probe's validation
   `score_raw` (`fit_platt`, Platt 1999 prior-corrected targets, BFGS). Outputs:
   `predictions/validation.parquet`, `metrics/validation.json` (AUCPR, AUROC, Brier, log
   loss, 10-bin equal-frequency ECE and reliability, per-trajectory Precision@top-10 % and
   NDCG@top-10 %, Spearman, RMSE, AUCPR gain, paired differences, 2,000-replicate clustered
   bootstrap), `probe_models/calibrator.json`. Calibration runs once per run; it cannot alter
   the pending selection (`tests/integration/test_stage9_pipeline.py` fingerprints the Stage 8
   artifacts before and after).
2. `preanchor freeze-selection --run-id <S9>` writes `metrics/analysis_registry.json` (the
   prespecified confirmatory analyses, secondary metrics, Stage 10 sensitivities, model and
   calibration identity, threshold, bootstrap specification, test partition identity; no
   realized test value) and `frozen_selection.json`: code commit, dirty-diff, lock, resolved
   and scientific config hashes, every input hash, the split file hash and problem sets, the
   primary/excluded feature lists and forbidden-feature check, the numeric training q90, the
   selected binary layer/`C` and Ridge layer/alpha, the frozen probe-family settings, the
   Stage 8 artifact hashes, the calibration method/parameters or skip reason and the
   calibrator hash, the validation output hashes, metric definitions, the bootstrap
   specification and the registry hash. Serialised with `json_bytes` (sorted keys); **the
   file's SHA-256 is the freeze hash**, recorded in the manifest (`stages.stage9_freeze`) and
   in `output_hashes`. Freezing refuses to run twice.
3. `preanchor evaluate-test --run-id <S9> --confirm-frozen <sha256>` (`evaluate_test`)
   fails closed unless, in order: the run identity check passes; `--confirm-frozen` is a
   lowercase 64-hex digest; `frozen_selection.json` exists, passes its sidecar check and
   hashes to exactly the supplied value; the manifest records that same hash as the freeze;
   the record belongs to this run and this Stage 8 run; code/lock/config identities equal
   the run's; every input hash equals the freeze; the split file and test problem list are
   unchanged; the Stage 8 run still verifies with identical hashes, layer, regularisation
   and threshold; the calibrator, registry and validation outputs hash as frozen and the
   calibrator was fitted on validation; the bootstrap replicate count is the frozen one; and
   no receipt or complete test state exists. Only then does it read a test row.

### Exactly-once semantics and interruption

- The claim (`stages.stage9_test = claimed`, attempt id, freeze hash) is saved as an
  immutable manifest revision **before** any test row is read.
- Outputs (`predictions/test.parquet`, `metrics/test.json`) contain no timestamps and are
  byte-deterministic for a freeze (fixed models, derived bootstrap seed); they are written
  with the non-replacing atomic writer, so a retry after a crash either reproduces identical
  bytes or fails on the first differing artifact. `metrics/test_receipt.json` (attempt
  history, freeze hash, output hashes, timing) is the commit point; the manifest then records
  `complete`.
- An interrupted attempt (state `claimed` or `failed`, no receipt) may be retried with the
  same hash; it is recorded as a further attempt with `resumed: true`. A receipt, a complete
  state, planted outputs without a claim, or a claim under another freeze all refuse.
- `preanchor validate-evaluation --run-id <S9>` re-runs the freeze checks against the
  current artifacts read-only and reports `ok: false` with reasons when any selection-relevant
  state changed after the freeze: the run is then invalid and a new `run_id` is required.

### Tests

`tests/scientific/test_test_lock.py`: refusal before the freeze, malformed/uppercase/wrong
hashes, another run's freeze (by argument and by copied file), code change after the freeze,
calibrator change, selection change (and `validate-evaluation` invalidation), Stage 4 label
change, crash before outputs and crash before the receipt with identical-byte recovery,
exactly-once after publication, planted outputs, and that `fit_platt` is only called from
`calibrate`. `tests/unit/test_clustered_bootstrap.py`: problem clusters keep both
trajectories and every sentence with multiplicity, one draw per replicate shared by all
models, 2,000 replicates under `configs/mvp.yaml`, derived seeds, determinism.

## Status (2026-09-15)

Stage 9 executed once on the canonical Stage 8 run `20260915T193326Z-57bcb41c84-7ebd0db6`
at commit `67f824a`: run `20260915T201107Z-484989d2fe-19f1ab0d`, freeze
`e68f1d829145431481d81d25c1546773d81e66c3cd3a39a1fdd98f298113e0bf`, calibration fitted on
680 validation rows / 54 anchors, test evaluated once (689 rows, 25 anchors, 4 problems,
1 attempt, receipt `a6ef2ba8…`). See `docs/stage9_validation.md`.
Independent audit (2026-09-15, `docs/stage9_audit.md`): ACCEPT; freeze-before-test,
exact-hash enforcement, exactly-once evaluation, train-only threshold, validation-only
calibration and the four test problems re-verified from the immutable manifest revisions
and the saved artifacts; every saved metric reproduces from the saved predictions.
