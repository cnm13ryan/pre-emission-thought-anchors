# Test lock protocol

Training-only selection, validation-only calibration and a single frozen-hash test
evaluation (SPEC §3.4, §3.10, §10.5, Stage 8-9; AGENTS §5.4).

## Partitions

The fixed problem split `configs/splits/mvp_20_v1.yaml` (12 train / 4 validation / 4 test
problems, both trajectories of a problem together) is the only partition. `config.py`
asserts the sets are disjoint; `tests/scientific/test_split_isolation.py` asserts no
row-wise splitter exists in `src/`. The four test problems are an exploratory held-out
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

## Stage 9 — validation calibration and the freeze

- `calibrate` reads validation rows only; the Platt calibrator is skipped, with the reason
  recorded, when validation has fewer than 20 positives or one class
  (`probes/calibrate.py::calibration_decision`).
- `freeze-selection` copies the pending selection, the calibrator and the threshold into
  `frozen_selection.json`; its SHA-256 is the freeze hash.
- `evaluate-test` runs once, requires `--confirm-frozen <sha256>` equal to the recorded
  freeze hash, refuses to run before the freeze exists, and any later change to the
  selection invalidates the run (new `run_id`).

## Status

Stage 8 implemented (`codex/stage8`); validated on synthetic runs and a two-site real-data
smoke. Stage 9 commands are not implemented. No validation or test metric has been
computed or inspected.
