# Data dictionary

Document table grain, field meaning, schema versions, provenance, eligibility and feature exclusions.

Stage interfaces are specified in SPEC §5.
The 12 observable column names in `configs/mvp.yaml` map in order to SPEC §3.9;
`generated_token_count` counts tokens since the start of `<think>`, and
`prefix_context_fraction` divides prefix token count by 32768. Neither uses completed length.

## Activation index and shards (Stage 6/7, `artifacts/runs/<run_id>/activations/`)

`units/<unit_key>/<problem>_<kind>.safetensors` holds `resid` `[n_sentences, 49, 5120]`
float16 (site 0 = embedding output, sites 1–48 = decoder-block outputs before the final
RMSNorm, at `boundary_token_idx = target_token_start − 1`) and `final_prev_position`
`[n_sentences, 5120]` float16 (site 48 at `boundary_token_idx − 1`, NaN when unavailable).
`shard_meta.json` records the unit key, trace and token-id hashes, the boundary table and
the architecture check; `SUCCESS.json` lists every file hash. `index.parquet` has one row
per stored sentence: SPEC §5 activation-index columns, the boundary indices,
`overlapping_token`, and the Stage 4 eligibility fields (`eligible`, `eligible_k4`,
`exclusion_reason`, `label_source`, `labels_sha256`) for selection only; importance values
stay in `data/processed/stage4/labels.parquet` and join on `sentence_id`.

## Feature table (Stage 7, `data/processed/stage7/features.parquet`)

Canonical table: 6,474 rows from run `20260915T182306Z-3ab408d80a-7452f1c2`, SHA-256
`f3227c386383701741aa284cc3bbf6b7c3205f2de9593616575bf1187e9a064d` (`docs/stage7_validation.md`);
protected with the run by `preanchor backup --for-stage 7` (`docs/backup_restore.md`).

Built by `preanchor build-dataset --config configs/mvp.yaml --run-id <extraction run>` from a
completed `extract-activations` run; the run keeps the identical table under
`features/features.parquet` with `feature_audit.json` and `feature_validation.json`. One
row per stored sentence (all 6,474 Stage 4 rows), sorted by `trajectory_id, sentence_idx`,
joined to the activation index and to `stage4/labels.parquet` by `sentence_id` only.
Schema and column classes live in `preanchor.features.observables` (`FEATURES_SCHEMA`,
`PRIMARY_FEATURES`, `DIAGNOSTIC_FEATURES`); `check_feature_names` asserts the primary list
equals `configs/mvp.yaml` `features.observables` and that no excluded quantity appears
outside a `diagnostic__` column.

| Class | Columns | Meaning |
|---|---|---|
| identity | `schema_version`, `sentence_id`, `trajectory_id`, `problem_id`, `trace_kind`, `split` | `split` is the frozen problem split (`train`/`validation`/`test`) of `configs/splits/mvp_20_v1.yaml` |
| primary (SPEC §3.9, the only probe inputs) | `sentence_idx`; `generated_token_count` = prefix tokens after the stored prompt (the prompt ends with `<think>\n`; counted from the full-trace offsets, never by re-tokenising); `prefix_token_count` = `boundary_token_idx + 1`; `log1p_prefix_token_count`; `prefix_context_fraction` = prefix / 32768; `previous_sentence_tokens` = token span `[target_token_start, target_token_end)` of `S_{i-1}` (0 for the first sentence); `previous_emitted_token_logprob` = log-probability of the boundary token under the distribution at `boundary − 1` (from `final_prev_position`); `next_token_entropy`, `next_token_max_probability`, `next_token_probability_margin` (top-1 − top-2), `next_token_top5_mass`, `next_token_top10_mass` from the boundary distribution | All next-token quantities are `lm_head(final_norm(h))` applied in float32 to the stored float16 `layer=48` state; full-vocabulary logits are never stored |
| diagnostic (never a probe input) | `diagnostic__first_target_token_logprob` (log-probability of the actual first target token at the boundary), `diagnostic__relative_completed_trace_position` (`sentence_idx / n_sentences` of the completed trace) | SPEC §3.9 exclusions kept for audits; the `diagnostic__` prefix is the frozen marker |
| boundary | `boundary_token_idx`, `target_token_start`, `overlapping_token`, `final_prev_position_available` | copied from the activation index |
| activation pointer | `activation_run_id`, `shard_path`, `row_idx`, `trace_sha256`, `token_ids_sha256` | `resid[row_idx]` of `artifacts/runs/<activation_run_id>/activations/<shard_path>` is this row's `[49, 5120]` float16 state |
| target (Stage 4, joined) | `eligible`, `eligible_k4`, `exclusion_reason`, `label_source`, `importance_released`, `importance_legacy`, `importance_canonical`, `importance_primary`, `importance_log1p`, `anchor_threshold`, `anchor` | byte-equal to `stage4/labels.parquet`; support counts, convergence and final-answer indices stay in the label table (join on `sentence_id`) because they describe the completed trace |

`validate-features` re-checks coverage (every index row and, under `configs/mvp.yaml`,
every label row exactly once), the joins, the allow-list, definitional and range
invariants of every primary/diagnostic column, the published-copy identity, and recomputes
a seeded sample of rows from the shards in float32 (exact for integer columns, `1e-5`
for the uncertainty and diagnostic columns).

## Stage 8 probe tables (`artifacts/runs/<run_id>/probe_models/`)

`units/<unit_key>/` is one (probe, site) work unit with `cv_results.parquet`,
`fold_predictions.parquet`, `unit.json` and `SUCCESS.json`; the unit key is
`ids.cache_key(stage="stage8_fit", ...)` over the scientific config hash, code commit,
Stage 7 input hashes, partition and fold seed. Units are byte-deterministic; timings live
only in `stages/stage8/summary.json`.

`cv_results.parquet`: one row per (`probe_name`, `layer`, `task`, `hyperparameter`,
`fold`) with fold sizes and class counts, the validation problems of the fold,
`average_precision`/`auroc` (binary) or `spearman`/`rmse` (continuous), `converged`.

`fold_predictions.parquet`: SPEC §5 prediction columns (`split` is always `train`,
`lead_sentences` 0, `threshold_quantile` 0.9) plus `task`, `hyperparameter`, `fold`;
`importance_true` and `importance_predicted` are on the `log1p` scale of the Ridge target.

`selected_model.joblib`: `{"models": {family:task -> record}, "pipelines": {...}}` refitted
on all training rows at each family's selected configuration.
`frozen_selection.pending.json`: the train-CV selection Stage 9 freezes.
