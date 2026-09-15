# Data dictionary

Document table grain, field meaning, schema versions, provenance, eligibility and feature exclusions.

Stage interfaces are specified in SPEC §5. No data tables have been produced.
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
