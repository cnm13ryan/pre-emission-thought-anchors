# Stage 6 validation — activation-extraction smoke (2026-09-15, H100 PCIe)

Branch `codex/stage6` from `abf7ea5`; implementation commits `4e00554` (extraction),
`434340c` (amendment and records), `f9dd294` (offset fix), `1907c1c` (heartbeat during
diagnostics). Persistent root `/lambda/nfs/preanchor/preanchor`; every run below is under
`artifacts/runs/`. SPEC Stage 7 (all 40 trajectories) was **not** run. Stage 3–5 canonical
artifacts were read only. Cost figures use `PREANCHOR_HOURLY_RATE=2.49` (planning rate).

Inputs (identical in every run): `stage3/trajectories.parquet` `4fd5fbee…39913d`,
`stage3/sentences.parquet` `83cf8c20…143cf8`, `stage4/labels.parquet` `fab9886e…a61b31`,
model and tokenizer revision `1df8507178afcc1bef68cd8c393f61a886323761`, lock
`1126f52a…`. Resolved `configs/smoke.yaml` hash `30994ebc…fab79` (scientific
`bc44349e…d74756`; it differs from `configs/mvp.yaml`'s `9bad4b36…` only by the execution
scope keys, as intended for smoke runs).

## Runs

| Run | Purpose | Commit | Result |
|---|---|---|---|
| `20260915T164208Z-29d5f2acb0-7a4c0acd` | SPEC Stage 6 smoke: `330:correct`, 16 boundaries | `434340c` | complete; 16 rows; primary gate pass (max abs 0.0); logit identity exact; 14/16 truncated-prefix diagnostics within 0.02, max 4.5 |
| `20260915T164439Z-29d5f2acb0-586ddfe5` | first interruption drill | `434340c` | failed on `2189:incorrect`: offset check rejected coinciding byte-level spans (`" ÷"`); fixed in `f9dd294`; retained as evidence |
| `20260915T165230Z-8b49a551f6-1dfccd21` | SIGTERM drill, `330:correct` + `2189:incorrect`, 16 each | `f9dd294` | `StageInterrupted: SIGTERM` inside unit 2; resume: unit 1 bytes and mtime unchanged, partial quarantined, unit 2 regenerated; 32 rows; primary pass; 29/32 diagnostics within 0.02, max 8.0 |
| `20260915T165641Z-8b49a551f6-c8fcfdfe` | SIGKILL drill, same scope | `f9dd294` | exit 137 with stage `running`; resume wrote the `UnfinishedAttempt` record, quarantined the partial, regenerated unit 2; unit 1 unchanged; 32 rows; primary pass |
| `20260915T170028Z-8b49a551f6-36843658` | memory envelope: `3935:incorrect`, all 389 boundaries (13,087 tokens) | `f9dd294` | complete; 389 rows; forward 3.45 s at 3,790 tok/s; peak VRAM 29,974 MiB (32,127 MiB during the full-logit identity check); primary gate pass (max abs 0.0 at every site); 388/389 truncated-prefix diagnostics within 0.02, the exception at prefix length 107 (max abs 12.0), shortest matching prefix 136; logit identity exact at three boundaries; attempt wall 1,652 s because the diagnostics replay the 13,087-token trace twice per boundary |

Determinism across runs: the `330:correct` shard hashes `987ce4ab…656fd` in all four
successful runs and the `2189:incorrect` shard `5a8f3222…` in both drills, including the
regenerated units; stored fp16 versus a fresh bf16 replay differs by at most 3.0e-8.

## Measured resources

| Quantity | Smoke (2,973 tokens, 16 rows) | Envelope (13,087 tokens, 389 rows) |
|---|---|---|
| Model load (NFS cache) | 71.2 s | 52.5 s |
| Forward with hooks | 1.52 s (first forward, includes warm-up) | 3.45 s (3,790 tok/s) |
| Peak VRAM allocated | 28,568 MiB | 29,974 MiB |
| Shard bytes | 8,192,168 (512,010 B/row) | 199,168,176 (512,000 B/row) |
| Peak host RSS | 9.8 GB | 9.8 GB |
| Wall for the unit | 89 s attempt (load dominated) | 70.6 s |

Storage: 512,000 B per row × 6,474 rows = 3.31 GB for Stage 7 plus index and sidecars.

## Gates

- Model: 14,770,033,664 parameters on `cuda:0`, bf16, SDPA, 48 layers, d=5120, untied
  head; tokenizer fast, `add_special_tokens=False`, decode round-trip exact.
- Boundaries: SPEC §10.2 structural audit passes for all 40 trajectories on CPU (6,474
  rows, no negative boundary, minimum boundary index 60); the smoke's 16 boundaries are
  69, 85, 111, 115, 159, 171, 181, 208, 235, 256, 268, 317, 335, 347, 364, 374.
- Shapes: `resid [n, 49, 5120]` float16, `final_prev_position [n, 5120]` float16, all
  finite; `final_prev_position_available` true for every stored row.
- Primary causal-invariance gate (amendment `docs/amendments/20260915T170000Z.md`):
  same-shape future-token mutation changes no site at any boundary (max abs 0.0 at every
  one of the 49 sites for all 16 + 32 + 32 + 389 boundaries).
- Diagnostic: truncated-prefix replay differs for prefixes shorter than about 100 tokens
  (smoke: prefixes 70 and 86 → 4.5 and 2.0 at layer 48, first difference at layer 1) and
  is bitwise identical otherwise. Envelope: the single exceedance is the shortest prefix (107 tokens, max abs 12.0 at layer 48 where |values| reach the hundreds); prefixes of 136+ tokens match bitwise. Across all 469 boundaries checked, every exceedance has a prefix below 108 tokens and every same-shape mutation is exactly zero.
- Logit identity: `lm_head(final_norm(layer48))` equals the model's boundary logits from
  the same pass, max abs 0.0, argmax equal, at the checked boundaries.
- Observables: `logit_features.parquet` (12 SPEC §3.9 columns plus
  `diagnostic__first_target_token_logprob`) computed in float32 from stored states; no
  full-vocabulary logits stored.
- Integrity: `validate-activations` passes on every completed run (unit checksums, index
  schema and coverage, index rows against shard metadata).
- Tests: 209 CPU tests pass (`tests/unit`, `tests/integration`, `tests/scientific`; the
  real-model scientific test is skipped without the GPU); ruff clean on every changed file.

## Stage 7 projection (`stages/stage6/resource_projection.json` of the envelope run)

40 units, 6,474 rows, 187,579 tokens; measured 3,790 tok/s and 30.0 GB peak;
projected 410 s ≈ 0.11 GPU-hours ≈ $0.28 at 2.49 USD/h, 3.31 GB output, 6.6 GB disk
growth upper bound. The projection is conservative (doubles the per-unit forward time
and adds a 100 MB/s write allowance); the smoke run's projection from its cold forward was
320 s.

## Stage 7 admission

| Gate | Status |
|---|---|
| Measured fit on this GPU (SPEC §7.1, AGENTS §8) | PASS |
| Resource projection newer than the current commit | must be regenerated at the Stage 7 commit (the envelope projection is from `f9dd294`; `1907c1c` changed only heartbeat pulses) |
| Stage 5 secondary backup verified | BLOCKED (`backup.verified: false`; MacBook receipt pending) |
| Explicit user authorisation of Stage 7 | BLOCKED |
