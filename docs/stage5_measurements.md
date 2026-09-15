# Stage 5 A6000 measurements and production launch (2026-09-15)

All runs live under `$PREANCHOR_ROOT/artifacts/runs/<run_id>`; benchmark and startup logs
under `$PREANCHOR_ROOT/artifacts/a6000-admission/`. Nothing listed here was deleted.
Runs marked **pre-fix** carry a `PRE_FIX_NON_CANONICAL.json` marker (index:
`artifacts/PRE_FIX_NON_CANONICAL_INDEX.json`) and must not be used as evidence for
canonical labels. Cost figures use `PREANCHOR_HOURLY_RATE=1.09` (user planning rate;
billed rate unavailable).

Frozen settings for every run: DeepSeek-R1-Distill-Qwen-14B revision
`1df8507178afcc1bef68cd8c393f61a886323761`, vLLM image digest
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` (v0.28.0),
dtype bfloat16, `max_model_len` 20480, `gpu_memory_utilization` 0.92, prefix caching on,
vLLM seed 20260914, 8 forks × 32 samples = 256 rollouts, single NVIDIA RTX A6000 (49,140 MiB).

## Benchmarks (8 units, one per fork)

| Run | Status | Commit | Config | Aggregate tok/s | Per-request decode tok/s | Generated tokens | Mean output tokens | Peak VRAM MiB | KV peak occupancy | Finish reasons | Cost (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260915T103143Z-2749b29e05-83fb8155` | pre-fix | a624171 | B2: concurrency 2, max_num_seqs 2 | 42.34 | 22.73 | 30,575 | 3,821.9 | 43,967 | 0.126 | stop 8 | 0.219 |
| `20260915T105529Z-2749b29e05-ad9de222` | pre-fix | a624171 | B3: concurrency 3, max_num_seqs 3 | 59.03 | 21.26 | 45,685 | 5,710.6 | 43,956 | 0.292 | stop 7, length 1 | 0.234 |
| `20260915T122814Z-d6cd3c11d8-0b463188` | **post-fix, canonical evidence** | f3c0a38 | B2: concurrency 2, max_num_seqs 2 | 38.18 | 22.77 | 30,366 | 3,795.8 | 43,967 | 0.125 | stop 8 | 0.241 |

Notes:

- B3 was rejected: one of eight completions hit the length cap (12.5% truncation),
  failing the 5% truncation gate, despite ~39% higher aggregate throughput.
- The post-fix B2 benchmark reproduced the pre-fix B2 measurement within noise
  (same peak VRAM, same prefix-cache hit rate 0.710, same finish reasons, per-request
  decode rate 22.7–22.8 tok/s). The lower aggregate figure (38.2 vs 42.3 tok/s) comes
  from the tail of the run where only one request was in flight; per-unit wall time
  was 99.4 s.
- Available KV cache: 14.42 GiB, 78,768 tokens; measured headroom 43,934 tokens at
  concurrency 2; min free HBM 4,574 MiB; no OOM, no request failures, zero benchmark
  contamination and zero duplicate rollout IDs in every run.
- Earlier pre-fix production attempts `20260915T111338Z-10dd1d2789-0135d659` and
  `20260915T111421Z-ab6207e1fa-98852b69` were stopped cleanly during the Stage 4
  remediation and are retained as non-canonical evidence. Their projection
  (6.42 GPU-hours, $7.00, 978,400 tokens) derived from the pre-fix B2 benchmark.

## Canonical production run

- Run id: `20260915T124322Z-d6cd3c11d8-5568422c` (namespace `production`)
- Code commit: `f3c0a38df233d3eb94d746f05ce23eb17a81fa3b`, clean tree (dirty diff hash
  is the empty diff), resolved config hash `9bad4b36…960ff`
- Inputs: repaired Stage 4 / Stage 3 hashes recorded in the manifest
  (`stage3/trajectories.parquet` 07867455…, `stage3/sentences.parquet` 83cf8c20…,
  `configs/pilot_targets_4682.yaml` 1243a6fe…)
- Selection: B2 chosen by rule "B3 only with ≥10% aggregate throughput improvement
  and safety gates"; adaptive extension **not** authorized
- Measured projection (from the post-fix benchmark): 971,712 generated tokens,
  25,452 s ≈ 7.07 GPU-hours ≈ $7.71; stop rules 10 h / $20
- Plan: 256 jobs, 256 unique rollout IDs, forks 6, 7, 12, 13, 14, 15, 46, 47
- Launched 2026-09-15 12:44:08 UTC in tmux session `a6000-production`; log
  `artifacts/a6000-admission/production-b2-postfix.log`; vLLM startup log
  `artifacts/a6000-admission/vllm-b2-postfix-startup.log`
- Resume: `python -m preanchor.labels.a6000 execute --run-id 20260915T124322Z-d6cd3c11d8-5568422c --startup <startup log>`
  (check_resume refuses if commit, config, inputs, or vLLM digest differ)
- Expected completion: roughly 19:45–20:15 UTC, followed by `finalize`.
