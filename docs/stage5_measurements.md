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

# Stage 5 H100 PCIe measurements and production run (2026-09-15, new region)

The H100 instance is in a new Lambda region with a fresh persistent filesystem
(virtiofs UUID `8dd29228-2370-430d-836b-a3e9de7cc148`). Stage 2–4 state was reacquired and
recomputed from the pinned upstream revisions, not copied; the provenance record with every
hash is `artifacts/reconstruction/20260915T145500Z-h100-cross-region.json`. All recorded
Stage 3/4 invariants reproduced (40 trajectories, 6,474 sentences, 6,474 raw files /
20,996,423,633 bytes, Spearman 0.857966, 4,019 eligible, 3,902 K=4, q90 1.354981, 344
anchors). `labels.parquet` hashes `fab9886e…a61b31` instead of the A6000-era `2049a33f…`:
the `similarity_sha256` column fingerprints raw float64 cosine scores of float32 CUDA
embeddings, which differ in the last bits between Ampere and Hopper; every threshold-derived
value agrees. Cross-run fingerprint over (sentence_id, importance_primary):
`7534066e845f29c9c4142811e296ddc51b2ff882d269995c63c5ae6f9c08da8d`. Stage 3
`trajectories.parquet` (`4fd5fbee…`) embeds the local Stage 2 run path, so Stage 5 rollout
IDs on this filesystem differ from the A6000 run's by construction; `sentences.parquet`
(`83cf8c20…`) is byte-identical to the canonical record.

Hardware: NVIDIA H100 PCIe, 81,559 MiB HBM2e (vendor 2.0 TB/s, 756 TFLOP/s dense BF16),
driver 580.105.08, CUDA 13.0 runtime, host PyTorch 2.7.0+cu126. Frozen settings unchanged
(same model revision, vLLM image digest `61fc8a89…` v0.28.0, bfloat16, `max_model_len`
20480, prefix caching, seed 20260914, temperature 0.6, top_p 0.95, max_new_tokens 16384,
8 forks × 32 samples). Execution-only change: `gpu_memory_utilization` 0.90 from
`configs/lambda/h100_pcie_80.yaml`; concurrency cap lifted in commit `be7d855`.
Rate `PREANCHOR_HOURLY_RATE=2.49` (Lambda on-demand H100 PCIe list price; billed rate
unverified).

Roofline (14.77 B parameters, 29.54 GB BF16; KV 196,608 B/token; worst case 17,417 tokens =
3.19 GiB per sequence): critical intensity 378 FLOP/B, weight-streaming ceiling 67.7 tok/s at
B=1; decode is bandwidth-bound through B=16. Predicted worst-context aggregate: B1 60.7,
B2 109.9, B4 185.0, B8 281.0, B12 339.8 tok/s.

vLLM startup (both launches identical): 29.29 GiB weights + non-torch, 1.58 GiB peak
activation, 0.12 GiB CUDA graphs, 40.39 GiB KV cache = 220,592 tokens, "10.77x" maximum
concurrency at 20,480. With the 15% worst-case margin (`required_capacity`) the admitted
ceiling is 11 in flight; 12 would need 240,355 tokens.

## Benchmarks (8 units, one per fork, seeds of sample 0)

| Run | Config | Aggregate tok/s | Per-request decode tok/s | Generated tokens | Mean output | Wall s | Peak VRAM MiB | Min free MiB | KV peak | Required / headroom KV tokens | Finish | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260915T143802Z-bb9ba695de-eee71ef4` | B4: concurrency 4, max_num_seqs 4 | 187.8 | 53.7 | 43,108 | 5,388.5 | 229.5 | 71,727 | 9,353 | 0.092 | 80,119 / 150,924 | stop 8 | $0.16 |
| `20260915T144551Z-bb9ba695de-4b18d0f3` | B8: concurrency 8, max_num_seqs 8 | 253.9 | 52.0 | 40,736 | 5,092.0 | 160.4 | 71,735 | 9,345 | 0.112 | 160,237 / 81,256 | stop 8 | $0.11 |

No OOMs, request failures, warnings or truncations in either. B8 improves aggregate throughput
by 35% over B4, so the ascending chain rule selected B8. The eight-unit workload cannot
exercise more than eight in flight, so concurrencies 9–11, although admitted by KV capacity,
were not benchmarked; the benchmark aggregate is bounded by the longest single request and
understates steady-state production throughput at high concurrency.

## Production run

- Run id: `20260915T144926Z-bb9ba695de-9fca4c21` (namespace `production`), commit
  `be7d855e653468b38e34e9e3a073835aba734719`, clean tree, resolved config hash `9bad4b36…`
  (unchanged from the A6000 run), image digest verified, 256 unique rollout IDs, 32 per fork.
- Projection (from B8): 1,303,552 tokens, 5,134 s = 1.43 GPU-hours, $3.55.
- Actual: launched 14:49:46 UTC, 3,520.8 s wall = 0.98 GPU-hours, $2.44 at the planning
  rate; 1,341,475 generated tokens (mean 5,240, median 5,004); aggregate 381.0 tok/s
  (1.50× the projection), per-request decode 49.6 tok/s; peak VRAM 71,105 MiB, min free
  9,975 MiB, KV peak occupancy 0.191, prefix-cache hit rate 0.979; finish reasons stop 255,
  length 1 (fork 47 sample 19 at 16,384 tokens; 0.4%, within the 5% gate); no failures, no
  OOMs, no warnings; first-window throughput check passed.
- Finalize: 256 valid completions and checksums, zero duplicate or contaminated units,
  support per fork (dissimilar valid): 6→32, 12→29, 13→20, 14→32, 46→10; no deficient fork,
  adaptive extension not required; `fresh_labels.parquet` written (state
  `labels_constructed`). Fork 46 sits exactly at the ten-sample support floor.
- Artifacts: `artifacts/runs/20260915T144926Z-bb9ba695de-9fca4c21/stages/stage5/`
  (`units/`, `summary.json`, `base_validation.json`, `fresh_labels.parquet`), startup and
  production logs under `artifacts/h100-admission/`. vLLM stopped; GPU memory 0 MiB.
- The A6000 runs above remain evidence only; the superseded A6000 production run stays on
  the old-region filesystem.
