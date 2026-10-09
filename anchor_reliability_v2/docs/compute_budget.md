# Compute budget

**Rate:** $2.49/h, the Lambda on-demand H100 PCIe list price; the billed rate is
unverified. **Hardware:** 1 × H100 PCIe 80 GB.

## Measured throughput (pilot benchmarks, 2026-10-09)

Measured on the instance with the pinned vLLM image and model, `max_model_len` 24,576,
prefix caching on, 200 Phase 2 prompts, and the frozen generation settings. The benchmark
units are the first jobs of the Phase 2 order. They are kept, and they are not
confirmatory analyses.

| Concurrency | New units measured | Aggregate tok/s | Mean tokens / rollout | Length finishes | Errors |
|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 530.5 | 4,164 | 0 | 0 |
| 32 | 160 | **907.3** | 3,913 | 0 | 0 |

Concurrency 32 is chosen by the Stage 5 chain rule (≥ 1.10× gain, zero errors, ≤ 5%
length finishes).

Measured mean tokens are 30–38% above the estimate from released rollouts (3,009). The
estimate converted characters to tokens using a ratio calibrated on truncated rollouts,
and that ratio underestimates. Projections below use the measured mean.

These benchmarks ran through the earlier Phase 2 driver (`research/redesign/`), not this
folder's `src/arv2/vllm_client.py`. The study's own client has been exercised only by the
memorisation check (~420 short requests). A dedicated throughput pilot with the study
client was not run while the GPU was occupied.

## Phase 2: reliability replication (launched)

- **Size:** 40,000 rollouts × 3,913 tokens ≈ 156.5M tokens.
- **Projection at 907 tok/s:** 47.5 h, **$118**.
- **Guard:** $250 / 100 h.
- **Status:** launched 2026-10-09 07:56 UTC at the owner's instruction ("the reliability
  study should continue to run"), before plan v2. See the status note in the README.

## Phase 3: new problems (OPEN; not authorised)

Per problem:
- 1 base trace;
- with **S** sentence labels, S+1 forks × n resamples.

| Design | Tokens / problem | GPU-h / problem | $ / problem |
|---|---:|---:|---:|
| All ~100 sentences, n = 100 | ≈ 39.5M | 12.1 | $30 |
| 30 sentences, n = 100 | ≈ 12.1M | 3.7 | $9.2 |
| 30 sentences, n = 200 | ≈ 24.3M | 7.4 | $18.5 |

With the sample-size readings of `evaluation_design.md`:

| Problems | Design | Cost |
|---:|---|---:|
| 7 | 30 sentences, n = 100 | ≈ $65 |
| 28 | 30 sentences, n = 100 | ≈ $260 |
| 28 | all sentences | ≈ $840 |

These are projections only. A Phase 3 run needs its own measured pilot and explicit
approval.
