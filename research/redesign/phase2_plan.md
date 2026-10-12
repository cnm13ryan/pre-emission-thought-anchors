# Phase 2 plan: run-to-run reliability of the candidate labels

**Status: EXPLORATORY.** `stage: exploratory`, `preregistered: false`. Written 2026-10-09,
before any Phase 2 generation. The reliability thresholds in §4 are fixed now and will
not be changed after results are seen. No frozen MVP setting changes. This is a separate
experiment with its own config hash (`phase2_config.json`, hash `23f5e3e6bc2a…`) and its
own run directory.

## 1. Design (FIXED)

| Item | Choice |
|---|---|
| Problems | 10 of the 16 non-test MVP problems. The 4 test problems are untouched. |
| Selection | Seeded rule in `phase2_select.py`, seed material `20260914:phase2_selection`, applied before any generation. Output is [`phase2_forks.json`](phase2_forks.json); no trajectory needed replacing. |
| Trajectories / windows | One trajectory per problem; 20 contiguous forks per trajectory, inside the frozen pre-convergence and pre-final-answer region. |
| Units | **200 forks** and **190 sentence labels** (forks i and i+1 are both rerun). |
| Reruns | **2 independent replicates × 100 samples per fork = 40,000 rollouts.** |
| Seeds | `seed_root_r = uint32(sha256("20260914:phase2_replicate:r")[:8])` = 20419384 for r = 1 and 419638246 for r = 2, then the frozen per-sample derivation. |
| Generation | Frozen MVP settings: pinned model revision, vLLM image digest `61fc8a89…`, bf16, temperature 0.6, top_p 0.95, max_new_tokens 16384, exact stored prompt `full_cot[:char_start]`, no chat template, no added special tokens. The only change is `max_model_len` 24576 instead of 20480, because the longest prompt is ~4.6k tokens. |
| Order | Sample-major across both replicates. An early stop leaves every fork with the same N in both replicates. |
| Labels | The Phase 1 rules, unchanged: (a) KL α = 1e-9, flag ≥ 1.355; (b) KL α = ½, flag ≥ 0.2462; (c) fork TV, flag ≥ 0.17; (d) log(1 + KL α = 1e-9). Similarity uses the pinned MiniLM, threshold < 0.8, and the first replacement sentence as extracted by `extract_first_replacement_sentence`. Answers use `fresh.answer_category` (first `\boxed{}`, normalised). Flag thresholds are the **Phase 1 training-only q90 values, fixed**; they are not refit per replicate. |

## 2. Statistics (FIXED)

For each label, compare replicate 1 against replicate 2 over the 190 sentence labels:

- **Continuous forms**, i.e. log(1 + KL) for (a)/(d) and (b), and the raw value of (c):
  Spearman ρ, and ICC(2,1) (two-way random effects, absolute agreement, single
  measurement).
- **Binary flags**, i.e. (a), (b) and (c) at their fixed thresholds: Cohen's κ.
- **Uncertainty:** 95% percentile intervals from 2,000 problem-cluster bootstrap
  replicates, resampling the 10 problems with all their sentences, seeded with
  `20260914:phase2_bootstrap`.
- **Support:** a sentence whose "without" group has no dissimilar sample in either
  replicate gets status NO_SUPPORT for the KL labels. It is excluded from those labels and
  counted. `fork_tv` needs no similarity filter.
- **Secondary, descriptive:** each replicate against the released 100 samples
  ("replicate 0"); `<NO_ANSWER>` classes using vLLM's exact `finish_reason == "length"`.

## 3. Missing and partial data (FIXED)

Every rollout has a status (`OK`, `MISSING`, `GENERATION_ERROR`). A failed request stops
new launches, is recorded, and is retried on resume. Completed units are never rewritten.
If a budget guard stops the run early, the analysis uses the largest common N per fork
across both replicates, and the report states that N.

## 4. Minimum reliability thresholds (FIXED before running)

A label form **passes** only if every one of its point estimates meets the threshold:

| Form | Threshold |
|---|---|
| Continuous | Spearman ρ ≥ **0.80** **and** ICC(2,1) ≥ **0.75** |
| Binary flag | Cohen's κ ≥ **0.60** |

- A passing form whose 95% interval lower bound falls below its threshold is reported as
  **"pass, not robust"**.
- Forms below threshold are dropped from Phase 3.
- **If no form passes, the report states that the counterfactual-importance target cannot
  be measured reliably at about 100 samples per fork, as a result in its own right.**
- Thresholds are never widened after results are seen.

## 5. Compute, runtime and cost

**Hardware:** 1 × NVIDIA H100 PCIe 80 GB (Lambda, `209.20.159.95`), driver 580.105.08, and
a fresh persistent filesystem at `/lambda/nfs/preanchor`.

**Planning rate:** $2.49/h, the Lambda on-demand H100 PCIe list price used in
`docs/stage5_measurements.md`. **The billed rate is unverified; the owner should
confirm it.**

**Token budget:** 40,000 rollouts × ~3,009 tokens. The mean is estimated from the released
rollouts of these exact forks; 0.08% are expected at the cap. That gives **≈ 120.4M
generated tokens**.

| Aggregate throughput | Source | Runtime | Cost at $2.49/h |
|---|---|---:|---:|
| 381 tok/s | Measured Stage 5 production on this GPU type, ~8 in flight | 88 h | $219 |
| 600 tok/s | Plausible at 16 in flight, given the shorter sequences here | 56 h | $139 |
| 900 tok/s | Optimistic at 32 in flight | 37 h | $93 |

The table is a pre-benchmark estimate. Per AGENTS §8, the run starts only after a measured
benchmark writes `resource_projection.json` and that projection is within the runtime and
cost limits **set by the owner**:

```
PREANCHOR_HOURLY_RATE=<billed $/h>
PREANCHOR_MAX_COST=<authorised $>
PREANCHOR_MAX_RUNTIME_HOURS=<authorised h>
```

The run also stops launching new requests when elapsed time × rate reaches either limit;
in-flight requests are completed.

**Benchmark procedure:** 64 units, which are the first jobs of the production order and
are kept. Run at max_num_seqs 16, then 32. Take 32 only if it improves aggregate
throughput by ≥ 10%, with zero request failures and length-finish share ≤ 5%. This is the
Stage 5 chain rule.

Similarity scoring and all statistics run on the MacBook (CPU/MPS) after results are
copied back. No GPU time is spent on them.

## 6. Commands (on the instance)

```
cd ~/preanchor-repo && set -a && source .env && set +a
export PREANCHOR_ROOT=/lambda/nfs/preanchor/preanchor
IN=$PREANCHOR_ROOT/data/processed/redesign_phase2/inputs.parquet   # sha256 f6e6d667…
RUN=$PREANCHOR_ROOT/artifacts/runs/<utc>-phase2-23f5e3e6
# leases (docs/concurrency.md), then:
bash research/redesign/phase2_vllm.sh start 16            # prints endpoint list
uv run --frozen python research/redesign/phase2_generate.py check     --inputs $IN --run-dir $RUN
uv run --frozen python research/redesign/phase2_generate.py benchmark --inputs $IN --run-dir $RUN --concurrency 16 --units 64
uv run --frozen python research/redesign/phase2_generate.py run       --inputs $IN --run-dir $RUN --concurrency <chosen> --confirm-projection <sha256>   # in tmux
uv run --frozen python research/redesign/phase2_generate.py index     --inputs $IN --run-dir $RUN
bash research/redesign/phase2_vllm.sh stop
```

## 7. Decision register for Phase 2

| Item | Status |
|---|---|
| Design, seeds, labels, statistics, thresholds (§1–4) | FIXED |
| Planning hourly rate | PROVISIONAL, until the owner confirms |
| Budget and runtime caps | **OPEN: owner must set them** |
| Concurrency | PROVISIONAL, until set by the benchmark chain rule |
