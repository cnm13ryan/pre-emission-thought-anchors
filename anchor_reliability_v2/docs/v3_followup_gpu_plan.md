# v3 follow-up GPU plan: one bundled run on one instance

**Status: DRAFT v2, NOT APPROVED, NOT LAUNCHED.** EXPLORATORY, POST HOC.
- Nothing in this document has been run.
- No instance is launched and no lease is taken until the owner approves this plan.
- The plan_v3 verdict `INCONCLUSIVE_QUALITY_GATE` stands.
- Inputs are in `docs/v3_followup_cpu_results.md`.
- **LIT** marks a parameter that waits on the pending literature review (§8).
- The cost model is `analysis/v3_followup/gpu_plan_cost/cost.py`. Its outputs are
  `cost_K8*.json` and `cost_K4*.json`, and rerunning it with other parameters updates every
  number below.

**Changes from draft v1:**
- **T6:** the nested arm is redesigned as 20 problems × 20 positions, K = 8 (or 4),
  M = 200, scored with an unbiased importance; signed targets are dropped.
- **T4:** the forward-pass arms target other positions and tokens, not more probe
  capacity.
- **Hardware:** 8 × A100 80 GB is now the recommended default, because a single H100 would
  take 11–40 days.

## 0. Inputs from the CPU follow-up

| Item | Result | Design consequence |
|---|---|---|
| 6 | Grouping is not a cause (174 → 170) | The canonical grouper is used everywhere |
| 7 | Anchor rate 2.0% overall; 3.6% for "before convergence ∩ top half of a 20-sample screen" | Fork selection: §2 |
| 1 | `tv_debiased` reliability 0.16; about 2,100 samples per fork for 0.80 | No brute-force continuous labels; anchors are rare tested events |
| 4 | Signed R²_max ≈ 0 by construction. Unsigned R²_max ≈ 0.5, but unreliable at real effect sizes. K = 8 (4) and M = 200 give an interval width of about 0.15 at 20 × 20 | Core arm A1: §3 |
| 8 | Difficulty, PCA and linear capacity ruled out for single-boundary-token probes | Forward arms: all-token, span-pooled, attention, logit lens (§4) |
| 10 | N ≈ 50 at about 20 forks for the level-readout and change contrasts; H1a and H4-level would need about 200 | N = 50 total; H1a and H4-level descriptive |
| 12 | Nothing depends on 1591 | No exclusion; generality via A6 |
| 5 | The released-vs-ours gap (−0.11 on the incorrect traces) is UNEXPLAINED; the newline-collapsed released prompt is the leading candidate | A5 included |

## 1. Shared base

- **Model and engine.** Unchanged from Phase 2 and the pilot:
  - DeepSeek-R1-Distill-Qwen-14B at revision `1df8507178afcc1bef68cd8c393f61a886323761`;
  - pinned vLLM image `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`;
  - bf16, `max_model_len` 24576, server seed 20260914;
  - T = 0.6 and top_p 0.95 unless an arm says otherwise.
- **Problems.** N = 50 new MATH problems, levels 4–5, drawn from a pool of 120 candidates;
  the 20 MVP problems are excluded.
  - **Screen** (fixed before screening; s08 procedure): no-reasoning accuracy ≤ 0.25,
    `next10_exact` false, continuation edit similarity < 0.5.
  - **Known-contaminated items** are excluded (**LIT**).
  - **Uncertainty filter.** Keep problems whose 8-sample base accuracy is in [0.15, 0.85].
    This is a recorded scope limit.
  - **Reserve.** Exclusions are recorded. If fewer than 50 problems pass, report the
    shortfall instead of loosening the screen.
- **Stage B0, base traces.** 8 samples per problem.
  - Keep one trajectory per problem: the first incorrect sample, or else the first correct
    one.
  - Store the text, the token ids and the sentence split.
  - Prefixes are built from token-faithful text, with newlines kept. Only A5 uses the
    released format.
- **Problem roles,** assigned at random with a fixed seed before any B1 data are seen:
  - **20 core problems** (nested arm A1);
  - **30 label problems** (arm A1b).

## 2. Fork selection: stage B1

- **Screen.** At every pre-convergence fork (about 110 per trace), draw 8 continuations.
  This gives entropy and accuracy per fork, and the descriptive H1a level.
- **Selection.** **10 sentence units** per trajectory, each unit needing forks i and i+1:
  - 5 by top screened entropy (T3's deployable criterion);
  - 5 uniform at random from the rest of the pre-convergence region.
  - That is 20 forks per trajectory.
- **Weights.** Inclusion probabilities are recorded. Population quantities, R²_max above
  all, use inverse-probability weights.

## 3. Generation arms, in priority order

### A1: nested sampling (item 4), the core

- **Scope.** 20 core problems × 20 positions.
- **Procedure at each position.**
  1. Sample **K** sentences with `n=K`, each stopping at the sentence boundary (up to 64
     tokens, cut with the study splitter).
  2. Add the original S_i as k = 0.
  3. For each of the K + 1 sentences, draw **M = 200** continuations (`n=200`).
- **K.** K = 8 is T6's choice. K = 4 already gives a width of 0.14 on the released variance
  components, but needs K = 8 on the Phase 2 components. **Owner choice.**
- **Primary estimand: R²_max for unsigned importance.**
  - Use an **unbiased** score: squared-L2 distance between P(A|h,S) and P(A|h), and
    (Δacc)². Both are computed as cross-half products over the M continuations, so the
    finite-sample bias cancels.
  - This replaces TV minus a null, which T6 found breaks down at the real effect size of
    about 0.002.
- **Secondary.**
  - Signed R²_max, a check against the theoretical 0.
  - m(h).
  - The fork-level labels.
  - Anchors: T3's permutation test with BH, using the k = 0 vs pooled samples.
  - Every rule is fixed in code before A1 is scored.
- **Validation before scoring.** Rerun T6's synthetic checks, including the weak-effect
  case, on the new estimator. Report recovery bias against the known truth before reading
  real data.
- **Cost.** 20 × 20 × (K + 1) × 200 continuations:
  - K = 8: 720,000 continuations, **3.05B tokens**;
  - K = 4: 400,000 continuations, **1.70B tokens**.

### A1b: label arm, the 30 label problems

- **Design.** At each of the 20 selected forks, draw **40 continuations**. At n = 40,
  accuracy reliability is high; Phase 2 measured ICC 0.91 already at n = 20.
- **What it serves.** The level-readout (S2-type) and change (S4–S7-type) contrasts at N = 50
  overall, together with A1's forks.
- **Cost.** 30 × 20 × 40 continuations: **0.10B tokens**.

### A2: token-level resampling around the key phrase (item 2)

- **Where.** In each detected anchor, about 36 expected.
- **What.** 100 continuations from just before and 100 from just after the key phrase.
- **LIT, the key-phrase definition.** A semantic unit, or a model-internal forking token
  (max-|Δlogit|)? The definition and the number of positions must be fixed before A1 is
  scored.
- **Cost:** 31M tokens.

### A3: counterfactual sensitivity (item 3)

- **Design.** Resample, delete or paraphrase, each at T 0.6 and 1.0. Resample at 0.6 comes
  from A1 and A1b, so there are 5 new conditions.
- **Sentences.** The anchors plus 2 matched non-anchors each, about 108 sentences, with 100
  continuations per condition.
- **LIT, the paraphrase source.** Same model with a fixed prompt, or an external
  paraphraser? Also the meaning-preservation check and the deletion whitespace.
- **LIT, the second temperature.** 1.0 or 0.8?
- **Cost:** 229M tokens.

### A4: short-budget continuation readout (item 9)

- **Design.** At all 1,000 selected forks, 32 continuations capped at **256** tokens, then
  forced closure.
- **Validation.** Against the A1/A1b labels and the anchors. The Q2-type coverage gate and
  the candidate-free control are mandatory, because the pilot's forced answers were 78%
  `<OTHER>`.
- **LIT:** the budget grid and the closure suffix.
- **Cost:** 9M tokens.

### A5: rerun under the released settings (item 5)

- **Forks.** The 40 Phase 2 window forks of 2238:incorrect and 4019:incorrect, plus 40
  correct-trace control forks.
- **Prompts.** The exact cached released prompts, newline-collapsed, with 100 continuations
  per fork.
- **Comparison.** Against r0 and r1/r2.
- **Cost:** 34M tokens.

### A6: second problem source (item 12)

- **Problems.** 10 problems released after the model's training cutoff, for example AIME
  2025.
- **Design.** A B1 screen, then nested sampling at 10 positions, K = 5, M = 20.
- **A second model** instead is a frozen-item change and needs separate approval.
- **Cost:** 88M tokens.

### C: hardware calibration (only if the hardware is not an H100)

- **Design.** Rerun 20 Phase 2 window forks × 100 continuations with the identical Phase 2
  prompts and seeds derivation.
- **Test.** TV against r1/r2, compared with the r1-vs-r2 TV (equivalence check).
- **Why.** A100 and H100 bf16 kernels are not bit-identical.
- **Cost:** 8.5M tokens.

## 4. Forward-pass arms: teacher forcing, a separate process after generation

- **Setup.** HF `transformers` in the same pinned image, run on the freed GPUs during the
  generation tail (§5). Inputs: the 50 base trajectories, plus the 40 MVP traces for
  continuity.
- **F1: all-token and span states (item 8).**
  - Store the residual stream at every token for layers {8, 16, 24, 32, 40, 48}.
  - Store sentence-span mean-pooled states at all 49 layer outputs.
  - Size is about 13 GB per 50 traces.
  - T4 found full width adds nothing over PCA for the single boundary token. So the open
    question is the **position and pooling**, not capacity.
  - **LIT:** pooling choice, and whether a public SAE or tuned lens exists.
- **F2: attention (item 1).**
  - Sentence-to-sentence attention for every head, used to select receiver heads by
    kurtosis, as in Thought Anchors.
  - Attention suppression: mask each sentence's keys and record the KL on later tokens,
    about 7,500 passes.
- **F3: logit lens (item 9).** Per-layer answer-span log-probability at each sentence
  boundary.
  - **LIT:** logit lens vs tuned lens. A tuned lens costs about 2 GPU-h more.
- **Cost:** about 6 GPU-h in total.

## 5. Hardware and efficiency

### Recommended: 8 × A100 80 GB SXM4, an owner decision

- **Why.** A single H100 PCIe would take 271–973 h (11–40 days) for this plan.
- **Topology.** Eight TP = 1 vLLM replicas, one per GPU. The workload is
  memory-bandwidth-bound, and A100 SXM4 bandwidth (2.0 TB/s) is about that of the H100 PCIe
  used before.
- **It is a GPU-class change**, so it requires owner approval. Arm C covers the numerics.
- **The 40 GB A100 is not viable.** With KV cache at about 0.19 MB per token, it fits
  roughly one long sequence.
- **Lease.** The lease name stays `gpu:h100`: `control/` is frozen and not modified, and
  the lease acts as the shared-GPU lock. This naming mismatch is recorded.

### Efficiency measures

- **Prefix-affinity routing.** All requests for a fork go to the replica chosen by
  hash(trajectory). Random routing across replicas would destroy prefix-cache hits.
- **Prefix caching and n.** `--enable-prefix-caching`, with `n=K` and `n=M` in
  `SamplingParams`. With M = 200 on one prefix, prefill is paid once per sentence.
- **Concurrency.** `--max-num-seqs` comes from the dry run ({32, 64, 128}).
- **Per-problem pipelining.**
  - A problem moves B0 → B1 → A1/A1b as soon as its own previous stage finishes; there are
    no global barriers.
  - A2 and A3 start per problem once that problem's anchors are scored.
  - A5, A6 and C have no dependencies, so they run as low-priority backfill on idle
    replicas.
- **Tail drain.** When fewer than 8 replicas have work, stop the idle ones and start the
  forward-pass arms on the freed GPUs.
- **max_tokens.** Per problem, min(16,384, 1.25 × p99 of that problem's B0 lengths).
  Truncation stays an outcome category.
- **Resumable and idempotent.**
  - request_id = sha256(arm | problem | trajectory | fork | condition | k | seed).
  - One JSON file per request, never regenerated; checkpointed; resume skips completed
    IDs.
- **Per-request log.**
  - vLLM version (`/version`) and image digest;
  - model revision;
  - replica id and GPU name;
  - prompt sha256 and the prefix token count;
  - full sampling parameters, including n, the seed, max_tokens and stop;
  - finish_reason per choice;
  - token counts and wall time.
- **Guards.** A cost and runtime guard; leases; a 10-minute mirror of the store and logs to
  the Mac (Phase 2 sync script with ServerAliveInterval); secrets are never logged.
- **Model weights.** Loaded once from the persistent NFS cache (28 GB already there), with
  all replicas loading in parallel.

## 6. Cost

**Assumptions:**
- Mean full continuation: 4,242 tokens (Phase 2).
- **8 × A100 node:**
  - per-GPU throughput 1,000 / 1,600 / 2,200 tok/s (low / mid / high);
  - 85% efficiency;
  - plus 2.5 h for setup and dry run.
  - The rate is **unverified**: shown at $14.32/h ($1.79 per GPU-hour) and $10.32/h
    ($1.29). The owner confirms it.
- **1 × H100 PCIe:**
  - 1,075 tok/s (measured in Phase 2) or 2,500 tok/s (target);
  - $2.49/h.

### Tokens per arm (millions)

| Arm | K = 8 | K = 4 |
|---|---|---|
| B0 problem screen | 4 | 4 |
| B1 fork screen | 187 | 187 |
| **A1 nested** | **3,054** | **1,697** |
| A1b label arm | 102 | 102 |
| A2 token-level | 31 | 31 |
| A3 sensitivity | 229 | 229 |
| A4 short budget | 9 | 9 |
| A5 released settings | 34 | 34 |
| A6 second source | 88 | 88 |
| C calibration (A100 only) | 8.5 | 8.5 |
| F forward passes | 6 GPU-h | 6 GPU-h |
| **Total** | **3.75B** | **2.39B** |

### Wall time and cost

| Hardware | K | Low | Mid | High |
|---|---|---|---|---|
| 8 × A100 @ $14.32/h | 8 | 156 h, $2,238 | 99 h, $1,416 | 73 h, $1,043 |
| 8 × A100 @ $14.32/h | 4 | 101 h, $1,444 | 64 h, $920 | 48 h, $682 |
| 8 × A100 @ $10.32/h | 8 | $1,613 | $1,021 | $751 |
| 8 × A100 @ $10.32/h | 4 | $1,041 | $663 | $491 |
| 1 × H100 @ $2.49/h | 8 | 973 h, $2,422 (measured rate) | — | 422 h, $1,052 (target rate) |
| 1 × H100 @ $2.49/h | 4 | 622 h, $1,549 (measured rate) | — | 271 h, $676 (target rate) |

**Proposed guard:** 1.3 × the mid estimate for the chosen K at the confirmed rate. For
example, K = 4 at $14.32/h gives about $1,200. If the dry-run throughput is below 1,400
tok/s per GPU, re-cost and bring it back to the owner before launch.

### Cut list (apply from the top until the plan fits the approved budget)

1. **A1, K = 8 → 4.** Saves 1.36B tokens: about $500 at mid and $14.32/h. The width on the
   Phase 2 variance components grows above about 0.15.
2. **A6 second source.** Drop it: about $30.
3. **A3.** Anchors only, T 0.6 only: about $60.
4. **B1.** Screen every second fork at n = 6: about $40.
5. **A1, M 200 → 150.** Only if the re-run T6 design simulation keeps the width ≤ 0.2.
   Saves about 25% of A1.
6. **A5.** Drop it only if the owner accepts the prompt-format explanation without a test:
   about $12.

A2, A4, C and F are cheap and are never cut.

## 7. Dry run (before the full launch)

- **Scope.** 2 problems, end to end, on the full node:
  - B0;
  - B1 on all forks;
  - A1 on 2 positions at K = 4, M = 200;
  - A1b on 4 forks;
  - A2 and A3 on 1 sentence each;
  - A4, A5 (4 forks), C (4 forks) and F on both traces.
- **What it measures:**
  - per-GPU tokens per second at `--max-num-seqs` 32, 64 and 128;
  - the prefix-cache hit rate with affinity routing;
  - per-arm mean tokens against this plan;
  - forward-pass throughput;
  - store size;
  - a kill-and-resume test (0 duplicate request IDs);
  - all 8 replicas serving.
- **Go / no-go.**
  - Throughput ≥ 1,400 tok/s per GPU.
  - Per-arm mean tokens within ±30% of the plan.
  - No integrity failure.
  - Otherwise, re-cost and return to the owner.
- **Cost:** about 1 h of node time, about $15.

## 8. Literature dependence (decisions needed before launch)

| Arm or parameter | Item | Decision needed |
|---|---|---|
| A2 key-phrase definition and positions | 2 | Semantic unit vs model-internal forking token; number of positions; n per position |
| A3 paraphrase generator, deletion form, temperatures | 3 | Paraphrase source and prompt; meaning-preservation threshold; 1.0 vs 0.8 |
| F1 pooling, SAE or tuned lens | 8 | Span pooling (mean vs last token vs attention-weighted); availability of an SAE or tuned lens for this model |
| A4 budget and suffix; F3 lens | 9 | Budget grid; closure suffix; logit vs tuned lens |
| Contamination list | base | Which public contamination reports apply to the candidate pool |
| A1 unbiased importance estimator | 4 | The squared-L2 / (Δacc)² cross-half estimator, checked against any established unbiased TV or divergence estimators |

## 9. Owner decisions requested

1. **Hardware:** 8 × A100 80 GB SXM4 (recommended; a GPU-class change), or 1 × H100
   (11–40 days).
2. **The confirmed hourly rate**, and the budget cap.
3. **A1:** K = 8 or K = 4.
4. **A6:** second problem source (default) or a second model (a frozen-item change).
5. **Problem source:** MATH levels 4–5 for the 120 candidates.

## 10. Stop

This document is a plan only. **No instance, GPU or lease is touched until the owner
approves it** and the LIT decisions in §8 are made.
