# v3 follow-up GPU plan: one bundled run on one instance

**Status: DRAFT, NOT APPROVED, NOT LAUNCHED.** EXPLORATORY, POST HOC. Nothing in this
document has been run. No instance is launched until the owner approves this plan. The
plan_v3 verdict `INCONCLUSIVE_QUALITY_GATE` stands.

**Pending markers:**
- **PENDING T6** marks parameters that wait on T6 (nested R²_max ceiling).
- **PENDING T4** marks parameters that wait on T4 (difficulty-stratified probes).
- **LIT** marks parameters that depend on the pending literature review.

The CPU results that feed this plan are in `docs/v3_followup_cpu_results.md`, written when
T4 and T6 return. The cost model is `analysis/v3_followup/gpu_plan_cost/cost.py`. Every
number below comes from `cost_default.json`, and rerunning the script with new parameters
updates them.

## 0. What the CPU follow-up already settled (inputs to this plan)

| Item | CPU result | Consequence here |
|---|---|---|
| 6, answer grouping (Step 1) | 174 → 170 categories; no number or decision moves | Use the canonical grouper (`src/arv2/canon.py`) in every arm; no grouping arm needed |
| 7, window selection (T3) | Anchor rate 2.0% overall, 2.8% before convergence, 3.6% for "before convergence ∩ top half by a 20-sample entropy screen". The Phase 2 window was typical (67th percentile) | Fork selection: a cheap screen, then stratified forks (§2) |
| 1, label reliability (T2) | `tv_debiased` split-half SB Spearman 0.16 [0.13, 0.19] at n = 100; about 2,100 samples per fork for 0.80. `fork_tv`'s 0.81 is the noise floor | Do not try to make the continuous score reliable by brute force. Use nested sampling (arm 1) to measure the ceiling, and treat anchors as rare detected events |
| 10, sample size (T7) | N ≈ 50 problems at about 20 forks for the level-readout and change contrasts. H1a or H4-level as primary would need N ≈ 200 | **N = 50, F = 20.** H1a and H4-level are demoted to descriptive |
| 12, problem 1591 (T5) | No conclusion depends on it | No special exclusion. Arm 6 tests generality instead |
| 5, released vs ours (T1) | Gap −0.109 on the incorrect traces, about 0 on the correct ones. Sampling settings and the 16,384-token cap match. The released prompt collapses newlines to spaces (0 of 200 byte-identical, 197 of 200 identical after normalising whitespace); the released engine is an unknown hosted API. **UNEXPLAINED** quantitatively | Arm 5 is **included**, using the cached exact released prompts |
| 4, R²_max ceiling (T6) | PENDING T6 | Sets K and M for arm 1 |
| 8, probe capacity and difficulty (s16, T4) | s16: null across 108 probe settings at one boundary token per sentence. T4: PENDING | Sets the layers and positions stored in forward-pass arm F1 |

## 1. Shared base

- **Model and engine:**
  - model `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` at revision `1df8507178afcc1bef68cd8c393f61a886323761`;
  - pinned vLLM image `vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`;
  - bf16, `max_model_len` 24576, server seed 20260914;
  - temperature 0.6 and top_p 0.95 unless an arm says otherwise.

  These are unchanged from Phase 2 and the pilot.
- **Problems: N = 50 new problems.**
  - Candidates are 120 MATH problems (levels 4–5) that are not among the 20 MVP problems.
    Problem-level splitting stays as it is.
  - **Contamination and memorisation screen.** This uses the s08 procedure, with
    thresholds fixed before screening: no-reasoning accuracy ≤ 0.25, `next10_exact` false,
    and continuation edit similarity < 0.5. A problem that fails the screen is excluded
    and recorded.
  - **Known-contaminated items** (anything listed in public benchmark-contamination
    reports for Qwen2.5 or R1-distill) are excluded. **LIT**: the review supplies that
    list.
  - **Uncertainty filter.** Keep problems whose 8-sample base accuracy (stage B0) lies in
    [0.15, 0.85], because anchors live where the outcome is open (T3: the top entropy
    tercile is 1.9× enriched). This is a selection, so it is recorded and stated as a
    scope limit.
- **Base traces (stage B0), generated once and reused by every arm.**
  - Per problem: 8 samples. Keep one trajectory per problem: the first incorrect sample if
    any exists, otherwise the first correct one. Incorrect traces carried more of the real
    effects in Phase 2.
  - Store the full text, the token ids, and the sentence split with the study's splitter.
    The split is applied identically everywhere.
- **Released-format control.** T1 found that whitespace collapses in the released prefixes,
  so every arm builds prefixes from token-faithful text (newlines kept). Arm 5 alone uses the
  released format.

## 2. Fork selection (stage B1)

- **Screen.** For every pre-convergence fork of each base trajectory (about 110 per trace),
  draw 8 continuations at T = 0.6. That gives an entropy and accuracy estimate per fork,
  which also serves as the level label for descriptive H1a.
- **Selection.** Choose F = 20 forks per trajectory:
  - **10 by entropy**: the top 10 by screened entropy, which is T3's deployable enriching
    criterion;
  - **10 at random**: drawn uniformly from the rest of the pre-convergence region.
  - The inclusion probabilities are recorded. Population quantities, R²_max above all, are
    estimated with inverse-probability weights, so entropy-based selection cannot inflate
    the ceiling.

## 3. Generation arms, in priority order

Every arm reuses the base traces and the selected forks.

### A1: nested sampling (item 4), the core arm

- **What it samples.** At each selected fork i:
  1. Sample **K** candidate sentences S with `n=K` in a single request. Each stops at the
     sentence boundary: generate up to 64 tokens, then cut at the first boundary of the
     study splitter.
  2. Add the original S_i as k = 0.
  3. For each of the K + 1 sentences, sample **M** continuations with `n=M`, from prompt =
     prefix + S.
- **Estimands** are T6's, now with real per-sentence replication instead of embedding
  clusters:
  - m(h) = E_S[TV(P(A|h,S), P(A|h))];
  - E[Var(I|H)];
  - R²_max, the ceiling on any pre-emission predictor;
  - P(A|h) as the mixture over the sampled S.
- **Bias correction.** A split-half null over M; T6's validated estimator carries over.
- **What it also yields:**
  - fork-level accuracy and entropy (level labels) from (K+1)·M samples;
  - the original-sentence effect, giving `fork_tv` and `tv_debiased` with a split-half
    reliability from M;
  - the anchors, using T3's permutation and BH rule, fixed in code before any A1 data
    are scored.
- **Parameters:** **K = 10, M = 20 (PENDING T6)**. T6 returns the variance components and
  the K and M needed for an R²_max interval width of about 0.2.
- **Cost:** 1,000 forks × 11 × 20 continuations = 220,000 continuations, about **0.93B
  tokens: 104 GPU-h ($258) at the target rate, 241 h ($600) at the measured rate.**

### A2: token-level resampling around the key phrase (item 2)

- **Where.** In each anchor sentence detected in A1 (expected about 36: 1,000 forks ×
  3.6%), locate the key phrase.
- **What.** Resample the continuation from just **before** and just **after** it, 100
  continuations at each of the 2 positions.
- **Question.** Does the sentence's effect concentrate in the phrase (a forking-token
  account), or is it spread across the sentence?
- **LIT, the key-phrase definition.** It could be a semantic unit (the first claim, number
  or operation) or a model-internal one (the max-|Δlogit| token, as in forking-token
  work). The decision needed: which definition, fixed before A1 is scored, and whether
  more than 2 positions are needed.
- **Cost:** about 30M tokens, **3.4 h ($8)** at target and 7.9 h ($20) measured.

### A3: counterfactual sensitivity (item 3)

- **Design.** 3 interventions × 2 temperatures:
  - interventions: resample S_i, delete S_i, or paraphrase S_i;
  - temperatures: 0.6 and 1.0.
  - Resample at 0.6 is already in A1, so 5 new conditions.
- **Which sentences.** The A1 anchors plus 2 matched non-anchors each (matched on
  position tercile and screened entropy), about 108 sentences, with 100 continuations per
  condition.
- **Question.** Does an anchor's measured importance depend on how the counterfactual is
  made?
- **LIT, the paraphrase source.** Either the same model under a fixed paraphrase prompt, or
  an external paraphraser. Also, whether the deletion keeps sentence-boundary whitespace.
  The decisions needed: the paraphrase generator and prompt, and the meaning-preservation
  check (an embedding similarity of at least a fixed threshold).
- **LIT, the second temperature.** Is 1.0 standard in the literature, or is 0.8?
- **Cost:** about 229M tokens, **25.5 h ($63)** at target and 59 h ($147) measured.

### A4: short-budget continuation readout (item 9)

- **Design.** At each of the 1,000 A1 forks: 32 continuations capped at **256** tokens,
  then forced closure (`</think>` plus answer suffix).
- **Validation.**
  - Against A1's (K+1)·M accuracy labels: the level-readout contrast, which T7 found binds
    at N ≈ 44.
  - Against the anchor flags.
  - It must pass a coverage gate like the pilot's Q2, and include a candidate-free
    control, because the pilot's forced answers were 78% `<OTHER>`.
- **LIT, the budget grid and suffix.** The decision needed is the budget grid ({128, 256,
  512}?) and the closure suffix. These are fixed before the run.
- **Cost:** about 9M tokens, **1.0 h ($3)** at target and 2.4 h ($6) measured.

### A5: rerun under the released settings (item 5, included because T1 is unexplained)

- **Forks:** the 40 Phase 2 window forks of 2238:incorrect and 4019:incorrect, plus 40
  forks from 2 correct-trace controls.
- **Prompts:** the **exact released prompts**, newline-collapsed, from
  `derived/v3_followup/t1_released_vs_ours/released_window_prompts.parquet`, with 100
  continuations per fork.
- **Comparison:** against r0 (released) and r1/r2 (ours).
- **What it tests.** The question is whether prompt format reproduces the −0.11 gap on
  our pinned engine. The hosted provider cannot be reproduced, and the plan says so.
- **Cost:** about 34M tokens, **3.8 h ($9)** at target and 8.8 h ($22) measured.

### A6: second problem source (item 12), on a subset

- **Problems:** 10 problems from a source released after the model's training cutoff, for
  example AIME 2025, so contamination is excluded by date.
- **Design:** the same B1 screen, then a reduced A1 with 10 forks per problem, K = 5,
  M = 20.
- **A second model instead** (for example a different R1-distill size) would need a new
  vLLM model load, and it changes a frozen item, so it needs separate owner approval. This
  plan defaults to a second problem source on the same model.
- **Cost:** about 88M tokens, **9.8 h ($24)** at target and 23 h ($57) measured.

## 4. Forward-pass arms: teacher forcing, a separate process after generation

- **Setup.** Run after all generation finishes, once vLLM is stopped. Use HF `transformers`
  inside the same pinned image, with weights loaded at the same revision. The inputs are
  the stored traces: the N = 50 base trajectories plus the 40 MVP traces for continuity.
- **F1: all-token activations (item 8).** Store the residual stream at **every token** for
  a layer subset (**PENDING T4**; default layers 8, 16, 24, 32, 40 and 48), and the
  sentence-mean-pooled state at **all 49** layer outputs. Full storage at 49 layers × 5,120
  × fp16 would be about 125 GB for 50 traces; the subset is about 13 GB.
  - **LIT:** pooling choice, and whether to add a public SAE or a tuned lens if one exists
    for this model.
- **F2: attention (item 1).**
  - Sentence-to-sentence attention aggregated per head, for every layer and head. Use it
    to pick receiver heads by kurtosis, as in the Thought Anchors paper.
  - **Attention suppression**: mask each sentence's keys, then record the KL on later
    tokens. That is one pass per sentence, about 7,500 passes in total.
- **F3: logit lens (item 9).** At each sentence boundary, read the probability of each
  candidate answer's first token, and the answer-span log-probability, through the
  final-norm-plus-unembedding at every layer.
  - **LIT:** logit lens vs tuned lens. Training a tuned lens needs extra GPU time, roughly
    2 h, if it is chosen.
- **Cost:** an estimated **6 GPU-h ($15)**, dominated by the 7,500 F2 suppression passes.
  The dry run measures it.

## 5. Efficiency and integrity

- **vLLM.**
  - `--enable-prefix-caching`, as in Phase 2.
  - `--max-num-seqs` raised from Phase 2's 16 per endpoint to the value the dry run finds
    best ({32, 64, 128}).
  - Requests are submitted **grouped by shared prefix**: for each fork, all K sentence
    requests, then all (K+1) continuation requests, back to back. Every arm then branches
    from a warm cache.
  - `n=K` and `n=M` go in `SamplingParams` instead of K or M separate requests.
- **max_tokens.** Capped per problem at min(16,384, 1.25 × p99 of that problem's B0
  base-trace continuation lengths). Truncation stays a recorded outcome category. In
  Phase 2 the p99 was 14,490 and the mean 4,242, so the saving is small but it removes
  pathological loops.
- **Resumable and idempotent.**
  - request_id = sha256(arm | problem | trajectory | fork | condition | k | seed).
  - One JSON file per request goes in a durable store. An existing file is never
    regenerated, which is the s18 store pattern.
  - The driver checkpoints, and resumes by skipping completed IDs.
- **Per-request log.** Each request records:
  - vLLM version, from `/version`, and the image digest;
  - model revision;
  - prompt sha256 and the prefix token count;
  - full sampling params, including n, the seed, max_tokens and stop;
  - finish_reason for each choice;
  - token counts;
  - wall time.
- **Guards.**
  - A cost and runtime guard, as in Phase 2: the driver stops at the owner-approved cap.
  - Leases `gpu:h100` and `service:vllm`.
  - Instance records and logs are mirrored to the Mac every 10 min with the Phase 2 sync
    script, which uses ServerAliveInterval.
  - Secrets are never logged.

## 6. Cost

**Throughput scenarios:**
- **Measured:** 1,075 tok/s, Phase 2 as realised: 168.4M tokens in 43.5 h, 16 concurrent
  requests, prefix caching on.
- **Target:** 2,500 tok/s, assumed for higher concurrency and must be confirmed by the
  dry run.

Rate: $2.49/h, 1 × H100 PCIe (list price; billed rate unverified). Mean full continuation:
4,242 tokens.

| Arm | Tokens | GPU-h (target / measured) | $ (target / measured) |
|---|---|---|---|
| B0 problem screen (120 × 8) | 4.1M | 0.5 / 1.1 | 1 / 3 |
| B1 fork screen (50 × 110 × 8) | 187M | 20.7 / 48.2 | 52 / 120 |
| A1 nested (1,000 forks × 11 × 20) | 934M | 103.7 / 241.2 | 258 / 600 |
| A2 token-level | 30M | 3.4 / 7.9 | 8 / 20 |
| A3 sensitivity | 229M | 25.5 / 59.2 | 63 / 147 |
| A4 short budget | 9M | 1.0 / 2.4 | 3 / 6 |
| A5 released settings | 34M | 3.8 / 8.8 | 9 / 22 |
| A6 second source | 88M | 9.8 / 22.8 | 24 / 57 |
| F forward passes | — | 6 / 6 | 15 / 15 |
| **Total** | **1.52B** | **174 / 398** | **$434 / $990** |

**Wall time on one GPU** is 7.3 days at target and 16.6 days measured.

- **Option (owner decision).** An 8 × H100 instance runs 8 vLLM replicas, as
  `phase2_vllm.sh` already supports. The GPU-hours stay the same and the wall time is
  divided by about 8. It changes the GPU class and the rate, so it needs approval.
- **Proposed guard:** a cost cap at 1.3 × the target-scenario total, about $565. If the dry
  run's throughput is below 1,800 tok/s, the plan is re-costed and brought back before the
  launch.

### Cut list (apply from the top until the plan fits the approved budget)

1. **A6 second source.** Drop it entirely: −$24 / −$57.
2. **A3 sensitivity.** Keep the anchors, drop the matched controls, and keep only T = 0.6:
   −$51 / −$118.
3. **B1 fork screen.** Screen every second pre-convergence fork, n = 6: −$32 / −$75.
4. **A1.** Set M 20 → 15 (only if T6 shows the R²_max interval width stays ≤ 0.25):
   −$64 / −$150.
5. **A1.** Set F 20 → 15 forks per trajectory. The change contrasts trade forks against
   problems (T7), but the level readout does not, so N stays 50: −$64 / −$150.
6. **A5 released settings.** Drop it if the owner accepts T1's prompt-format explanation as
   enough: −$9 / −$22.

A2, A4 and F are cheap and are never cut.

## 7. Dry run (before the full launch)

- **Scope.** 2 problems, end to end, through every stage and arm at reduced size:
  - B0 with 8 samples;
  - B1 on all forks;
  - A1 on 4 forks;
  - A2 and A3 on 1 sentence each;
  - A4;
  - A5 on 4 forks;
  - F on both traces.
- **What it measures:**
  - tokens per second at `--max-num-seqs` 32, 64 and 128;
  - the prefix-cache hit rate, from vLLM metrics;
  - the mean generated tokens per request per arm, against this plan's assumptions;
  - the forward-pass throughput;
  - the size of the store on disk;
  - that resume works: kill the driver mid-run, restart it, and check for 0 duplicate
    request IDs.
- **Go / no-go.**
  - Throughput ≥ 1,800 tok/s.
  - Each arm's mean tokens within ±30% of the plan.
  - No integrity failure.
  - Otherwise, re-cost and return to the owner.
- **Dry-run cost:** about 2 GPU-h plus setup, so about $10.

## 8. Literature dependence (decisions needed before launch)

| Arm or parameter | Item | Decision needed |
|---|---|---|
| A2 key-phrase definition and positions | 2 | Semantic unit vs model-internal (forking-token, max-Δlogit) definition; how many positions; continuations per position |
| A3 paraphrase generator, deletion form, temperatures | 3 | Same-model paraphrase prompt vs external paraphraser; meaning-preservation check; 1.0 vs 0.8 as the second temperature |
| F1 layers, pooling, SAE or tuned lens | 8 | The stored layer subset (also PENDING T4); sentence pooling; whether a public SAE or tuned lens exists for this model |
| A4 budget grid and suffix; F3 logit vs tuned lens | 9 | Budget values; closure suffix; whether to train a tuned lens (about +2 GPU-h) |
| Known-contamination list | base | Which public contamination reports apply to the candidate pool |

## 9. Owner decisions requested (beyond approving the plan)

1. **Budget cap.** About $565 is proposed at target throughput. It is about $990 if the
   throughput stays at the measured rate.
2. **Instance.** 1 × H100 PCIe (default; about 7–17 days) or 8 × H100 (a GPU-class change;
   about 1–2 days).
3. **A6.** Second problem source (default) or a second model (a frozen-item change).
4. **Problem source.** Whether to use MATH levels 4–5 as the main source.

## 10. Stop

This document is a plan only. **No instance, GPU or lease is touched until the owner
approves it**, and until the PENDING T4/T6 parameters and the LIT decisions in §8 are
filled in.
