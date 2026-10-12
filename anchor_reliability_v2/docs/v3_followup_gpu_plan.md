# v3 follow-up plan: what to do next

**Status: DRAFT v3, NOT APPROVED, NOT LAUNCHED.** EXPLORATORY, POST HOC.
- Nothing here has been run.
- No instance, lease or GPU is touched until the owner approves.
- The plan_v3 verdict `INCONCLUSIVE_QUALITY_GATE` stands.

**Inputs:**
- `docs/v3_followup_cpu_results.md`: all CPU results (Steps 0–2, T1–T7).
- `reports/Thought anchors novelty audit.md`: the literature and novelty review.
- `reports/Out of distribution math problems.md`: problem sourcing.

**Costs** come from `analysis/v3_followup/gpu_plan_cost/cost_v3.py`; outputs are `cost_v3_*.json`.

**What changed from v2:**
1. New problems now come from post-release competitions, with screening, de-duplication and a dataset amendment.
2. The work is split into **Run 1** (screening, dry run, pilots) and **Run 2** (main), with an owner decision in between.
3. Arms are reprioritised by the novelty audit.
4. The cap becomes 20k tokens on the whole trace.
5. Costs are recomputed for traces 2–3× longer.
6. A CPU stage, Stage 0, comes first.

## 1. The research question and what is already settled

**The question.** For a reasoning model, can the size of a sentence's effect on the final
answer (its unsigned importance) be predicted before the sentence is written? How large is
that ceiling (R²_max), on problems the model has surely never seen? And is "anchor" a
stable property, or an artefact of how importance is measured?

**Settled, and not re-tested:**
- **Phase 2 reliability failure.** Resampling importance on the released data is mostly
  entropy-driven noise:
  - Phase 2 fork_tv Spearman 0.44, ICC 0.56;
  - debiased split-half reliability 0.16;
  - about 2% real effects.
- **Signed importance** cannot be predicted before emission. This is the zero-mean-advantage
  identity, which is known; its consequence for interpretability probes is the overlooked
  part.
- **Explanations ruled out** for the probe null: difficulty, grouping, problem 1591, PCA
  and linear capacity.
- **The forced-answer readout fails:** 78% of its answers are outside the set that
  continued reasoning reaches.

**Novelty position**, from the literature review:
- **Novel:**
  - the reliability audit, since no published critique of Thought Anchors importance
    exists;
  - the unsigned R²_max from nested sampling;
  - the forced-answer outside-set measurement;
  - the released-prompt / provider check;
  - the short-budget readout.
- **Partially anticipated:**
  - activation probes and span pooling (Nikankin et al. 2026, arXiv 2604.18307, same model,
    post-emission, a different target);
  - logit lens (Li et al. 2026, arXiv 2605.11746, same model);
  - resample vs delete vs paraphrase (Thought Branches, arXiv 2510.27484: no delete or
    paraphrase arm, one temperature);
  - token-level forking (Bigelow et al., arXiv 2412.07961; Forking Fast, arXiv 2608.19611).
- **Already done:** receiver heads and suppression (replication, weak baselines, open code
  issue #7); AIME as a source (standard).
- **Closest methodological neighbour:** Du et al. 2026 (arXiv 2609.04194, COLM 2026),
  resampled step importance with split-half noise ceilings on Qwen3.

**Main internal limit.** The current 20 MATH problems are partly too easy and
contaminated:
- MATH-500 is 93.9% for this model;
- the Qwen2.5 family is documented to memorise MATH;
- problem 1591 is memorised (0.80 accuracy with no reasoning).

So the new runs use post-release problems, and the MATH set is kept only as "the audited
released dataset".

## 2. Stage 0: CPU and owner work, before any GPU

| # | Item | Output |
|---|---|---|
| 0.1 | **Dataset amendment record.** `AGENTS.md` freezes the dataset and the 20 problems in `configs/mvp.yaml`. Adding post-release problems needs an amendment and owner approval | `docs/amendments/v3_problem_pool.md`, owner sign-off |
| 0.2 | **Preregistration of Run 2.** Primary estimands, thresholds, estimators (incl. the three-level variance model with binomial correction), selection rules, exclusion rules, decision table. Committed and pushed before any Run 2 data | `plan_v4_prereg.md` |
| 0.3 | **Candidate pool.** All 8 MathArena answer-based 2025–26 sets: AIME 2025/2026, BRUMO 2025, HMMT Feb 2025/2026, HMMT Nov 2025, SMT 2025, CMIMC 2025 (276 problems). Store dataset ID, revision, index and SHA-256 only, never problem text (CC BY-NC-SA 4.0; MAA copyright). OlymMATH-EN-EASY becomes a separately tagged reserve stratum | `configs/v3_pool.json` |
| 0.4 | **Pre-screen from public traces.** Use MathArena's per-sample 14B/32B traces to drop near-certain out-of-band items (all 8 correct, or all 12 wrong across 14B/32B/70B). This leaves about 240–260 | pool with prior estimates |
| 0.5 | **Near-duplicate search.** Lexical pass (10-gram and 13-gram/LCS ≥ 0.6), embedding top-10 with manual review, against AoPS before 2025, MSE, NuminaMath-1.5, Omni-MATH, OlympiadBench, MATH, GSM8K, historical contests and OlymMATH. Classify each match as verbatim / renumbered / same lemma / novel; the first two are excluded and kept as a labelled contamination stratum. Start with MathArena's 8 flagged AIME 2025 items. Core candidates also get a manual web search | `analysis/v3_followup/dedup/` |
| 0.6 | **Grader.** Extend `src/arv2/canon.py` for symbolic answers (radicals, π, factorials, sets). SMT-style transcendental answers are confined to the label set unless the grader is validated | grader plus tests |
| 0.7 | **Stratified re-report of the existing audit.** Recommended by the review: by accuracy-from-start and entropy strata, with and without memorised items | addendum to `docs/v3_followup_cpu_results.md` |
| 0.8 | **Literature-dependent settings fixed** (§7) | entries in 0.2 |
| 0.9 | **Run drivers.** Prefix-affinity router across 8 vLLM replicas; idempotent request store; per-problem pipelining; total-trace cap; logging of vLLM version, image digest, prompt hash, sampling parameters, seed, replica and GPU. Plus tests | `scripts/s19_*`, tests |
| 0.10 | **Own head-ablation hook.** Do not reuse Thought Anchors `ablation.py`; issue #7 says it does not ablate real heads | for arm F2 |

## 3. Run 1: screening, dry run and pilots (one short session on the node)

- **Target hardware:** 8 × H100 SXM 80 GB (recommended; see the §5 rate update). The
  fallback is 8 × A100 80 GB SXM4. Run 1 could also run on a single GPU, more slowly.
- **Shared settings:**
  - the same pinned model revision and vLLM image as before;
  - T 0.6, top_p 0.95;
  - **cap = 20,000 tokens on the total trace** (prefix plus continuation). It fits the
    24,576 context with about 4.5k headroom.

| Stage | What | Gate or output | Tokens |
|---|---|---|---|
| S1 | 16 samples per pool problem from the start | keep 3–13 / 16 correct | 44M |
| S2 | +32 samples on survivors (48 total) | keep estimate in [0.15, 0.85] | 35M |
| S3 | Puzzle screen: generated Knights-and-Knaves (4–7 people) and ZebraLogic, 3 difficulty settings, about 200 instances × 16 | in-band instances per family | 10M |
| S4 | Contamination red-flag probes | no-reasoning accuracy (closed `<think></think>`); base Qwen2.5-14B prefix completion at 40/60/80%; Min-K%++; token-probability deviation; constant-changed twins for core candidates. Each is calibrated against a positive control (the 20 MATH problems) and a negative control (AIME 2026 / puzzles). A hit drops the item; a miss proves nothing (detectors are near chance on R1 distills, arXiv 2510.02386). If the controls do not separate, report the probes as uninformative | about 1M + base-model load |
| P1 | Short-budget readout pilot: K ∈ {0, 64, 256, 1024} tokens, then forced closure, 5 problems × 20 forks × 32 | outside-set rate and accuracy tracking against K. **Drop arm A4** if the outside-set rate at 256 stays near the pilot's 78% | 4M |
| D | Dry run on 2 problems through every Run 2 arm | per-GPU tok/s at 32 / 64 / 128 sequences; prefix-cache hit rate with affinity routing; per-arm mean tokens; resume test (0 duplicate IDs) | 26M |
| C | A100 vs H100 calibration: 20 Phase 2 forks × 100 with identical prompts | equivalence of TV against r1/r2 | 9M |

**Run 1 total:** about 128M tokens, **4.5–7.3 node-hours, about $46–105**.

**Screen accuracies are never reused as accuracy-from-start estimates.** Selection on noisy
estimates causes regression to the mean, as the current set's median of 0.73 showed. Run 2
re-estimates accuracy from fresh samples.

### Owner decision point after Run 1

The owner receives, in one note:
- the in-band yield;
- the de-duplication and probe exclusions;
- measured throughput and trace lengths;
- the short-budget pilot result;
- the re-costed Run 2.

**Selection rules** (preregistered in 0.2):
- **Core 20:** 48-sample accuracy in 0.30–0.70; integer or simple rational answer, with
  AIME first; median trace ≤ 12k tokens; truncation ≤ 10%; passed manual near-duplicate
  review; at least 3 competitions represented.
- **Label problems:** the rest of the in-band set, 30–50 problems.
- **Thin yield.** The expected yield is about 60–70 after exclusions, a thin margin. If it
  is fewer than 50, fill from the OlymMATH reserve stratum, reported separately, or reduce
  the label set. Never loosen the screen.

## 4. Run 2: main experiment, in priority order (from the novelty audit)

Every arm uses the new problems unless stated otherwise. Rules and thresholds are fixed in
the preregistration (0.2).

### A1: nested sampling, the flagship (novel)

- **Design.** 20 core problems × 20 positions. At each position, K sampled sentences plus
  the original, each with **M = 200** continuations.
- **K = 4** is the default. It gives an R²_max width of about 0.14 on the released variance
  components. K = 8 is an owner option, safer on the Phase 2 components.
- **Positions:** stratified by screened entropy, half high and half random, with
  inverse-probability weights.
- **Estimator:**
  - a three-level variance model: between histories, between sentences within a history,
    and binomial noise within each history-sentence cell;
  - an **unbiased** importance score: squared-L2 or (Δacc)², computed as cross-half
    products;
  - validated on synthetic data, including the weak-effect regime where T6's TV-based
    estimator failed, before any real data are scored.
- **Primary result:** the unsigned R²_max with a problem-cluster interval. Secondary: a
  signed check against 0, m(h), anchors, and the reliability of the new labels.

### A1b: labels and replication of the reliability audit (novel extension)

- **Design.** 30 label problems × 20 forks × **100 continuations**, in two independent
  halves of 50.
- **What it gives.** Split-half reliability, real-effect rates with permutation tests and
  BH, and level labels on post-release problems.
- **The planned headline.** If real effects rise well above 2% on hard, uncontaminated
  problems, that is a result in itself (Du et al. 2026 found more answer-changing steps on
  AIME).

### A3: resample vs delete vs paraphrase × T ∈ {0.6, 1.0} (partially anticipated; the temperature factor is new)

- **Sentences.** The A1/A1b anchors plus 2 matched non-anchors each. The **same sentences**
  go through all arms, with 100 continuations per condition.
- **Noise floors.** Each arm gets its own noise floor.
- **Preregistered prediction** (from Thought Branches): on-policy resampling gives larger,
  more stable effects than off-policy edits.

### A5: released format × generation path (novel, cheap)

- **Forks.** The 80 Phase 2 forks: 2238 and 4019 incorrect, plus 2 correct controls.
- **Prompt format:** newline-collapsed, as released, vs faithful.
- **Generation path:** our pinned vLLM. A hosted-provider cell (the released data used
  Novita) runs **only if the owner approves an API account**, since it sends prompts
  outside; otherwise the arm is prompt format only.
- **Question.** Does format explain the −0.11 gap on the incorrect traces?

### A4: short-budget readout (novel; runs only if P1 passes)

- **Design.** At all 1,000 selected forks, 32 continuations at the budget chosen in P1, then
  forced closure.
- **Safeguards:** the coverage gate and the candidate-free control.
- **Validation:** against A1/A1b labels and anchors.

### F: forward passes, run on freed GPUs during the generation tail (partially anticipated; framed as comparisons with the ceiling)

- **F1, all-token states.** Store layers {8, 16, 24, 32, 40, 48}, plus sentence-span pooled
  states at all 49 layer outputs.
- **Probe analysis.** Pre-emission and post-emission positions are kept strictly separate.
  Every probe is benchmarked against the A1 R²_max and an entropy-only baseline, and
  includes a **Nikankin-style pooled-span baseline**. The write-up reconciles our
  pre-emission null with their post-emission positive result.
- **F3, logit lens** at sentence boundaries, scoring candidate sets against resampled
  accuracy (not ground-truth probability), framed as an extension of Li et al.
- **F2, receiver heads and suppression.** Run last, using our own ablation hook. The only
  new question is whether head scores track debiased importance or the entropy floor.

### A2: token-level resampling, small (partially anticipated)

- **Design.** In anchors, resample before and after the key phrase, 2 × 100 continuations.
- **Pooling.** Forking-Fast-style change-point pooling.
- **Question.** Do *reliable* sentence effects localise to tokens?

### A6: puzzle domain (novel as a resampling study; out of distribution by instance, not by task type)

- **Design.** 10 in-band puzzles from S3: nested sampling at 10 positions (K = 4, M = 100),
  plus label resampling.
- **What it tests.** Whether anchors and the ceiling change outside maths. A non-peer-reviewed
  replication found that anchors outside maths were fact retrieval rather than planning.

### A1m: MATH nested comparison (optional)

- **Design.** 5 MATH problems in the A1 design (K = 4), linking the new ceiling to the
  audited released dataset.

### Not included

- MATH-P-Hard: built on a contaminated original, so it is useful only as a paired
  contamination contrast.
- Putnam-AXIOM: only 37 variants change the answer.
- A second model: a frozen-item change. It can be proposed after Run 2.

## 5. Cost of Run 2

**Rate update, 2026-10-11, verified on lambda.ai/pricing and lambda.ai/instances.** The $14.32/h
figure in the tables below came from a stale third-party listing. Lambda's current on-demand
prices are:

| Instance | Per GPU-hour | Per hour |
|---|---|---|
| **8 × A100 SXM 80 GB** | **$2.79** | **$22.32** |
| 8 × H100 SXM 80 GB | $3.99 | $31.92 |
| 8 × A100 SXM 40 GB | $1.99 | $15.92 (not viable: KV-cache headroom) |
| 1 × H100 PCIe (the Phase 2 class) | $3.29 | $3.29 (was $2.49) |
| Persistent filesystem | $0.20/GB-month, no ingress or egress charges | |

To convert the $14.32 column below to the current A100 rate, multiply by 1.559.

| Option | 8 × A100 @ $22.32/h, low / mid / high |
|---|---|
| K = 4, M = 200 | $3,290 / $2,081 / $1,532 |
| K = 4, M = 150 | $2,743 / $1,739 / $1,283 |
| K = 8, M = 200 | $5,040 / $3,175 / $2,328 |

Run 1 on 8 × A100: $100–164.

**8 × H100 SXM is now likely as cheap or cheaper, and faster.**
- Decoding is bandwidth-bound, and H100 SXM has about 1.7× A100 bandwidth (3.35 vs
  2.0 TB/s). At that assumed 1.68× throughput, K = 4, M = 200 costs **$2,847 / $1,816 /
  $1,346** (low / mid / high) over **89 / 57 / 42 h**.
- It also keeps the H100 family that Phase 2 used. Arm C still runs, because SXM is not PCIe.
- The 1.68× is an assumption; the Run 1 dry run measures it.
- **Recommended hardware: 8 × H100 SXM**, subject to the owner's approval of the GPU-class
  change.

*(The tables below keep the original $14.32 / $10.32 columns for traceability.)*

**Assumptions:**
- Tokens per continuation: 6,000 for new problems (about half of an ≤12k trace), 4,242 for
  MATH, 3,000 for puzzles.
- Per-GPU throughput: 1,000 / 1,600 / 2,200 tok/s (low / mid / high), at 85% efficiency.
- Forward passes: 12 GPU-h.
- **All of these are replaced by Run 1 measurements.**

| Arm | Tokens (K = 4, M = 200) |
|---|---|
| B1 fork screen (every ~4th pre-convergence boundary, n = 8) | 120M |
| **A1 nested core** | **2,400M** (K = 8: 4,320M) |
| A1b labels and reliability | 360M |
| A3 counterfactual | 324M |
| A6 puzzles | 210M |
| A5 format | 68M |
| A2 token-level | 43M |
| A4 short budget | 10M |
| A1m MATH nested (optional) | 424M |

| Option | Tokens | Node-hours (low / mid / high) | $ at $14.32/h | $ at $10.32/h |
|---|---|---|---|---|
| **K = 4, M = 200 (recommended)** | 3.53B | 147 / 93 / 69 | 2,111 / 1,335 / 983 | 1,521 / 962 / 708 |
| K = 4, M = 150 | 2.93B | 123 / 78 / 57 | 1,760 / 1,116 / 823 | 1,268 / 804 / 593 |
| K = 8, M = 200 | 5.45B | 226 / 142 / 104 | 3,234 / 2,037 / 1,493 | 2,330 / 1,468 / 1,076 |
| K = 4 + A1m | 3.96B | 165 / 104 / 77 | 2,359 / 1,490 / 1,096 | 1,700 / 1,074 / 790 |

**Guard.** 1.3 × the re-costed mid estimate after Run 1, at the confirmed rate. The driver
stops at the guard.

### Cut list (apply from the top)

1. **A6 puzzles.** Drop them, or reduce to label resampling only. Saves about $75 at mid.
2. **A1 M 200 → 150.** Only if the design simulation keeps the R²_max width ≤ 0.2. Saves
   about $220.
3. **A3.** Anchors only, T 0.6 only. Saves about $80.
4. **B1.** Screen every 6th boundary. Saves about $15.
5. **A1 positions 20 → 15.** Saves about $230. Hurts the within-trajectory component.

A1 at K = 4 and M ≥ 150, A1b, A5, A4 (if P1 passes), C and F are never cut.

## 6. Hardware, efficiency, integrity

**Hardware:**
- 8 × H100 SXM 80 GB (recommended), with 8 × A100 80 GB SXM4 as the fallback, run as 8
  TP = 1 replicas. Either is a GPU-class change, so it needs owner approval. The code
  currently admits only H100 PCIe and A6000 hardware names (`src/preanchor/labels/a6000.py`),
  so that list must be extended before launch.
- A single H100 would take 11–40 days for Run 2.
- The lease name stays `gpu:h100`, because `control/` is frozen; the mismatch is recorded.

**Efficiency:**
- **Prefix-affinity routing.** All requests for a fork go to the replica chosen by
  hash(trajectory). Without it, the prefix cache is lost across replicas.
- **Prefix caching and n.** `n=K` / `n=M` in `SamplingParams` share each prefix.
- **Per-problem pipelining,** with no global barriers. A5, A6 and C are low-priority filler.
  Forward passes run on GPUs freed during the generation tail.
- **The total-trace cap is defined identically** in screening and in every arm, because the
  in-band definition depends on the cap.

**Integrity:**
- The idempotent request store is never regenerated, and a resumable checkpoint is kept.
- Per-request provenance is logged.
- The cost and runtime guard is enforced.
- Leases are held.
- The store and logs are mirrored to the Mac every 10 minutes.
- Secrets are never logged.

## 7. Literature-dependent settings (decide in Stage 0.8)

| Setting | Decision needed | Recommendation from the review |
|---|---|---|
| A2 key-phrase definition | Semantic unit vs model-internal forking token | Use Forking-Fast-style change-point detection on token-level resamples instead of choosing a phrase in advance |
| A3 paraphrase source | Same model with a fixed prompt vs an external paraphraser; meaning check | Same model, fixed prompt (keeps it on-policy-adjacent); MiniLM similarity ≥ 0.8 as the meaning check |
| A3 temperatures | 1.0 vs 0.8 | 0.6 and 1.0. Thought Branches used 0.7, so 0.6 vs 1.0 brackets it |
| A4 budget and suffix | K grid and closure text | Set by P1. The suffix follows DEER / s1-style closure (`</think>` + "Final Answer"), as in the pilot |
| F1 pooling, SAE or lens | Mean vs last-token vs attention-weighted; tuned lens? | Mean span pooling (matches Nikankin); logit lens first, tuned lens only if cheap |
| A1 estimator | Unbiased importance | Squared-L2 / (Δacc)² cross-half products, checked on synthetic data |

## 8. Owner decisions requested

1. **Approve the dataset amendment** (0.1): post-release competition problems, references
   only, under CC BY-NC-SA.
2. **Hardware:** 8 × H100 SXM 80 GB, recommended at the verified $31.92/h (8 × A100 80 GB
   at $22.32/h is the fallback). Either is a GPU-class change.
3. **Run 1 budget:** about 35–63 GPU-hours (4.4–7.9 node-hours), about $140–250 on
   8 × H100 SXM.
4. **For Run 2, after the Run 1 note:** K (4 or 8), M (200 or 150), whether to include A1m,
   and the budget cap.
5. **A5 hosted-provider cell:** approve an external API account, or run prompt format only.
6. **Second model:** defer until after Run 2 (default).

## 9. Stop

This is a plan only. **No instance, GPU, lease or external API is used until the owner
approves Stage 0 and Run 1, and separately approves Run 2 after the Run 1 note.**
