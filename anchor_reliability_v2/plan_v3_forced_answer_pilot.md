# anchor_reliability_v2: plan v3, forced-answer readout (PILOT preregistration)

**Version 1 — 2026-10-11.** Written before any forced-answer data exist. Labels as in
`phase2_plan.md`: **FIXED** (not changed after pilot data exist), **PROVISIONAL**
(revisable before the run, with a changelog entry, §13), **OPEN** (needs an owner
decision).

**Stage: exploratory pilot (`stage: exploratory_pilot`, `preregistered: true`).** All 10
problems were seen in the MVP and in Phase 2, and the validation labels come from the same
forks. Nothing here is confirmatory evidence about new problems. A PASS licenses a
confirmatory plan on new problems (Phase 3); it is not itself that evidence.

Code: `scripts/s18_forced_answer.py` (`build-requests`, `run`, `score`), tested in
`tests/test_forced_answer.py`. Output: `results/plan_v3_pilot.json`.

## 0. Motivation (facts carried in)

- Residual-stream probes read answer **uncertainty**, not anchoring
  (`docs/representation_rootcause.md`). The `fork_tv` probe correlates 0.52 with the
  noise floor, and its partial correlation with `fork_tv` given the floor is −0.01.
- The stored boundary state does not linearly encode the **outcome**. Accuracy level has
  a within-trajectory ceiling of 0.96 but decodes at 0.04. Δaccuracy is not decodable at
  any layer.
- The resampled accuracy curve is highly reliable: ICC 0.98 at n = 100
  (`docs/phase2_diagnostics.md`, Finding 4). At BH 10%, 7 of 190 Phase 2 sentences carry
  a real effect (`results/s13_sentence_effects.csv`).
- **The question for this pilot:** does a direct, one-pass readout of the model's own
  current answer distribution track the resampled outcome? If it does, the change between
  consecutive boundaries is a candidate cheap estimator of counterfactual effect.

## 1. Hypotheses (directional)

- **H1 (primary; level).** The forced-answer probability of the correct answer,
  p_forced(i), is **positively** monotonically related to the resampled accuracy
  acc(i) at the same fork. Two parts:
  - **H1a**, across all 200 forks: Spearman ≥ 0.60.
  - **H1b**, within trajectory: the median over the 10 trajectories of the per-trajectory
    Spearman (20 forks each) is ≥ 0.30.
  - H1 is the composite of the two (§7).
- **H2 (secondary; change).** The forced change between consecutive boundaries is
  **positively** related to the resampled change:
  - S4: Spearman(Δp_forced, signed Δacc) ≥ 0.30;
  - S7: forced TV ranks the 7 real-effect sentences, AP ≥ 0.25 (base rate 0.037).
- **H3 (secondary; not just uncertainty again).**
  - S3: forced-answer entropy tracks the resampled answer entropy, Spearman ≥ 0.50.
    This is expected, and on its own not interesting.
  - S5: forced TV carries effect beyond the floor. Spearman(forced TV, `tv_debiased`)
    ≥ 0.20.
  - S6: the partial Spearman of forced TV with `fork_tv`, controlling for the analytic
    floor, is ≥ 0.10. The residual probe scored −0.01 on the same quantity.
- **H4 (secondary; incremental).** The forced readout beats the baselines on the same
  rows:
  - position, for the level and for AP;
  - the s14 residual-probe out-of-fold predictions on the 190 sentences, for AP;
  - a per-problem constant, the s08 no-reasoning accuracy. This shows how much of H1a
    problem identity alone explains.

## 2. Units and validation targets (FIXED)

- **Forks and sentences.** The 200 Phase 2 forks (`configs/phase2_forks.json`; 10
  trajectories × 20 contiguous forks) and 190 sentence units (fork i and fork i+1).
- **PRIMARY labels: replicates 1 and 2 only** (label set `r12`, 200 resamples per fork).
  - These were generated with exactly the vLLM image, model revision and sampling
    settings that this pilot uses.
  - The released replicate 0 shows systematically lower accuracy on both incorrect-trace
    trajectories: 2238 is 0.50 against 0.62, and 4019 is 0.085 against 0.18. The other 8
    trajectories differ by ≤ 0.05.
  - That points to a generation difference between the release and our run, so r0 is
    not pooled into the primary labels.
- **Sensitivity labels (prespecified):** the 300-sample pool r0 + r1 + r2 (`r012`, as used
  in s13). The same metrics are reported. No decision is taken on them.
- **Per-fork targets:**
  - acc(i) (v2 parser, ground truth from `trajectories.parquet`);
  - answer entropy, in bits;
  - category counts.
- **Per-sentence targets:**
  - signed and absolute Δacc;
  - `fork_tv`;
  - the analytic floor (normal approximation, as s13 `tv_null_analytic`);
  - `tv_debiased` = `fork_tv` − floor.
- **Real effects (secondary target, with a caveat):** `real_tv_bh10 OR real_dacc_bh10`
  from `results/s13_sentence_effects.csv` (7 sentences in 3 problems).
  - **Caveat:** they were defined on the 300-sample pool, which includes r0, and they are
    not re-derived here.
  - The 2238 pair (s25/s26) is one accuracy spike, so the effective number of
    independent positives is about 6.
  - AP on 7 positives across 3 problems is fragile. Its bootstrap often has resamples
    with no positive; these are counted as non-finite and reported.

## 3. Forced-answer construction (FIXED)

### Prompt

- **Base.** The exact stored Phase 2 fork prompt (`inputs.parquet` `prompt`, sha-checked),
  with no chat template. It is the raw `"... Solution: \n<think>\n" + trace prefix up to
  S_i` that Phase 2 generated from.
- **Fail-closed checks.** Every prompt has exactly one `<think>` and no `</think>`.
  `forced_prompt` refuses anything else, so the suffix can never be appended twice.
- **Primary suffix.** The stored prefix's trailing whitespace is stripped, then:
  ```
  forced_i = prompt_i.rstrip() + "\n</think>\n\n**Final Answer**\n\\[\n\\boxed"
  ```
- **Deviation from the brief: the prefix ends at `\boxed`, not `\boxed{`.**
  - The model's tokenizer merges `{` with a following punctuation character. Checked
    locally: `{\` in `{\frac`, and `{(`.
  - It never merges `{` with the letters of `boxed`.
  - So the forced prefix stops at `\boxed`, and the brace is part of the scored or
    generated continuation.
  - With the local tokenizer, `build-requests --tokenizer-check` found **0 of 2,552**
    scored spans off a token boundary.
- **Alternative suffixes** (format check only, §6):
  - `alt1`: `prompt_i + "\n</think>\n\nThe final answer is \\boxed"`. No strip; this is the
    s08 memorisation format.
  - `alt2`: `prompt_i.rstrip() + "\n\nI'll stop here and give my answer.\n</think>\n\nThe
    answer is $\\boxed"`.

### Option (a): candidate scoring (PRIMARY readout)

- **Request.** For each candidate surface string s:
  - completions API, prompt = `forced_i + "{" + s + "}"`;
  - `echo: true`, `logprobs: 1`, `max_tokens: 1`, `temperature: 0`,
    `add_special_tokens: false`.
- **Span log-probability.**
  - log P(s) is the sum of the prompt-token logprobs of the shortest run of final prompt
    tokens whose text covers `{s}`.
  - The generated token is dropped using `usage.prompt_tokens`. This avoids relying on
    vLLM `text_offset`, which drifts after split multi-byte characters.
  - A span that does not start on a token boundary is flagged (`span_ok = false`).
- **Distribution.**
  - Surfaces of one category are alternative spellings of the same answer, so their
    probabilities add (logsumexp).
  - Categories are then renormalised over the candidate set to give q_i.
  - The raw unnormalised mass Σ P is kept as a coverage diagnostic.
  - These are raw (T = 1) model probabilities. Phase 2 sampled at T = 0.6. Spearman-based
    metrics are insensitive to that difference, and calibration (absolute agreement) is
    descriptive only.
- **Candidate sets (FIXED rule, built by `build-requests`):**
  - categories with share ≥ 0.5% of the source samples, at most 12, by count;
  - plus the ground truth, always;
  - non-answers (`<TRUNCATED>`, `<NON_ANSWER>`) excluded;
  - up to 2 raw surface strings per category, which must normalise to it (v2 normaliser),
    most frequent first. The ground truth's reference string comes first.
  - **PRIMARY `out`:**
    - Source: released resamples of the same problem at forks *outside* this trajectory's
      Phase 2 window, plus the other trace kind of that problem.
    - These samples are disjoint from every validation label (§9).
    - Coverage of the in-window resample mass is 0.92–1.00 per trajectory
      (`candidates.json`).
  - **Sensitivity `in`:** the pooled in-window resamples (r0 + r1 + r2), as in the brief.
    This set is circular (§9).
  - Both sets are scored in one pass. There are 1,160 distinct (fork, surface) pairs, 5.8
    per fork.

### Option (b): sampling (candidate-free readout)

- **Request.** K = 32 completions of `forced_i`:
  - `max_tokens` 30, temperature 0.6, top_p 0.95;
  - field set identical to `arv2.vllm_client.Client`.
- **Parsing.**
  - The answer is the first balanced box of `"\boxed" + completion`, normalised with v2.
  - A completion that does not open a brace, or does not close it within 30 tokens, is
    `<NON_ANSWER>`.
  - Answers outside the candidate set are `<OTHER>`.
- **Seeds and passes.**
  - Seed = `sample_seed(seed_from("20260914:arv2:plan_v3:sample:<pass>"), traj, fork, k)`.
  - Two passes, `s1` and `s2`.
  - `s1` is the readout; `s2` is the reproducibility control.

### Derived quantities

| Level | Quantity |
|---|---|
| Per fork | p_forced = q_i(ground truth) |
| | H_forced = entropy of q_i, in bits |
| | p_sampled = share of correct samples (option b) |
| Per sentence | Δp_forced = p_forced(i+1) − p_forced(i) |
| | forced TV = ½ Σ \|q_{i+1} − q_i\| (same candidate support within a trajectory) |

## 4. Estimands and metrics (FIXED)

All statistics are on PRIMARY labels (`r12`), with PRIMARY candidates (`out`) and the
`main` store. Thresholds and decisions:

| Id | Statistic | Rows | Threshold |
|---|---|---|---|
| **H1a** | Spearman(p_forced, acc) | 200 forks | 0.60 |
| **H1b** | median over trajectories of within-trajectory Spearman(p_forced, acc) | 10 trajectories | 0.30 |
| S2 | Spearman(p_sampled [s1], acc), the candidate-free replication of H1a | 200 | 0.60 |
| S3 | Spearman(H_forced, resampled entropy) | 200 | 0.50 |
| S4 | Spearman(Δp_forced, signed Δacc) | 190 | 0.30 |
| S5 | Spearman(forced TV, `tv_debiased`) | 190 | 0.20 |
| S6 | partial Spearman(forced TV, `fork_tv` \| floor) | 190 | 0.10 |
| S7 | AP(forced TV; real effects) | 190 | 0.25 |
| H4-level | Spearman(p_forced, acc) − \|Spearman(fork index, acc)\| | 200 | 0 (paired) |
| H4-AP | AP(forced TV) − AP(baseline), for each baseline | 190 | 0 (paired) |

**Descriptive, with no decision:**
- Pearson, ICC(2,1) and mean absolute error of p_forced against acc;
- Spearman of the per-problem constant (s08 no-reasoning accuracy) against acc, and the
  paired difference;
- AP of |Δp_forced|;
- Spearman of forced TV with `fork_tv` and with the floor;
- within-trajectory Spearman of Δp_forced with signed Δacc;
- Spearman of the option (b) and option (a) p_forced;
- the mean `<OTHER>` and `<NON_ANSWER>` rates of option (b);
- every statistic again on the `in` candidates and on the `r012` labels.

**Baselines for H4** (fixed list; s14 out-of-fold columns joined by
`trajectory_id:sentence_idx`; trained on `fork_tv`):
- `position_None`;
- `text+position_None`;
- `pre_8`;
- `pre+post_32`;
- `|valueprobe_dphat_32|`;
- raw sentence index.

For reference, s14 reported AP of 0.04–0.24 for these on the same 190 rows.

## 5. Statistics and decision rule (FIXED)

- **Intervals.** 95% percentile intervals from 2,000 **problem-cluster** bootstrap
  replicates. Each statistic's seed is `seed_from("20260914:arv2:plan_v3:boot:<id>")`.
- **H1b bootstrap.** Whole trajectories are resampled; there is one trajectory per
  problem.
- **Three-way rule per statistic** (as `phase2_plan.md` §7):
  - PASS if the lower bound ≥ threshold;
  - FAIL if the upper bound < threshold;
  - INCONCLUSIVE otherwise, or if the interval is undefined.
- **Paired H4 comparisons.** PASS means the lower bound of the difference is ≥ 0.
- **H1 composite:**
  - PASS if H1a and H1b both PASS;
  - FAIL if either FAILS;
  - otherwise INCONCLUSIVE.
- **Quality gates.** If any gate fails, every decision is reported but the conclusion is
  `INCONCLUSIVE_QUALITY_GATE`.

  | Gate | Condition |
  |---|---|
  | Q1 | the share of option (a) responses with `span_ok = false` (span not on a token boundary, or text ≠ `{s}`) is ≤ 1% |
  | Q2 | the median raw candidate mass Σ P over forks is ≥ 0.50 (the candidates carry most of the model's boxed-answer mass) |
  | Q3 | ≥ 95% of the primary score requests are present |

- **Advisory flags** (reported, and they qualify the wording of conclusions):

  | Flag | Condition |
  |---|---|
  | R1 deterministic | main vs replay p_forced Spearman ≥ 0.99 and max \|Δ\| ≤ 0.02 |
  | Format robust | p_forced primary vs each alternative suffix Spearman ≥ 0.80 on the 20-fork subset |
  | Sample coverage | mean option (b) `<OTHER>` rate ≤ 0.20 |

## 6. Reproducibility and format controls (FIXED)

- **R1 (option a is deterministic).**
  - Every primary scoring request is sent twice, to separate stores (`main`, `replay`).
    The payloads are identical, so a separate store is the only way to force a second
    computation.
  - Expectation: identical up to bf16 batching noise.
  - If R1 fails, the primary analysis also reports every statistic on `replay`. A
    conclusion stands only if both agree.
- **R2 (option b across seeds).** Two passes, `s1` and `s2`, each K = 32. The report gives
  ICC and Spearman of p_sampled between them. At K = 32 the binomial SD at p = 0.5 is
  0.09, so ICC ≈ 0.8 is expected if the forced distribution varies across forks.
- **Format sensitivity.**
  - The two alternative suffixes (§3) are scored with option (a) on a fixed 20-fork
    subset: 2 per trajectory, the forks with the smallest
    `sha256("20260914:arv2:plan_v3:format:<traj>:<fork>")`.
  - The report gives Spearman and mean |Δ| of p_forced.
  - If it is not format-robust, any PASS is reported as "suffix-specific".

## 7. What we will conclude (FIXED; `score` applies it mechanically)

| Gates | H1 | Change (S4, S7) | Conclusion code | Interpretation and next step |
|---|---|---|---|---|
| any fail | — | — | `INCONCLUSIVE_QUALITY_GATE` | Fix the readout mechanics, then rerun (no new data decision). |
| ok | PASS | at least one PASS | `READOUT_VALID_AND_CARRIES_CHANGE` | The one-pass readout tracks the outcome and its change. Write a confirmatory Phase 3 plan on new problems, with forced Δ as a cheap anchor estimator and resampling only to confirm. |
| ok | PASS | all FAIL | `READOUT_TRACKS_LEVEL_ONLY` | A valid outcome readout, but boundary-to-boundary differences do not recover the effects. Use it as an "anchor opportunity" or uncertainty prior to allocate resamples, not as a detector. |
| ok | PASS | otherwise | `READOUT_TRACKS_LEVEL_CHANGE_INCONCLUSIVE` | The level is valid. The change cannot be judged with 7 positives in 3 problems, so more labelled effects are needed before any claim. |
| ok | FAIL | — | `READOUT_NOT_REPRESENTATIVE` | The answer the model would give now is not the answer it reaches after continued reasoning. Drop the forced readout as an outcome representation. |
| ok | INCONCLUSIVE | — | `INCONCLUSIVE` | 10 clusters are too few. Before any conclusion, extend to the 30 other MVP trajectories using released labels (with the r0 caveat). |

**Qualifiers that always apply:**
- If H1a PASSES but the problem-constant difference is not > 0, the level result is
  reported as "problem-level, not boundary-level". H1b carries the boundary-level claim.
- A PASS on `in` candidates only, but not on `out`, is reported as circular (§9) and does
  not count.
- Any result that does not hold on the `r012` sensitivity labels is reported as "label
  source dependent".

## 8. Compute budget and stopping rules (FIXED, except the rate)

### Requests (from `build-requests`; requests sha256 `c2cfa4cc…c8e3`)

| Kind | Count | Prompt tokens | Generated tokens |
|---|---:|---|---|
| score, primary, main | 1,160 | ~1,525 each (mean prompt 1,506 + suffix; max 4,470) | 1 |
| score, primary, replay | 1,160 | same | 1 |
| score, alt1 + alt2 (20 forks) | 232 | same | 1 |
| sample, s1 + s2 (K = 32) | 12,800 | 200 distinct prefixes (prefix-cached) | ≤ 30 |

### Estimate on one H100 (pinned image; `phase2_vllm.sh`, `max_model_len` 24,576, prefix caching on)

- **Scoring.** About 3.9M prompt tokens. Requests that ask for prompt logprobs may bypass
  the prefix cache, so these are assumed to be fully prefilled. At 3k–15k tok/s
  effective, that is **4–22 min**.
- **Sampling.**
  - At most 0.38M generated tokens over 12,800 short requests: **3–10 min** at
    concurrency 32.
  - Worst case, if the prefix cache misses: about 19.5M prefill tokens, roughly 35 min.
- **Setup.** Instance boot, image pull and model load (from a warm HF cache, or a 29 GB
  download): **20–45 min**. Copying the build files over and mirroring the store back:
  about 5 min.
- **Total.** About 1.0 h expected (**$2.49** at $2.49/h). About 2.0 h pessimistic
  (**$5.00**).

### Hard guard and stopping rules

- **Guard: $25 in total instance cost (10.0 h).** Run with `run --max-cost 25 --max-hours
  10 --spent-hours <setup hours already used>`. The guard stops launching requests;
  in-flight requests finish.
- **Stopping rules, in order:**
  1. **Smoke:** `run --limit 8`. Every stored score response must have
     `span_method = "suffix"` and `span_ok = true`. Otherwise stop. The fix is a code
     change before data, with a changelog entry.
  2. **Measure:** `run --only score:primary:main`, then project the rest from measured
     throughput. If the projected remainder exceeds 3 h or $10, stop and ask the owner.
  3. **Then:** `score:primary:replay`, `score:alt1:main,score:alt2:main`, `sample`.
  4. **Always:** no request outside the built set, and no re-sending (the durable store
     never regenerates). Failures are retried 4 times with exponential backoff, then
     recorded, never silently dropped.
  5. **Early stop:** the analysis runs on the forks complete at stop time, with Q3
     applied.

## 9. Leakage, circularity and mitigations

- **Same prefixes.** The forced readout and the resample labels condition on the same fork
  prefix. That is the point: both measure the outcome distribution *at that boundary*.
  It is not leakage of the label, because the readout sees no resample.
- **Candidates from resamples (circularity).**
  - Scoring only answers that the resamples produced, and renormalising over them,
    injects label information into q_i: which wrong answers exist, and how many.
  - Mitigations:
    1. The PRIMARY candidate set comes only from samples disjoint from every label: other
       forks of the same problem, outside the validation window.
    2. Option (b) is candidate-free (S2) and replicates H1a without any candidate list.
    3. The raw mass Σ P is reported, so heavy renormalisation is visible (Q2).
  - p_forced for the correct answer always includes the ground truth. That uses only the
    reference answer, which the accuracy target uses too.
  - A candidate set from *training problems only* is not meaningful here, because answers
    are problem-specific. The out-of-window set is the leak-free analogue.
- **Real-effect flags.** These were selected on the 300 pool by a permutation test on the
  same forks. They are a fixed external list for this pilot; nothing is tuned against
  them.
- **Seen problems.** Problem 1591 is memorisation-flagged (s08). Every statistic is also
  reported without 1591 (`sensitivity_without_1591`), descriptively. The other 9 problems
  are not flagged, but all were seen.
- **The baselines come from s14 out-of-fold predictions.** They were trained on MVP labels
  (r0-like released data) by leave-one-problem-out. That favours neither side on these
  190 rows.

## 10. Implementation notes (PROVISIONAL)

- **Outputs.**
  - `build-requests` writes `$ARV2_DATA/derived/plan_v3/`: `requests.jsonl`,
    `prompts.parquet`, `targets.parquet` (both label sets), `candidates.json` and
    `build_report.json`.
  - `run` writes `$ARV2_DATA/generations/plan_v3_forced/{main,replay}/<sha256(request)>.json`.
    The layout is the same as `arv2.vllm_client.Client`. Score responses keep only the
    last 48 echoed tokens.
- **Verification.** `run` re-derives every payload from the stored prompt and verifies its
  sha256. HTTP is imported lazily.
- **Version risk.** Echo with logprobs has not been exercised against the pinned vLLM
  image (version unrecorded). The smoke step (§8.1) is the check.

## 11. Resolved before any forced-answer request (2026-10-11)

1. **Thresholds.** Kept as stated: H1a 0.60 and H1b 0.30. They are not to be changed
   after the data are seen.
2. **Scored-probability temperature.** Only the T = 1 option (a) distribution is computed.
   No T = 0.6 tempered variant will be added after the data are seen.
3. **Primary candidate set.** Stays `out` (out-of-window, leak-free). `in` remains the
   circularity sensitivity (§7 qualifier).

The owner authorised the GPU run. These defaults were settled by the analysis lead
before the run and recorded here; the owner may override them only by a dated amendment
made before scoring.

## 12. Not in scope

- New problems.
- Attention or receiver-head analyses.
- Probes on forced readouts.
- Re-deriving real effects on `r12`. That would be a separate, later step, and it would
  have to keep the s13 BH procedure fixed.

## 13. Changelog

- v1 (2026-10-11): initial pilot preregistration, before any forced-answer request was
  sent. Before finalising, the primary labels were switched to replicates 1 and 2 only,
  and the 300 pool became a prespecified sensitivity, because the release and our run
  differ on the incorrect traces (§2). The request set is unchanged by this (requests
  sha256 identical).
- v1.1 (2026-10-11): §11 open items resolved to the stated defaults, before any request
  was sent.
