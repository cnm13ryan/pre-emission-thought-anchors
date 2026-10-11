# Can a chain-of-thought monitor see a pivotal step coming?

**Reliability and pre-emission predictability of "thought anchors": project proposal and
BlueDot Rapid Grant application.**

Draft prepared 2026-10-11 from the committed results on branch `study/anchor-reliability-v2`.
Items in `[BRACKETS]` must be filled in by the applicant; nothing personal has been assumed.

---

## Part A: project proposal

### A1. Summary

- **What is being audited.** "Thought anchors" are sentences in a reasoning model's chain of
  thought (CoT) that disproportionately determine its final answer. They are found by
  resampling the reasoning from each sentence and measuring how the answer distribution
  shifts (Bogdan et al. 2025, arXiv 2506.19143).
- **Why it matters for safety.** The method and its follow-ups are now used to locate the
  reasoning steps that drive safety-relevant behaviour, such as sycophancy and
  blackmail-style agentic misbehaviour (Thought Branches, arXiv 2510.27484, ICLR 2026;
  Sycophantic Anchors, arXiv 2601.21183).
- **What we found.** On the released data, the standard importance score is mostly
  finite-sample noise that tracks answer uncertainty.
  - Replicate agreement: Spearman 0.44, ICC 0.56.
  - Split-half reliability of the uncertainty-corrected score: 0.16.
  - Only about 2% of sentences carry a statistically real effect.
- **What this project does.** It finishes the audit and answers one question that matters
  for CoT monitoring: **can a pivotal reasoning step be predicted before the model writes
  it?** If yes, monitors can intervene before the step is written. If no, they can only
  react after it.

### A2. Safety motivation

1. **CoT monitoring is a safety layer that depends on knowing which steps matter.** Current
   proposals treat legible reasoning as a monitorable signal (e.g. Korbak et al. 2025,
   "Chain of Thought Monitorability"). A monitor is far more useful if it can tell the step
   that actually commits the model to an action from the many steps that don't.
2. **Safety findings now rest on resampling-based importance.** Thought Branches applies it
   to blackmail and resume-screening scenarios, and Sycophantic Anchors trains probes on
   anchor labels built from 20 completions per sentence.
   - Our results show the measurement is noise-dominated at 100 samples, and the
     literature review found no published reliability check.
   - So claims of the form "this step caused the misbehaviour" may be partly artefacts.
     A reliability-corrected methodology protects safety conclusions from that failure.
3. **Prevention vs detection hinges on a measurable ceiling.**
   - We showed that whether a step helps or hurts is unpredictable from the history alone,
     by construction: the average effect over the steps the model might write is zero.
   - *How much* a step will move the outcome may be predictable: our preliminary ceiling
     estimate is about 0.5, against about 0.04 for current activation probes.
   - Measuring this ceiling (R²_max) properly tells us whether "flag the pivotal step
     before it is written" is achievable, or whether monitors should be built for
     after-the-fact detection.

### A3. What already exists (self-funded so far)

- **A preregistered replication (Phase 2).**
  - 40,000 regenerations on 10 traces from the released Thought Anchors dataset
    (DeepSeek-R1-Distill-Qwen-14B), with full provenance.
  - About $108 of compute, self-funded.
  - Result: FAIL on reproducibility. The reproducible part of the score is the
    noise floor of answer entropy.
- **Diagnostics and root cause.**
  - The noise floor predicts the score better than the score predicts itself.
  - Real effects are rare and concentrated where the model is uncertain.
  - 108 probe configurations show that activation probes read uncertainty, not importance.
  - Problem difficulty, answer grouping and probe capacity are all ruled out as causes.
- **A preregistered forced-answer pilot.** It is a negative result: 78% of the answers forced
  mid-reasoning are premature guesses outside the set of answers that continued reasoning
  reaches.
- **A released-data discrepancy.** On incorrect traces, the released data are 11 points less
  accurate than faithful regeneration. The leading candidate is a prompt-format artefact:
  newlines collapsed.
- **Theory.** Signed importance has a predictability ceiling of zero (a zero-mean-advantage
  identity). The unsigned ceiling is estimated at about 0.5, but the estimator is unreliable
  at real effect sizes.
- **Literature audits.**
  - A novelty audit, with 13 closest prior works: no published reliability analysis of
    resampling importance exists.
  - An out-of-distribution problem-sourcing report for this model.
- **A fully specified next step.** A staged GPU plan with costs from measured throughput, a
  preregistration and integrity checks (`docs/v3_followup_gpu_plan.md`).

### A4. Aims

- **Aim 1. The predictability ceiling, the flagship.** Measure R²_max for unsigned sentence
  importance with nested sampling:
  - 20 problems the model cannot have seen (2025–26 competitions, screened and
    de-duplicated) × 20 positions × 5 sentences × 200 continuations;
  - unbiased estimators with an explicit sampling-noise correction.
- **Aim 2. Reliability on clean problems.** Replicate the reliability audit on 30 more
  post-cutoff problems: split-half reliability, a per-sentence significance test, and the
  anchor rate.
- **Aim 3. Is "anchor" a stable property?** On the same sentences, compare resampling,
  deletion and paraphrase, at two temperatures, each against its own noise floor.
- **Aim 4. A safety-relevant transfer test.** Repeat the reliability audit in a sycophancy
  setting (hinted multiple-choice questions, as in Sycophantic Anchors and Thought
  Branches). This tests whether anchor labels used for safety probes are reproducible.
- **Aim 5. What monitors could read.** Benchmark activation probes, sentence-span states and
  logit lens against the measured ceiling and an uncertainty-only baseline. Also run a cheap
  short-budget readout as a monitor-friendly proxy for expensive resampling.

### A5. Methods (abridged; the full plan is in `docs/v3_followup_gpu_plan.md`)

- **Model and data.** DeepSeek-R1-Distill-Qwen-14B with a pinned vLLM image and model
  revision. Problems come from AIME 2025/26, BRUMO, HMMT, SMT and CMIMC 2025, all first
  published after the model's January 2025 release.
  - Two-stage screening (16 + 32 samples) for problems solved 15–85% of the time.
  - Near-duplicate search, and contamination red-flag probes.
- **Run 1** (about 5–7 node-hours): screening, a dry run measuring throughput, and pilots.
  It ends at a go/no-go gate.
- **Run 2** (about 42–90 node-hours on 8 × H100 SXM): Aims 1–5, run by a resumable,
  idempotent driver with per-request logging and a hard cost guard.
- **Analysis.** Problem-clustered bootstrap intervals. Thresholds fixed in a preregistration
  before Run 2 data are scored. Everything is labelled exploratory or confirmatory.

### A6. Outputs (within about 6–8 weeks of funding)

1. **A short paper / LessWrong post:** "Thought anchors are mostly noise at standard sample
   sizes; here is how to measure them reliably, and how predictable they are."
2. **An open toolkit.** Debiased importance estimators, a per-sentence significance test, a
   split-half reliability check, and the canonical answer grouper. The point is that safety
   researchers using resampling importance can check their own labels.
3. **The R²_max estimate**, with intervals, and its implication for preventive vs reactive CoT
   monitors.
4. **Released traces and labels** for post-cutoff problems (licence permitting:
   CC BY-NC-SA, references rather than problem text).

### A7. Risks and mitigations

- **Too few clean problems pass screening.** Mitigation: OlymMATH as a labelled reserve, and
  the Run 1 gate.
- **Throughput below assumption.** Mitigation: a re-cost at the gate, and a pre-ordered cut
  list.
- **The ceiling estimator is still unstable.** Mitigation: synthetic validation, including
  weak effects, before real data are scored.
- **A null on predictability is still informative.** It points monitors to after-the-fact
  detection.

---

## Part B: application answers

### Q1. How much funding are you requesting?

**$3,650**

### Q2. Project overview

> I'm auditing "thought anchors", a resampling method now used to find which reasoning steps
> cause behaviours such as sycophancy in reasoning models. My replication shows the standard
> score is mostly noise. This grant funds the decisive experiment: can a pivotal reasoning
> step be predicted before it is written? The answer informs whether CoT monitoring can aim
> to prevent, or only to detect.

*(about 60 words)*

### Q3. What have you already done?

> - Preregistered replication of Thought Anchors (DeepSeek-R1-Distill-Qwen-14B): 40,000
>   regenerations, self-funded (~$108). Importance scores fail reproducibility (Spearman
>   0.44); corrected split-half reliability 0.16; only ~2% of sentences carry real effects.
> - Root-cause analysis: activation probes read uncertainty, not importance (108 probe
>   configurations); a theoretical proof that signed importance is unpredictable
>   pre-emission.
> - Preregistered forced-answer pilot: negative result (78% premature guesses).
> - Found an 11-point accuracy gap in the released dataset (likely a prompt-format artefact).
> - Literature audit: no published reliability check exists. Plus an out-of-distribution
>   problem-sourcing report and a costed, staged GPU plan.
>
> Links: `[REPO OR PUBLIC SUMMARY LINK]`

*(about 115 words)*

### Q4. What specifically would this grant fund?

| Item / service | What it is for | Estimated cost |
|---|---|---|
| Lambda 8 × H100 SXM, Run 1 (~6 node-hours at $31.92/h) | Screening ~250 post-cutoff problems, dry-run throughput measurement, pilots; go/no-go gate | $200 |
| Lambda 8 × H100 SXM, Run 2 (~89 node-hours at the conservative throughput bound) | Nested-sampling ceiling (Aim 1), reliability replication (Aim 2), counterfactual comparison (Aim 3), probes and readouts (Aim 5); ~3.5B generated tokens | $2,850 |
| Lambda 8 × H100 SXM, safety transfer arm (~4.5 node-hours) | Reliability audit of sycophancy anchors on hinted questions (Aim 4); ~180M tokens | $150 |
| Hosted inference API credits | The released-data generation path (Novita) for the prompt-format × provider test | $50 |
| Lambda persistent filesystem (~150 GB × 3 months at $0.20/GB-month) | Traces, activations and model cache between runs | $90 |
| Contingency (~10% of compute) | Reruns after failures, throughput shortfall | $310 |
| **Total** | | **$3,650** |

**Notes on the budget.**
- **Rates.** Lambda's on-demand prices, verified 2026-10-11 on lambda.ai/pricing:
  8 × H100 SXM at $3.99 per GPU-hour ($31.92/h); filesystem at $0.20/GB-month, no
  transfer fees.
- **Hardware choice.** 8 × H100 SXM is chosen over 8 × A100 80 GB ($2.79 per GPU-hour,
  $22.32/h). Generation is bandwidth-bound and H100 SXM has about 1.7× the bandwidth, so
  the total cost is similar (A100 conservative bound about $3,290) and the run is about 40%
  faster. The cheap Run 1 dry run measures this before Run 2 starts.
- **Estimate.** Run 2 is budgeted at the low-throughput bound; the mid estimate is about
  $1,820. Any unspent compute is returned, or used to extend the problem count.
- **Staged option.** If a smaller ask is preferred, about $2,000 funds Run 1 plus a reduced
  Run 2 (150 continuations per sentence, no puzzle arm), which still answers Aim 1.

### Q5. How does this reduce catastrophic risk from AI?

> CoT monitoring is a leading near-term safety layer for reasoning models, and
> resampling-based "thought anchors" are increasingly used to say which reasoning steps
> cause unsafe behaviour. I'll deliver (1) a reliability toolkit that safety researchers can
> run on their own anchor labels, and (2) a measured ceiling on whether pivotal steps can be
> predicted before they are written. The load-bearing link is (2). It indicates to monitor
> builders whether preventive intervention is achievable in principle, or whether to invest
> in after-the-fact detection. This is research evidence, not a deployable monitor.

*(about 80 words)*

### Q6. What would you do without this grant?

> Reduce scope and slow down. I can self-fund the ~$200 screening run. But the nested-sampling
> experiment (~3.5B tokens) is unaffordable personally, so I would drop the predictability
> ceiling: the central safety question. I'd publish only the reliability critique on the
> released data, likely 2–3 months later, on smaller rented GPUs.

*(about 60 words)*

### Q7. Why you? Why now? Why will it succeed?

> **Why me:** I built and ran the whole pipeline: preregistration, a 40k-generation
> replication with hash-verified provenance, diagnostics, and the negative pilots. I found
> the reliability problem myself. `[BACKGROUND; BLUEDOT AFFILIATION, e.g. course and cohort]`
>
> **Why now:** Thought Anchors and Thought Branches (ICLR 2026) are being adopted for safety
> analyses such as sycophancy and agentic misbehaviour, but no one has checked whether the
> measurements reproduce. CoT monitorability is a time-limited opportunity while models
> still reason legibly.
>
> **Why it will work:** the infrastructure is proven at scale, and the costs come from
> measured throughput. A cheap Run 1 gates the expensive run, and the analysis plan is
> preregistered. At the end: a public write-up, an open toolkit, and the ceiling estimate.

*(about 130 words, excluding the placeholder)*

---

## Pre-submission check

- [x] **Amount matches the budget.** Q1 equals the Q4 total ($3,650): a multiple of $50,
      within $50–$10,000.
- [ ] **Q3 has a link.** Q3 names concrete work; **a public link is still needed** (the repo
      is private).
- [x] **Q4 is in scope.** It is itemised, and every line is compute, API credits or storage.
- [x] **Q5 is one plausible chain.** The load-bearing link is named.
- [x] **Q6 is an honest counterfactual.** It shows that money is the bottleneck for the core
      experiment.
- [ ] **Q7 is complete.** It still needs the applicant's background and BlueDot affiliation.
- [x] **Nothing is fabricated.** Every number comes from committed results; personal items
      are left as placeholders.
