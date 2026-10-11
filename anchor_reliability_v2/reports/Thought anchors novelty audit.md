# Reliability audit stays novel; easy MATH limits it

The programme's core contribution is still open in the literature: the reliability audit of Thought Anchors' resampling importance and the nested-sampling R²_max ceiling on pre-emission prediction. No published critique, replication or reliability analysis of per-sentence Thought Anchors importance was found. The closest neighbour, Du et al. (COLM 2026), uses split-half noise ceilings for Monte Carlo step importance, but on Qwen3 models and for a different question. The task, however, is partly too easy. MATH is near ceiling for R1-Distill-Qwen-14B (93.9% on MATH-500). Thought Anchors picked its 20 problems with only a 10-sample screen. Qwen2.5-family models demonstrably memorise MATH-500. Our own check (median accuracy from the start 0.73, one memorised problem) shows the set sits at the top of its intended 25–75% band. That does not invalidate the audit, because the released dataset is the object being audited. It does mean the headline "about 2% of sentences carry real effects" describes an easy, partly contaminated problem set, not reasoning in general. Most of the novelty damage is concentrated in the peripheral arms. The signed R²_max = 0 result is the textbook zero-mean-advantage identity. Receiver heads and suppression are a replication. Logit lens on R1-Distill-Qwen-14B / MATH-500 was done in May 2026. Sentence-span probes on the same model and benchmark appeared in April 2026, and that paper reports a *positive* result we must reconcile with our negative one. The GPU plan should therefore put first the nested-sampling arm, a properly screened uncontaminated second source, and the per-arm noise floor. It should demote or reframe the activation and attention arms.

**How to read this report.** Claims marked **Verified** come from the research notes and were checked against a primary page (arXiv abstract or HTML, venue page, GitHub, LessWrong), each with an inline link. Most arXiv content was read through a summarising fetch, so small paraphrase errors in numbers are possible. Claims marked **Inference** are our reasoning from those facts. **Programme results** are the user's own findings as given in the brief; they cannot be checked externally. Claims that the notes held only from background knowledge are not used as evidence. They appear in the final list of unverified items.

## Verdicts at a glance

| Question | Verdict | One-line reason |
|---|---|---|
| A. Task difficulty | **Partly too easy; not fatal** | MATH is near-saturated and contaminated for Qwen2.5-family models. The 10-sample selection screen lets easy problems in. Fine for an audit of the released data; weak for general claims. |
| B(i) Reliability of resampling importance | **Open; our strongest claim** | No reliability analysis of Thought Anchors exists. Du et al. 2026 is the nearest methodological precedent (split-half ceilings). |
| B(ii) Nested sampling / forking paths / variance decomposition | **Mostly open** | Bigelow's K×30 token design is a structural precedent. The zero-mean advantage is textbook (PAV). No three-level decomposition or unsigned R²_max ceiling was found. |
| B(iii) Pre-emission prediction of important steps | **Partially occupied** | Nikankin et al. 2026 probe "importance" on R1-Distill-Qwen-14B / MATH-500, but their importance is removability, read from pooled step activations. Sycophantic Anchors probes *post*-emission. No resampling-ceiling benchmark exists. |
| B(iv) Correctness and answer probes | **Crowded** | Many papers decode correctness mid-CoT (AUROC 0.7–0.95). None targets resampled prefix accuracy with position and entropy controls. |
| B(v) Forced early answering and truncation | **Method done; our measurement new** | Lanham, DEER, Answer Convergence and Reasoning Theater all force answers. None reports the share of forced answers outside the continuation answer set. No "continue K tokens, then force" estimator was found. |
| B(vi) Resample vs delete vs paraphrase | **Partially anticipated** | Thought Branches: on-policy beats off-policy insertion, at one temperature, without delete or paraphrase arms. Nobody compares all three on the same sentences across temperatures. |
| B(vii) Receiver heads and suppression | **Already done (replication)** | Thought Anchors itself plus a legal-domain replication. Attention–importance correlation is weak (ρ = .20, r = .23). There is an open ablation-hook bug report. |
| B(viii) Critical and high-entropy tokens | **Adjacent; conceptually useful** | cDPO and the 80/20 paper define critical or forking tokens without resampling importance. That supports our "entropy ≠ importance" result rather than pre-empting it. |
| C. Methodological critiques | **None published** | Only GitHub issues (API-error rollouts, a head-ablation hook) and a LessWrong replication. Entropy dominance, finite-sample TV bias, threshold and prompt-format sensitivity are all unclaimed. |
| D. Overall novelty threats | **Moderate, localised** | Main threats: Du et al. (ceilings), Nikankin et al. (probes, conflicting result), Li et al. (logit lens), PAV (tower property), Thought Branches (on/off-policy). The main limit is 20 easy MATH problems on one model. |

## The task is easy enough to bias the audit's headline numbers (A)

**Verified.** The model card reports R1-Distill-Qwen-14B at **93.9% pass@1 on MATH-500** and 69.7% on AIME 2024, at temperature 0.6, top-p 0.95 and 64 samples ([HF model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)). On competitions held after release, MathArena measures **AIME 2025 at 49.17%**, HMMT Feb 2025 at 31.67%, BRUMO 2025 at 68.33% and SMT 2025 at 54.72% ([MathArena](https://matharena.ai/models/deepseek_deepseek_distill_14b)).

Thought Anchors chose "20 challenging but doable questions that are correctly solved 25-75% of the time, identified by testing on 1,000 problems 10 times each" ([arXiv 2506.19143](https://arxiv.org/html/2506.19143)). Neither the paper nor the repo mentions a memorisation filter ([GitHub](https://github.com/interp-reasoning/thought-anchors)).

Wu et al. show that Qwen2.5 models reproduce MATH-500 problems from partial prompts. From a 60% prefix, Qwen2.5-7B reaches **21.2% exact match** and Qwen2.5-Math-7B 54.6%, against 3.8% for Llama-3.1-8B. They conclude that contamination explains Qwen's "spurious reward" gains ([arXiv 2507.10532](https://arxiv.org/html/2507.10532v3)). Their study does not test R1-Distill models. MATH-Perturb finds large drops on hard perturbations of Level-5 MATH, consistent with models applying memorised solution templates ([ICML 2025](https://icml.cc/virtual/2025/poster/45435)).

Difficulty also changes what resampling finds. In Du et al., "the easier and better-attested GSM8K and MATH500 show a lower proportion of self-advantage steps", while AIME problems almost always contain at least one causally important step ([arXiv 2609.04194](https://arxiv.org/html/2609.04194v1)). Thought Anchors' own MMLU analysis finds that high-accuracy questions have stronger close-range and weaker long-range links ([arXiv 2506.19143](https://arxiv.org/html/2506.19143)).

**Programme results.** Accuracy from the problem start has median 0.73 (range 0.22–0.90). Problem 1591 reaches 0.80 with no reasoning.

**Inference.** A 10-sample screen has a binomial standard error of about 0.16 at p = 0.5. Re-measured with many samples, problems will regress toward the model's high base rate on MATH. Our median of 0.73 is exactly what that predicts: half the set sits at or above the top of the intended band. R1-Distill-Qwen-14B is fine-tuned from the Qwen2.5 family, so it most likely inherits that family's MATH exposure. Problem 1591 is direct evidence of this for at least one item, and others may be partly memorised.

The consequence for the audit is specific. Our noise floor is entropy-driven, and real effects sit where the answer is uncertain. Easy or memorised problems have few such points. The "about 2% of sentences" rate and the low debiased split-half reliability (0.16) are therefore partly properties of an easy, contaminated set. The task is not "too easy" for an *audit of the released dataset*: that dataset is the object under test, and its easiness is itself a finding about the original paper. It is too easy to support general claims about how often reasoning steps matter. That requires the second-source arm, and difficulty (accuracy from start, answer entropy) has to be a stratifier rather than something averaged away.

## Prior work leaves the reliability and ceiling questions open (B)

### (i) Nobody has measured the reliability of resampling importance

**Verified.** Thought Anchors reports a single variance statement: per-problem accuracy CIs are "at worst ±10%". It calls compute cost "a primary limitation", and its only split-half reliability figure concerns receiver-head kurtosis (r = .84), not sentence importance ([arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)). Thought Branches (ICLR 2026 poster) reports boxplots and scatterplots without CIs, significance tests or a rollout-sufficiency analysis ([arXiv 2510.27484v2](https://arxiv.org/html/2510.27484v2); [ICLR](https://iclr.cc/virtual/2026/poster/10008605)). Du et al. estimate importance as the change in expected reward from 50 rollouts per prefix. They build split-half noise ceilings and say they "must account for estimator variance" ([arXiv 2609.04194](https://arxiv.org/abs/2609.04194)). Their setting is Qwen3-1.7B/4B/8B, and their question is whether LLM judges can recover importance. Forking Fast finds that noise in uncertainty dynamics is largely multinomial sampling noise scaling as 1/√S "rather than an LLM's sensitivity to each individual token" ([arXiv 2608.19611](https://arxiv.org/html/2608.19611v1)).

**Inference.** Our replicate Spearman 0.44, ICC 0.56, entropy-driven floor, debiased split-half 0.16 and BH survival of about 2% are unclaimed. Du et al. must be cited as the methodological precedent for split-half ceilings. Forking Fast's sampling-noise result is independent corroboration of the floor mechanism and strengthens our story.

### (ii) The tower property is textbook, but the unsigned ceiling is not

**Verified.** Bigelow et al. (ICLR 2025) resample at each token: up to 10 alternatives with at least 5% probability, then 30 completions each. They locate forking tokens with Bayesian change-point detection on GPT-3.5 and do not analyse hidden states ([arXiv 2412.07961](https://arxiv.org/html/2412.07961); [ICLR](https://iclr.cc/virtual/2025/poster/30769)). Zur et al. probe hidden states for the outcome distribution. They use only four examples and find that steerability correlates with uncertainty (R = 0.57–0.64) ([arXiv 2511.04527](https://arxiv.org/html/2511.04527)).

PAV defines progress as A = Q − V. It notes that using the base policy's advantage gives "exactly the same policy gradient update as only optimizing outcome evaluation", so it switches to a separate prover policy ([arXiv 2410.08146](https://arxiv.org/html/2410.08146)). VinePPO and OmegaPRM estimate V or Q from Monte Carlo rollouts ([VinePPO](https://arxiv.org/html/2410.01679v1); [OmegaPRM](https://arxiv.org/abs/2406.06592v1)). Real-time progress prediction reports a rollout-based error floor of about 5–10% for *progress* ([arXiv 2506.23274](https://arxiv.org/html/2506.23274v2)).

**Inference.** Signed importance is acc(h,S) − acc(h), which is the advantage A(h,S), and E_S[A | h] = 0 is the zero-mean-advantage identity. Signed R²_max = 0 will be recognised as known. The ceiling for *unsigned* importance, Var_h E[|A| | h] / Var|A|, is not covered by that identity. So is the empirical three-level split into between-history, between-sentence-within-history, and within-cell binomial noise. Neither was found anywhere and both look novel. Bigelow's design is the token-level structural ancestor of our K×M design.

### (iii) Pre-emission importance prediction has one close, conflicting neighbour

**Verified.** Nikankin et al. probe step "importance" on **R1-Distill-Qwen-1.5B/7B/14B** with HARP and **MATH-500**. Their activations are mean-pooled per step and flattened across layers. Their importance is removability (sufficiency and attainability), not resampling. They report **80% or higher probe accuracy against 60% or lower for an LLM judge**, low correlation with position and length, and importance "encoded before later steps are generated" ([arXiv 2604.18307](https://arxiv.org/html/2604.18307)).

Sycophantic Anchors defines Imp(s_k) from 20 completions and probes at the *final token of the sentence*. It reaches R² 0.48–0.74 for anchor strength, with no ceiling analysis ([arXiv 2601.21183](https://arxiv.org/html/2601.21183)). Other probes target history-level quantities: semantic entropy before generation ([arXiv 2406.15927](https://arxiv.org/pdf/2406.15927)), accuracy from the question alone ([arXiv 2509.10625](https://arxiv.org/abs/2509.10625)), and features preceding "wait" ([arXiv 2510.04128](https://arxiv.org/html/2510.04128v1)).

**Inference.** This is the biggest novelty risk to the probe arm. A reviewer will set "Reasoning models know what's important" (same model, same benchmark family, positive) against our "probes read uncertainty, not importance". The results can be reconciled in three ways:

- Removability and resampling importance are different targets.
- Mean-pooled step activations include the step itself, so they are post-emission, as in Sycophantic Anchors. That lets them exceed a pre-emission ceiling.
- Their surface controls (position, length) are not entropy controls.

We have to demonstrate this reconciliation, not just assert it.

### (iv) Correctness probes are crowded, but none target resampled accuracy

**Verified.** Zhang et al. decode intermediate-answer correctness at ROC-AUC above 0.7, and above 0.9 on AIME for R1-Distill-Qwen-32B. The signal appears before the answer is written. They do not test the 14B model ([arXiv 2504.05419](https://arxiv.org/html/2504.05419v1)). Yuan et al. reach 0.95 AUROC on trace correctness, and 0.79 from the first step, but interventions fail: "diagnostic, not causal" ([arXiv 2605.09502](https://arxiv.org/abs/2605.09502)). Reasoning Theater trains attention probes on random prefixes of R1 and the R1-Distill 1.5B–32B models, on MMLU and GPQA-D only ([arXiv 2603.05488](https://arxiv.org/html/2603.05488)). Afzal et al. predict CoT success "before a single token is generated" ([arXiv 2505.24362](https://arxiv.org/abs/2505.24362)).

**Inference.** These labels are mostly between-trace or between-problem correctness, which difficulty and entropy can explain. Our position-plus-entropy-controlled, within-trace negative result is sharper and unclaimed. It should be framed as an explanation of *why* these probes succeed.

### (v) Forced answering is well trodden; our measurement of it is not

**Verified.** Lanham et al. introduced truncation with early answering ([arXiv 2307.13702](https://arxiv.org/abs/2307.13702)). DEER inserts a `\boxed{` inducer at transition points and trusts the result only at confidence 0.95 ([arXiv 2504.15895](https://arxiv.org/html/2504.15895)). Answer Convergence (EMNLP 2025) finds math answers converge after about 60% of steps and requires 10 identical consecutive answers before stopping ([arXiv 2506.02536](https://arxiv.org/html/2506.02536v2)). Xiong et al. (NeurIPS 2025) find that immediate answering changes the answer in about 30% of GPQA cases on R1-Distill models including 14B, and that the answer stage "often introduces new computation" ([arXiv 2505.13774](https://arxiv.org/html/2505.13774v2)). Reasoning Theater finds forced answers near chance early on GPQA-D ([arXiv 2603.05488](https://arxiv.org/html/2603.05488)).

**Inference.** Our pilot method is DEER-style and not novel. The measurement is new: 78% of forced answers fall outside the continuation answer set. Existing gating heuristics implicitly concede the problem without quantifying it. No paper continues for K tokens and then forces closure.

### (vi) Resample vs delete vs paraphrase: the spirit is anticipated, the design is not

**Verified.** Thought Branches compares on-policy resampling with three off-policy insertions: handwritten sentences, other models' sentences, and same-model sentences from different seeds. The off-policy arms "cluster near zero effect", and the paper uses one temperature (0.7) with no deletion or paraphrase arm ([arXiv 2510.27484v2](https://arxiv.org/html/2510.27484v2)). Lanham et al. paraphrase the remaining CoT ([arXiv 2307.13702](https://arxiv.org/abs/2307.13702)). Nikankin et al. use greedy deletion ([arXiv 2604.18307](https://arxiv.org/html/2604.18307v1)). Thought Editing compares substitution, insertion and resampling for steering ([LessWrong](https://www.lesswrong.com/posts/KXR5FNs4hHT5sMRti/thought-editing-steering-models-by-editing-their-chain-of)).

**Inference.** Three things remain novel: all three arms on the same sentences, temperature as a factor, and each arm scored against its own noise floor. Thought Branches predicts that delete and paraphrase will be weaker or noisier, and that prediction can be pre-registered.

### (vii) Receiver heads are a replication with weak baselines

**Verified.** In Thought Anchors, attention suppression and resampling importance correlate positively in 19 of 20 traces, but with **mean ρ = .20** ([arXiv 2506.19143v3](https://arxiv.org/html/2506.19143v3)). A legal-domain replication finds receiver heads in later layers and an attention–importance correlation of r = 0.23 ([LessWrong](https://www.lesswrong.com/posts/f9cnGHCiJgo5eDkSQ/understanding-reasoning-with-thought-anchors-and-probes)). GitHub issue #7 (open, September 2026) reports that the head-ablation hook zeroes blocks *after* `o_proj`, which are not individual heads. The report is based on source inspection only ([issue #7](https://github.com/interp-reasoning/thought-anchors/issues/7)). The Preplan-and-Anchor paper (ICML 2026) finds globally focused heads that highlight anchor tokens ([arXiv 2510.13554](https://arxiv.org/abs/2510.13554)).

**Inference.** The only novel angle left is to ask whether these scores track *debiased, reliable* importance or the entropy floor.

### (viii) Critical and high-entropy tokens support "entropy ≠ importance"

**Verified.** cDPO identifies critical tokens by contrasting likelihoods under positive and negative fine-tuned models ([arXiv 2411.19943](https://arxiv.org/abs/2411.19943v1)). The 80/20 paper (NeurIPS 2025) calls the top-20% entropy tokens "forking tokens". Training only on them matches or beats full updates ([arXiv 2506.01939](https://arxiv.org/html/2506.01939v2)).

**Inference.** "Forking" means high entropy in the RL literature but outcome-changing under resampling for Bigelow. Our probe result is evidence that these two notions separate at the sentence level, and that is a useful clarifying contribution.

## Closest prior works and the gap each leaves

| Work (arXiv, venue, date) | What they did | Overlap with our arms | Remaining gap |
|---|---|---|---|
| Bogdan, Macar, Nanda, Conmy, *Thought Anchors* ([2506.19143](https://arxiv.org/html/2506.19143v4)); under review at ICLR 2026, decision unknown; v4 27 Oct 2025 | 100 resamples per sentence, 20 MATH problems in the 25–75% band, R1-Distill-Qwen-14B; receiver heads; suppression | Object of the audit; receiver-head and suppression arm | No reliability, bias or significance analysis of sentence importance |
| Macar et al., *Thought Branches* ([2510.27484](https://arxiv.org/abs/2510.27484)); ICLR 2026 poster; Oct 2025, revised Apr 2026 | On-policy resampling vs off-policy insertion; resilience metric; blackmail, resume and hint scenarios | Resample vs delete vs paraphrase | No delete or paraphrase arm, one temperature, no MATH, no reliability analysis |
| Du, Hoyle, Ruis, Locatelli, *Legibility is Not Interpretability* ([2609.04194](https://arxiv.org/abs/2609.04194)); COLM 2026; Sep 2026 | Monte Carlo step importance (N = 50) with split-half ceilings; Qwen3; AIME 24/25/26, AMC, MATH-500, GSM8K | Reliability audit; AIME second source; difficulty effect | Not an audit of Thought Anchors; no TV, no entropy decomposition of the floor, no R1-Distill-14B |
| Nikankin et al., *Reasoning Models Know What's Important* ([2604.18307](https://arxiv.org/html/2604.18307)); preprint; Apr 2026, revised Jun 2026 | Removability importance; pooled-step MLP probes reach 80% or higher; R1-Distill-Qwen-14B, MATH-500 | Probes; span activations | Removability rather than resampling target; no entropy control; no pre-emission ceiling; positive result conflicts with ours |
| Li, Yang, Mehta, Hazarika, Onoue, *When Reasoning Traces Become Performative* ([2605.11746](https://arxiv.org/html/2605.11746)); preprint; May 2026 | Logit lens at every CoT step with tuned-lens and Patchscopes checks; R1-Distill-Qwen-7B/14B/32B; MATH-500 | Logit lens | Single-token ground-truth proxy; no answer distribution; no validation against resampled accuracy |
| Bigelow, Holtzman, Tanaka, Ullman, *Forking Paths* ([2412.07961](https://arxiv.org/abs/2412.07961)); ICLR 2025; Dec 2024 | Token-level nested resampling (K ≤ 10 × 30); change-point and survival analysis; GPT-3.5 | Token-level resampling; nested design | No hidden states, no R1, no variance decomposition, no ceiling |
| Bigelow, Zur, Grant et al., *Forking Fast* ([2608.19611](https://arxiv.org/html/2608.19611v1)); NeurIPS 2026 listing; Aug 2026 | Efficient forking estimation; noise is multinomial sampling noise; R1-Distill-Llama-8B | Noise floor; token-level arm | No history/sentence decomposition; no hidden-state prediction |
| Zur, Geiger, Lubana, Bigelow ([2511.04527](https://arxiv.org/html/2511.04527)); preprint; Nov 2025 | Probes predict the outcome distribution; steerability tracks uncertainty; 4 examples, Llama-3.2-3B | Probes read uncertainty | Tiny n; no importance target; no ceiling |
| Setlur et al., *PAV* ([2410.08146](https://arxiv.org/abs/2410.08146)); venue unverified; Oct 2024 | Advantage as progress; base-policy advantage adds no signal | Signed R²_max = 0 | Not applied to interpretability probes; no unsigned ceiling |
| Duszenko, *Sycophantic Anchors* ([2601.21183](https://arxiv.org/html/2601.21183)); preprint; Feb 2026 | Post-emission probes regress anchor strength (R² 0.48–0.74) | Probes | Post-emission; small rollout counts (20); no ceiling |
| Boppana et al., *Reasoning Theater* ([2603.05488](https://arxiv.org/html/2603.05488)); preprint; Mar 2026 | Attention probes vs forced answering vs CoT monitor; R1 and distills incl. 14B | Forced answering; prefix probes | MCQ only; no resampled-accuracy target |
| Xiong et al., *Faithfulness of Thinking Drafts* ([2505.13774](https://arxiv.org/html/2505.13774v2)); NeurIPS 2025 | Step insertion and draft perturbation; immediate answering changes about 30% of answers; incl. R1-Distill-14B | Forced-answer pilot; intervention arms | GPQA and MMLU only; no resampling comparison |
| Yang et al., *DEER* ([2504.15895](https://arxiv.org/html/2504.15895)); preprint; Apr 2025 | `\boxed` inducer at transitions, confidence-gated early exit | Forced-answer pilot | Never measures agreement with the continuation distribution |

## Per-arm novelty and what to change

| Arm | Status | What to change |
|---|---|---|
| Reliability audit (ICC, split-half, entropy floor, BH) | **Novel** | Cite Du et al. for split-half ceilings and Forking Fast for multinomial noise. Report results stratified by accuracy from start and by answer entropy. Report with and without problem 1591. |
| Signed R²_max = 0 | **Already done** (as an identity) | Present it as the zero-mean-advantage identity, citing PAV. The contribution is its consequence: any pre-emission probe of signed importance is predicting noise. |
| Nested sampling K × M = 200 for unsigned R²_max | **Novel** (structural precedent in Bigelow) | Fit a three-level variance model with explicit binomial within-cell correction. Without it, unsigned importance is biased upward and the ceiling downward. Choose positions stratified by entropy, and include second-source problems. |
| Residual probes / teacher-forced all-token and span activations | **Partially anticipated** (Nikankin; Sycophantic Anchors; Zhang; Yuan) | Run a Nikankin-style pooled-span probe on our traces as a baseline. Separate pre- from post-emission positions. Benchmark every probe against R²_max and an entropy-only baseline. |
| Token-level resampling around a key phrase | **Partially anticipated** (Bigelow; Forking Fast; Thought Anchors' own token-vs-sentence claim) | Frame it as a test of whether *reliable* sentence effects localise to tokens. Reuse Forking Fast's change-point pooling to cut cost. Keep it small. |
| Resample vs delete vs paraphrase × 2 temperatures | **Partially anticipated** (Thought Branches) | Same sentences across all arms. Pre-register the Thought Branches prediction. Give each arm its own noise floor. The temperature factor is the clearest new element. |
| 256-token short-budget readout | **Novel**, but at risk of failing | Sweep K (e.g. 0, 64, 256, 1024) and report the outside-set rate as a function of K. Convergence after about 60% of steps suggests 256 tokens may be too short. Pilot it on a few problems before scaling. |
| Forced-answer pilot (completed) | **Method already done; measurement novel** | Publish it as a negative result, positioned against DEER, Answer Convergence and Xiong et al.'s roughly 30% answer-change figure. |
| Rerun with released prompt format | **Novel** (no public report), modest weight | Design it to separate prompt format (collapsed newlines) from provider effects. The repo defaults to the Novita completions endpoint ([generate_rollouts.py](https://github.com/interp-reasoning/thought-anchors/blob/main/generate_rollouts.py)), and issue #4 shows API failures already contaminated the data once ([issue #4](https://github.com/interp-reasoning/thought-anchors/issues/4)). |
| AIME 2025 second source | **Already standard practice** (Du et al. use AIME 24/25/26) | Necessary but not novel. Pool AIME 2025 with AIME 2026 and BRUMO 2025 to raise the yield of intermediate problems. Screen with many samples, not 10. |
| Receiver heads and attention suppression | **Already done** (replication) | Lowest priority. The only new question is whether these scores track debiased importance or the entropy floor. Check issue #7 before reusing `ablation.py`. |
| Logit lens | **Partially anticipated** (Li et al. 2026, same model and benchmark) | Score candidate answer sets at sentence boundaries and validate against resampled accuracy, not ground-truth probability. Cheap, so run it, but frame it as an extension. |

## Methodological critiques and limits (C, D)

**Verified.** Searches of arXiv, LessWrong, the Alignment Forum and GitHub found no published critique of Thought Anchors' importance measure. They found nothing on entropy dominance, finite-sample TV/KL bias, similarity-threshold sensitivity or prompt-format sensitivity ([LessWrong search hits](https://www.lesswrong.com/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter)). The reported data problems are API-error rollouts counted as incorrect, since fixed ([issue #4](https://github.com/interp-reasoning/thought-anchors/issues/4)), and the open head-ablation hook report ([issue #7](https://github.com/interp-reasoning/thought-anchors/issues/7)). Thought Anchors' counterfactual importance conditions on resampled sentences with cosine below 0.8 under all-MiniLM-L6-v2 ([arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)). Thought Branches uses a different embedder, a "below median" threshold and temperature 0.7 ([arXiv 2510.27484v1](https://arxiv.org/html/2510.27484v1)). Neither reports sensitivity to these choices.

**Inference.** The critique space is unoccupied. Our entropy-floor, debiasing, prompt-format and provider findings are the first public methodological critique, and they should be framed that way.

The overall novelty threats (D) are bounded and identifiable:

- Du et al. for noise ceilings.
- Nikankin et al. for activation-based importance, with a result that conflicts with ours.
- Li et al. for logit lens.
- PAV and textbook RL for the signed identity.
- Thought Branches for on- versus off-policy edits.
- Bigelow and Forking Fast for token-level forking.

None of these pre-empts the combination of a reliability audit and a resampling-derived predictability ceiling.

The limits are internal rather than external: one model, 20 problems, an easy and partly memorised set, and a 0.11 accuracy gap on incorrect traces between released and regenerated data. That gap means part of any discrepancy with the original paper may come from the generation pipeline rather than the method. Thought Branches deliberately selects scenarios with "considerable variability in final answers" ([arXiv 2510.27484](https://arxiv.org/html/2510.27484)), which is the right instinct, but it shows that the field has not standardised how to select problems.

## Prioritised recommendations

These recommendations are inferences that build on the verified facts above.

**GPU plan, in order.**

1. Before spending GPU time, run a no-reasoning (forced immediate answer) accuracy check on all 20 problems, not just 1591. Ideally also run the Wu et al. partial-prompt completion test with the Qwen2.5-14B base. Tag memorised items, and report every result with and without them.
2. The nested K × M = 200 arm is the flagship. Its unsigned R²_max is the one result with no precedent. Spend most compute here, with explicit binomial-noise correction and entropy-stratified positions, on both MATH and second-source problems.
3. Build the second source from AIME 2025 plus AIME 2026 and/or BRUMO 2025. MathArena puts the model at 49% and 68% on AIME 2025 and BRUMO 2025 ([MathArena](https://matharena.ai/models/deepseek_deepseek_distill_14b)), so these sit near the middle of the accuracy range. Screen with many samples so that "intermediate difficulty" is real. Replicate the reliability audit there: if real effects rise well above 2% on AIME, that is itself a headline.
4. Run resample vs delete vs paraphrase on the same sentences, with per-arm noise floors. The temperature factor is the defensible novelty.
5. Run the released-prompt rerun as a 2×2 of prompt format against generation path. It is cheap, and it closes the loophole that our discrepancies come from the pipeline.
6. Pilot the short-budget readout with a K sweep before committing. Drop it if the outside-set rate stays near the pilot's 78% at 256 tokens.
7. Fold the teacher-forced activation probes and logit lens into the R²_max comparison, including a Nikankin-style pooled-span baseline. They are cheap once activations are cached.
8. Shrink the token-level arm and use Forking Fast-style pooling.
9. Run receiver heads and suppression last, as a replication against debiased importance, after resolving issue #7.

**Framing.**

- Lead with a unified claim: resampling importance on the released data is mostly entropy-driven noise, and the predictable part of importance is bounded by a measurable ceiling.
- State the signed result as a known identity with an overlooked consequence. Present the unsigned ceiling as the new quantity.
- Explicitly reconcile our probe result with Nikankin et al. (different target, pre- vs post-emission, entropy controls).
- Position MATH as "the audited released dataset" and AIME as the generalisation test, not the reverse.
- Cite Du et al., Thought Branches, Li et al., PAV, Bigelow and Forking Fast up front, so reviewers see we know the neighbourhood.

## Items that could not be verified

- The ICLR 2026 decision on Thought Anchors, and any OpenReview reviews of it or of Thought Branches: OpenReview was blocked. Reviewers may already have raised variance concerns.
- PAV's venue (believed to be ICLR 2025) and Wu et al.'s AAAI 2026 publication. The latter is suggested by an AAAI OJS listing but was not confirmed.
- Bigelow's Appendix C.2 claim of minimal correlation between forking and entropy (seen only via fetch summary).
- The third-party claim that Thought Anchors importance is high-variance when fewer than 10 resampled sentences diverge.
- Details of Thought Anchors' token-level vs sentence-level comparison.
- Whether published Thought Anchors figures depend on the `ablation.py` hook in issue #7.
- Whether the 20 problems fall inside MATH-500, and their MATH levels.
- Direct memorisation measurements for R1-Distill-Qwen-14B. These exist only for 7B Qwen2.5 models.
- R1-Distill release date and base-model lineage (from background knowledge only).
- Datasets used by AttriCoT (arXiv 2606.21821) and Forking Fast.
- Whether Zur et al. report pre- vs post-emission probe performance separately.
- Du et al.'s appendix, in particular whether they relate noise to entropy.
- Newline handling in `generate_rollouts.py`, and Novita's model build or quantisation.
- Background-only citations not re-fetched:
  - Math-Shepherd (2312.08935);
  - Turpin et al. (2305.04388);
  - Chen et al. (2505.05410);
  - Arcuschin et al. details;
  - Tutek et al. FUR (2502.14829);
  - Orgad et al.;
  - tuned lens and Patchscopes specifics;
  - the NoThinking prefill string;
  - the Qwen3 budget suffix;
  - the source of "most high-entropy tokens are not planning tokens".
- Two deletion-based CoT-pruning papers (arXiv 2603.00296, 2509.12464), which surfaced but were not inspected.
- Unpublished critiques on X, Slack or in MATS channels.
- Exhaustive PRM literature on fixed-length partial rollouts, relevant to the short-budget arm's novelty.

## Conclusion

The literature has converged on resampling as the gold standard for step importance (Thought Branches' title says so outright), yet nobody has checked whether the gold standard is reliable. That asymmetry is the programme's opening. The finding that most measured importance is entropy-driven noise is more consequential than any single new probe or attention result. It retroactively qualifies every probe and attention paper that validates against resampling scores, Thought Anchors' own suppression correlation (ρ = .20) included.

The real risk is not being scooped on the core. It is diluting the core with peripheral arms that are already done, and drawing general conclusions from an easy, partly memorised MATH subset. Concentrating GPU time on the nested ceiling and on a properly screened uncontaminated second source turns the audit from "the original dataset was noisy" into a general, falsifiable statement about how much of reasoning-step importance is predictable at all.
