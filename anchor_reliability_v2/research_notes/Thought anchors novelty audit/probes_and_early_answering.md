# Reading out a reasoning model's eventual answer or correctness mid-CoT: probes, forced early answering, short-budget continuations, logit/tuned lens

Scope: prior and concurrent work through Oct 2026, compared with our study: DeepSeek-R1-Distill-Qwen-14B on MATH. Our arms are (a) a forced-answer pilot that failed: 78% of forced answers fell outside the set of answers that continued reasoning reaches, and the readout did not track resampled accuracy; (b) a planned short-budget readout (continue about 256 tokens, then force closure); (c) residual probes, teacher-forced all-token activations and sentence-span pooling; (d) logit lens at sentence boundaries.

Verification legend: **[V]** means the claim was checked this session against the arXiv abstract or HTML full text (via a summarising fetch, so exact wording or numbers can carry small paraphrase errors). **[BG]** means it comes from background knowledge and was not re-fetched this session, so verify before citing.

## Q1. Probing correctness or the eventual answer from hidden states during reasoning

### Takeaway
Several works decode either the correctness of an intermediate answer or the eventual final answer from hidden states mid-CoT. The reported numbers are strong: AUROC 0.7 to above 0.9, calibrated, with "lookahead" before the answer is written. But nearly all label correctness per chunk or per trace against ground truth, or decode the final answer of a single trace. None that I found targets *resampled accuracy from a prefix* (the thought-anchors quantity), and none controls for position or entropy the way our negative result does. The closest concurrent work is Nikankin et al. 2026, which probes step importance on R1-Distill-Qwen-14B and MATH-500 using mean-pooled step activations.

### Cited Findings
- **Zhang et al. 2025, "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"** [V]. arXiv 2504.05419 (submitted 7 Apr 2025). Authors: Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He. The arXiv ID matches. — [arXiv abs](https://arxiv.org/abs/2504.05419)
  - Models: R1-Distill-Llama-8B and -70B, R1-Distill-Qwen-1.5B, -7B and -32B, and QwQ-32B. **R1-Distill-Qwen-14B is not included.** Data: GSM8K, MATH, AIME and KnowLogic. — [arXiv HTML v1](https://arxiv.org/html/2504.05419v1)
  - Method: traces are split into paragraphs on "\n\n". Chunks are grouped into reasoning paths at keywords ("wait", "double-check", "alternatively", "verify"...). Intermediate answers are labelled correct or incorrect by Gemini 2.0 Flash. The probe is a two-layer MLP with weighted BCE; many probes converge to linear (hidden size 0). — [arXiv HTML v1](https://arxiv.org/html/2504.05419v1)
  - Results: ROC-AUC above 0.7 across datasets, ECE below 0.1, and above 0.9 ROC-AUC on AIME for R1-Distill-Qwen-32B. MATH and GSM8K probes transfer to each other. Transfer from easy datasets to AIME degrades calibration, and transfer from logic to math is unstable. — [arXiv HTML v1](https://arxiv.org/html/2504.05419v1)
  - Lookahead: correctness is decodable "even before an intermediate answer has been explicitly generated". At the 10% position within a chunk, accuracy is about 14 points below peak (peak 79%); at the 95% position it is 5 points below. — [arXiv HTML v1](https://arxiv.org/html/2504.05419v1)
  - Early exit: at threshold 0.85, accuracy is about the same (88.2%) with about 24% fewer tokens; at threshold 0.9, 88.6% with 19% fewer. — [arXiv HTML v1](https://arxiv.org/html/2504.05419v1)
- **Afzal et al. 2025, "Knowing Before Saying: LLM Representations Encode Information About CoT Success Before Completion"** [V]. arXiv 2505.24362 (30 May 2025). A probe on representations predicts CoT success "even before a single token is generated". It beats a BERT baseline on the generated text. "Using later reasoning steps does not always improve classification." Truncated CoT beats no CoT but loses to full CoT. — [arXiv abs](https://arxiv.org/abs/2505.24362)
- **Boppana et al. 2026, "Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"** [V]. arXiv 2603.05488 (5 Mar 2026; revised 28 May 2026; Goodfire-affiliated authors including Atticus Geiger and Jack Merullo). — [arXiv abs](https://arxiv.org/abs/2603.05488)
  - Models: DeepSeek-R1 671B and GPT-OSS 120B, plus R1-Distill 1.5B, 7B, 14B and 32B. Data: MMLU and GPQA-Diamond (multiple choice only, not MATH).
  - Method: per-layer attention probes (learned pooling), trained on random prefixes. They are compared with forced answering and a CoT monitor. The forced-answer suffix for R1 is `</think>\n{\n "answer": "` (temperature 0, logits over A–D).
  - On MMLU, probes and forced answering both predict the final answer far earlier than a CoT monitor does. On GPQA-D, all three start near chance and rise together ("genuine" reasoning).
  - "Performativity rate" for R1: 0.417 on MMLU and 0.012 on GPQA-D.
  - Probes are well calibrated. Early exit at 95% confidence keeps 97% of MMLU performance with 68% fewer tokens; at 80% confidence on GPQA-D, 50% fewer tokens with 97% of performance retained.
  - Probe and forced answers disagree inconsistently: forced answering sometimes wins on MMLU, and both often err together early on GPQA-D.
  - Larger models show more performative reasoning on easy tasks. — [arXiv HTML](https://arxiv.org/html/2603.05488)
- **Yuan et al. 2026, "Hidden Error Awareness in CoT Reasoning: The Signal Is Diagnostic, Not Causal"** [V]. arXiv 2605.09502 (10 May 2026; Mechanistic Interpretability Workshop @ ICML 2026). Models: Qwen, Llama and Phi (1.5B–72B) plus DeepSeek-R1.
  - A linear probe predicts trace correctness at 0.95 AUROC, and at 0.79 AUROC from the first reasoning step.
  - Verbalized confidence barely separates the classes: 4.55/5 for incorrect traces vs 4.87/5 for correct ones.
  - Steering, probe-guided best-of-N, self-correction and patching all fail to fix errors. — [arXiv abs](https://arxiv.org/abs/2605.09502)
- **Nikankin, Tutek, Ashuach, Rosenfeld, Belinkov 2026, "Reasoning Models Know What's Important, and Encode It in Their Activations"** [V]. arXiv 2604.18307 (20 Apr 2026; revised 11 Jun 2026). — [arXiv abs](https://arxiv.org/abs/2604.18307)
  - Models: GPT-OSS-20B, **R1-Distill-Qwen-1.5B, -7B and -14B**, and OLMo-3-7B-Think. Data: HARP and **MATH-500**.
  - "Importance" here means removability (sufficiency and attainability), **not** resampling-based counterfactual importance.
  - Activations are averaged over each step's tokens and flattened across layers, then fed to a 2-layer MLP probe. The probe reaches 80% or higher accuracy vs 60% or lower for an LLM judge.
  - Probe predictions show low correlation with surface features (position, length). Importance is encoded before later steps are generated, and probes transfer across models. — [arXiv HTML](https://arxiv.org/html/2604.18307)
- **Cox, Kianersi, Garriga-Alonso 2026, "Post-Hoc Reasoning in CoT: Decoding and Steering Pre-Committed Answers"** [V]. arXiv 2603.01437 (v2 23 Jul 2026; OpenReview forum UMUYpeXtJQ).
  - Probes at the last pre-CoT token reach AUC above 0.9 on most binary tasks for Gemma-2 and Qwen-2.5 instruct models (0.69–0.88 on Logical Deduction). Steering flips answers.
  - **For the reasoning model GPT-OSS-20B (appendix), probe AUC is "considerably lower" and steering effects are near negligible.** This suggests reasoning models compute during CoT rather than pre-committing. — [arXiv HTML](https://arxiv.org/html/2603.01437); [OpenReview](https://openreview.net/forum?id=UMUYpeXtJQ)
- **Qian, Liu et al. 2025, "Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning"** [V, abstract-level]. arXiv 2506.02867, NeurIPS 2025. Mutual information between intermediate representations and the correct answer spikes suddenly at certain steps. The peaks fall on "thinking tokens" such as "Hmm", "Wait", "So" and "Therefore". Higher MI bounds error lower. — [arXiv](https://arxiv.org/abs/2506.02867); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119169)
- **LessWrong/AF post "Understanding reasoning with thought anchors and probes"** [V]. Authors: JeaniceK, Matthew Robbins, Johannes Taraz. Model: R1-Distill-Llama-8B on synthetic legal cases.
  - Probes predict the "early stopping logit difference" (guilty vs innocent) at each sentence position.
  - Attention probes reach 95% sign-match on held-out cases; linear, EMA and multimax probes underperform.
  - Trajectories are non-monotonic: the model "can change their mind late". — [LessWrong](https://www.lesswrong.com/posts/f9cnGHCiJgo5eDkSQ/understanding-reasoning-with-thought-anchors-and-probes)
- **Orgad et al. 2024, "LLMs Know More Than They Show"** [BG]. arXiv 2410.02707, ICLR 2025. Truthfulness signal concentrates at the exact-answer tokens. Probes detect errors, but generalization across tasks is poor. The study covers non-reasoning models with short answers, not long CoT. — [arXiv](https://arxiv.org/abs/2410.02707)

### Inferences
- **Overlap with our probe arm.** Zhang et al. and Yuan et al. establish that correctness is decodable mid-trace, but their labels are per-trace or per-chunk ground-truth correctness. That is largely a *between-problem* or *between-trace* signal, which difficulty and entropy can explain. Our finding is that probes decode entropy but not within-trace accuracy, and that answer identity is explained by position plus entropy. That result is consistent with, and sharper than, these papers: none of them reports within-trace or prefix-level resampled-accuracy decoding with position and entropy controls. This is a plausible novelty claim, but it is inference.
- **Overlap with sentence-span pooling.** Nikankin et al. already use mean-pooled step activations on R1-Distill-Qwen-14B and MATH-500, with a surface-feature control. Their target, removability-based importance, differs from resampling importance and resampled accuracy. Our pooling arm must cite them and distinguish its target.
- Reasoning Theater's attention probes on random prefixes are the strongest existing "answer decoding over CoT position" method. It covers MCQ only, so a four-way softmax is trivial compared with open-ended MATH answers.

### Gaps
- I did not find a paper that probes for *resampled* (Monte Carlo) accuracy of a prefix on R1-style traces. Absence of evidence, not a proof.
- Orgad et al. details were not re-fetched.
- I could not verify a distinct paper titled "Do LLMs plan their answer before CoT", or "can we predict the answer from the first tokens", beyond Afzal et al. and Cox et al.

## Q2. Early exit and answer convergence: suffixes, and the quality or prematurity of forced intermediate answers

### Takeaway
The early-exit literature routinely forces trial answers mid-trace, using a `</think>` or `\boxed` suffix, and reports that many traces contain an early point where the forced answer is already correct. DEER reports about 75% "pearl reasoning". But all of these methods *gate* on confidence or on consecutive-answer stability precisely because raw forced answers are unreliable. None reports the fraction of forced answers that fall outside the continuation's answer set, so our 78% figure is a new kind of measurement.

### Cited Findings
- **DEER, Yang et al. 2025, "Dynamic Early Exit in Reasoning Models"** [V]. arXiv 2504.15895 (22 Apr 2025; revised 28 Sep 2025). Authors: Chenxu Yang, Qingyi Si et al.
  - Method: at transition points ("Wait", or entropy above 0.672), the transition word is replaced with an answer inducer that contains `\boxed{`. Confidence is the geometric mean of the max token probabilities over the trial answer, with threshold λ = 0.95.
  - Results on R1-Distill-Qwen-7B, MATH-500: 87.4% → 89.8% accuracy, 3,858 → 2,143 tokens.
  - Pearl reasoning: about 75% of samples contain an early exit point that yields a correct answer, and 36.7% stay correct with less than 50% of the steps. On MATH-500, 60.8% of correct answers survive exit at 20% of the reasoning.
  - Across 11 models and 10 benchmarks: 19.1–80.1% shorter CoT and +0.3 to +5.0% accuracy. — [arXiv abs](https://arxiv.org/abs/2504.15895); [HTML](https://arxiv.org/html/2504.15895)
- **Liu & Wang 2025, "Answer Convergence as a Signal for Early Stopping in Reasoning"** [V]. arXiv 2506.02536, EMNLP 2025 main.
  - Method: the CoT is split into sentences (NLTK); `</think>` is appended after each chunk and the answer is greedy-decoded.
  - On math, models converge to their final answer after about 60% of the steps (Answer Convergence Ratio, measured on R1-Distill-Qwen-32B).
  - They propose three methods: Answer Consistency (stop after k = 10 identical consecutive answers), Think-Token logit boosting (α = 0.6), and Learn-to-Stop (an LSTM probe on hidden states, labelled by convergence rather than ground truth).
  - GSM8K with QwQ-32B: 44% fewer tokens, -0.2% accuracy. NQ: more than 40% fewer tokens and higher accuracy.
  - In a manual GPQA check, 3% of outputs were unfaithful. — [arXiv HTML v2](https://arxiv.org/html/2506.02536v2); [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.904)
- **ES-CoT, Mao et al. 2025** [V]. arXiv 2509.14004 (Sep 2025). Appends "The final answer is" after each step and stops when the run length of identical step-answers jumps significantly (d_min = 10). Models: QwQ-32B, Qwen3-8B and R1-Distill-Llama-8B. About 41% fewer tokens with comparable accuracy. — [arXiv HTML](https://arxiv.org/html/2509.14004v1)
- **s1, Muennighoff et al. 2025** [V]. arXiv 2501.19393. Budget forcing appends the end-of-thinking delimiter "and optionally 'Final Answer:'" at the maximum budget. To extend thinking, it suppresses the delimiter and appends "Wait". Extending more than 6 times causes repetitive loops. — [arXiv HTML](https://arxiv.org/html/2501.19393)
- **NoThinking, Ma et al. 2025, "Reasoning Models Can Be Effective Without Thinking"** [V]. arXiv 2504.09858. NoThinking prefills a fabricated thinking block. Model: R1-Distill-Qwen-32B. It uses 2.0–5.1× fewer tokens, lags at pass@1, and catches up or surpasses at pass@k. Under budget forcing it emits "Final Answer:" at the limit. — [arXiv HTML](https://arxiv.org/html/2504.09858)
  - [BG, unverified] The exact prefill string is "Okay, I think I have finished thinking.</think>".
- **Certainty-Guided Reasoning, Nogueira et al. 2025** [V, abstract]. arXiv 2509.07820. Periodically checks answer-token probabilities and stops at a certainty threshold. Evaluated on AIME2025 over 64 seeds. — [arXiv abs](https://arxiv.org/abs/2509.07820)
- **Reasoning Theater** (see Q1) uses `</think>\n{\n "answer": "` as its forced suffix. On GPQA-D, forced answers are near chance early and frequently wrong together with the probe. — [arXiv HTML](https://arxiv.org/html/2603.05488)
- **The suffix "Considering the limited time by the user, I have to give the solution based on the thinking directly now."** An exact-phrase web search returned no primary source. [BG] I believe it comes from Qwen3's thinking-budget recipe; unverified.
- **ThinkPrune** [BG]. arXiv 2504.01296. RL with length limits; not a readout method. Not re-fetched.

### Inferences
- **Forced-answer pilot.** The pilot reproduces DEER and Answer Convergence style elicitation; our `</think>\n\n**Final Answer**\n\[\n\boxed` is essentially DEER's `\boxed` inducer. These papers implicitly confirm that ungated forced answers are noisy: they need high confidence (0.95), or 10 identical consecutive answers, before trusting them. DEER's "60.8% of correct answers survive at 20%" is conditioned on the final answer being correct. It does not measure agreement with the *distribution* of continuations. Our "78% outside the continuation answer set" and "does not track resampled accuracy" measure a different property, namely representativeness for resampled outcomes, which as far as I found has not been reported.
- **Short-budget arm.** None of these papers continues for K tokens before forcing.

### Gaps
- None of these papers reports how often forced answers are "premature guesses" in our sense.
- I did not verify the Qwen3 suffix source or the NoThinking prefill string.

## Q3. Truncation and early-answering faithfulness tests

### Takeaway
Lanham et al. (2023) introduced early answering (truncate the CoT, then force an answer). R1-era replications mostly test other faithfulness facets (cue articulation, draft perturbation). For R1-Distill models, the answer stage itself performs new computation (about 30% answer changes on GPQA). That supports our finding that forced closure is not a faithful readout of what continuation would produce.

### Cited Findings
- **Lanham et al. 2023, "Measuring Faithfulness in Chain-of-Thought Reasoning"** [V]. arXiv 2307.13702 (17 Jul 2023).
  - Tests: truncation with early answering, adding mistakes, paraphrasing, and filler tokens.
  - Reliance on the CoT varies greatly by task. Larger models give "less faithful reasoning on most tasks". — [arXiv abs](https://arxiv.org/abs/2307.13702)
- **Chua & Evans 2025, "Are DeepSeek R1 and Other Reasoning Models More Faithful?"** [V]. arXiv 2501.08156 (14 Jan 2025; revised 15 Jul 2025). This is a **cue-articulation** test, not truncation. R1 articulates the cue's influence 59% of the time vs 7% for non-reasoning DeepSeek. — [arXiv abs](https://arxiv.org/abs/2501.08156)
- **Xiong, Chen, Qi, Lakkaraju 2025, "Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models"** [V]. arXiv 2505.13774, NeurIPS 2025.
  - Models: R1-Distill 7B, 8B, **14B** and 32B, QwQ-32B and Skywork-OR1-32B. Data: GPQA-D and MMLU.
  - Tests: intra-draft counterfactual step insertion, and draft-to-answer perturbation of the draft's conclusion.
  - "Immediate answering" changes the answer in about 30% of GPQA cases relative to standard answering. Draft-answer consistency ranges from 13.39% to 97.71%, and QwQ is lowest at 19.54% on GPQA. The answer stage "often introduces new computation". — [arXiv HTML v2](https://arxiv.org/html/2505.13774v2); [NeurIPS](https://neurips.cc/virtual/2025/poster/120231)
- **Li et al. 2026, "When Reasoning Traces Become Performative"** [V]. arXiv 2605.11746. See Q4: confabulated vs contradictory steps across R1-Distill 7B, 14B and 32B. — [arXiv HTML](https://arxiv.org/html/2605.11746)

### Inferences
- Early-answering curves (Lanham) are computed against the *full-trace* answer, so they share the weakness our pilot exposed: a forced answer can match or mismatch the final answer without reflecting the distribution of resampled continuations. The thinking-drafts result (about 30% answer changes from immediate vs standard answering, on R1-Distill-14B among others) is direct evidence that the closure format matters.

### Gaps
- I did not fetch Lanham's exact early-answering prompt or its AOC numbers.
- I found no R1-specific *truncation-curve* replication on MATH. [V-negative, search-based]

## Q4. Logit lens and tuned lens on reasoning models

### Takeaway
One concurrent paper, Li et al. 2026 (arXiv 2605.11746), applies the logit lens at every CoT step to R1-Distill-Qwen-7B, -14B and -32B on MATH-500 among other datasets. It cross-checks with Patchscopes and tuned lens. Our logit-lens arm overlaps heavily on the method and differs in its target: they score probability on the ground-truth answer token for commitment timing, whereas we want a readout of resampled accuracy or answer distribution.

### Cited Findings
- **Li, Yang, Mehta, Hazarika, Onoue 2026, "When Reasoning Traces Become Performative: Step-Level Evidence that CoT Is an Imperfect Oversight Channel"** [V]. arXiv 2605.11746 (12 May 2026).
  - Models: 9 open models including **R1-Distill-Qwen-7B, -14B and -32B** and the matched Qwen2.5-Instruct. Data: 7 benchmarks including **MATH-500**, with 60–300 instances each.
  - Method: logit lens at every even layer; the commitment proxy is the probability on the ground-truth answer token, thresholded at τ = 0.3. Validation: Patchscopes agree 75.6% on mid-depth steps, tuned-lens agreement is 99.2% or higher, and direction ablation is used.
  - Mean Belief-CoT Agreement is 0.619. 58% of mismatches are "confabulated" steps, where the model keeps writing after its latent answer has stabilized.
  - R1 distillation at 32B lowers confabulated steps from 57.6% to 37.4% and raises contradictory states from 7.1% to 28.9%.
  - Lower alignment correlates with larger CoT utility (r = -0.42).
  - Limitation: a single-token ground-truth proxy. — [arXiv HTML](https://arxiv.org/html/2605.11746)
- **Tuned lens, Belrose et al. 2023** [BG]. arXiv 2303.08112. — [arXiv](https://arxiv.org/abs/2303.08112)
- **Patchscopes, Ghandeharioun et al. 2024** [BG]. arXiv 2401.06102, ICML 2024. — [arXiv](https://arxiv.org/abs/2401.06102)
- A LessWrong sprint applied tuned and logit lens to CODI, a latent-CoT model, not R1. — [LessWrong](https://www.lesswrong.com/posts/c4LYmzEC6cFQzrDZz/latent-reasoning-sprint-1-tuned-lens-and-logit-lens-on-codi)

### Inferences
- Li et al.'s single-token, ground-truth-probability proxy is poorly suited to multi-token MATH answers and cannot measure *outcome distributions*. Two things in our arm remain novel: candidate-set scoring at sentence boundaries, and validation against resampled accuracy. Their 58% "confabulated" rate suggests the latent answer often stabilizes before the text does. That may make logit lens track *answer identity* while still failing on *accuracy*, mirroring our probe result. This is speculative.

### Gaps
- I found no paper applying tuned lens specifically to long R1 traces as a primary method, beyond Li et al.'s validation use.

## Q5. Short-budget continuation readouts ("continue K tokens, then force an answer")

### Takeaway
I found **no** paper that uses "continue about K tokens, then force closure" as a cheap estimator of resampled accuracy or of step importance on reasoning traces. Nearby practice falls into three groups:
- full-rollout Monte Carlo (Math-Shepherd, OmegaPRM, Thought Anchors);
- immediate forced answers (DEER, Answer Convergence, ES-CoT);
- classic truncated rollouts in game AI, where a value estimate replaces the tail.

That makes the short-budget arm the most novel component, pending a check of the PRM literature for "partial-rollout" variants.

### Cited Findings
- **Thought Anchors, Bogdan, Macar, Nanda, Conmy 2025** [V-partial]. arXiv 2506.19143. Counterfactual sentence importance comes from resampling replacement sentences and *continuing the CoT to completion*. — [arXiv](https://arxiv.org/abs/2506.19143v4)
- **Truncated rollouts in classic MC search** [V, secondary]. A rollout runs only a few steps and substitutes a value estimate for the outcome, giving "at least an order of magnitude speed-up" and lower variance. — [emergentmind summary](https://www.emergentmind.com/topics/monte-carlo-rollouts)
  - The original source is Tesauro & Galperin, "On-line Policy Improvement using Monte-Carlo Search" — [arXiv 2501.05407](https://arxiv.org/pdf/2501.05407)
- **PRM Monte Carlo annotation uses multiple full rollouts per step from a completer.** — [SCAN, OpenReview](https://openreview.net/pdf/90e683c8f88b8fe14fd41acb175f70e1cbcdd305.pdf)
  - [BG] Math-Shepherd (arXiv 2312.08935, ACL 2024) and OmegaPRM (arXiv 2406.06592) do the same.
- **Afzal et al.** report that truncated CoT still beats no CoT but loses to full CoT. — [arXiv](https://arxiv.org/abs/2505.24362)
- **Budget-forcing papers study accuracy as a function of the total budget.** Examples: s1, NoThinking, and Budget-Constrained Accuracy in [2506.11274](https://arxiv.org/pdf/2506.11274). They do not estimate accuracy from a fixed prefix plus a K-token continuation.

### Inferences
- A 256-token continuation interpolates between DEER-style immediate forcing, which our pilot found unrepresentative, and full resampling. It is the LLM analogue of a truncated rollout. One risk follows from the convergence results: models converge only after about 60% of the steps (Liu & Wang), and late answer changes occur (the LessWrong legal-probe post). So 256 tokens may often be too short to leave the "premature guess" regime, and the arm should report the outside-set rate as a function of K. This is inference.

### Gaps
- I searched for a PRM paper that uses fixed-length partial rollouts plus forced answers, or a K-token continuation estimator compared with full MC, but did not exhaustively check the PRM literature (e.g., "lookahead"-limited PRM labelling). This should be flagged as "not found", not "does not exist".
