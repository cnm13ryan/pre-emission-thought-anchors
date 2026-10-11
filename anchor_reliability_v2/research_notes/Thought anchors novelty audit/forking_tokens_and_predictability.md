# Forking Tokens, Critical Tokens, Variance Decomposition and Pre-Emission Predictability: Prior and Concurrent Work (through Oct 2026)

Scope note: this audit used about 18 search and fetch calls. "Verified" means I read the claim on the arXiv abstract or HTML page, or on the venue page, during this session (abstracts were read through a summarising fetch tool, so exact wording may differ slightly). Items marked [background, not fetched] come from prior knowledge and should be checked before they are cited in a paper.

Our arms, for reference: (A1) nested sampling, with K candidate sentences per position and M=200 continuations each, on R1-Distill-Qwen-14B MATH CoTs; (A2) a variance decomposition of importance into a history part and a sampled-S part, giving R²_max, with signed R²_max = 0 by the tower property and unsigned R²_max about 0.5; (A3) residual-stream probes, which predict entropy and uncertainty but not importance; (A4) token-level resampling before and after key phrases in anchor sentences.

## Q1. Forking paths and forking tokens (Bigelow et al. and follow-ups)

### Takeaway
The ID is confirmed: Bigelow et al., "Forking Paths in Neural Text Generation", arXiv 2412.07961, is an ICLR 2025 poster. It runs token-level resampling with outcome distributions, Bayesian change-point detection and survival analysis on GPT-3.5, with no hidden states. Its follow-ups are Zur et al. (Nov 2025), which probes hidden states for the outcome distribution, and "Forking Fast" (Aug 2026), which makes the resampling more efficient. Neither follow-up does a nested variance decomposition, and neither builds a ceiling on predicting forking before emission.

### Cited Findings
- **Bigelow, Holtzman, Tanaka, Ullman, "Forking Paths in Neural Text Generation"**: arXiv 2412.07961, submitted 10 Dec 2024. It was an ICLR 2025 poster. The paper hypothesises "forking tokens", where resampling at that token but not at others leads to very different outcomes. The method is model-agnostic and needs no weight access. — [arXiv](https://arxiv.org/abs/2412.07961); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30769)
- Method details from the HTML:
  - Model: GPT-3.5 (gpt-3.5-turbo-instruct-0914). Gemini Flash extracts the final answers.
  - At each token, the top K≤10 alternative tokens with ≥5% probability are taken, with 30 completions per alternative token.
  - Outcomes are mapped to semantic answer vectors, weighted by token probability.
  - Change points are found with Bayesian CPD (the BEAST algorithm), and a discrete-time hazard and survival analysis estimates whether the base path "survives".
  - Cost is about 1M tokens (about $2) per analysed sample.

  — [arXiv HTML](https://arxiv.org/html/2412.07961)
- Tasks: 7 tasks in 4 domains, namely CoinFlip, LastLetter, AQuA, GSM8k, MMLU, HotpotQA and StoryCloze.
  - Change points are most frequent in LastLetter (63%) and least frequent in StoryCloze (7%).
  - Final survival S(T) < 0.2 for GSM8k and MMLU.
  - Forking tokens include punctuation, for example a space instead of a colon.

  — [arXiv HTML](https://arxiv.org/html/2412.07961)
- Bigelow et al. do not analyse hidden states and list it as future work. According to the fetch summary, their Appendix C.2 shows minimal correlation between forking and token entropy or probability (the appendix was not read directly; see Gaps). — [arXiv HTML](https://arxiv.org/html/2412.07961)
- **Zur, Geiger, Lubana, Bigelow, "Are language models aware of the road not taken? Token-level uncertainty and hidden state dynamics"**: arXiv 2511.04527, 6 Nov 2025, arXiv preprint. — [arXiv](https://arxiv.org/abs/2511.04527)
  - Main model is Llama-3.2-3B-Instruct, compared against Gemma-2-2B-Instruct.
  - Only four examples were analysed, from GSM8k, AQuA and GPQA, chosen so that the modal answer appears 4–6 times out of 10.
  - Linear probes predict the outcome distribution o_t at each token. The KL loss is 0.11 from the model's own activations versus 0.19 from another model's embeddings at layer 8.
  - Steerability correlates with uncertainty: R = 0.57 in the main example and 0.64 averaged over the four examples.
  - Per the fetch summary, they do not predict which tokens will become forking tokens from pre-emission states.

  — [arXiv HTML](https://arxiv.org/html/2511.04527)
- **Bigelow, Zur, Grant et al., "Forking Fast: Efficiently Estimating Uncertainty Dynamics in Text Generation"**: arXiv 2608.19611, Aug 2026. — [arXiv HTML](https://arxiv.org/html/2608.19611v1)
  - Setup: Llama-3-8B-Instruct and DeepSeek-R1-Distill-Llama-8B on 100 tinyMMLU questions, about 1.77B tokens in total.
  - Method: PELT change-point detection plus kernel-weighted pooling between change points, giving a 3.3–5× gain in effective sample efficiency.
  - Finding: the noise in uncertainty dynamics is largely multinomial sampling noise, falling as 1/√S, "rather than an LLM's sensitivity to each individual token". The outcome distribution is piecewise stationary, with sharp jumps at forks.
  - It does not predict forks from hidden states, and it does not decompose variance beyond the sampling-noise model.
- **Macar, Bogdan, Rajamanoharan, Nanda, "Thought Branches: Interpreting LLM Reasoning Requires Resampling"**: arXiv 2510.27484, 31 Oct 2025, revised 13 Apr 2026. It argues that interpreting a single CoT sample is inadequate and resamples from partial prefixes. Its case studies cover self-preservation statements, CoT steering, regeneration of removed steps, and unfaithful CoT. — [arXiv](https://arxiv.org/abs/2510.27484)

### Inferences
- Bigelow's design is a two-level nested sample: K alternative tokens at a position, then 30 continuations each. It is structurally the token-level analogue of our K×M design, so arm A4 is closest to Bigelow and Forking Fast. Arm A4's distinct contribution would be (a) applying it to R1-Distill-Qwen-14B anchor sentences, (b) placing it before and after a key phrase, and (c) linking it to sentence-level importance.
- Forking Fast's 1/√S result supports our M=200 choice. It also warns that importance estimated with few continuations is dominated by sampling noise. Our R²_max estimate must be corrected for within-cell binomial noise; otherwise unsigned importance will be biased upward and R²_max downward.
- No forking-paths paper found frames a "history versus sampled token" law-of-total-variance decomposition, or an R²_max ceiling on pre-emission prediction. This is inference from absence across these three papers.

### Gaps
- I did not read Bigelow's Appendix C.2 directly. The "minimal correlation with entropy" claim comes from the fetch summary, so check it before citing.
- Whether Zur et al. report probe performance specifically at pre-emission positions versus post-emission positions was not stated in what I read.

## Q2. Critical tokens, high-entropy "forking" tokens, and rollout-based token/step values

### Takeaway
"Critical tokens" (cDPO) and RLVR "forking tokens" (the 80/20 paper) both concern which tokens matter. cDPO identifies critical tokens by contrastive likelihood, and the 80/20 paper defines forking tokens purely by entropy. Neither paper's main pipeline estimates importance from resampling at the sentence level, and neither tests how importance relates to entropy. Rollout-based step values (Math-Shepherd, OmegaPRM, VinePPO, PAV) are Monte Carlo estimates of V or Q, the same quantity as our acc(h).

### Cited Findings
- **Lin et al., "Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability" (cDPO)**: arXiv 2411.19943, Nov 2024.
  - Critical tokens are defined as tokens that lead to incorrect trajectories; forcing other tokens there tends to produce correct outcomes.
  - The method identifies them automatically by contrasting the likelihoods of positive and negative fine-tuned models, then weights token-level DPO by that difference.
  - Evaluated on GSM8K and MATH500 with Llama-3 8B/70B and deepseek-math-7B.

  — [arXiv](https://arxiv.org/abs/2411.19943v1)
- **Wang et al. (Qwen team / Tsinghua LeapLab), "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning"**: arXiv 2506.01939, June 2025, NeurIPS 2025 poster. — [arXiv](https://arxiv.org/pdf/2506.01939); [NeurIPS](https://neurips.cc/virtual/2025/poster/115123)
  - About 20% of CoT tokens have high entropy and are called "forking tokens". Over 50% of tokens have entropy below 1e-2, and the 80th-percentile threshold is 0.672.
  - Restricting policy gradients to the top-20% entropy tokens matches full updates on Qwen3-8B. On Qwen3-32B it improves AIME'25 by 11.04 and AIME'24 by 7.71. Training only on the bottom 80% hurts.

  — [HF paper page](https://huggingface.co/papers/2506.01939); [arXiv HTML](https://arxiv.org/html/2506.01939v2)
- Counterpoint from the RL literature: "while a majority of planning tokens exhibit high entropy, most high-entropy tokens are not planning tokens". This was surfaced in search results; I did not confirm which paper in the result set (likely 2509.03646, "Emergent Hierarchical Reasoning in LLMs through RL") makes it. — [search result set incl. arXiv 2509.03646](https://arxiv.org/pdf/2509.03646)
- **OmegaPRM** (arXiv 2406.06592): divide-and-conquer MCTS that finds the first error in a CoT by binary search over Monte Carlo rollouts. It collects more than 1.5M process annotations and reaches 69.4% on MATH with weighted self-consistency. — [arXiv](https://arxiv.org/abs/2406.06592v1)
- **VinePPO** (arXiv 2410.01679): replaces the PPO value network with unbiased Monte Carlo value estimates from rollouts at intermediate states, exploiting the fact that a language environment can be reset to any prefix. — [arXiv](https://arxiv.org/html/2410.01679v1)
- **Math-Shepherd** (Wang et al., arXiv 2312.08935): labels each step's quality by the fraction of rollouts from that step that reach the correct answer. [background, not fetched]
- **Thought Anchors** (Bogdan, Macar, Nanda, Conmy, arXiv 2506.19143, v4 27 Oct 2025): the sentence-level counterpart. — [arXiv HTML](https://arxiv.org/html/2506.19143)
  - Same model as ours: R1-Distill-Qwen-14B, with R1-Distill-Llama-8B and Qwen3-30B-A3B as secondary models.
  - Data: 20 MATH problems with 25–75% accuracy, and MMLU.
  - 100 rollouts per sentence position.
  - Importance is defined as a KL divergence between answer distributions, conditioned on the resampled sentence being semantically different (cosine < 0.8, all-MiniLM-L6-v2).
  - Plan-generation and uncertainty-management sentences are the most important.
  - No activation probe predicts importance.

### Inferences
- The 80/20 paper uses the word "forking" for high entropy, which differs from Bigelow's operational definition (resampling changes the outcome). Our probes predict entropy but not importance, which is empirical evidence that these two notions come apart at the sentence level. Bigelow's reported weak entropy–forking correlation, if confirmed, would be a direct precedent at the token level.
- Rollout PRMs estimate V(h) = acc(h). Our signed importance acc(h,S) − acc(h) is exactly the Monte Carlo advantage of step S under the policy itself. See Q3 for what this means for novelty.

### Gaps
- I did not verify the exact identification procedure of cDPO's rollout-based "critical token" analysis. The paper also reportedly samples continuations after candidate tokens, but I did not confirm this.
- I found no paper that directly correlates rollout-measured token or sentence importance with entropy in R1-style models, apart from Bigelow's appendix note.

## Q3. Variance decomposition and the zero-mean-advantage (tower-property) result

### Takeaway
The fact that the advantage has zero mean under the policy, E_{a~π}[A^π(s,a)] = 0, is textbook RL. The PAV paper (Setlur et al.) explicitly relies on its consequence: process advantages taken under the base policy carry no extra signal, so they use a different "prover" policy. Our "signed importance has R²_max = 0" is therefore a restatement of a standard fact and should be framed as such. I found no prior paper that applies the law of total variance to rollouts to bound the R² of any pre-emission predictor of unsigned importance, or that estimates that ceiling empirically. That part appears novel.

### Cited Findings
- **Setlur et al., "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning"** (PAV): arXiv 2410.08146, 10 Oct 2024 (ICLR 2025 per prior knowledge; venue not verified here). — [arXiv](https://arxiv.org/abs/2410.08146); [arXiv HTML](https://arxiv.org/html/2410.08146)
  - It defines A^μ(s_h,a_h) := Q^μ(s_h,a_h) − V^μ(s_h) as "progress", the change in the likelihood of success before and after the step.
  - It states that using A^π in the policy gradient "results in exactly the same policy gradient update as only optimizing outcome evaluation".
  - It notes that for a poor π, A^π ≈ 0, so beam search on it would "not be informative at all".
  - Hence it uses a complementary prover μ ≠ π.
  - The fetch found no explicit "zero mean" sentence in the paper.
- Zero-mean advantage, E_{a~π}[Q^π(s,a)] = V^π(s), is a standard identity, used for example in the GAE and baseline literature (Schulman et al., arXiv 1506.02438). [background, not fetched]
- "Real-Time Progress Prediction in Reasoning Language Models" (arXiv 2506.23274) generates rollouts from random prefixes. It reports that the mean absolute error of progress is about 5–10%, a lower bound on how well progress can be predicted. This is a rollout-based irreducible-noise floor, though for progress rather than importance. — [arXiv HTML v2](https://arxiv.org/html/2506.23274v2); [v4](https://arxiv.org/html/2506.23274v4)
- "Forking Fast" (2608.19611) models the sampling noise in outcome estimates as multinomial (∝1/√S), but it does not decompose variance into history and next-step parts. — [arXiv HTML](https://arxiv.org/html/2608.19611v1)

### Inferences
- In our framing, acc(h,S) = Q(h,S), acc(h) = V(h), and signed importance = A(h,S). So E_S[signed importance | h] = 0 is the zero-mean-advantage identity, and Var(E[A|h]) = 0 gives R²_max = 0 immediately. Reviewers will recognise it. Present it as "a known identity with a consequence the interpretability literature has overlooked": any pre-emission probe of signed importance is predicting noise.
- The unsigned case is not covered by that identity. E[|A| | h] is a history-level quantity, such as the expected absolute advantage or a local "forkiness" or volatility. The fraction of variance it explains, Var_h(E[|A||h]) / Var(|A|), is a new empirical quantity. Its estimate of about 0.5 is new, and it is close in spirit to Zur et al.'s finding that uncertainty and steerability correlate.
- A nested K×M design is needed to separate three things: between-h variance, between-S-within-h variance, and within-(h,S) binomial noise. PRM papers (Math-Shepherd, OmegaPRM, VinePPO) estimate only V or Q and never this three-level split, as far as I found.

### Gaps
- I did not find a paper explicitly computing Var_h E[|Q−V| | h] / Var|Q−V| or any equivalent ceiling. Absence of evidence comes from targeted searches only; a dedicated search of the statistics and RL literature on "advantage variance decomposition" was not done.
- I did not confirm the PAV venue (ICLR 2025) in this session.

## Q4. Predicting important steps or tokens from hidden states before emission

### Takeaway
Many probes predict quantities at the level of the history from activations: correctness of future answers, outcome distributions, semantic entropy before generation, accuracy from the question alone, progress, and the upcoming "wait"/backtracking. The closest work to "probing importance" is Sycophantic Anchors (Jan/Feb 2026). It probes at sentence boundaries, after the anchor sentence has been emitted, to classify anchor type and regress anchor strength. I found no work that probes pre-emission activations for counterfactual importance and benchmarks the probe against a ceiling derived from resampling. Our negative result (probes recover entropy, not importance) together with the R²_max ceiling remains unoccupied.

### Cited Findings
- **Duszenko, "Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models"**: arXiv 2601.21183, v2 7 Feb 2026. — [arXiv HTML](https://arxiv.org/html/2601.21183)
  - Importance definition: Imp(s_k) = acc(prefix s_{1:k−1}) − acc(prefix s_{1:k}), from N=20 completions. An anchor requires Imp ≥ 0.50.
  - Models: R1-Distill-Llama-8B, R1-Distill-Qwen-7B and -1.5B, and Falcon-H1R-7B.
  - Probes read hidden states at the final token of the sentence boundary.
  - Results: 73.8–84.6% balanced accuracy for sycophantic versus correct anchors, and R² 0.48–0.74 for regressing anchor strength. No ceiling analysis.
- **"NLA Thought Anchors"** (LessWrong): Natural Language Autoencoders on Qwen2.5-7B-Instruct with GSM8K, 40 rollouts per sentence and a cosine threshold of 0.8. Counterfactually important sentences correlate with lower NLA reconstruction loss. — [LessWrong](https://www.lesswrong.com/posts/6HnnMHRoJLff46kgw/nla-thought-anchors)
- **Zur et al.** (2511.04527): linear probes on hidden activations predict the future outcome distribution o_t, and steerability peaks where uncertainty is high. — [arXiv HTML](https://arxiv.org/html/2511.04527)
- **Zhang et al., "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"**: arXiv 2504.05419 (NYU). Probes verify intermediate answers with high accuracy and calibration, and hidden states encode correctness of future answers before those answers are formulated. Used for early exit, the probe saves 24% of inference tokens. — [arXiv HTML](https://arxiv.org/html/2504.05419v1)
- **"No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes"**: arXiv 2509.10625, ICLR 2026 workshop. A probe on the question alone predicts accuracy before any answer. — [arXiv](https://arxiv.org/abs/2509.10625)
- **Semantic Entropy Probes** (Kossen et al., arXiv 2406.15927): probes capture semantic entropy from hidden states, even before generation. — [arXiv](https://arxiv.org/pdf/2406.15927)
- **"Internal states before wait modulate reasoning patterns"**: arXiv 2510.04128, Oct 2025. Latent attribution patching finds SAE features that promote or suppress the "wait" token. Steering promoters negatively removes "wait" entirely. — [arXiv HTML](https://arxiv.org/html/2510.04128v1)
- **Venhoff et al., "Understanding Reasoning in Thinking Language Models via Steering Vectors"**: arXiv 2506.18167, ICLR 2025 Workshop on Reasoning and Planning for LLMs. Backtracking, uncertainty expression and example testing are mediated by linear directions. Steering vectors control these behaviours in three R1-Distill models. — [arXiv](https://arxiv.org/pdf/2506.18167)
- **Pal et al., "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State"**: CoNLL 2023 (arXiv 2311.04897). From a single hidden state, a model's subsequent tokens can be predicted with more than 48% accuracy at some layers. — [ACL Anthology](https://aclanthology.org/2023.conll-1.37/)
- "Progress prediction" probes (2506.23274) predict reasoning progress from hidden states in real time. — [arXiv HTML](https://arxiv.org/html/2506.23274v4)

### Inferences
- All verified probes target history-level quantities: V(h), entropy, semantic entropy, progress, or the propensity to "wait" next. By the tower property, these are exactly the predictable component. This fits our finding that the probes recover entropy or uncertainty but not importance. It suggests a reframing: the best a pre-emission probe can do is predict E[|A| | h], a volatility or forkiness quantity that is plausibly what entropy and uncertainty probes already capture.
- The Sycophantic Anchors strength regression (R² up to 0.74) is post-emission: the probe reads activations at the end of the sentence, so S is already in context. That is not comparable to our pre-emission ceiling. The comparison is itself a useful point: post-emission probes can exceed R²_max because they observe S.
- The "wait"/backtracking work (2510.04128, 2506.18167) predicts or controls behaviour type, which the Thought Anchors taxonomy links to high importance for uncertainty management. A probe for "upcoming uncertainty-management sentence" may therefore partly predict unsigned importance. This is untested inference.

### Gaps
- I found no paper that predicts Thought-Anchors-style counterfactual importance from pre-emission activations, and none that reports a resampling-derived R² ceiling. I flag this as "not found" rather than proven absent.
- I did not check whether Thought Anchors v4 or the Thought Branches revision (Apr 2026) added probe experiments; the summaries I read mention none.
- I did not verify "lookahead" or planning-probe papers beyond Future Lens, such as "Do language models plan ahead for future tokens?" (Wu et al., 2404.00859) [background, not fetched].
