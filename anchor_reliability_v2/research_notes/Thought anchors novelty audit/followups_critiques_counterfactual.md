# Follow-ups, critiques, replications and counterfactual-CoT literature relevant to a Thought Anchors reliability audit (as of 2026-10-11)

Scope note: about 25 searches and fetches were run on 2026-10-11. Items marked **[verified this session]** were checked against a primary page (arXiv abs/HTML, GitHub API, LessWrong). Items marked **[from prior knowledge, not re-fetched]** are standard citations whose IDs I am confident of but did not re-open in this session. OpenReview pages (reviews of Thought Anchors / Thought Branches) were blocked by a bot check and could not be read.

## Q1. Follow-ups by the same authors and others (2025–2026): what do they do, and do they report variance/reliability/noise of resampling importance?

### Takeaway
The main follow-up, Thought Branches (Macar, Bogdan, Rajamanoharan, Nanda; arXiv 2510.27484, ICLR 2026), extends resampling to agentic/blackmail, resume and hint settings. It adds a "resilience" metric and compares on-policy resampling with off-policy insertions. Neither it nor Thought Anchors reports replicate-level reliability (test-retest, ICC, split-half) of per-sentence importance, a finite-sample bias correction, or a significance test per sentence. Both name compute cost as their main limitation. The closest independent work on estimator noise is Du et al. 2026 (COLM 2026, arXiv 2609.04194). It uses split-half noise ceilings for Monte-Carlo step importance, but on Qwen3 models and a different question (whether LLM judges can recover importance).

### Cited Findings
**Thought Anchors (base paper) [verified this session]**
- Setup and status:
  - The arXiv PDF header reads "Under review as a conference paper at ICLR 2026". The latest arXiv HTML version is v4. — [arXiv 2506.19143 PDF](https://arxiv.org/pdf/2506.19143); [v4 HTML](https://arxiv.org/html/2506.19143v4)
  - I could not determine the ICLR 2026 decision, because OpenReview was blocked. An anonymous OpenReview PDF exists. — [OpenReview PDF](https://openreview.net/pdf/f5416cc2c39a560a1694883f61b4ea9cb2d3c614.pdf)
- Sampling parameters: 100 resamples per sentence on 20 MATH problems with 25–75% accuracy, temperature 0.6, top-p 0.95. Average trace: 144.2 sentences and 4,208 tokens. — [arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)
- Similarity filter: counterfactual importance treats a resampled sentence as different from the original when cosine similarity falls below 0.8, described as the median over sentence pairs. Embeddings come from all-MiniLM-L6-v2. — [arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)
- Variance statements:
  - The only quantitative variance statement found is for single-problem plots: "95% CI corresponds to at worst ±10%" on accuracy.
  - The authors note that "fewer resamples for any one CoT may suffice" for aggregate patterns.
  - They call compute cost "a primary limitation".
  - Source: [arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)
- Robustness that is reported is at the head level, not the sentence-importance level:
  - Receiver-head kurtosis has split-half reliability across problems of r = .84.
  - Attention suppression correlates positively with resampling-based importance (appendix).
  - The authors acknowledge a position confound for receiver-head scores (Appendix C.4).
  - Source: [arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)
- Attention suppression vs resampling: the matrices correlated positively in 19 of 20 traces, with a mean Spearman ρ of only .20. — [arXiv 2506.19143v3](https://arxiv.org/html/2506.19143v3)
- Third-party summary (not primary): counterfactual importance "can yield high-variance estimates when the number of semantically divergent resampled sentences is low (e.g., <10)". I could not find this sentence in the primary text during this session. — [search-result summary of arXiv 2506.19143 / learnmechinterp](https://learnmechinterp.com/topics/counterfactual-resampling/)

**Thought Branches [verified this session]**
- Bibliographic details:
  - Authors are Uzay Macar, Paul C. Bogdan, Senthooran Rajamanoharan and Neel Nanda.
  - v1 was posted on 2025-10-31 and revised on 2026-04-13.
  - It was published as an ICLR 2026 poster.
  - Sources: [arXiv 2510.27484](https://arxiv.org/abs/2510.27484); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008605); [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/954d303839bf7fc458acea7384322255-Paper-Conference.pdf)
- Four case studies — [arXiv 2510.27484](https://arxiv.org/abs/2510.27484):
  1. In agentic-misalignment blackmail scenarios, stated self-preservation sentences have small causal impact.
  2. Off-policy interventions give "small and unstable effects" compared with resampling.
  3. A **resilience** metric repeatedly resamples so that similar content cannot reappear downstream.
  4. Unverbalized hints have subtle, cumulative influence.
- Models and parameters — [arXiv 2510.27484v1](https://arxiv.org/html/2510.27484v1); [v2](https://arxiv.org/html/2510.27484v2):
  - Models: qwen-32b, qwen3-235b, llama-3.1-nemotron-ultra-235b and deepseek-r1.
  - Temperature 0.7, top-p 0.95, applied uniformly.
  - 100 rollouts per sentence.
  - Counterfactual importance uses KL divergence, bert-large-nli-stsb-mean-tokens embeddings and a "cosine similarity below median" threshold.
  - Scenarios: 20 blackmail base traces, 500 resume CoTs and 40 MMLU hint problems.
- The off-policy arms were "handwritten sentences crafted to discourage blackmail", "sentences extracted from other models' CoTs", and "sentences from the same model's CoTs but from different seeds". Deletion and paraphrase are not listed as separate arms, and no intervention was run at more than one temperature. — [arXiv 2510.27484v2](https://arxiv.org/html/2510.27484v2)
- Uncertainty reporting is minimal:
  - Results are boxplots and scatterplots. The main results have no confidence intervals or significance tests.
  - Threshold robustness is not reported.
  - Whether 100 rollouts are enough is not discussed.
  - Stated limitation: "computational cost, which currently makes it more suitable for offline analysis than real-time monitoring".
  - Sources: [arXiv 2510.27484v1](https://arxiv.org/html/2510.27484v1); [v2](https://arxiv.org/html/2510.27484v2)

**Other 2026 follow-ups**
- Thought Editing (de la Fuente & Engels, LessWrong, 2026-02-03) [verified this session]:
  - Compares off-policy target-word substitution, off-policy random insertion and on-policy rejection resampling for steering. Models: R1, Nemotron and GPT-OSS-20b.
  - Finds that "resampling produces different phrasings of the same behavior" and that random insertion works best.
  - It is about steering, not importance estimation.
  - Source: [LessWrong](https://www.lesswrong.com/posts/KXR5FNs4hHT5sMRti/thought-editing-steering-models-by-editing-their-chain-of)
- Legibility is Not Interpretability (Du, Hoyle, Ruis, Locatelli; COLM 2026; posted 2026-09-03) [verified this session]:
  - Importance is "the change in expected reward ... from including that step, estimated via Monte Carlo rollouts", using N=50 rollouts per prefix and V/Q-value estimates.
  - Ceilings are split-half: advantage estimates from half of the rollouts are scored against labels from the other half (conservative) or from the full set (optimistic).
  - The authors say they "must account for estimator variance" and handle it with changepoint detection.
  - Models: Qwen3-1.7B/4B/8B. Data: AIME, AMC, MATH500 and GSM8K.
  - Conclusion: "step importance is only partially recoverable from the text."
  - Sources: [arXiv 2609.04194](https://arxiv.org/abs/2609.04194); [HTML](https://arxiv.org/html/2609.04194v1)
- Reasoning Models Know What's Important (Nikankin, Tutek, Ashuach, Rosenfeld, Belinkov; arXiv 2604.18307, 2026-04-20) [verified this session]:
  - Defines importance by removability, using greedy deletion guided by gradient attribution. Uses K=5 completions for sufficiency.
  - Models include R1-Distill-Qwen-1.5B/7B/14B, GPT-OSS-20B and OLMo-3-7B-Think. Data: HARP and MATH-500.
  - Cites Bogdan et al. but makes no empirical comparison with resampling importance, and reports no CIs.
  - Source: [arXiv 2604.18307](https://arxiv.org/html/2604.18307v1)

### Inferences
- Thought Branches already covers on-policy versus off-policy insertion. Our resample vs delete vs paraphrase arm at two temperatures is therefore **partially pre-empted in spirit** (on-policy beats off-policy) but not in design:
  - Thought Branches covers no deletion or paraphrase arm, no temperature sweep and no MATH or R1-Distill-14B setting.
  - It runs no reliability analysis of any arm.
- Our reliability results look novel relative to the Bogdan/Macar line:
  - replicate Spearman 0.44 and ICC 0.56;
  - the noise floor tracks answer entropy;
  - floor-debiased split-half reliability is 0.16;
  - about 2% of sentences survive a BH permutation test.
- Du et al. 2026 is the closest methodological precedent for split-half noise ceilings on Monte-Carlo step importance, and it should be cited as such. It does not audit Thought Anchors, does not use TV, does not decompose the noise floor by entropy and does not use R1-Distill-14B.

### Gaps
- I could not read the OpenReview reviews of Thought Anchors (ICLR 2026) or of Thought Branches. Reviewers may have raised variance or sample-size concerns.
- I could not determine whether Thought Anchors was accepted anywhere.
- I could not verify the third-party "<10 divergent samples" claim in the primary text.
- I did not inspect the full Du et al. appendix, for example to see whether they relate noise to answer entropy.

## Q2. Critiques and replications: has anyone published a reliability analysis, or reported entropy dominance, finite-sample TV/KL bias, threshold sensitivity or prompt-format sensitivity?

### Takeaway
I found **no published critique or reliability analysis of Thought Anchors' per-sentence importance**. That covers arXiv, LessWrong, the Alignment Forum and GitHub. In particular I found no claim that scores are dominated by answer entropy, that TV/KL estimates carry finite-sample bias, or that results are sensitive to the similarity threshold or prompt format. The only replication found is a legal-domain extension posted on LessWrong. The only reported data problems are GitHub issues about failed API rollouts and a head-ablation hook bug.

### Cited Findings
- **Legal-reasoning replication** (JeaniceK, Matthew Robbins, Johannes Taraz; LessWrong, 2026-03-10) [verified this session]:
  - Re-applies resampling, attention, receiver heads and causal dependencies to legal reasoning with R1-Distill-Llama-8B, plus Llama-70B and Qwen-1.5B.
  - Finds that anchors centre on fact retrieval rather than planning, and that attention correlates weakly with causal importance (Pearson r = 0.23).
  - Finds that receiver heads concentrate in later layers, consistent with the original paper.
  - The authors acknowledge that apparent long-range dependencies may be token-overlap artifacts, and that statistical power and sample sizes are limited.
  - Source: [LessWrong](https://www.lesswrong.com/posts/f9cnGHCiJgo5eDkSQ/understanding-reasoning-with-thought-anchors-and-probes)
- **GitHub issue #4** (chrismerck, 2025-08-06) [verified this session]:
  - The Hugging Face math-rollouts dataset contained many rollouts with an `"error": "'text'"` field.
  - Counting these as incorrect produced extra downward spikes in Fig. 4A. The reporter raised possible content dependence, such as hitting the token limit or a missing `\boxed{}`.
  - Author response (uzaymacar, 2025-08-10): these were "unintended failures ... due to concurrency limits or other intermittent network or API issues. They should be retried." The analysis scripts filter them out.
  - A new dataset (20 problems × one correct and one incorrect CoT × 100 rollouts per sentence) was pushed with the errors removed.
  - Source: [GitHub issue #4](https://github.com/interp-reasoning/thought-anchors/issues/4)
- **GitHub issue #7** (Sagar300100, open, 2026-09-05) [verified this session]:
  - The head-ablation hook in `whitebox-analyses/pytorch_models/ablation.py` zeroes blocks of the output *after* `o_proj`. Those blocks are not individual heads.
  - The suggested fix is a forward pre-hook on `o_proj`.
  - Scope stated in the issue: "source inspection only."
  - Source: [GitHub issue #7](https://github.com/interp-reasoning/thought-anchors/issues/7)
- **Other issues** [verified this session]:
  - #5: dataset column/cast error.
  - #3: missing `normalize_answer`.
  - #2: question about the accuracy of GPT-labelled taxonomy.
  - Source: [GitHub issues](https://github.com/interp-reasoning/thought-anchors/issues?q=is%3Aissue)
- Searches for "thought anchors" combined with noise, entropy, critique, finite-sample or total variation found nothing beyond the above. — [search summary: LessWrong/AF hits only the original post and the legal replication](https://www.lesswrong.com/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter)
- Papamichalis & Ruane (arXiv 2609.25039, 2026-08-26) decompose CoT *token* entropy into scaffold and content channels. They do not discuss Thought Anchors or noise floors in importance estimates. — [arXiv 2609.25039](https://arxiv.org/abs/2609.25039)

### Inferences
- Several of our findings appear to be unclaimed in the literature:
  - the noise floor tracks answer entropy;
  - debiased importance is barely reliable;
  - only about 2% of effects are real and they sit where the answer is uncertain;
  - the released prompts collapse newlines;
  - accuracy differs by 0.11 on incorrect traces between the hosted-API data and our local regeneration.
- Issue #4 shows that API failures in the hosted generation pipeline already affected the released data once. This supports treating the provider/API path as a reproducibility variable.

### Gaps
- Unpublished critiques may exist on Twitter/X or in Slack/MATS channels. I could not search X.
- I found no report on newline collapsing in the prompts. I did not open the prompt-construction code of `generate_rollouts.py` beyond grepping it.

## Q3. Counterfactual CoT interventions (truncate, delete, paraphrase, add mistakes, resample): who did what, and has anyone compared arms across temperatures?

### Takeaway
The classic faithfulness papers use off-policy edits:
- truncation, adding mistakes, paraphrasing and filler tokens (Lanham et al. 2023);
- biasing features (Turpin et al. 2023);
- hint verbalization (Chen et al. 2025);
- inserted steps (Xiong et al. 2025);
- greedy deletion (Nikankin et al. 2026).

Thought Branches compares on-policy resampling against off-policy insertion. **No work found compares resample vs delete vs paraphrase of the same sentence on the same traces at more than one temperature.**

### Cited Findings
- Thought Branches: off-policy insertions (handwritten, cross-model, same-model different seed) "cluster near zero effect", whereas on-policy resampling reaches "up to a 100% reduction in blackmail rates". The temperature was fixed at 0.7. — [arXiv 2510.27484v1](https://arxiv.org/html/2510.27484v1); [v2](https://arxiv.org/html/2510.27484v2)
- Measuring the Faithfulness of Thinking Drafts (Xiong et al.; arXiv 2505.13774; NeurIPS 2025) [verified this session]:
  - A counterfactual framework with two parts. Intra-draft faithfulness uses counterfactual step *insertions*. Draft-to-answer faithfulness perturbs the concluding logic.
  - Six LRMs show "selective faithfulness".
  - Sources: [arXiv 2505.13774](https://arxiv.org/abs/2505.13774); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/120231)
- Chain-of-Thought Is Not Explainability (Barez, Wu, Arcuschin, … Bengio; Oxford AIGI, July 2025) [verified this session]:
  - A position paper: CoT rationales are "neither necessary nor sufficient for trustworthy interpretability".
  - It cites answer changes of up to 36% under option reordering that the CoT never mentions.
  - Sources: [Oxford AIGI PDF](https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf); [Oxford Martin](https://www.oxfordmartin.ox.ac.uk/publications/chain-of-thought-is-not-explainability)
  - I did not confirm an arXiv ID. An alphaXiv page exists.
- Reasoning Models Know What's Important (arXiv 2604.18307): greedy-deletion removability, with no temperature sweep. — [arXiv 2604.18307](https://arxiv.org/html/2604.18307v1)
- Legibility is Not Interpretability (arXiv 2609.04194): resample-based (Monte Carlo) importance, compared with LLM-judge importance. No deletion or paraphrase arm was reported in the parts I read. — [arXiv 2609.04194 HTML](https://arxiv.org/html/2609.04194v1)
- Thought Editing compares substitution, random insertion and on-policy resampling for steering, with no temperature sweep. — [LessWrong 2026-02-03](https://www.lesswrong.com/posts/KXR5FNs4hHT5sMRti/thought-editing-steering-models-by-editing-their-chain-of)
- From Reasoning to Answer (Zhang, Lin, Rajmohan, Zhang; arXiv 2509.23676; EMNLP 2025 main) [verified this session]: activation patching of key reasoning tokens in distilled R1 models "can reliably alter the final answers". — [arXiv 2509.23676](https://arxiv.org/abs/2509.23676)
- **[From prior knowledge, not re-fetched]**
  - Lanham et al. 2023, "Measuring Faithfulness in Chain-of-Thought Reasoning", arXiv 2307.13702 (Anthropic). Interventions: early answering (truncation), adding mistakes, paraphrasing, filler tokens. Models are pre-R1 Anthropic models. This is the closest prior use of *paraphrase* as an intervention, but it targets the whole remaining CoT rather than single sentences, and it does not compare against resampling.
  - Turpin et al. 2023, "Language Models Don't Always Say What They Think", arXiv 2305.04388, NeurIPS 2023. Biasing features (e.g., "answer is always A") alter answers without being mentioned in the CoT.
  - Arcuschin et al. 2025, "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful", arXiv 2503.08679. Implicit post-hoc rationalization on paired comparative questions, plus "unfaithful illogical shortcuts".
  - Chen et al. 2025, "Reasoning Models Don't Always Say What They Think", arXiv 2505.05410 (Anthropic). Hint-verbalization rates for Claude 3.7 Sonnet and R1 are often below 20%.
  - Tutek et al. 2025, "Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps" (FUR), arXiv 2502.14829. It unlearns individual steps, which is another parametric alternative to deletion. I believe the ID and the EMNLP 2025 venue are right but did not verify them.
  - Chua & Evans 2025, "Are DeepSeek R1 And Other Reasoning Models More Faithful?", arXiv 2501.08156. Cue articulation in reasoning models.

### Inferences
- Remaining novelty for our **resample vs delete vs paraphrase × 2 temperatures** arm:
  1. All three arms applied to the *same* sentences of R1-Distill-14B MATH traces.
  2. Temperature as a factor. No paper found varies sampling temperature for any CoT-intervention comparison.
  3. Each arm's reliability measured against its own noise floor.
- The Thought Branches finding that off-policy edits are near zero and unstable predicts that delete and paraphrase will show weaker or noisier effects. Our design can test that prediction under the noise-floor correction.
- Our token-level resampling around key phrases is related to Thought Anchors' own token-level vs sentence-level comparison. The paper says "sentence-level resampling data mirrors patterns seen when resampling tokens" ([search-result summary of arXiv 2506.19143](https://openreview.net/pdf/e2eb4189bc2250be1718a88fcb63c2423b280109.pdf)). Its details were not inspected here, so our arm should be positioned as a replication or extension of that appendix claim.

### Gaps
- I did not re-verify Lanham, Turpin, Arcuschin, Chen, Tutek or Chua & Evans in this session.
- I did not search exhaustively for overthinking or "thought pruning" papers that score step importance by deletion (e.g., length-efficient CoT pruning). Two such papers surfaced but were not inspected: [arXiv 2603.00296](https://arxiv.org/html/2603.00296v1) and [arXiv 2509.12464](https://arxiv.org/html/2509.12464v1).
- I could not find a paper on "causal importance of reasoning steps" that adopts ROSCOE-style scoring in R1 models.

## Q4. Receiver heads and attention suppression: replications, extensions and related attention-to-steps work

### Takeaway
Receiver heads have been partly replicated: they appear in later layers in legal reasoning. However, the link between attention and causal importance looks weak everywhere it has been measured:
- Thought Anchors itself: mean ρ = .20;
- the legal replication: r = .23.

Related independent work finds analogous "anchor" tokens and reasoning-focus heads. A head-ablation code bug was reported in Sept 2026. Our receiver-head and suppression arm is therefore a replication with a known weak baseline correlation and a known code concern.

### Cited Findings
- Thought Anchors receiver heads: kurtosis-based, with split-half r = .84 across problems. Top receiver heads intercorrelate at mean r = .56 against a baseline of .35. — [arXiv 2506.19143v4](https://arxiv.org/html/2506.19143v4)
- Attention suppression vs resampling: 19 of 20 traces correlate positively, mean Spearman ρ = .20. — [arXiv 2506.19143v3](https://arxiv.org/html/2506.19143v3)
- Legal replication: receiver heads appear in later layers with vertical attention stripes. Attention and causal importance correlate at r = 0.23. — [LessWrong 2026-03-10](https://www.lesswrong.com/posts/f9cnGHCiJgo5eDkSQ/understanding-reasoning-with-thought-anchors-and-probes)
- Attention Illuminates LLM Reasoning: the Preplan-and-Anchor Rhythm (Yang Li, Zhichen Dong, Yuhan Sun et al.; arXiv 2510.13554; ICML 2026) [verified this session]:
  - Introduces "Future Attention Influence" and windowed attention distance. Globally focused heads highlight anchor tokens that organise later reasoning.
  - Uses these for fine-grained RL credit assignment.
  - No explicit Thought Anchors comparison.
  - Source: [arXiv 2510.13554](https://arxiv.org/abs/2510.13554)
- From Reasoning to Answer (arXiv 2509.23676; EMNLP 2025): mid-layer "Reasoning-Focus Heads" track the reasoning trajectory, and activation patching confirms that information flows from reasoning to answer. — [arXiv 2509.23676](https://arxiv.org/abs/2509.23676)
- Understanding Aha Moments (arXiv 2504.02956): links aha moments to anthropomorphic, uncertainty-marked language and to difficulty representations across layers. — [arXiv 2504.02956](https://arxiv.org/pdf/2504.02956)
- Understanding and Steering the Cognitive Behaviors of Reasoning Models at Test-Time (arXiv 2512.24574): identifies attention heads correlated with cognitive behaviours. — [arXiv 2512.24574](https://arxiv.org/html/2512.24574v2)
- GitHub issue #7: the head-ablation hook acts after `o_proj`, so it does not ablate individual heads. Whether any published numbers depend on it is undetermined. — [issue #7](https://github.com/interp-reasoning/thought-anchors/issues/7)

### Inferences
- Novel angles for our attention arm:
  - Test whether receiver-head and suppression scores track *reliable*, floor-debiased importance or the entropy-driven noise floor. No one has done this.
  - Check whether the issue #7 ablation bug affects receiver-head ablation results.

### Gaps
- I did not verify which Thought Anchors figures use `ablation.py`.
- I did not read the 2504.02956 or 2512.24574 bodies in detail.

## Q5. Released-data issues: math-rollouts dataset, prompt construction, Novita hosted API, regeneration differences

### Takeaway
The rollout script defaults to the Novita hosted API, with Together, Fireworks and Local as options, at temperature 0.6 and top-p 0.95. One data problem has been publicly reported and fixed: API error rollouts in the HF dataset (issue #4). I found no public report of newline collapsing in prompts, or of accuracy differences between hosted-API and local regeneration.

### Cited Findings
- `generate_rollouts.py` [verified this session via GitHub API]:
  - Reads `NOVITA_API_KEY`.
  - `--provider` defaults to `"Novita"` with choices Novita, Together, Fireworks and Local.
  - Novita requests go to `https://api.novita.ai/v3/openai/completions`, the completions endpoint, so the prompt string is built client-side.
  - Default temperature is 0.6 and default top-p is 0.95.
  - There is a min_p option that Fireworks does not support.
  - For Together, a leading `"<think>\n"` is stripped from the output.
  - Source: [generate_rollouts.py](https://github.com/interp-reasoning/thought-anchors/blob/main/generate_rollouts.py)
- Issue #4: the HF dataset had error rollouts. The author attributes them to concurrency, network or API failures and re-pushed a cleaned dataset `uzaymacar/math-rollouts` (20 problems × correct/incorrect × 100 rollouts per sentence). — [issue #4](https://github.com/interp-reasoning/thought-anchors/issues/4)
- Issue #5: column-mismatch or cast error when loading the dataset (closed). — [issue #5](https://github.com/interp-reasoning/thought-anchors/issues/5)
- The repository's last push was 2025-10-27. [verified this session via GitHub API] — [repo](https://github.com/interp-reasoning/thought-anchors)

### Inferences
- Several sources of drift between providers and local regeneration are plausible:
  - the API completions endpoint with client-built prompts;
  - provider-specific `<think>` handling;
  - min_p differences;
  - retry and failure behaviour;
  - provider-side quantization or kernels.
  
  Our 0.11 accuracy gap on incorrect traces and the newline collapse appear to be new public findings. Attribution to any single cause is inference.

### Gaps
- I did not inspect the exact prompt-template lines in `generate_rollouts.py` that would show newline handling. The grep did not surface them.
- I could not verify which Novita model build or quantization served R1-Distill-Qwen-14B at generation time.
- I found no public discussion of Novita-specific nondeterminism for this dataset.
