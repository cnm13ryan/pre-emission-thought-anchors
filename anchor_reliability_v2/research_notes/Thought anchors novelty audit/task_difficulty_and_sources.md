# Task difficulty, saturation/contamination of MATH for R1-Distill-Qwen-14B, and problem sources in CoT-interpretability work

Research date: 2026-10-11. All URLs below were fetched or returned by search in this session unless marked "[from memory, not re-verified]".

## Q1. Reported accuracies of DeepSeek-R1-Distill-Qwen-14B on math benchmarks

### Takeaway
On aggregate benchmarks the model is near ceiling on MATH-500 (93.9% pass@1). It is far from ceiling on AIME 2024 (69.7%) and on the post-release competitions AIME 2025 (~49%) and HMMT Feb 2025 (~32%). So MATH as a *whole* is close to saturated for this model, but a filtered subset can still sit at intermediate accuracy.

### Cited Findings
- Model card table (from DeepSeek-R1 paper, arXiv 2501.12948): R1-Distill-Qwen-14B scores AIME 2024 pass@1 69.7, AIME 2024 cons@64 80.0, MATH-500 pass@1 93.9, GPQA-Diamond 59.1, LiveCodeBench 53.1, CodeForces rating 1481 — [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- Eval settings for those numbers: max generation 32,768 tokens, temperature 0.6, top-p 0.95, 64 samples per query; the card recommends no system prompt and forcing output to begin with "<think>\n" — [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- MathArena (independent, temp 0.6, top-p 0.95): AIME 2025 49.17% ± 8.94 (rank 51/61); HMMT Feb 2025 31.67% ± 8.32 (rank 50/60); BRUMO 2025 68.33% ± 8.32 (40/45); SMT 2025 54.72% ± 6.70 (44/44). No results are listed for CMIMC 2025, AIME 2026 or HMMT 2026 — [MathArena model page](https://matharena.ai/models/deepseek_deepseek_distill_14b)
- MathArena describes itself as evaluating on "uncontaminated problems from recent human competitions" (paper arXiv 2505.23281) — [MathArena paper](https://arxiv.org/pdf/2505.23281)

### Inferences
- The drop from AIME 2024 (69.7) to AIME 2025 (~49) is ~20 points. Part of this could be year-to-year difficulty and part could be contamination of AIME 2024, which predates the model. The two cannot be separated from these numbers alone.
- With MATH-500 at 93.9%, a random MATH problem is likely to be solved nearly always. Intermediate-accuracy MATH problems for this model are therefore a selected tail. They are disproportionately (a) hard Level-5 items, (b) items with ambiguous or awkward answer formats or grader issues, or (c) items where the model "knows" the answer but is noisy. This bears on whether such problems are representative.

### Gaps
- I did not find an official DeepSeek figure for R1-Distill-14B on AIME 2025 or HMMT. The R1 paper predates them. MathArena is the main independent source.
- I could not confirm a MATH-500 breakdown by level for the 14B distill.

## Q2. Is MATH / MATH-500 saturated, and is there evidence of contamination or memorisation in Qwen2.5-based models?

### Takeaway
Yes on both counts, with one caveat: the direct evidence is for the Qwen2.5 and Qwen2.5-Math *base* models, not for R1-Distill-Qwen-14B specifically. R1-Distill-Qwen-14B is fine-tuned from Qwen2.5-14B, so it inherits the same pretraining. Strong 2025 evidence shows Qwen2.5 base models can reproduce MATH-500 problems verbatim from partial prompts. That memorisation, rather than learning, explains the "spurious rewards" RLVR gains on Qwen.

### Cited Findings
- Wu et al., "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination," arXiv 2507.10532 (v3 dated 17 Dec 2025). The fetched page described it as a preprint, but search returned an AAAI proceedings listing (ojs.aaai.org article 40687), which suggests publication at AAAI 2026 — [arXiv v3](https://arxiv.org/html/2507.10532v3); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40687)
  - Partial-prompt completion on MATH-500 with a 60% problem prefix: Qwen2.5-Math-7B ROUGE-L 78.06%, exact match 54.60%; Qwen2.5-7B ROUGE-L 60.98%, EM 21.20%; Llama3.1-8B ROUGE-L 40.55%, EM 3.80% — [arXiv 2507.10532v3](https://arxiv.org/html/2507.10532v3)
  - With an 80% prefix: AMC EM 42.17% for Qwen2.5-Math vs 0% for Llama; AIME 2024 EM 20% vs 0%. On LiveMathBench both drop to ~0–5%, so it is treated as clean along with their synthetic RandomCalculation set — [arXiv 2507.10532v3](https://arxiv.org/html/2507.10532v3)
  - Conclusion: contamination is "the main factor behind the 'magical' success of spurious rewards on Qwen." On a leakage-free dataset only accurate rewards improve performance — [arXiv 2507.10532](https://arxiv.org/html/2507.10532v1)
  - The paper does **not** test DeepSeek-R1-Distill models — [arXiv 2507.10532v3](https://arxiv.org/html/2507.10532v3)
- Shao et al., "Spurious Rewards: Rethinking Training Signals in RLVR," arXiv 2506.10947 (12 Jun 2025). RLVR improves Qwen2.5-Math-7B MATH-500 accuracy by +21.4 points with random reward, +13.8 format reward, +24.1 incorrect labels, +26.0 one-shot RL and +27.1 majority vote, vs +29.1 with ground truth. These gains often fail to transfer to Llama3 or OLMo2 — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947?context=cs)
- Huang et al., "MATH-Perturb," arXiv 2502.06453 (ICML 2025, PMLR v267). It perturbs 279 Level-5 MATH problems into MATH-P-Simple and MATH-P-Hard. Large drops on hard perturbations (o1-mini −16.49 points, gemini-2.0-flash-thinking −12.9) indicate a form of memorisation in which learned solution templates are applied where they no longer fit — [ICML poster](https://icml.cc/virtual/2025/poster/45435); [PMLR](https://proceedings.mlr.press/v267/huang25k.html)

### Inferences
- R1-Distill-Qwen-14B is SFT'd from Qwen2.5-14B on R1-generated traces [from memory: R1 paper states distills are fine-tuned from Qwen2.5 base models, not re-verified]. It most likely inherits whatever MATH memorisation Qwen2.5-14B has. Wu et al. show this for 7B models, so the 14B case is an inference. The study's flagged problem (80% accuracy with no reasoning) fits this.
- Contamination has a particular effect on CoT-importance analyses. If the model can retrieve the answer, the CoT can be partly decorative. Sentence-level counterfactual importance should then be low or diffuse, and "anchors" may reflect answer formatting or verification rather than computation. The study could detect this by measuring accuracy with forced-immediate-answer or no-think for every problem, not just the flagged one.

### Gaps
- I found no study that directly measures MATH memorisation in R1-Distill-Qwen-14B, for example by partial-prompt completion. The study could run the Wu et al. 60%/80% prefix completion test itself on its 20 problems with the Qwen2.5-14B base.
- I did not verify whether the 20 Thought Anchors problems fall inside MATH-500 or the MATH train/test split.

## Q3. How did Thought Anchors select its problems?

### Takeaway
They screened 1,000 MATH problems with 10 samples each and kept 20 "challenging but doable" problems solved 25–75% of the time. For each they chose one correct and one incorrect base trace and ran 100 rollouts per sentence.

### Cited Findings
- Exact wording (arXiv 2506.19143v4, 27 Oct 2025): "20 challenging but doable questions that are correctly solved 25-75% of the time, identified by testing on 1,000 problems 10 times each." One correct and one incorrect trace per problem gives 40 responses. Mean trace length is 144.2 sentences (95% CI 116.7–171.8), ~4,208 tokens — [arXiv 2506.19143 HTML](https://arxiv.org/html/2506.19143)
- Setup: R1-Distill-Qwen-14B (48 layers), temperature 0.6, top-p 0.95, 100 rollouts per sentence for counterfactual resampling — [arXiv 2506.19143](https://arxiv.org/html/2506.19143)
- Repo: `selected_problems.json` is "a list of problems identified in the 25% - 75% accuracy range," "sorted in increasing order by average length of sentences" — [GitHub interp-reasoning/thought-anchors](https://github.com/interp-reasoning/thought-anchors)
- HF dataset `uzaymacar/math-rollouts` includes R1-Distill-Llama-8B and R1-Distill-Qwen-14B (temp 0.6, top-p 0.95). There are ~20 problems per solution type per model, tagged with MATH Level 1–5 and subject. Solution types are correct base, incorrect base, forced-answer variants and sanity checks — [HF dataset card](https://huggingface.co/datasets/uzaymacar/math-rollouts)
- The paper's only difficulty analysis covers MMLU, not MATH: "questions with high average accuracy elicit CoTs with stronger close-range links and weaker long-range links" — [arXiv 2506.19143](https://arxiv.org/html/2506.19143)

### Inferences
- With a 10-sample screen, the binomial SE at p=0.5 is ~0.16. Problems screened as 0.25–0.75 will spread when re-measured with many samples. The study's 0.22–0.90 range and 0.73 median fit selection noise plus regression toward the model's high base accuracy on MATH (93.9% on MATH-500). That is, several problems may really be "easy-ish" (~0.85–0.90), above the intended band.
- The released problems were not screened for memorisation, for example by no-reasoning accuracy. Nothing in the paper or repo mentions such a filter.

### Gaps
- The repo README does not give per-problem MATH levels or the exact selection script parameters. The HF problem.json files carry the levels, but I did not tabulate them.

## Q4. Problem sources used by comparable 2025–2026 CoT-interpretability / faithfulness papers

### Takeaway
Resampling and attribution papers are moving away from MATH alone. They use AIME 24/25/26, AMC 23 and MATH-500/GSM8K as an easy-vs-hard contrast, as well as MMLU/GPQA hint setups, Putnam, and non-math behavioural scenarios. Most still use a small number of problems (20–40), and some filter for answer variability.

### Cited Findings
- Du, Hoyle, Ruis, Locatelli, "Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-of-Thought Reasoning," arXiv 2609.04194 (Sep 2026). It uses 30 problems from each of AIME 24, AIME 25, AIME 26, AMC 23, MATH500 and GSM8K, with 10 responses per prompt, on Qwen3-1.7B/4B/8B. No accuracy filter is mentioned — [arXiv 2609.04194](https://arxiv.org/html/2609.04194v1)
- Macar, Bogdan, Rajamanoharan, Nanda, "Thought Branches: Interpreting LLM Reasoning Requires Resampling," arXiv 2510.27484 (v2, 13 Apr 2026). It uses blackmail scenarios (20 base CoTs per model), 40 professor-hinted MMLU problems where the hint shifts answers by ≥10% (R1-Distill-Qwen-14B), and resume screening (8 resumes with yes-rate ~46%; Qwen3-8B). Math appears only in an appendix. Selection explicitly targets "considerable variability in final answers" — [arXiv 2510.27484](https://arxiv.org/html/2510.27484)
- Wei, Belkhiter, Miehling, Marinescu, "Local Causal Attribution of Chain-of-Thought Reasoning" (AttriCoT), arXiv 2606.21821 (20 Jun 2026), ICML 2026 Mechanistic Interpretability Workshop. It covers 5 datasets and 4 reasoning models; the dataset names were not in the abstract — [arXiv 2606.21821](https://arxiv.org/abs/2606.21821)
- Bigelow et al., "Forking Fast: Efficiently Estimating Uncertainty Dynamics in Text Generation," arXiv 2608.19611 (20 Aug 2026; listed at NeurIPS 2026). This is a cheaper forking-paths analysis with sentence-level resampling on R1-Distill-Llama-8B. Datasets were not in the abstract — [arXiv 2608.19611](https://arxiv.org/abs/2608.19611); [NeurIPS listing](https://neurips.cc/virtual/2026/157612); predecessor "Forking Paths in Neural Text Generation" arXiv 2412.07961 — [arXiv](https://arxiv.org/pdf/2412.07961)
- Arcuschin, Janiak, Krzyzanowski, Rajamanoharan, Nanda, Conmy, "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful," arXiv 2503.08679 (ICLR 2025 workshop). It uses Putnam problems for "unfaithful illogical shortcuts" and comparative yes/no questions for post-hoc rationalisation — [arXiv 2503.08679v4](https://arxiv.org/html/2503.08679v4); [mlanthology](https://mlanthology.org/iclrw/2025/arcuschin2025iclrw-chainofthought)
- [from memory, not re-verified] Chen et al. (Anthropic), "Reasoning Models Don't Always Say What They Think," arXiv 2505.05410, uses hinted MMLU and GPQA multiple-choice questions.

### Inferences
- Pairing MATH with AIME 2025 matches current practice; Du et al. 2026 is the closest precedent. AIME 2026 is now available and is cleaner still. A variability or intermediate-accuracy filter (as in Thought Anchors and Thought Branches) is standard and defensible, as long as it is re-measured with enough samples.

### Gaps
- I could not confirm the dataset lists for AttriCoT and Forking Fast without full-text access in this session.
- I found no 2025–2026 CoT-resampling paper that uses HMMT, OlympiadBench, Omni-MATH or LiveBench as its main source. This does not mean none exists.

## Q5. Does problem difficulty affect the number or kind of important sentences, or answer uncertainty?

### Takeaway
Yes, there is direct evidence. On easy, well-attested benchmarks (GSM8K, MATH500), far fewer reasoning steps have measurable causal effect on the answer than on AIME. Thought Anchors' own MMLU analysis finds easier questions give more local and fewer long-range sentence dependencies. Reasoning length also varies non-monotonically with difficulty.

### Cited Findings
- Du et al. 2026: "the easier and better-attested GSM8K and MATH500 show a lower proportion of self-advantage steps, whereas the harder AIME datasets almost always have at least one reasoning step" with a causal effect — [arXiv 2609.04194](https://arxiv.org/html/2609.04194v1)
- Thought Anchors (MMLU analysis): "questions with high average accuracy elicit CoTs with stronger close-range links and weaker long-range links" — [arXiv 2506.19143](https://arxiv.org/html/2506.19143)
- Forking Fast: "uncertainty dynamics converge to stable patterns, and noise is largely an artifact of sampling." This matters for low-rollout importance estimates — [arXiv 2608.19611](https://arxiv.org/abs/2608.19611)
- Reasoning length vs difficulty: LLMs "overthink" simple problems and "underthink" hard ones (arXiv 2505.00127) — [arXiv 2505.00127](https://arxiv.org/pdf/2505.00127). For R1-Distill-7B on MATH-500, accuracy peaks at a mid-length decile (91.1%) and falls to 24.4% in the longest decile, so long traces signal struggle (arXiv 2604.10739) — [arXiv 2604.10739](https://arxiv.org/pdf/2604.10739)

### Inferences
- For the study, easy or memorised MATH problems likely produce fewer and weaker anchors and lower answer entropy along the trace. Importance is then concentrated near the end, or is noise-dominated. An AIME 2025 arm should yield more anchors and more mid-trace uncertainty. Difficulty, measured as accuracy from start, should therefore be a covariate or stratifier in any anchor-reliability analysis, not something to average over.
- Since the study retains both correct and incorrect base traces, incorrect traces on easy problems may be atypical, for example grader or format failures. These deserve separate inspection.

### Gaps
- I found no study that directly regresses the *count* of Thought-Anchors-style counterfactual anchors on per-problem accuracy for R1-Distill-Qwen-14B.

## Q6. Uncontaminated post-cutoff math sources for a January 2025 model, and R1-Distill-14B accuracy on them

### Takeaway
AIME 2025 (Feb 2025), HMMT Feb 2025, BRUMO 2025 and SMT 2025 all postdate the model's 20 Jan 2025 release, and MathArena reports accuracies for all four (49%, 32%, 68% and 55% respectively). AIME 2026, HMMT 2026, CMIMC and LiveMathBench are further options, but MathArena reports no 14B numbers for them.

### Cited Findings
- R1-Distill-14B on post-release competitions (MathArena): AIME 2025 49.17%, HMMT Feb 2025 31.67%, BRUMO 2025 68.33%, SMT 2025 54.72% — [MathArena](https://matharena.ai/models/deepseek_deepseek_distill_14b)
- AIME 26 is already used as a source in a Sep 2026 CoT-importance paper — [arXiv 2609.04194](https://arxiv.org/html/2609.04194v1)
- LiveMathBench shows ~0–5% partial-prompt completion for Qwen, i.e. it looks clean — [arXiv 2507.10532v3](https://arxiv.org/html/2507.10532v3)

### Inferences
- AIME 2025 at ~49% mean accuracy is a good fit for a 25–75% intermediate-accuracy filter: many of its 30 problems should fall in band. Yield will be small (30 problems, of which perhaps 10–15 are intermediate), so pooling AIME 2025 + 2026 or adding HMMT Feb 2025 or BRUMO 2025 would raise the count. AIME answers are integers from 0 to 999, which makes grading clean.
- HMMT Feb 2025 (~32%) is harder. More problems will be near 0, which gives low-variance outcomes that are uninformative for resampling importance.
- "Post-release" does not guarantee clean data. Problems can resemble older AoPS material. MathArena's uncontamination claim depends on timing.

### Gaps
- No MathArena 14B results for AIME 2026, HMMT 2026 or CMIMC 2025. The study would need to measure these itself.
- Release date of R1-Distill (20 Jan 2025) and the R1 paper's Nature publication (2025) are [from memory, not re-verified in this session].
