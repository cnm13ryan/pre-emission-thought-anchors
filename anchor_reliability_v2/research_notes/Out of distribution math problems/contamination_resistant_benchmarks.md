# Contamination-resistant, perturbed, and live math benchmarks for out-of-distribution problems for DeepSeek-R1-Distill-Qwen-14B (as of 2026-10-11)

Target model: DeepSeek-R1-Distill-Qwen-14B (released 2025-01-20/21; Qwen2.5-14B base). Its reference numbers are 69.7% pass@1 on AIME 2024 and 93.9% on MATH-500 ([DeepSeek-R1 paper, arXiv 2501.12948](https://arxiv.org/html/2501.12948v1)). AIME 2024 is a known contamination risk, so those numbers overstate how it does on novel problems. The study needs 50–100 problems with short answers that can be checked automatically, that the model solves 15–85% of the time, and whose traces fit in about 16k tokens.

Labels used below: **[V]** means I checked it against a primary source during this session. **[U]** means it comes from prior knowledge or a secondary source and was not checked this session. Treat [U] items as leads to confirm.

## Q1. Perturbation and functional-variant benchmarks

### Takeaway
MATH-Perturb (P-Hard) and Putnam-AXIOM Variation are the strongest options for "same skill, new answer". Both change the final answer and both show accuracy drops of 10–25 points (absolute) for reasoning models. Putnam-AXIOM is the only one where I verified a distilled-R1 number: R1-Distill-Qwen-32B drops from 33% to 24%. GSM-Symbolic, functional MATH and UGMathBench come with generators or templates that can produce new instances after today. However, GSM-level problems are far too easy for R1-Distill-14B (well above 85%). MATH-P-Hard and Putnam-AXIOM Variation are the likeliest to land in the 15–85% band.

### Cited Findings
**MATH-Perturb (arXiv 2502.06453, v2 12 Feb 2025)**
- [V] 279 problems, all Level-5 MATH (164 from the train split, 115 from test). Each has two hand-made variants: MATH-P-Simple (non-essential change, solvable by the same method) and MATH-P-Hard (fundamental change, the original method no longer works). Both variants have answers that differ from the original. Twelve PhD-level annotators built them with cross-validation. — [arXiv HTML](https://arxiv.org/html/2502.06453)
- [V] All 18 models tested drop 10–25% on P-Hard, e.g. o1-mini −16.49% and gemini-2.0-flash-thinking −12.9%. The main failure mode is "subtle memorization": the model applies the original problem's method and ignores the changed assumptions. — [arXiv abs/HTML](https://arxiv.org/html/2502.06453); [alphaXiv overview](https://www.alphaxiv.org/overview/2502.06453)
- [V] The paper excerpts I read have no R1-Distill-14B figure. — [arXiv HTML](https://arxiv.org/html/2502.06453)
- [U] The dataset is believed to be on Hugging Face/GitHub (project page "math-perturb"), but I did not find the link in the fetched text. There is no generator: the variants are hand-written and fixed.
- [V] A later paper, "Can Large Reasoning Models Improve Accuracy on Mathematical Tasks Using Flawed Thinking?" (arXiv 2512.17079), cites or uses MATH-Perturb with distilled reasoning models. I did not read its numbers. — [arXiv 2512.17079](https://arxiv.org/pdf/2512.17079)

**Putnam-AXIOM (arXiv 2508.08292, Aug 2025; ICML 2025; earlier NeurIPS 2024 MATH-AI workshop version)**
- [V] 522 Putnam problems, plus Putnam-AXIOM Variation: 100 functional variants made by programmatically perturbing variables and constants. Of these, 37 change constants (and so change the answer) as well as variables, and 63 change variable names only (answer unchanged). — [arXiv 2508.08292v2](https://arxiv.org/html/2508.08292v2)
- [V] Answers are boxed and checked by an equivalence function after normalisation. Licence is CC BY 4.0. Data and code are at https://github.com/brando90/putnam-axiom — [arXiv HTML](https://arxiv.org/html/2508.08292v2)
- [V] The authors say the variation protocol can produce an "unlimited stream of equally difficult, unseen instances". o1-preview drops 19.6 points (a 46.8% relative drop) in one version of the paper. — [papers.cool / abstract](https://papers.cool/arxiv/2508.08292)
- [V] Table 2: **DeepSeek-R1-Distill-Qwen-32B goes from 33% on Original to 24% on Variation**, GPT-4o from 30% to 22%, and o1-preview from 51% to 43%. — [arXiv HTML](https://arxiv.org/html/2508.08292v2)
- Caveat: only the 37 constant-change items have new answers. The 63 variable-rename items keep the original answer, so they only change the surface form.

**GSM-Symbolic (Apple, arXiv 2410.05229, Oct 2024; ICLR 2025)**
- [V] GSM8K problems are turned into parsable templates with variable domains (names, numbers) and constraints that keep the answer correct. New instantiations change both numbers and answers. GSM-NoOp adds an irrelevant clause and causes drops of up to 65%. — [arXiv 2410.05229](https://arxiv.org/html/2410.05229v1)
- [U] Apple released the templates and generated sets on GitHub (apple/ml-gsm-symbolic). Licence is probably Apple's sample-code licence (unverified).
- Inference: grade-school difficulty. R1-Distill-14B almost certainly scores above 90%, so this is out of the 15–85% band except perhaps the P2/NoOp variants.

**Functional MATH (Srivastava et al., arXiv 2402.19450, Feb 2024)** — [U] entirely
- [U] "Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap" (Saxena/Consequent AI). It rewrites MATH problems as code functions and releases quarterly snapshots (MATH() Dec-2023, etc.). It reports "reasoning gaps" of 58–80% between static and functional accuracy for 2023-era models. I did not verify a generator release or licence this session (repo believed to be consequentai/fneval). My first search with this ID did not surface the paper, so check the ID.

**GSM-Plus (Li et al., arXiv 2402.19255, ACL 2024)** — [U]
- [U] 10,552 perturbed GSM8K items across 8 perturbation types; some types change the answer (numerical substitution, digit expansion, adding an operation). The items are static with no generator, and the difficulty is GSM-level, so too easy.

**VarBench (Qian et al., arXiv 2406.17681, 2024)** — [U]
- [U] Variable-perturbation benchmarks for GSM8K, ARC, CommonsenseQA and TruthfulQA. Variables are sampled again each evaluation, so the answers change. Too easy for this model. My search did not surface the primary source.

**UGMathBench (arXiv 2501.13766, Jan 2025; ICLR 2025)**
- [V] 5,062 undergraduate problems across 16 subjects and 111 topics, with 10 answer types. Each problem has 3 randomized versions with changed numbers and answers. Metrics are effective accuracy (EAcc: correct on all 3 versions) and the reasoning gap Δ. The best EAcc is 56.3% (o1-mini). — [arXiv 2501.13766](https://arxiv.org/pdf/2501.13766)
- [U] It comes from WeBWorK-style randomizable problems. Whether a generator for new versions is public is unverified.

**DynaMath (arXiv 2411.00836, NeurIPS-era 2024)**
- [V] 501 seed questions, each written as a Python program that generates visual and textual variants. — [SlidesLive/abstract](https://slideslive.com/39033813/dynamath-a-dynamic-visual-benchmark-for-evaluating-mathematical-reasoning-robustness-of-vision-language-models?locale=cs)
- Inference: it is **multimodal** (aimed at vision-language models) and many items need an image, so it is largely unusable for a text-only 14B model.

**MATH-Rephrase** — I found no reliable primary source under this name. It may refer to rephrased MATH sets in data-augmentation papers such as MetaMath. **Gap.**

### Inferences
- For "new answer, same skill", the best ready-made sources are the 279 MATH-P-Hard problems and the 37 constant-changed Putnam-AXIOM variants. R1-Distill-14B scores about 94% on MATH-500, so Level-5 originals are probably at 85–95%. A 10–25 point P-Hard drop would put many P-Hard items roughly in the 60–85% range: the upper part of the target band. Putnam variants are probably at 20–35% (inferred from the 32B distill's 24%).
- Putnam-AXIOM's code is the only verified generator for hard problems. GSM-Symbolic's templates are a usable generator, but the difficulty is wrong for this model.
- Released fixed sets (MATH-P, GSM-Plus) were published after the model's January 2025 release. That makes them out of distribution for this model's training data, but not for later models.

### Gaps
- No verified R1-Distill-Qwen-14B accuracy on MATH-P-Simple/Hard, GSM-Symbolic or UGMathBench.
- I did not verify the licence or URL of the MATH-Perturb dataset.
- Functional MATH, GSM-Plus and VarBench details are unverified ([U]).

## Q2. Live and refreshed benchmarks

### Takeaway
The best post-January-2025 sources with verified R1-Distill-14B numbers are MathArena's 2025 competition sets. R1-Distill-14B scores 31.7% on HMMT Feb 2025, 49.2% on AIME 2025, 54.7% on SMT 2025 and 68.3% on BRUMO 2025, with average outputs of about 11–16k tokens. All four sit inside the 15–85% band, and together they give roughly 120+ problems. LiveAoPSBench's forum-timestamp pipeline can in principle be filtered to post-January-2025 posts, but the released split ends in August 2024. LiveBench holds back its newest questions.

### Cited Findings
**MathArena (ETH Zurich SRI Lab, arXiv 2505.23281, v1 29 May 2025, v3 14 Jan 2026; CC BY 4.0)**
- [V] Evaluates models on newly released competitions as soon as they are released. The paper covers 162 problems from 7 competitions (including AIME 2024, CMIMC 2025 and IMO 2025) and more than 50 models. It finds "strong signs of contamination in AIME 2024". — [arXiv abs](https://arxiv.org/abs/2505.23281)
- [V] **DeepSeek-R1-Distill-Qwen-14B on MathArena:**

  | Competition | Accuracy | Average output tokens |
  |---|---|---|
  | AIME 2025 | 49.17% ± 8.94 | 12,352 |
  | HMMT Feb 2025 | 31.67% ± 8.32 | 15,559 |
  | BRUMO 2025 | 68.33% ± 8.32 | 10,864 |
  | SMT 2025 | 54.72% ± 6.70 | 12,399 |

  Source: [MathArena model page](https://matharena.ai/models/deepseek_deepseek_distill_14b)
- [V] MathArena also runs an "ArXivMath" track built from recent arXiv papers. The search snippet says new versions are released monthly. — [matharena.ai/arxivmath](https://matharena.ai/arxivmath/)
- [U] The final-answer problem sets are on Hugging Face (MathArena/aime_2025, hmmt_feb_2025, etc.), with integer or short LaTeX answers. Sizes are believed to be AIME 30, HMMT Feb 30, BRUMO 30 and SMT about 53; confirm on the dataset cards.
- [V] Other post-cutoff competition sets with dates exist, such as HMMT Nov 2025 and AIME 2026, which are tracked by aggregators. — [benchlm AIME 2026](https://benchlm.ai/benchmarks/aime2026); [benchlm HMMT Nov 2025](https://benchlm.ai/benchmarks/hmmtnov2025)
- [V] There is a 2026 Korean CSAT math study (arXiv 2511.18649) in which questions were digitised within 2 hours of public release, to avoid leakage. — [arXiv 2511.18649](https://arxiv.org/pdf/2511.18649)

**LiveAoPSBench (arXiv 2501.14275, Jan 2025, v2 27 Jun 2025; ICML 2025)**
- [V] Built by an automated pipeline from Art of Problem Solving forum threads. The companion training set is AoPS-Instruct (more than 600k QA pairs). Each item is timestamped with its forum post date. LiveAoPSBench-2024 has 5,328 items from January 2023 to August 2024, and the main evaluation uses the 3,863 items from 2024. Answers are boxed. Labels are cross-checked by Llama-3.1-70B and Qwen2.5-72B, and equivalence is checked with SymPy. Previously seen problems are removed by substring and 8-gram matching. Licence is CC BY 4.0. — [arXiv HTML](https://arxiv.org/html/2501.14275); [GitHub DSL-Lab/aops](https://github.com/DSL-Lab/aops)
- [V] Accuracy falls steadily as post dates get more recent: 2023 scores are higher than 2024 scores for all 18 open models tested. This is direct evidence of a contamination effect. — [Pith summary](https://pith.science/paper/2501.14275); [emergentmind](https://www.emergentmind.com/topics/art-of-problem-solving-aops)
- [V] Because the pipeline is timestamped, a future split can be built from any cutoff date. The paper points to a leaderboard for current results. — [arXiv HTML](https://arxiv.org/html/2501.14275)
- Caveats: answers are labelled by LLMs, so some will be noisy. The released split predates the model's January 2025 release, so it is **not** out of distribution for R1-Distill-14B unless you rerun the pipeline on 2025–2026 posts with the public code.
- [V] The leaderboard URL quoted in the paper is livemathbench.github.io/leaderboard.html. This name collides with the separate "LiveMathBench" project; I did not resolve this. — [arXiv HTML](https://arxiv.org/html/2501.14275)

**LiveBench (arXiv 2406.19314; livebench.ai)**
- [V] The math category has competition problems, olympiad proof-ordering and synthetic hard (AMPS_Hard) tasks. The stated policy is to replace about 1/6 of the questions each update, a full refresh roughly every 6 months. — [arXiv 2406.19314](https://arxiv.org/pdf/2406.19314)
- [V] The GitHub README says the current release is 2025-04-25, "not all questions for this release are public on Huggingface", and 2024-11-25 is the latest fully public release. The `--livebench-release-option` flag filters by release. — [GitHub LiveBench](https://github.com/LiveBench/LiveBench)
- [U] Each question carries a `livebench_release_date` field on Hugging Face, so items released after 2025-01 can be filtered. I did not check whether any release after 2025-04 exists as of October 2026.

**LiveMathBench (Liu et al., arXiv 2412.13147, Dec 2024; "Are Your LLMs Capable of Stable Reasoning?", G-Pass@k metric)** — [U]
- [U] Versions 202412 and 202505 have been reported. They draw on recent contests (CNMO, AIME, etc.). Not verified this session.

**AntiLeak-Bench (Wu et al., arXiv 2412.13670, Dec 2024)** — [U]
- [U] Builds questions automatically from knowledge (Wikidata/Wikipedia) that appeared after a model's cutoff. It targets factual QA, not math, so it is of limited use here.

**LemmaBench (arXiv 2602.24173, Feb 2026)**
- [V] A "live, research-level" benchmark built from the latest results proved by mathematicians on arXiv. — [arXiv 2602.24173](https://arxiv.org/pdf/2602.24173)
- Inference: research-level, so probably below 15% for a 14B model.

### Inferences
- **Most practical choice for the study:** MathArena's 2025 final-answer sets (AIME 2025, HMMT Feb 2025, BRUMO 2025, SMT 2025, plus CMIMC 2025 and HMMT Nov 2025 / AIME 2026 if they have been evaluated). They give about 120–180 short-answer problems released after January 2025, with **verified per-model accuracy for R1-Distill-14B inside the band**. Per-problem accuracies are visible on the MathArena site, so items can be picked by their actual 14B solve rate.
- The token budget is tight. HMMT averages 15.6k output tokens, so a 16k cap will truncate many traces and lower accuracy. AIME, SMT and BRUMO average 11–12k and fit better.
- Temporal novelty is reliable here because the competitions were written after the model's weights were frozen (January 2025). Contamination in a fixed checkpoint can only come from its training data, which ended before release.

### Gaps
- I did not verify per-problem MathArena accuracy for 14B or exact dataset sizes.
- I did not check whether the LiveAoPSBench pipeline or leaderboard has a split after January 2025.
- I did not check LiveBench releases after 2025-04-25.

## Q3. Newly authored hard sets

### Takeaway
Most new hand-written olympiad sets (AMO-Bench, IMO-AnswerBench hard items, OlymMATH-HARD, BeyondAIME) are probably below 15% for a 14B distill and produce traces far longer than 16k tokens. OlymMATH-EASY and parts of IMO-AnswerBench are the plausible in-band candidates. Omni-MATH and HARDMath (October 2024) predate the model and are not out of distribution.

### Cited Findings
**OlymMATH (RUCAIBox, arXiv 2503.21380, v1 27 Mar 2025, v3 12 Apr 2026; ACL 2026 Main)**
- [V] 200 computational problems with numerical answers, split into EASY and HARD (100 each, reportedly) and given in English and Chinese, plus 150 Lean 4 problems (OlymMATH-LEAN). Problems were "manually sourced from printed publications to minimize data contamination". The release includes 582k+ reasoning trajectories. — [arXiv abs](https://arxiv.org/abs/2503.21380); [GitHub RUCAIBox/OlymMATH](https://github.com/RUCAIBox/OlymMATH)
- Caveat: the problems come from print, so they are novel only to the extent they were not digitised before. That is weaker than original authorship.
- [U] The paper reportedly evaluates R1-Distill models: EASY is moderate for 32B-class models, and HARD is very low. Not verified.

**AMO-Bench (Meituan / UCAS / HIT, arXiv 2510.26768, 30–31 Oct 2025)**
- [V] 50 entirely original problems, cross-validated by experts to be at least IMO difficulty. Answer types are numerical, set, variable expression and descriptive. 39 are graded by parser; 11 by an LLM (o1-mini, 5-vote majority), with 99.2% grading accuracy. The best model scores 52.4%, and most of the 26 models are below 40%. On the parser subset DeepSeek-R1 scores 11.7%. GPT-5-Thinking (High) averages about 37k output tokens. — [arXiv HTML](https://arxiv.org/html/2510.26768v1); [amo-bench.github.io](https://amo-bench.github.io)
- Inference: R1-Distill-14B would be close to 0–5% and over the token budget. Not suitable.

**IMO-Bench / IMO-AnswerBench (Google DeepMind, arXiv 2511.01846, Nov 2025)**
- [V] 400 "robustified" olympiad problems with short verifiable answers, 100 each in algebra, combinatorics, geometry and number theory. The suite also includes ProofBench (60) and GradingBench (1000). Gemini Deep Think scores 80.0%. — [arXiv 2511.01846](https://arxiv.org/html/2511.01846v1); [HF OpenEvals/IMO-AnswerBench](https://huggingface.co/datasets/OpenEvals/IMO-AnswerBench/blob/main/README.md)
- [U] "Robustified" means existing olympiad problems were changed (numbers, wording) so that they have new answers. That is a perturbation guarantee, not original authorship. A difficulty-tiered subset might contain items in the 14B band. No 14B number was found.

**BeyondAIME (ByteDance Seed, 2025)** — [U]
- [U] About 100 problems, harder than AIME, with integer answers. Reported as written or adapted to resist contamination, and available on HF (ByteDance-Seed/BeyondAIME). The search confirmed only that it is a curated advanced-math test set. — [awesomepapers entry](https://awesomepapers.io/reinforcement-learning/datasets/beyondaime)

**Omni-MATH (arXiv 2410.07985, Oct 2024)** — [U] ID
- Inference: about 4,428 olympiad problems scraped from public competitions before October 2024. **This predates the model's January 2025 release, and the problems are public, so it is not out of distribution.**

**HARDMath (arXiv 2410.09988, Oct 2024) / HARDMath2 (2025)** — [U]
- [U] Applied-math asymptotics problems. HARDMath is generated by a pipeline, but the October 2024 release predates the model. Answers are often expressions, which are harder to check automatically. Not verified.

**MathConstruct (Balunović et al., arXiv 2502.10197, Feb 2025)** — [U]
- [U] Constructive proofs checked by verifier code, with parameterised variations. The answers are constructions, not short answers, so it is not suitable for exact-match grading.

**RealMath (ETH SPY Lab, arXiv 2505.12575, May 2025; NeurIPS 2025)**
- [V] A refreshable pipeline that pulls verifiable statements with short answers from arXiv papers (math and cs) and Math StackExchange. The HF release has 633 math.arXiv, 111 cs.arXiv and 542 Math.SE samples. Frontier models score 43–49% on math.arXiv and up to 70% on Math.SE. — [arXiv 2505.12575](https://arxiv.org/pdf/2505.12575); [GitHub ethz-spylab/realmath](https://github.com/ethz-spylab/realmath)
- Inference: you can rerun the pipeline on arXiv papers posted after a chosen date to get problems that are new by date. The 14B score is probably low (perhaps 10–30% on Math.SE). Unverified.

**"EternalMath" or similar 2026 sets** — I found no source under that name. LemmaBench (Feb 2026) and MathArena ArXivMath are the 2026 live research-level options I found. **Gap.**

### Inferences
- Hand-written frontier sets are calibrated so that frontier models score below 50%. A model scoring about 50% on AIME 2025 will mostly fall below 15% on them.
- The in-band candidates are OlymMATH-EASY, possibly the easier tiers of IMO-AnswerBench, and the MathArena 2025 competitions.

### Gaps
- No verified R1-Distill-14B numbers for OlymMATH, IMO-AnswerBench, BeyondAIME or RealMath.
- I did not verify any licence other than AMO-Bench (arXiv licence only; the dataset licence is unclear) and those already listed.

## Q4. Contamination-free evaluation methodology (2024–2026)

### Takeaway
The literature points to layered defences. The core is a temporal cutoff: use problems created after the model's weights were frozen. The second layer is perturbation that changes the answer, so memorised answers fail. The third is to compare paired original and perturbed versions, or accuracy by date, as a contamination check. Hold back part of the set where possible.

### Cited Findings
- [V] A survey, "Benchmarking LLMs Under Data Contamination: From Static to Dynamic Evaluation" (arXiv 2502.17521), covers the move from static to dynamic and timestamped benchmarks, and lists LiveAoPSBench among them. — [arXiv 2502.17521](https://arxiv.org/html/2502.17521v2)
- [V] Evidence for a temporal signal: LiveAoPSBench accuracy falls as post dates approach the present ([arXiv 2501.14275](https://arxiv.org/html/2501.14275)), and MathArena flags contamination in AIME 2024 ([arXiv 2505.23281](https://arxiv.org/abs/2505.23281)).
- [V] Evidence for a perturbation signal: paired original/variant gaps (MATH-P-Hard: 10–25 points; Putnam-AXIOM: R1-Distill-32B 33 to 24) are read as memorisation. — [MATH-Perturb](https://arxiv.org/html/2502.06453); [Putnam-AXIOM](https://arxiv.org/html/2508.08292v2)
- [V] Evidence for holding items back: LiveBench withholds part of its newest release. — [GitHub](https://github.com/LiveBench/LiveBench)
- [V] Evidence for releasing items immediately after they are written: the CSAT 2026 study digitised questions within 2 hours of release. — [arXiv 2511.18649](https://arxiv.org/pdf/2511.18649)
- [V] There is a "Benchmark Health Index" framework for auditing benchmarks (arXiv 2602.11674). I did not read it. — [arXiv 2602.11674](https://arxiv.org/pdf/2602.11674)

### Inferences (recommended protocol for this study)
1. **Temporal cutoff (the strongest guarantee for a frozen January 2025 checkpoint).** Take problems from competitions held after 2025-01-20, e.g. the MathArena sets AIME 2025, HMMT Feb 2025, SMT 2025 and BRUMO 2025, where the 14B model scores 32–68%. For this fixed model, any problem written after that date is out of distribution by construction.
2. **Answer-changing perturbation as a second layer,** or for older problems. Use MATH-P-Hard items, or change the constants in Putnam-AXIOM / GSM-Symbolic-style templates and recompute the answer by hand or with code.
3. **Calibrate solve rates.** Run, say, 8–16 samples per problem with a 16k-token cap, and keep only problems with a 15–85% empirical solve rate. MathArena's 14B averages are about 11–16k tokens, so expect truncation to remove some HMMT items.
4. **Contamination check.** Compare accuracy on paired originals and their perturbations. Optionally run a completion or n-gram probe on the problem text. This is cheap insurance, though it is barely needed for problems written after the model's release.

### Gaps
- I did not read a dedicated 2025–2026 best-practice paper in full. The protocol above is my synthesis.
- MATH-Rephrase and EternalMath were not found.
