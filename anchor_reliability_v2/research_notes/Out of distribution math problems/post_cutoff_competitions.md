# Post-January-2025 math competition problem sets as OOD test problems for DeepSeek-R1-Distill-Qwen-14B

Notes compiled 2026-10-11. Labels used below:
- **[verified]**: checked against a primary source (MathArena page, Hugging Face API or dataset, arXiv abstract).
- **[computed]**: worked out by me from the raw MathArena or OlymMATH output datasets on Hugging Face, queried with DuckDB over the parquet files.
- **[unverified]**: comes from a secondary source, a search snippet or background knowledge, and should be checked before it is relied on.

R1-Distill-Qwen-14B was released on 2025-01-20 [unverified date; the brief says January 2025]. So every competition below that was held after February 2025 is after the model's release.

## Q1. Inventory of post-January-2025 competitions (dates, counts, format, availability, licence, accuracy)

### Takeaway
MathArena evaluated R1-Distill-Qwen-14B on four post-cutoff answer-based sets: AIME 2025, HMMT Feb 2025, BRUMO 2025 and SMT 2025. That is 143 problems, with 4 samples per problem at a 32k-token cap, temperature 0.6 and top-p 0.95. The 14B model's accuracy was 49.2%, 31.7%, 68.3% and 54.7% respectively. MathArena publishes every raw trace with per-sample correctness, so per-problem accuracy can be computed. The licence is CC BY-NC-SA 4.0. Later sets (CMIMC 2025, HMMT Nov 2025, AIME 2026, HMMT Feb 2026) are on MathArena too, but without the 14B model or any R1-Distill model.

### Cited Findings

**MathArena results for R1-Distill-Qwen-14B (the model page)**
- MathArena's model page reports the following for "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" (settings: vllm, max tokens 32,000, temperature 0.6, top-p 0.95) [verified] — [MathArena model page](https://matharena.ai/models/deepseek_deepseek_distill_14b):

  | Set | Accuracy | Rank |
  |---|---|---|
  | AIME 2025 | 49.17% ± 8.94% | 51/61 |
  | HMMT Feb 2025 | 31.67% ± 8.32% | 50/60 |
  | BRUMO 2025 | 68.33% ± 8.32% | 40/45 |
  | SMT 2025 | 54.72% ± 6.70% | 44/44 |

- The raw output datasets confirm 4 samples per problem: n=120 for the 30-problem sets and n=212 for SMT's 53 problems [computed] — [MathArena/aime_2025_outputs](https://huggingface.co/datasets/MathArena/aime_2025_outputs).

**Comparable models in the same MathArena runs [computed]** — from the [MathArena HF org](https://huggingface.co/MathArena) `*_outputs` datasets

| Model | AIME 2025 | HMMT Feb 2025 | BRUMO 2025 | SMT 2025 |
|---|---|---|---|---|
| R1-Distill-1.5B | 20.0% | 11.7% | – | – |
| **R1-Distill-14B** | **49.2%** (mean 12.4k output tokens, p90 20.9k) | **31.7%** (mean 15.6k, p90 27.7k) | **68.3%** (mean 10.9k, p90 24.6k) | **54.7%** (mean 12.4k, p90 24.7k) |
| R1-Distill-32B | 60.0% | 33.3% | 68.3% | 60.4% |
| R1-Distill-70B | 55.0% | 33.3% | 66.7% | 60.8% |
| QwQ-32B | 65.8% | 47.5% | – | – |
| Qwen3-30B-A3B | 70.0% | 50.8% | 77.5% | 67.9% |
| Similar to 14B: LIMO | 49.2% | 30.0% | – | – |
| Similar to 14B: s1.1-32B | 50.0% | 37.5% | – | – |
| Similar to 14B: Claude-3.7-Sonnet (Think) | 49.2% | 31.7% | 65.8% | 56.6% |

- MathArena lists no Qwen3-14B result in these datasets [computed].

**Self-reported accuracies (Qwen3 technical report)**
- Table 15 reports AIME'25 scores (64 samples per question, max output 38,912 tokens): Qwen3-14B (Thinking) 70.4, Qwen3-8B (Thinking) 67.3, QwQ-32B 69.5, R1-Distill-Qwen-32B 49.6, R1-Distill-Qwen-14B 56.3 — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/html/2505.09388v1).
- Caveat: these values were pulled out by a summarising fetch tool. The 14B scoring above the 32B is unusual and should be re-checked against the PDF [unverified extraction].

**Per-competition inventory**

**AIME 2025 I/II**
- MathArena: 30 problems (I and II combined); answers are integers 0–999 [verified] — [MathArena competitions](https://matharena.ai/competitions).
- Dates: AIME I on 2025-02-06 and AIME II on 2025-02-12 [unverified; standard MAA schedule; see the AoPS pages [2025 AIME I](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I) and [2025 AIME II](https://artofproblemsolving.com/wiki/index.php/2025_AIME_II)].
- Problems and answers are on the AoPS wiki (links above). The MathArena HF dataset `MathArena/aime_2025` (created 2025-05-10) is CC BY-NC-SA 4.0 [verified via HF API].
- MAA terms: reproducing problems or solutions "for revenue-generating purposes requires written permission" from the MAA, and problems should be credited, e.g. "2017 AMC 12 B, Problem #21" — [MAA notice as reproduced in search results; primary MAA page not fetched](https://maa.org/student-programs/amc/) [unverified wording].

**AIME 2026**
- Dates: AIME I on 2026-02-05 and AIME II on 2026-02-11 — [AoPS 2026 AIME I](https://artofproblemsolving.com/wiki/index.php/2026_AIME_I); [MAA thresholds news](https://maa.org/news/2025-26-aime-thresholds-are-now-available/) [verified via search snippet].
- MathArena: 30 problems; dataset `MathArena/aime_2026` created 2026-02-13; CC BY-NC-SA 4.0 [verified].
- No R1-Distill or QwQ result. The weakest models evaluated were Qwen3-4B-2507-Think and QED-Nano, both at 82.5% [computed].

**HMMT February 2025**
- Date: 2025-02-15 [unverified].
- MathArena: 30 problems. The dataset `MathArena/hmmt_feb_2025` was created 2025-04-16 and is CC BY-NC-SA 4.0. It states "Questions were extracted, converted to LaTeX and verified" [verified] — [HF card](https://huggingface.co/datasets/MathArena/hmmt_feb_2025).
- Answers are mixed: integers, fractions, radicals (e.g. `\frac{9\sqrt{23}}{23}`, `1-\frac{2}{\pi}`), one multi-value answer, and one `2^{25}\cdot 26!` [computed from gold answers].
- 14B accuracy: 31.7%.

**HMMT February 2026**
- Date: 2026-02-14 — [Wikipedia: HMMT](https://en.wikipedia.org/wiki/HMMT) via search snippet [verified as snippet only].
- MathArena: 33 problems; dataset created 2026-02-19; CC BY-NC-SA 4.0 [verified].
- No R1-Distill result. The weakest model evaluated was Qwen3-4B-2507-Think at 53.0% [computed].

**HMMT November 2025**
- Date: early November 2025, probably 2025-11-08 [unverified].
- MathArena: 30 problems; dataset `MathArena/hmmt_nov_2025` created 2025-11-17; CC BY-NC-SA 4.0 [verified].
- No R1-Distill result. The weakest model evaluated was QED-Nano at 75.0% [computed].

**BRUMO 2025**
- Held on Brown's campus on 2025-04-04/05 (inaugural year, about 250 students) — [Brown Daily Herald](https://www.browndailyherald.com/article/2025/04/how-a-team-of-brown-students-hosted-a-math-olympiad-for-250-high-school-students).
- MathArena: 30 problems; dataset created 2025-05-13; CC BY-NC-SA 4.0 [verified].
- Answers are mixed integer/expression; one answer is a set of complex roots [computed].
- 14B accuracy: 68.3%.

**SMT 2025**
- Held 2025-04-11/12 [search snippet; unverified primary].
- MathArena: 53 problems; dataset created 2025-05-14; CC BY-NC-SA 4.0 [verified].
- Answers include transcendental expressions, e.g. `-\frac{\pi e}{2}`, `-1013\ln(2026)` and `\frac{\sin(6\pi/13)}{\sin...}` [computed]. These may need a symbolic checker.
- 14B accuracy: 54.7%.

**CMIMC 2025**
- Date: spring 2025, likely April [unverified].
- MathArena: 40 problems; dataset created 2025-08-11; CC BY-NC-SA 4.0 [verified].
- No R1-Distill result. The weakest models were Grok 3 Mini (low) at 37.5% and o4-mini (low) at 46.3% [computed]. For comparison, those two models score 65.0% and 61.7% on AIME 2025.

**PUMaC 2025, ARML 2025, AMC 10/12 2025**
- None of these is on MathArena [verified absence from the competitions list — [MathArena competitions](https://matharena.ai/competitions)].
- AMC dates: 10A/12A on 2025-11-05 and 10B/12B on 2025-11-13 — [MAA AMC page via search](https://maa.org/student-programs/amc/).
- AMC is multiple-choice with 25 problems per paper [background, unverified]. It can be turned into a short-answer test by dropping the options.
- No accuracies found for the R1-Distill models on any of these.
- No dates or counts were found for PUMaC 2025 (November 2025) or ARML 2025 (late May/June 2025). See Gaps.

**USAMO 2025, IMO 2025, Putnam 2025 (proof-based)**
- MathArena: USAMO 2025 has 6 problems, IMO 2025 has 6, Putnam 2025 has 12, plus IMC 2025 (10) and Miklós Schweitzer 2025 (10). All are proof-based and human-graded [verified] — [MathArena competitions](https://matharena.ai/competitions); [GitHub README](https://github.com/eth-sri/matharena).
- They are unsuitable for automatic answer checking.
- MathArena's paper says top models reach "slightly less than 40%" on IMO 2025 proofs — [arXiv 2505.23281](https://arxiv.org/abs/2505.23281).

**Other MathArena sets**
- Apex 2025 has 12 problems, chosen because top models did poorly on them; Apex Shortlist has 47. Both are far too hard for a 14B model.
- Project Euler (50 problems) is computational.
- Kangaroo 2025 is multiple-choice and visual.
- ArXivMath is research-level, released monthly from 12/2025.
- Source: [MathArena competitions](https://matharena.ai/competitions).

**National olympiads with answer rounds (China, Brazil, Indonesia, etc.)**
- No leaderboard covers them, and none of their 2025 papers turned up in a structured dataset. See Gaps.

### Inferences
- **Available now with 14B ground truth:** AIME 2025, HMMT Feb 2025, BRUMO 2025 and SMT 2025 (143 problems in total). Per-problem pass rates already exist for these, but each rests on only 4 samples.
- **CMIMC 2025 and HMMT Feb 2026 would be harder for the 14B than AIME 2025.** The weak anchor models score 25–30 points lower on CMIMC than on AIME 2025. QED-Nano drops from 76.7% on HMMT Feb 2025 to 64.4% on HMMT Feb 2026. My guess for the 14B is roughly 25–35% on both.
- **AIME 2026 and HMMT Nov 2025 look similar to, or easier than, AIME 2025 for small models.** QED-Nano scores 82.5% on AIME 2026 vs 77.5% on AIME 2025, and 75% on HMMT Nov 2025. My guess for the 14B is around 45–60%.
- All of these guesses are inference and must be confirmed by sampling.
- **Licensing:** the MathArena redistributions are CC BY-NC-SA 4.0, which suits non-commercial research but requires share-alike for derived datasets. The underlying copyright stays with the contest organisers (MAA, HMMT and others). MathArena's README says nothing about permissions.

### Gaps
- Exact dates for AIME 2025 I/II, HMMT Feb 2025, HMMT Nov 2025, CMIMC 2025, USAMO 2025, IMO 2025 and Putnam 2025 were not checked against primary pages. The values given are standard-calendar recollections.
- I did not find the licence or terms pages for HMMT, SMT, CMIMC or BRUMO themselves.
- PUMaC 2025, ARML 2025 and AMC 2025: no dates, counts or model accuracies were checked, and no public LLM evaluations were found.
- No 14B results exist anywhere for CMIMC 2025, HMMT Nov 2025, AIME 2026 or HMMT Feb 2026.
- No Qwen3-14B result appears in MathArena's datasets.

## Q2. Aggregated live benchmarks

### Takeaway
MathArena is the main source. It publishes per-model results and the full raw traces, with a per-sample `correct` flag, for every competition, all on Hugging Face under CC BY-NC-SA 4.0. That makes per-problem accuracy fully recoverable. LiveAoPSBench and LiveMathBench exist but are less directly useful. AIMO-3's 110 problems are private apart from a 10-problem public reference set.

### Cited Findings

**MathArena**
- Paper: "MathArena: Evaluating LLMs on Uncontaminated Math Competitions" by Balunović, Dekoninck, Petrov, Jovanović and Vechev. Submitted 2025-05-29, revised 2026-01-14, and appeared at NeurIPS 2025 Datasets & Benchmarks. It covers 162 problems across seven competitions in the paper version — [arXiv 2505.23281](https://arxiv.org/abs/2505.23281); [NeurIPS](https://neurips.cc/virtual/2025/poster/121385).
- The site lists ArXivLean, BrokenArXiv, ArXivMath, Kangaroo 2025, AIME 2025/2026, HMMT Feb 2025/2026, HMMT Nov 2025, BRUMO 2025, SMT 2025, CMIMC 2025, Apex, USAMO 2025/2026, IMO 2025, IMC 2025, Miklós Schweitzer 2025, Putnam 2025 and Project Euler — [MathArena competitions](https://matharena.ai/competitions).
- Per-problem data:
  - The `MathArena/*_outputs` datasets have the fields `problem_idx, model_name, idx_answer, answer, output_tokens, gold_answer, parsed_answer, correct`, plus the full `all_messages` trace — [HF datasets-server info for aime_2025_outputs](https://datasets-server.huggingface.co/info?dataset=MathArena/aime_2025_outputs).
  - Output datasets confirmed to exist [verified]: aime_2025, hmmt_feb_2025, brumo_2025, smt_2025, cmimc_2025, hmmt_nov_2025, aime_2026 and hmmt_feb_2026.
  - The GitHub repo is MIT-licensed and also offers a local per-problem viewer — [eth-sri/matharena](https://github.com/eth-sri/matharena).
  - The website's main page does not show per-problem tables in its static content [verified via fetch], but the raw data covers this.

**LiveAoPSBench**
- Paper: "Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation", arXiv 2501.14275, v1 2025-01-24, CC BY 4.0.
- It builds an "evolving evaluation set with timestamps" from new AoPS forum posts. It reports "a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure" — [arXiv 2501.14275](https://arxiv.org/abs/2501.14275).
- The size, time window and any R1-Distill results were not found in the abstract.

**LiveMathBench (OpenCompass)**
- The HF dataset `opencompass/LiveMathBench` was created 2024-12-23 and last modified 2025-08-05. It is CC BY 4.0 and gated, so login is required [verified via HF API] — [HF](https://huggingface.co/datasets/opencompass/LiveMathBench).
- Release versions and contents were not inspected.

**AIMO Progress Prize 3**
- 110 "entirely original problems", a $2.2M prize pool, and closing on 2026-04-15. Only a 10-problem reference set is public. Example answer format: "m/n ... final answer m+n" (integer) — [search summary; AIMO site](https://aimo.xtxmarkets.com/about) [unverified details; not fetched directly].
- An "AIMO Interpretability Challenge" paper exists — [arXiv 2607.13899](https://arxiv.org/pdf/2607.13899). Not read.

### Inferences
- For this study, MathArena's raw outputs let us pick problems by the 14B's empirical pass rate. The same parquet files can also serve as reference traces under MathArena's own sampling settings.

### Gaps
- Whether the AIMO-3 full problem set was released after the competition closed (April 2026) is unknown.
- LiveAoPSBench's current size and any results for small distilled models were not found.

## Q3. How many problems fall in the 15–85% band for R1-Distill-Qwen-14B

### Takeaway
On MathArena's four 14B-evaluated sets, 44 of 143 problems had 1–3 of 4 samples correct, which is the closest a 4-sample estimate gets to the 15–85% band. Another roughly 14 problems scored 0/4 for the 14B but were solved at least once by R1-Distill-32B, so they are plausible low-band candidates. Only about 34 of the 44 have correct traces averaging 16k tokens or fewer. With 4 samples, the band assignments are very noisy and need re-estimating with at least 16–32 fresh samples.

### Cited Findings
All per-problem results below are **[computed]** from the MathArena `*_outputs` datasets: 4 samples per problem, 32k token cap.

| Set | Problems | 14B 1–3/4 correct (≈ band) | …of which mean tokens of correct 14B traces ≤16k | 14B 0/4 but 32B > 0 |
|---|---|---|---|---|
| AIME 2025 | 30 | 13 (P2, 8, 9, 12, 18, 20–25, 27, 29) | 13 | 4 (P7, 11, 14, 26) |
| HMMT Feb 2025 | 30 | 8 (P5–10, 12, 28) | 4 | 4 (P4, 11, 16, 27) |
| BRUMO 2025 | 30 | 8 (P3, 14, 17, 21, 22, 24–26) | 8 | 3 (P13, 27, 29) |
| SMT 2025 | 53 | 15 (P1, 4–6, 9, 10, 17, 21, 24, 33, 35, 36, 39, 50, 52) | 9 | 3 (P18, 49, 51) |
| **Total** | **143** | **44** | **34** | **14** |

- Problem counts outside the band:
  - AIME 2025: 11/30 at 4/4 and 6/30 at 0/4 for both 14B and 32B.
  - HMMT Feb 2025: 14/30 at 0/4 for the 14B. It has the most problems the 14B never solves.
- Token lengths: for in-band problems, mean 14B output length is typically 10–20k tokens. On some HMMT and SMT problems, some samples hit the 32k cap.
  - Examples: HMMT P7, P8, P10 and P12 have a maximum of 32,000 tokens and correct-trace means of 17–21k.
  - SMT P9 averages 24k tokens in correct traces, and SMT P52 averages 28k.
- Overall 14B output-length p90 is 20.9k on AIME 2025, 27.7k on HMMT Feb 2025, 24.6k on BRUMO and 24.7k on SMT [computed].

**OlymMATH (64 samples per problem, from `RUC-AIBOX/OlymMATH-eval`) [computed]** — [HF OlymMATH-eval](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH-eval)

| Subset | 14B accuracy | Problems in 15–85% | …with mean correct-trace tokens ≤16k | In 25–75% |
|---|---|---|---|---|
| EN-EASY (100 problems) | 61.8% | 67 | 45 | 53 |
| EN-HARD (100 problems) | 13.3% | 29 | 14 | 21 |

- For comparison on EN-EASY / EN-HARD:
  - R1-Distill-7B: 47.5% / 11.1%
  - R1-Distill-32B: 67.3% / 16.9%
  - QwQ-32B: 84.0% / 23.1%
- The generation cap used in OlymMATH-eval was not checked.

### Inferences
- **Combining MathArena's four 2025 sets** (44 core plus about 14 borderline problems) gives about 50–60 candidate post-cutoff problems in the band. That is enough for the 50–100 target only if the borderline problems survive re-estimation.
  - Applying the 16k trace limit cuts the core to about 34.
  - AIME 2025 and BRUMO 2025 are the most "16k-friendly". HMMT Feb 2025 and SMT 2025 in-band problems often need more than 16k tokens.
- **Adding sets not yet tested on the 14B** (CMIMC 2025, HMMT Nov 2025, AIME 2026, HMMT Feb 2026: 133 problems) would plausibly add another 30–50 in-band problems. This is inferred from the band yield of about 30% on the 2025 sets, and must be measured.
- **OlymMATH-EN-EASY alone yields about 67 in-band problems** (45 within 16k), measured with 64 samples. It is the richest single source, but its problems come from older printed sources rather than post-2025 contests. See Q5.
- **Statistical caveat:** with 4 samples, a problem with a true pass rate of 15% shows 0/4 about 52% of the time, and one at 85% shows 4/4 about 52% of the time. Many true-band problems are therefore hidden in the 0/4 and 4/4 groups, and some 1/4 or 3/4 problems are really outside the band.
- **Differences from the study's setup:** MathArena's settings (temperature 0.6, top-p 0.95, 32k cap) match DeepSeek's recommended settings. However, a 16k cap would turn some currently-correct long traces into failures and push those problems down in accuracy.

### Gaps
- MathArena reports no 14B per-problem results for CMIMC 2025, HMMT Nov 2025, AIME 2026 or HMMT Feb 2026. Fresh sampling is needed.

## Q4. Contamination incidents

### Takeaway
The best-documented contamination finding is MathArena's evidence that AIME 2024 is contaminated, which is pre-cutoff and so out of scope. For AIME 2025 there is a secondary-source report that a problem identical to Problem 1 existed on Quora before the contest. I found no MathArena-documented leaks for HMMT, BRUMO, SMT or CMIMC 2025.

### Cited Findings
- MathArena found "strong signs of contamination in AIME 2024" and reports that harder 2025 competitions such as CMIMC 2025 still show strong reasoning — [arXiv 2505.23281](https://arxiv.org/abs/2505.23281).
- A search summary reports that someone used OpenAI Deep Research to look for problems similar to AIME 2025 and found an identical copy of AIME 2025 Q1 on Quora. The conclusion drawn was that "problems near identical to the test set can be found online" — surfaced via search; the original source appears to be a LessWrong/GreaterWrong shortform comment ([greaterwrong.com](https://www.greaterwrong.com/posts/RDG2dbg6cLNyo6MYT/thane-ruthenis-s-shortform/comment/vxHcdFb3KLWEQbmRv)) [unverified; original post not read; problem number and detail not confirmed].
- LiveAoPSBench found LLM accuracy falls on newer AoPS problems compared with older ones, which is consistent with pre-training exposure inflating older-problem scores — [arXiv 2501.14275](https://arxiv.org/abs/2501.14275).

### Inferences
- **Training-data timing:** R1-Distill-Qwen-14B's weights were fixed in January 2025, and its Qwen2.5 base was pretrained in 2024. So training-data contamination by the actual 2025 or 2026 contest problems is impossible.
- **Remaining risk:** near-duplicates of older problems that are in the training data, like the AIME 2025 P1 report. Easy, early-numbered problems (e.g. AIME P1) are the most likely to be standard textbook types. They mostly fall in the "always solved" group anyway.

### Gaps
- I did not find a systematic near-duplicate audit of the 2025/2026 sets, by MathArena or anyone else.
- MathArena's paper may contain more analysis. A local copy of the PDF was saved but not fully parsed.

## Q5. Other new problem collections

### Takeaway
- **Newly written (2025):** AMO-Bench (50 problems, Oct 2025) and AIMO-3 (private).
- **Old problems revised or curated offline (2025):** BeyondAIME (100 integer problems, CC0, June 2025), OlymMATH (200 numeric-answer problems, March 2025, MIT) and IMO-AnswerBench (400 problems, Nov 2025).
- **Best fit for the 14B:** OlymMATH-EN-EASY, which has measured 14B per-problem data.
- **Probably too hard for a 14B (inferred):** AMO-Bench and BeyondAIME.

### Cited Findings

**OlymMATH (RUC AIBOX)**
- arXiv 2503.21380, submitted 2025-03-27; ACL 2026 Main — [arXiv](https://arxiv.org/abs/2503.21380).
- Size: 200 natural-language problems with numerical answers (EASY and HARD, 100 each), each in English and Chinese, plus 150 Lean problems.
- Novelty: problems were "manually sourced from printed publications" rather than online sources.
- Licence: MIT. HF created 2025-03-27 — [HF](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH).
- 14B accuracy: 61.8% on EASY and 13.3% on HARD [computed; see Q3].

**BeyondAIME (ByteDance Seed)**
- HF `ByteDance-Seed/BeyondAIME`, created 2025-06-17, CC0-1.0 — [HF](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME).
- 100 problems with positive-integer answers [count from a secondary source: [wizwand](https://www.wizwand.com/dataset/beyond-aime)].
- The card says:
  - difficulty is "≥ AIME Problems #11-15";
  - "every problem has been manually revised to be unique";
  - pseudo-proof problems are reformulated.
- Problems are sourced from high-school and university competitions.

**AMO-Bench (Meituan LongCat)**
- arXiv 2510.26768, submitted 2025-10-30.
- 50 "entirely original" human-crafted problems at IMO difficulty or above. Final-answer only. Best model 52.4%, with most models below 40% — [arXiv](https://arxiv.org/abs/2510.26768).
- HF dataset has 50 test rows with `answer_type` and expert `solution`; MIT licence; created 2025-10-29 — [HF](https://huggingface.co/datasets/meituan-longcat/AMO-Bench).

**IMO-AnswerBench (Google DeepMind, part of IMO-Bench)**
- arXiv 2511.01846, submitted 2025-11-03; EMNLP 2025; paper licence CC BY 4.0.
- 400 olympiad problems with verifiable short answers, 100 each in algebra, combinatorics, geometry and number theory, drawn from national, regional and international olympiads.
- Gemini Deep Think scores 80.0% — [arXiv](https://arxiv.org/abs/2511.01846); [imobench.github.io](https://imobench.github.io).
- Whether problems were modified ("robustified") to resist memorisation was not confirmed from the abstract.

**MathOdyssey, HLE math subset, RealMath, FrontierMath (not researched in this pass)**
- None of these was checked in this pass. Background knowledge only, all [unverified]:
  - MathOdyssey: 2024, so pre-cutoff.
  - Humanity's Last Exam: released January 2025, research-level.
  - FrontierMath (Epoch AI): private, research-level.
  - RealMath: 2025, built from arXiv content.
- All four are expected to be far too hard (near 0%) for a 14B distill.

### Inferences
- **OlymMATH-EN-EASY:** about 67 problems in the 15–85% band for the 14B, measured with 64 samples. It is the best non-competition supplement. Novelty depends on offline sourcing rather than post-2025 authorship, so near-duplicates in pretraining data are possible.
- **BeyondAIME:** given its "≥ AIME 11–15" difficulty and the 14B's 0–25% on AIME 2025 problems 10–15, overall 14B accuracy is plausibly about 10–25%, with a minority of problems in the band. CC0 makes it the easiest to redistribute.
- **AMO-Bench:** with frontier models below 40–52%, a 14B distill is likely in the single digits.
- **IMO-AnswerBench:** likely low for a 14B, and its sources include pre-2025 olympiads.

### Gaps
- No 14B (or Qwen3-14B) results were found for BeyondAIME, AMO-Bench or IMO-AnswerBench.
- MathOdyssey, HLE-math, RealMath and FrontierMath details were not verified.
- The generation cap used for OlymMATH-eval was not checked.
