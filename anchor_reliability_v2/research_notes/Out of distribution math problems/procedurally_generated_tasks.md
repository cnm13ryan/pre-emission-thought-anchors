# Procedurally generated reasoning tasks as a second problem source for a thought-anchor study of DeepSeek-R1-Distill-Qwen-14B

Research date: 2026-10-11. All arXiv IDs and URLs were checked during this session unless marked **[UNVERIFIED]**. "Fact" bullets carry a source. Interpretation is kept in the "Inferences" subsections.

---

## Q1. Library inventory: task families, difficulty controls, verifiers, licence, reported R1-Distill/QwQ accuracy

### Takeaway
Only a few of the listed resources are true **generators** with code-level difficulty controls and rule-based verifiers. These are Reasoning Gym, Enigmata, SynLogic, ZebraLogic (generator method), the K&K generator (mem-kk-logic / Logic-RL), GSM-Infinite, CLRS-Text, SATBench and Mathador-LM. BBEH, KOR-Bench, LogicBench, ProcBench and PuzzleWorld are mostly **fixed** datasets, so they are not "fresh by construction". Published R1-Distill numbers are almost always for the **32B** distill, not 14B. The 14B accuracy band has to be calibrated by running the model.

### Cited Findings

**Reasoning Gym (RG)**: arXiv 2505.24760, submitted 2025-05-30, revised 2025-10-20, NeurIPS 2025 Spotlight
- Stojanovski, Stanley, Sharratt, Jones, Adefioye, Kaddour, Köpf. It has more than 100 generators and verifiers spanning algebra, arithmetic, computation, cognition, geometry, graph theory, logic and common games. Each has "adjustable complexity" — [arXiv abs](https://arxiv.org/abs/2505.24760)
- Difficulty is set by keyword-argument configs. The paper classes these as *difficulty parameters* (e.g. node counts, polynomial degrees, word lengths), *structural parameters* (e.g. dimensionality, constraint types) and *stylistic parameters* (e.g. variable names and formats, which do not change difficulty). Together they support "dynamic curricula" — [arXiv HTML](https://arxiv.org/html/2505.24760)
- Each dataset has a `score_answer` verifier, with a `cascade_score` fallback that tries string, then numeric, then symbolic matching. Install with `pip install reasoning-gym`. The code is **Apache-2.0** and the paper is CC BY 4.0. Example tasks named in the README include countdown, leg_counting, Rubik's cube and figlet_font — [GitHub](https://github.com/open-thought/reasoning-gym); [arXiv abs](https://arxiv.org/abs/2505.24760)
- Zero-shot results in the paper: o3-mini 63.5%, DeepSeek-R1 59.5%, Grok-3-Mini 55.1%, Llama-4-Maverick 41.5%, Claude 3.5 Sonnet 40.3%. Accuracy falls sharply between "easy" and "hard" configs (o3-mini: code −71.9 pts, graphs −33.8, geometry −33.1, algorithms −25.6; R1: code −61.8) — [arXiv HTML](https://arxiv.org/html/2505.24760)
- **No R1-Distill or QwQ numbers were found in the paper.** The README points to a separate `reasoning-gym-eval` repository for model results, which I did not inspect — [GitHub](https://github.com/open-thought/reasoning-gym)

**Enigmata**: arXiv 2505.19914, 2025-05-26 (ByteDance Seed and collaborators)
- 36 tasks in 7 categories. Each has a generator with controllable difficulty and a rule-based verifier — [arXiv abs](https://arxiv.org/abs/2505.19914)
- The categories and tasks are:
  - Crypto: KPA, KKA.
  - Arithmetic: Game24, Countdown.
  - Logic: Knights & Knaves, FOLIO, Zebra.
  - Grid: Binario, Campsite, Magic Square, Skyscraper, Sum Skyscraper, Star Battle, Sudoku, Sudoku2, Crosswords, Symbolic Hard.
  - Graph: Hamiltonian cycle and path, NL navigation, Maze.
  - Search: Hitori, Kakurasu, Light Up, Minesweeper, Slant, Checkmate-in-one, Tic-tac-toe.
  - Sequential: Twiddle, Car Painting, Stack Permutation, BB Symbolic, 8/9/15/16-puzzles.

  Each task has Easy, Medium and Hard levels, set by parameters such as grid size and number of blank cells — [arXiv HTML](https://arxiv.org/html/2505.19914)
- Enigmata-Eval overall accuracy: **DS-R1-Distill-Qwen-32B 31.1%**, QwQ-32B 43.8%, DeepSeek-R1 49.2%, o3-mini-high 59.9% — [arXiv HTML](https://arxiv.org/html/2505.19914)
- Per category, R1-Distill-Qwen-32B scored: Crypto 16.7, Arithmetic 47.3, Logic 62.9, Grid 38.2, Graph 36.4, Search 12.4, Sequential 18.8 — [arXiv HTML](https://arxiv.org/html/2505.19914)
- Licence is stated as **CC BY-NC-SA 4.0** (non-commercial). Project site: seed-enigmata.github.io — [arXiv HTML](https://arxiv.org/html/2505.19914)

**SynLogic**: arXiv 2505.19641, 2025-05-26 (MiniMax)
- 35 logic tasks, e.g. Sudoku, Game of 24, ciphers, Skyscraper, Minesweeper, Cryptarithm, Futoshiki, Numbrix, ARC-AGI-style tasks and Dyck language. Difficulty is set by task-specific parameters such as grid size. Verification is rule-based. Code at GitHub MiniMax-AI/SynLogic. Licence CC BY 4.0 per arXiv — [arXiv abs](https://arxiv.org/abs/2505.19641); [arXiv HTML](https://arxiv.org/html/2505.19641)
- SynLogic-Easy has 27 tasks, aimed at 7B models. SynLogic-Hard has all 35, aimed at 32B models — [arXiv HTML](https://arxiv.org/html/2505.19641)
- Table 2 scores for **R1-Distill-Qwen-32B**: SynLogic-Val 33.0%, KOR-Bench 66.6%, BBH 88.3%, BBEH 19.2% — [arXiv HTML](https://arxiv.org/html/2505.19641)
- RL on SynLogic lengthened CoT to about 2.5k tokens (7B) and about 4k tokens (32B) — [arXiv HTML](https://arxiv.org/html/2505.19641)

**ZebraLogic**: arXiv 2502.01100, submitted 2025-02-03, revised 2025-07-15, ICML 2025
- Lin, Le Bras, Richardson, Sabharwal, Poovendran, Clark, Choi. Logic-grid (Einstein) puzzles are generated as CSPs.
- Difficulty is controlled by grid size N×M (N, M ∈ {2..6}, 25 configurations) and measured by search-space size and **Z3 conflict count**.
- The benchmark has 1,000 puzzles, averaging 10.4 clues — [arXiv abs](https://arxiv.org/abs/2502.01100); [arXiv HTML](https://arxiv.org/html/2502.01100)
- DeepSeek-R1 scored 78.7% overall: Small 98.4%, Medium 95.7%, X-Large 28.5%. o1-full scored 81.0% overall and 42.5% on X-Large. Performance collapses once the search space exceeds about 10^7 or Z3 conflicts exceed about 20 (the "curse of complexity") — [arXiv HTML](https://arxiv.org/html/2502.01100)
- Hidden reasoning tokens correlate positively with Z3 conflicts. o1-preview averaged about 5,346 hidden tokens, against GPT-4o's 544 visible tokens — [arXiv HTML](https://arxiv.org/html/2502.01100)
- Paper licence CC BY 4.0. **No R1-Distill numbers were seen in the fetched content.** Code and generator availability (WildEval/ZeroEval repository) **[UNVERIFIED in this session]**.

**Knights-and-Knaves (K&K)**: Logic-RL, arXiv 2502.14768, 2025-02-20, and its K&K source arXiv 2410.23123
- The K&K generator comes from Xie et al., "On Memorization of LLMs in Logical Reasoning" (arXiv 2410.23123, 2024-10-30, revised 2025-03-04). It is a dynamic benchmark with **2–8 people** and supports perturbations of statements and leaves. Code and data are at memkklogic.github.io and GitHub AlphaPav/mem-kk-logic — [arXiv abs](https://arxiv.org/abs/2410.23123); [Logic-RL GitHub](https://github.com/Unakar/Logic-RL)
- Logic-RL trained Qwen2.5-7B with RL on about 5k K&K problems. Reflection, verification and summarisation emerged, and the gains transferred to AIME/AMC. The repository is **Apache-2.0** — [arXiv abs](https://arxiv.org/abs/2502.14768); [GitHub](https://github.com/Unakar/Logic-RL)
- Accuracy by number of people (2→8), from the Logic-RL README:
  - o3-mini-high: 0.99 → 0.83.
  - o1-2024-12-17: 0.83 / 0.51 / 0.38 / 0.38 / 0.35 / 0.30 / 0.20.
  - GPT-4o: 0.68 → 0.11.
  - Logic-RL-7B: 0.99 → 0.67.

  [GitHub](https://github.com/Unakar/Logic-RL)
- **No R1-Distill numbers in the README.** A licence for mem-kk-logic itself was **[UNVERIFIED]**.
- Enigmata reports the Logic category (which includes K&K) at 62.9% for R1-Distill-32B — [arXiv HTML](https://arxiv.org/html/2505.19914)

**GSM-Infinite**: arXiv 2502.05252, 2025-02-07
- Zhou, Liu, Chen, Tian, Chen (Infini-AI Lab / CMU / Meta). GSM-style word problems are generated from computation graphs.
- The knobs are the **number of operations** and the **amount of distractor context** (added nodes and edges). There are three subsets: Symbolic, Medium and Hard. Licence CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2502.05252)
- Code is at github.com/Infini-AI-Lab/gsm_infinite, and the paper describes a generation pipeline for arbitrary complexity — [arXiv HTML](https://arxiv.org/html/2502.05252)
- Results are reported as an AUC-style score plus the op count at which accuracy falls below 50%:
  - DeepSeek-R1: Symbolic 7280, Medium 9750.85, Hard 8573.8; below 50% at op 70 (Symbolic).
  - o3-mini: below 50% at op 70.
  - o1-mini: below 50% at op 50.
  - **QwQ-32B-preview**: Symbolic 3530, Medium 3205.75, Hard 1846.19; below 50% at **op 21**.

  Accuracy follows a "consistent sigmoid decline" as complexity rises — [arXiv HTML](https://arxiv.org/html/2502.05252)
- **No R1-Distill numbers in the fetched content.**

**BIG-Bench Extra Hard (BBEH)**: arXiv 2502.19187, 2025-02, ACL 2025
- Each of the 23 BBH tasks is replaced by a harder task that tests a similar skill — [arXiv](https://arxiv.org/abs/2502.19187v2); [ACL Anthology](https://aclanthology.org/2025.acl-long.1285)
- **R1-Distill-Qwen-32B**: harmonic-mean 5.2%, micro-average 19.2%, BBEH-Mini 15.4%. o3-mini-high leads at 44.8 harmonic mean — [search summary of arXiv 2502.19187](https://arxiv.org/abs/2502.19187v2). The 19.2% figure is independently matched by [SynLogic Table 2](https://arxiv.org/html/2505.19641)
- It is a **static** dataset, not a generator. Whether some subtasks have generator code was not checked **[UNVERIFIED]**.

**KOR-Bench**: arXiv 2410.06526, ICLR 2025
- "Knowledge-orthogonal" rule-application tasks in 5 categories: Operation, Logic, Cipher, Puzzle, Counterfactual. o1-preview scored 72.88% and o1-mini 70.16% — [arXiv](https://arxiv.org/abs/2410.06526v3); [ICLR](https://iclr.cc/virtual/2025/poster/29599)
- R1-Distill-Qwen-32B scored 66.6% — [SynLogic HTML](https://arxiv.org/html/2505.19641)
- It is a **static** dataset. Licence **[UNVERIFIED]**.

**CLRS-Text**: arXiv 2406.04229, 2024-06-06, ICML 2024 workshop context
- Text traces for 30 classical algorithms from the CLRS textbook (sorting, graph search, DP, etc.). Traces are procedurally generated over any input distribution, and difficulty is set by input size — [arXiv abs](https://arxiv.org/abs/2406.04229); [HTML](https://arxiv.org/html/2406.04229v1)
- Licence (likely Apache-2.0, as with google-deepmind/clrs) **[UNVERIFIED]**. No R1-Distill numbers found.

**SATBench**: arXiv 2505.14615, 2025-05, NeurIPS 2025
- SAT formulas are turned into story puzzles by an LLM. Difficulty is set by the number of clauses. There are 2,100 puzzles, checked by solver and LLM consistency checks. o4-mini scores only 65.0% on hard UNSAT, near the 50% chance rate. Failure modes include satisfiability bias and condition omission — [arXiv HTML](https://arxiv.org/html/2505.14615v2); [NeurIPS](https://neurips.cc/virtual/2025/126546)
- The answer is binary (SAT/UNSAT), with an optional assignment.

**LogicBench**: arXiv 2404.15522, ACL 2024
- Parmar et al. Covers 25 single-inference-rule patterns across propositional, first-order and non-monotonic logic. Evaluated GPT-4, ChatGPT, Gemini, Llama-2 and Mistral. Code at GitHub mihir3009/logicbench — [arXiv](https://arxiv.org/abs/2404.15522); [ACL](https://aclanthology.org/2024.acl-long.739)
- Each item uses a single inference rule, so it is shallow by design.

**ProcBench**: arXiv 2410.03117, 2024-10-04
- 23 task types (string manipulation, list processing, arithmetic) that take 2–25 steps of exact procedure-following with minimal prior knowledge — [arXiv](https://arxiv.org/abs/2410.03117v1)

**PuzzleWorld**: arXiv 2506.06211, 2025-06-06, ICLR 2026
- 667 **human-authored** puzzlehunt puzzles. They are multimodal, open-ended and annotated with reasoning traces. Most SOTA models score 1–4%, and the best scores 14%. Data at github.com/MIT-MI/PuzzleWorld — [arXiv](https://arxiv.org/abs/2506.06211); [ICLR](https://iclr.cc/virtual/2026/poster/10011416)

**Countdown / 24-game / Sudoku generators**
- Countdown and Game24 are in Enigmata ([HTML](https://arxiv.org/html/2505.19914)) and SynLogic ([HTML](https://arxiv.org/html/2505.19641)). Countdown is also in Reasoning Gym ([GitHub](https://github.com/open-thought/reasoning-gym)).
- Sudoku is in Enigmata and SynLogic. Reasoning Gym also lists Sudoku-type tasks **[UNVERIFIED: exact dataset names]**.

**Mathador-LM**: arXiv 2406.12572, EMNLP 2024
- Kurtic, Moeini, Alistarh (ISTA). The task is to reach a target number from base numbers using arithmetic, with a scored ruleset. Instances are generated dynamically at a target difficulty. In 2024, SOTA LLMs scored below 15%, against 43.7% for 3rd graders. Code at github.com/IST-DASLab/Mathador-LM — [arXiv](https://arxiv.org/pdf/2406.12572); [ACL](https://preview.aclanthology.org/credits/2024.emnlp-main.946)

### Inferences
- Taking the R1-Distill-32B anchors (Enigmata overall 31%, SynLogic-Val 33%, BBEH micro 19%), the **14B distill probably sits a few points lower** on mixed puzzle suites. Per-task configs therefore need to be set *easier* than "hard" to land in the 15–85% band. This is an inference, not a measurement.
- These tasks are generators with a single verifiable answer, tunable difficulty, and a likely trace length in the 16k range:
  - **Enigmata:** Logic, Arithmetic, Grid and Graph categories.
  - **Reasoning Gym:** logic, games, graph and algebra tasks.
  - **K&K:** about 4–7 people.
  - **ZebraLogic:** roughly 3×3 to 4×5 grids, i.e. Medium/Large.
  - **GSM-Infinite:** Symbolic or Medium.
  - **SATBench-style generation** done in-house, without the LLM story layer.
- Use BBEH, KOR-Bench, LogicBench, ProcBench and PuzzleWorld as calibration or reference only. They are fixed and public, so instances may be memorised. PuzzleWorld is also multimodal and far too hard.
- **Licence caution:** Enigmata is CC BY-NC-SA. Reasoning Gym, Logic-RL and SynLogic (per arXiv) are permissive.

### Gaps
- **No published accuracy found for R1-Distill-Qwen-14B** on any of these suites. R1-Distill-Qwen-7B numbers were also not found in fetched sources.
- Reasoning Gym's separate eval repo may hold distill results. Not inspected.
- These licences were not verified: ZebraLogic generator code, mem-kk-logic, KOR-Bench, BBEH, CLRS-Text, SATBench.
- The Logic-RL paper may include a DeepSeek-R1 row for K&K. Not verified.

---

## Q2. Which task families produce rich, branching CoT (backtracking, case analysis) rather than rote computation?

### Takeaway
There is direct evidence that **search and constraint-satisfaction puzzles** draw out verification, backtracking and case analysis. These include Countdown, K&K, Zebra/CSP grids and SAT. Their reasoning length scales with search difficulty (Z3 conflicts). Rote, procedure-following tasks (ProcBench, CLRS traces, GSM-style arithmetic chains) mainly test step accuracy and state tracking. I found **no head-to-head study** of "aha"/backtracking frequency on puzzles versus math for R1-Distill-14B.

### Cited Findings
- On **Countdown**, four behaviours were identified as key to self-improvement: verification, backtracking, subgoal setting and backward chaining. Qwen-2.5-3B showed them natively, but Llama-3.2-3B did not. Priming with *incorrect* solutions that contain these behaviours worked about as well as priming with correct ones (Gandhi et al., arXiv 2503.01307, 2025-03-03) — [arXiv](https://arxiv.org/abs/2503.01307)
- RL on **K&K** alone produced reflection, verification and summarisation behaviours in Qwen2.5-7B (Logic-RL, arXiv 2502.14768) — [arXiv](https://arxiv.org/abs/2502.14768)
- In ZebraLogic, hidden reasoning tokens correlate positively with **Z3 conflicts**, a solver-side measure of how much backtracking an instance needs. Accuracy collapses above about 20 conflicts — [arXiv HTML](https://arxiv.org/html/2502.01100)
- SATBench is pitched as testing **search-based** reasoning, unlike rule-application benchmarks such as LogicBench. Even top models fail hard UNSAT instances, which require exhausting the cases — [arXiv HTML](https://arxiv.org/html/2505.14615v2)
- "Illusion of Thinking" (Shojaee et al., Apple, arXiv 2506.06941, 2025-06-07, NeurIPS 2025) used controllable puzzle environments such as Tower of Hanoi. It found three regimes:
  - at low complexity, non-thinking models beat thinking ones;
  - at medium complexity, thinking models win;
  - at high complexity, both collapse.

  Reasoning effort rises with complexity and then *falls* despite remaining token budget — [arXiv](https://arxiv.org/abs/2506.06941). (Rebuttals exist, but I did not fetch them in this session **[UNVERIFIED]**.)
- About 20% of CoT tokens are high-entropy "forking" tokens that steer reasoning paths, and RL gains come mainly from them. This was shown on math (AIME) with Qwen3 models, not on puzzles (Wang et al., arXiv 2506.01939) — [arXiv](https://arxiv.org/abs/2506.01939)
- In R1-Distill-Qwen-32B, "wait" appears in 15.40% of correct trajectories versus 25.75% of incorrect ones — [arXiv 2601.17421](https://arxiv.org/pdf/2601.17421) (seen via search snippet only; domain mix not checked **[UNVERIFIED detail]**)
- Aha moments in LRMs show a more anthropomorphic, self-reflective tone on harder problems. Internally, perceived difficulty inverts across layers (Yang et al., arXiv 2504.02956; programming, math and commonsense tasks, no puzzles) — [arXiv](https://arxiv.org/abs/2504.02956)
- FINEREASON (ACL 2025) is a logic-puzzle benchmark built explicitly to test reflection, backtracking and exploration of alternatives — [ACL Anthology](https://aclanthology.org/2025.acl-long.333.pdf) (not read in full)

### Inferences
- **Best fit for a study of decision points:**
  - **K&K:** case splits on each person's identity.
  - **Zebra/CSP grids:** branching tracked by Z3 conflicts, which gives a solver-side "branching ground truth" to correlate with thought anchors.
  - **Countdown / Game24 / Mathador:** explicit search with dead ends.
  - **Small SAT instances.**

  All of these leave a readable trace of commitments ("assume A is a knight…") that resampling can test.
- **Weaker fit:** GSM-Infinite, ProcBench and CLRS-Text traces. Errors there are mostly arithmetic or state-tracking slips, so counterfactual importance may concentrate on computation sentences rather than planning sentences.
- The Z3-conflict and "number of people" knobs let you **separate difficulty from branching**. You could, for example, hold accuracy fixed while varying the required search.
- **Trace length:** the curse-of-complexity and illusion-of-thinking results suggest the hardest tiers produce very long or prematurely abandoned traces. The intermediate tiers are where traces are likely to fit under 16k and still contain real search.

### Gaps
- No study found that directly compares the frequency of backtracking or "aha" tokens on puzzles versus math for R1-Distill models.
- No reasoning-length measurements found for the 14B distill on any of these generators.

---

## Q3. Math-flavoured generators producing competition-style problems with new answers: can they hit the intermediate band for a 14B reasoning model?

### Takeaway
Yes, in principle, through GSM-Infinite (op count), Mathador/Countdown (number of operands and target), and Reasoning Gym's algebra, arithmetic and number-theory configs. But none of them produces true *competition-style* (AIME-like) problems. They are templated arithmetic or algebra. R1-Distill-14B is very strong at math, so templated algebra will likely be either too easy or hard only through length rather than insight.

### Cited Findings
- GSM-Infinite difficulty is continuous in the number of operations, with a sigmoid accuracy decline. Even QwQ-32B-preview stays above 50% up to about op 21 on Symbolic, and R1 up to about op 70 — [arXiv HTML](https://arxiv.org/html/2502.05252)
- Reasoning Gym categories include algebra and arithmetic, with difficulty parameters such as polynomial degree — [arXiv HTML](https://arxiv.org/html/2505.24760)
- Mathador-LM generates instances at a target difficulty and was explicitly designed against test-set leakage — [arXiv](https://arxiv.org/pdf/2406.12572)
- In Enigmata's Arithmetic category (Game24 and Countdown), R1-Distill-32B scored 47.3% versus 77.1% for R1 — [arXiv HTML](https://arxiv.org/html/2505.19914). This is already near the middle of the band for a distill.

### Inferences
- **Most promising "math-flavoured but branching" source:** Countdown/Game24/Mathador, at about 47% for R1-Distill-32B on Enigmata's settings. Tuning the number of operands and the target range should reach 15–85% for 14B.
- GSM-Infinite can reach the band only at fairly high op counts. That probably means long, rote traces, possibly close to the 16k limit. Prefer the Symbolic subset at moderate ops.
- For "competition style with new answers", the cleanest option is probably in-house templating of MATH-like problems with resampled parameters (functional-variation approaches). This is outside the generators reviewed here **[not researched in this session]**.

### Gaps
- No R1-Distill-14B or 7B accuracy-vs-difficulty curves found for GSM-Infinite, Reasoning Gym algebra, or Mathador.
- I did not find a maintained generator that produces AIME-style problems.

---

## Q4. Prior interpretability or CoT-faithfulness studies using procedurally generated puzzles with R1-style models

### Takeaway
Interpretability work on procedurally generated puzzles exists, but most of it concerns **memorisation** (K&K) or **RL training dynamics** (Countdown, K&K), not sentence-level causal importance. The original Thought Anchors work used math. One follow-up applied thought anchors to synthetic legal cases. **I found no published thought-anchor or resampling study on K&K, Zebra or Countdown** — this looks like an open niche.

### Cited Findings
- Thought Anchors (Bogdan et al., arXiv 2506.19143, 2025) uses counterfactual resampling (about 100 resamples per sentence), attention analysis, and attention suppression. Anchors tend to be planning or uncertainty-management sentences, and "receiver" heads attend to them — [arXiv HTML](https://arxiv.org/html/2506.19143); [GitHub](https://github.com/interp-reasoning/thought-anchors). The follow-up "Thought Branches" (arXiv 2510.27484) argues that interpreting reasoning requires resampling — [arXiv](https://arxiv.org/html/2510.27484v1) (not read in detail).
- An ARENA 7.0 capstone (March 2026; JeaniceK, Robbins, Taraz) applied thought anchors and probes to DeepSeek-R1-Distill-Llama-8B on 92 synthetic criminal-court cases generated by Claude. Anchors centred on **fact retrieval rather than planning**, unlike in math. Attention correlated only weakly with causal importance. Attention probes at layers 16 and 24 reached up to 95% sign-match on judgement-leaning — [LessWrong](https://www.lesswrong.com/posts/f9cnGHCiJgo5eDkSQ/understanding-reasoning-with-thought-anchors-and-probes) (blog post, not peer-reviewed)
- Xie et al. (arXiv 2410.23123) used K&K with perturbations, cross-difficulty transfer, **internal probing**, and fine-tuning on wrong answers. They found that fine-tuning causes heavy memorisation (failing on slight variations) while still improving generalisation — [arXiv](https://arxiv.org/abs/2410.23123)
- On Countdown, the presence of reasoning behaviours matters more than correctness for RL self-improvement — [arXiv 2503.01307](https://arxiv.org/abs/2503.01307)
- An empirical, attention-based and mechanistic study of distilled R1 models found that answer tokens attend to reasoning segments, and identified "Reasoning-Focus Heads" (arXiv 2509.23676) — [arXiv](https://arxiv.org/pdf/2509.23676) (seen via search snippet only; tasks not checked **[UNVERIFIED]**)

### Inferences
- The legal-case result suggests anchor *type* depends on the domain. Puzzles might push anchors towards case-split or hypothesis sentences ("Suppose B is a knave"), which would be a testable contrast with the planning anchors seen on math.
- The K&K perturbation machinery (statement and leaf perturbations) from 2410.23123 could double as a memorisation control.

### Gaps
- No study found that runs sentence-resampling importance on K&K, ZebraLogic, Countdown or SAT for any R1-Distill model.
- The thought-anchors problem-selection protocol (which MATH accuracy band, which model) was not re-verified in this session.

---

## Q5. Caveats: "OOD by instance" versus "OOD by task type", and whether R1 distillation data included such puzzles

### Takeaway
The R1 paper **explicitly** includes logic, cryptography and arithmetic puzzles in R1's RL data. The 800k distillation set was rejection-sampled from that RL checkpoint over "reasoning prompts". So R1-Distill-14B has very likely seen puzzle *formats* such as K&K, ciphers and 24/Countdown-style arithmetic, even though fresh instances are new. Treat generated puzzles as "new instances of familiar task types", not as OOD tasks.

### Cited Findings
- R1 paper (arXiv 2501.12948 v1, §2.3.2): reasoning-oriented RL focuses on "coding, mathematics, science, and logic reasoning, which involve well-defined problems with clear solutions" — [arXiv v1 PDF](https://arxiv.org/pdf/2501.12948v1)
- §2.3.3: "we collect about 600k reasoning related training samples" by rejection sampling from the RL checkpoint, plus "approximately 200k training samples that are unrelated to reasoning", for "about 800k samples". The distilled models are fine-tuned on "800K training samples" generated by R1 as the teacher — [arXiv v1 PDF](https://arxiv.org/pdf/2501.12948v1)
- The later revision, Table 4, App. B.3.1, lists RL datasets with 26k math, 17k code, 22k STEM and **15k logic** questions. The logic set includes "cryptography puzzles", "logic puzzles focus on deductive reasoning" and "arithmetic puzzles" — [arXiv HTML (current version)](https://arxiv.org/html/2501.12948)
- Fine-tuning on K&K produces memorisation alongside generalisation, with brittleness to small perturbations — [arXiv 2410.23123](https://arxiv.org/abs/2410.23123)
- Mathador-LM and GSM-Infinite were both designed to counter test-set leakage through dynamic generation — [Mathador](https://arxiv.org/pdf/2406.12572); [GSM-Infinite](https://arxiv.org/abs/2502.05252)

### Inferences
- **Order of familiarity.** Inferring from the R1 data description, the most familiar formats are likely K&K, Zebra, ciphers and 24/Countdown. Less familiar are SAT-as-story, CLRS algorithm traces, Enigmata's Search and Sequential puzzles (e.g. Kakurasu, Twiddle, sliding puzzles) and Mathador's ruleset scoring.
- R1-Distill-32B is weakest on Search (12.4%) and Sequential (18.8%) in Enigmata. That is consistent with less exposure, but it is not proof.
- **Mitigations:**
  - use stylistic-parameter variation (Reasoning Gym), new surface stories, or novel rule variants (KOR-style rule descriptions);
  - check perturbation robustness (K&K perturbations);
  - report results by task-type familiarity.
- The 15k logic set was used for R1's **RL** stage. Whether those prompts reappear in the 600k SFT reasoning set is not stated. It is likely, because SFT prompts were drawn from the same reasoning-prompt pool, but this is an inference.
- Template-generated instances from public generators (e.g. the K&K dataset released in 2024) may overlap *in template* with DeepSeek's private logic data. This cannot be checked, because the R1 data is not released.

### Gaps
- DeepSeek has not released the R1 SFT or RL prompt sets, so the exact puzzle types (Zebra? Sudoku? K&K?) are unknown.
- No contamination audit of R1-Distill on any of these generators was found.
