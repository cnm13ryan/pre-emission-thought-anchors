# T1 (item 5): released r0 vs our r1/r2

**Status: EXPLORATORY, POST HOC.** Nothing here was preregistered. Every grid and threshold was fixed in `run.py` before it ran, and every cell is reported in `results.json`. The analysis ran on CPU only. It also made public, read-only fetches from Hugging Face (the 200 window `solutions.json` files at the pinned revision `a3a2fe1e`, each sha256-verified against the MVP manifest) and from GitHub (to check that the Thought Anchors repo head is still `b53ed8c`).

**Verdict: UNEXPLAINED (quantitatively). One concrete, unquantified candidate cause is identified.**

- None of the data-side factors we can test removes the accuracy gap on the incorrect traces: truncation, no-answer samples, answered-only conditioning, length matching, answer grouping and sampling noise.
- The released rollouts were generated from **prompts formatted differently from ours**, through a hosted API with an unknown engine. Every newline in the CoT prefix is collapsed to a single space, and the prompt ends with no trailing whitespace.
- The released continuations carry that format forward: they have fewer paragraph breaks and are about 30% shorter in all 10 trajectories.
- How much of the accuracy gap the prompt format explains cannot be estimated without generating. That is exactly what **arm 5** would measure, and our setup can run it directly, because the exact released prompt strings are now extracted (cache below).

Intervals are cluster bootstraps over problems (2,000 replicates). With only 2 problems, the incorrect-group interval is close to degenerate. For per-trajectory evidence, see the permutation tests.

## Step 1: fork by fork on the 200 window forks

### Accuracy gap

The gap is r0 − (r1∪r2), using canonical grouping. Raw grouping gives identical numbers.

| | incorrect (2) | correct (8) | incorrect − correct |
|---|---|---|---|
| gap, all samples | **−0.109** [−0.121, −0.096] | −0.005 [−0.020, 0.013] | **−0.103** [−0.127, −0.081] |
| reference r1 − r2 | −0.0003 [−0.001, 0.0005] | −0.002 [−0.009, 0.004] | |

- **Per trajectory:**
  - 2238: 0.499 vs 0.620/0.621.
  - 4019: 0.085 vs 0.182/0.181.
  - On both, r0 is lower at **20 of 20** forks.
- **Permutation test** (r0/r1/r2 labels exchanged within each fork, 2,000 permutations): p = 0.0005 for both incorrect trajectories.
- **Correct traces:** the gap is small and mixed in sign. The range is −0.029 (4605) to +0.050 (2189), and 5 of 8 trajectories have p < 0.05.

### Answer distribution

TV is computed per fork with canonical categories. Excess = mean(TV(r0,r1), TV(r0,r2)) − TV(r1,r2), all comparisons at 100 vs 100 samples.

| | TV r0–r1 | TV r1–r2 (noise) | excess |
|---|---|---|---|
| incorrect | 0.111 | 0.046 | **0.065** [0.059, 0.072] |
| correct | 0.062 | 0.049 | **0.013** [0.009, 0.018] |

- r0 is distinguishable from ours on 8 of 10 trajectories (permutation p ≤ 0.022 on 7; 2189 has p = 0.067).
- So **r0 ≠ ours is a general shift**. It is about 5× larger on the incorrect traces, where the outcome is close to binary and near 50/50 (2238) or rare (4019).

### Length and caps

Token counts come from the pinned tokenizer on the generated text, on both sides. Our server counts differ from these by a median of 1 token.

- **Caps:**
  - Both sides cap at **16,384** generated tokens.
  - Released: the window maximum is 16,384, and truncated rollouts sit at a median of 16,384 over all 647k samples. The script default is `max_tokens=16384`.
  - Ours: `max_new_tokens` is 16,384, and finish_reason=length occurs at exactly 16,384.

| median tokens (p10–p90) | r0 | r1∪r2 |
|---|---|---|
| incorrect | 1,480 (827–3,213) | 2,212 (1,283–5,788) |
| correct | 3,059 (1,353–6,342) | 3,972 (2,188–7,601) |

- **Shorter in every trajectory:** r0 is shorter in all 10 trajectories. The median difference is −917 [−1,339, −496] tokens on incorrect traces and −807 [−1,177, −531] on correct ones. The length shift is **not** specific to the incorrect traces.
- **Truncation rates:** 0.44% for r0 and 0.67% for ours (87 and 269 rollouts), so truncation is rare on both sides.
- **No-answer rates:** 0.07% for r0 and 0.01% for ours.
- **Paragraph breaks per 1,000 characters of continuation:**
  - r0: 5.7.
  - Ours: 9.4.
  - Incorrect traces: 2.1 vs 6.0.

### Decomposing the gap

Each row gives the remaining gap and the fraction of the original gap it removes.

| | incorrect | correct |
|---|---|---|
| D0: all samples | −0.109 | −0.005 |
| D1: exclude truncated and no-answer on both sides | −0.108 (removes 0.2%) | −0.005 |
| D3: answered (boxed) samples only | −0.108 (removes 0.2%) | −0.005 |
| D2: length-matched, within-trajectory deciles | −0.157 (**−44%**, the gap grows) | −0.019 |
| D2b: length-matched, within-fork quintiles | −0.156 (−44%) | −0.015 |
| D1 + D2 | −0.156 | −0.016 |

- In every length-matching cell, 0% of r0 samples were dropped for lack of support.
- **Length matching works differently on the two incorrect trajectories:**
  - 4019: the gap goes from −0.096 to −0.001.
  - 2238: the gap goes from −0.121 to −0.312.
- Length is downstream of the outcome, so length matching is a description of the gap, not a correction for it. Truncation and no-answer samples explain none of the gap.

## Step 2: released generation settings vs ours

All released settings come from `generate_rollouts.py` at `b53ed8c`, which is identical to the repo head. The dataset card and the paper state only T = 0.6 and top_p = 0.95.

| | Released (rollouts; base solutions use the same template) | Ours (Phase 2) |
|---|---|---|
| Engine | Hosted API `/completions`, raw prompt. The default provider is **Novita** (non-streaming, httpx timeout 240 s, max_retries 1). Together, Fireworks and Local HF are options. Which one built the dataset, and its engine version, is **not stated anywhere**. | vLLM `vllm/vllm-openai@sha256:61fc8a89…`. v0.28.0 according to `docs/stage5_measurements.md` (the image log does not record the version). bf16, max_model_len 24,576. |
| Model | `deepseek/deepseek-r1-distill-qwen-14b` (provider-hosted; revision and quantisation unknown) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` at revision `1df85071…` |
| Sampling | T 0.6, top_p 0.95, max_tokens 16,384, no seed, no penalties or top_k/min_p by default | T 0.6, top_p 0.95, max_tokens 16,384, per-sample seeds. Verified in `phase2_config.json`. |
| Chat template / system prompt | none / none | none / none |
| Special tokens | Left to the provider. BOS is unknown. | `add_special_tokens=False` and no BOS in the prompt |
| Template | `Solve this math problem step by step. You MUST put your final answer in \boxed{}. Problem: {problem} Solution: \n<think>\n{prefix}` | Same header, byte-identical on all 200 forks |
| **Prefix** | `" ".join(chunks[:i+1]).replace(chunk_i, "").strip()` | `full_cot[:char_start_i]`, the base trace verbatim |
| Error handling | Failed or timed-out requests are dropped and regenerated until 100 valid samples exist. | none needed (40,000 of 40,000 complete) |

**Measured prompt diff over the 200 window forks**, released prompt = `full_cot` minus `rollout`, one distinct prompt per fork:

- **Exact equality:** 0 of 200 prompts are byte-identical.
- **After normalising whitespace:** 197 of 200 are identical.
- **Newlines collapsed:** the CoT part of a released prefix has **0** paragraph breaks (mean 1.2 `\n`, which come from inside chunks). Ours has a mean of 27.0 paragraph breaks and 56 `\n`. On 4019 the gap is 96 paragraph breaks per prompt vs 0.
- **Trailing whitespace:**
  - Released prompts never end in whitespace, so the model writes the separator itself. 94.6% of r0 continuations start with a space, and only 0.02% start with `\n\n`, even where the original trace had a paragraph break.
  - Ours end with `' '` (97), `'\n\n'` (95), `'\n'` (5) or other (3).
- **Text deleted:** on 3 forks, released prompts lost text because `str.replace` removes every earlier occurrence of the resampled chunk. These are 4019 forks 162 and 165 (16 and 42 characters) and 4164 fork 105.
- **Prompt length:** ours are longer by a mean of 2.9 tokens.
- **Incorrect-trace gaps by our separator:** −0.123 at forks where our prompt ends in a space and −0.095 where it ends in `\n\n`. A gap is present in both cases.

## What is resolved and what remains

**Resolved:**

- The sampling settings are identical: temperature, top_p, max_tokens cap, no chat template and no system prompt.
- Truncation and no-answer samples explain none of the gap.
- Canonical vs raw grouping makes no difference.
- Our two replicates agree (r1 − r2 ≈ 0), so the r0 difference is systematic, not sampling noise.
- The released prompts are off-format relative to the trace they resample: collapsed newlines, no trailing separator, and occasional deleted text.

**Remaining:**

- How much of the gap the prompt format accounts for, compared with the provider engine (unknown kernels, quantisation and BOS handling) and the regeneration of timed-out requests.
- Arm 5 can separate the format from the engine at no extra cost by feeding the extracted released prompt strings to our pinned vLLM.
- The provider itself cannot be reproduced.
- This feeds the GPU decision: the case for arm 5 is supported.

## Files

- `run.py` reproduces everything. Without the cache it re-fetches 0.52 GB of public files.
- `results.json` holds all numbers.
- `trajectory_metrics.csv`, `fork_tv_canon.csv` and `prompt_diff_by_fork.csv` hold the per-trajectory, per-fork and prompt-diff tables.
- The cache is in `~/preanchor-root/anchor_reliability_v2/derived/v3_followup/t1_released_vs_ours/`:
  - `released_window_prompts.parquet` holds the exact released prompts;
  - two other files hold per-sample text features.
