# Reading the 7 sentences with a real effect

> **EXPLORATORY.** This is a qualitative reading done after the confirmatory s12 result and the s13 diagnostics. It was not preregistered.
> It covers the 7 of 190 sentences flagged by the s13 pooled permutation test (BH 10%, `real_tv_bh10 | real_dacc_bh10`).
> The analysis ran on CPU only, with `scripts/s15_read_real_effects.py --index <phase2 completions_index.parquet>`.
> Accuracy is pooled over about 300 resamples per fork: 100 released (r0) plus 200 new (r1, r2). Answers use `parse_v2` / `category_v2`.
> Replacement sentences come from the new run only (`first_replacement_sentence`).

**Fork convention.** Fork i regenerates S_i, and fork i+1 keeps it. The effect of S_i is acc(fork i+1) − acc(fork i).

**Boundary check, applied to all 7.** For every flagged sentence:

- the fork i+1 prompt equals the fork i prompt followed by the original S_i and its trailing separator (`' '` or `'\n\n'`);
- `target_text` equals the `sentences.parquet` text exactly;
- every fork prompt ends on a sentence boundary followed by whitespace, never mid-token, mid-expression or mid-sentence.

Across all 200 Phase 2 prompts, 103 end in `\n` and 97 in a single space. The flagged forks follow the same convention as their neighbours. No flagged S_i is a duplicated sentence in its trace.

---

## 2238:incorrect: what the problem means by "coefficient"

**Problem:** the coefficient of a²b² in (a+b)⁴(c+1/c)⁶.

- Ground truth is **120**: 6 × C(6,3), using the c⁰ term.
- The original trace ends at **384**: 6 × 2⁶, multiplying by the *sum of all* coefficients of the c-factor.
- Outcomes are almost exactly binary, 120 vs 384, so accuracy works as a readout of which reading of the problem the model adopts.

The baseline over forks 14–32 is about 0.50–0.63, which is high outcome uncertainty.

### S24 (0.513 → 0.667, +0.15)

> "But the problem is asking for the coefficient of \( a^2b^2 \) in the entire product."

- **Context:**
  - S21–23: "that doesn't make sense … the term a²b² would be multiplied by every term in (c+1/c)⁶".
  - S25: a²b² produces "terms like a²b²c^k".
- **What it does:** it re-poses the question neutrally, without committing.
- **Distributions:**
  - Fork 24: 120: 0.51, 384: 0.48.
  - Fork 25: 120: 0.67, 384: 0.32.
- **Replacements:**
  - Only 9% reproduce S24 verbatim.
  - Common alternatives jump straight to a computation: "…multiplied by the **sum of all coefficients** in (c+1/c)⁶" (21% of first sentences contain "sum of coefficients", with accuracy 0.42), or "regardless of the powers of c" (18%, accuracy 0.42).
  - Keeping the neutral restatement postpones that commitment, and the later sentences then lean towards the c⁰ reading.
  - So S24's effect comes from *not* committing to the sum-of-coefficients reading.

### S25 (0.667 → 0.930, +0.26) and S26 (0.930 → 0.623, −0.31)

> S25: "Wait, but \( a^2b^2 \) doesn't involve \( c \) or \( 1/c \), so when multiplied by terms from the second polynomial, it would create terms like \( a^2b^2c^k \) or \( a^2b^2/c^k \)."
>
> S26: "But the original problem is just asking for the coefficient of \( a^2b^2 \), regardless of the powers of \( c \)."

- **What S25 does:** it states the key structural fact that every product term carries a power of c. That sets up the correct c⁰ reading.
- **What S26 does:** it overrides S25 and commits to "regardless of the powers of c", which is the reading that sums all coefficients (384). S27 ("times any power of c?") and S32 continue from it.
- **Distributions:**

| Fork | 120 | 384 |
|---|---|---|
| 25 | 0.67 | 0.32 |
| 26 | 0.93 | 0.07 |
| 27 | 0.62 | 0.38 |

- **Replacements at fork 26** (S25 present, S26 regenerated):
  - None of them reproduce S26.
  - 38% of first sentences immediately name the **constant / c⁰ term**, with accuracy 0.99. For example: "the question is only about a²b², without any c terms. So … multiply a²b² by the constant term in (c+1/c)⁶".
  - Only 7% say "regardless", and even those reach 0.85.
  - Median continuation length falls to 1,561 tokens at fork 26, against about 2,500–3,300 at forks 23–28. The model resolves the problem quickly.
  - The 12 wrong samples at fork 26 mostly re-derive "regardless of the power of c" on their own. One reproduces S26 almost word for word.
- **Replacements at fork 25** (S25 regenerated): these are often near-paraphrases of S25, but about a third pivot to "regardless" or the sum of coefficients without first stating the c^k structure.

### S32 (0.523 → 0.313, −0.21)

> "Therefore, the coefficient of \( a^2b^2 \) in the entire product would actually be the sum of all the coefficients from \( \left( c + \frac{1}{c} \right)^6 \) …"

- **What it does:** it *commits explicitly to the wrong computation* that the original trace follows to 384.
- **Distributions:**
  - Fork 32: 120: 0.52.
  - Fork 33: 120: 0.31, 384: 0.69.
- **Replacements:**
  - 56% of first sentences contain "sum of coefficients", with accuracy 0.42, so the commitment is the modal next step.
  - The remainder ("only come from the term where the second part is 1, i.e., c⁰") reach 0.88.
  - Keeping the explicit "Therefore … sum of all the coefficients" lowers accuracy further, below even the resampled variants of the same idea.

---

## 3935:correct: exact fraction vs decimal approximation

**Problem:** f(3x) = 3f(x), and f(x) = 1−|x−2| on [1,3]. Find the smallest x with f(x) = f(2001). Ground truth is **429**.

- The main wrong answers are **53703/125 = 429.624**, written either as `\frac{53703}{125}` or as `429.624`.
- These come from using the decimal m ≈ 2.744 instead of m = 2001/729. That gives f(m) = 0.256 and 729 × 0.256 = 186.624 instead of exactly 186, and so x = 429.624.

### S21 (0.943 → 0.857, −0.09)

> "So, m is 2 + 181/243, which is 2.744 approximately."

- **What it does:** it *introduces the decimal approximation* of m.
- **Distributions:**
  - Fork 21: 429: 0.94; 429.624-type answers: 0.01.
  - Fork 22: 429: 0.86; 429.624-type answers: 0.13.
- **Replacements:**
  - Almost all are "181 is a prime number, I think, so that's as simplified as it gets", which stays in exact form.
  - None of the first sentences contain "2.74". Several still mention ≈2.745 later, but by then the exact form is established.

### S26 (0.783 → 0.960, +0.18)

> "So, m is 2001 / 3^6, which is approximately 2.744, but exactly, it's 2001/729."

S27 follows: "Let me keep it as a fraction for exactness."

- **What it does:** it *corrects towards the exact value*. It is a recovery from the risk that S21 introduced.
- **Distributions:**
  - Fork 26: 429: 0.78; 53703/125: 0.17.
  - Fork 27: 429: 0.96; 53703/125: 0.03.
- **Replacements:**
  - Typically "2001 = 3^6 × (2001/729) ≈ 3^6 × 2.744": 85% contain "2.74".
  - Their continuations then compute "f(m) = 1 − 0.744 = 0.256 … 729 × 0.256", which produces the 33 wrong samples.
  - The original's explicit "*but exactly*" contrast is what matters, not merely mentioning 2001/729. 90% of replacements also write 2001/729, and they still reach only 0.85.

S21 and S26 are two ends of one mechanism: the decimal approximation is introduced at S21 and retracted at S26.

---

## 330:correct: framing the nested expression

**Problem:** evaluate 3(1+3(1+…(1+3)…)), with 10 threes. Ground truth is **88572**.

- The main wrong answer is **36084**. It comes from mis-assigning the innermost layer (taking (1+3) = 4 as a layer value and then applying 3(1+·) nine times; the next step gives 108255).
- Off-by-one layer counts give 29523 and 265719.

### S6 (0.707 → 0.857, +0.15)

> "First, I notice that the expression is a nested set of terms where each term is 1 plus 3 times another similar expression."

- **What it does:** it gives the correct recursive description, term = 1 + 3·term, before any computation.
- **Distributions:**
  - Fork 6: 88572: 0.71, 36084: 0.15, 265719: 0.03, 29523: 0.03.
  - Fork 7: 88572: 0.86, 36084: 0.04.
- **Replacements:**
  - These are almost all paraphrases with the same opening, "First, I notice that the expression is a series of nested parentheses, **each multiplied by 3**".
  - The 82% that describe it only as multiplication by 3 reach 0.67; the rest reach 0.80, though that subgroup is small (about 36).
  - So the semantic difference is subtle: S6 states the "1 plus 3 times" recursion precisely, and paraphrases blur which part is the repeated unit.
- **Assessment:** this is the least interpretable of the 7. The step is consistent across replicates, but the mechanism is only weakly supported by the replacement text.

---

## Is the 2238 S25/S26 spike genuine or an artefact?

**Per-replicate accuracy, 100 samples per cell:**

| Fork | r0 (released) | r1 | r2 | Pooled |
|---|---|---|---|---|
| 24 | 0.44 | 0.59 | 0.51 | 0.513 |
| 25 | 0.63 | 0.72 | 0.65 | 0.667 |
| **26** | **0.91** | **0.93** | **0.95** | **0.930** |
| 27 | 0.48 | 0.67 | 0.72 | 0.623 |
| 28 | 0.52 | 0.61 | 0.57 | 0.567 |

**Evidence:**

- **The spike reproduces in all three independent replicates.**
  - r0 was generated separately by the original release, and r1/r2 come from the new run.
  - The spike rises 0.21–0.28 above fork 25 in each replicate, and fork 26 is the maximum of every replicate.
  - A sampling fluke at one fork would not repeat three times at this size: with n = 100 and p ≈ 0.65, the SE is about 0.05.
- **The prompt boundaries are clean.**
  - The fork 26 prompt ends `"…terms like \\( a^2b^2c^k \\) or \\( a^2b^2/c^k \\). "`: a complete sentence, balanced `\( \)`, and a trailing space, the same as forks 24, 25 and 27.
  - The fork 27 prompt is exactly the fork 26 prompt + S26 + `' '`.
  - Prompt token counts rise smoothly (736 → 802 → 831).
  - Neither S25 nor S26 is duplicated in the trace, and both match `target_text` exactly.
  - No continuations are truncated (`finish_reason == length` is 0 at forks 23–28).
- **The mechanism is coherent.**
  - At fork 26 the model sees S25 ("creates terms like a²b²c^k") without S26's "regardless of the powers of c".
  - 38% of resamples immediately name the c⁰/constant term, with accuracy 0.99, and continuations are about 45% shorter.
  - Re-inserting S26 restores the "regardless" reading that leads to 384.
  - The same reading recurs later and is penalised again at S32, with "regardless" and "sum of all coefficients" appearing as first sentences at forks 29–32.

**Verdict: genuine.**

- S25 and S26 are a real push–pull pair. S25 moves the model towards the correct interpretation, and S26 immediately pulls it back. Their net effect is near zero (0.667 → 0.623).
- **Interpretation caveat:** the two effects share fork 26, so their estimation errors are perfectly anti-correlated. Treat them as one event (a window in which the right reading is available) rather than two independent anchors.

**Side finding: a replicate offset.**

- For 2238, the released replicate r0 is systematically lower than the new replicates across *all* forks: mean 0.499 vs 0.620 / 0.620.
- 4019:incorrect shows the same pattern: 0.085 vs 0.182 / 0.181.
- The other 8 trajectories differ by 0.05 or less.
- This does not explain the spike, which r0 also shows. It does point to a generation difference between the release and the Phase 2 run (sampling settings, prompt or template) that affects the two "incorrect" trajectories most. It is worth checking before pooling r0 in level-based analyses.

---

## Patterns across the 7

1. **Effects concentrate where the outcome is uncertain.**
   - 5 of 7 flagged sentences have acc_before in [0.2, 0.8], against 19% of the other 183.
   - Mean p(1−p) before the sentence is 0.17 for flagged sentences and 0.11 for the rest.
   - 4 of 7 come from 2238, the only trajectory whose accuracy sits near 0.5 throughout and has a clean two-answer split.
   - 7 of the 10 trajectories have no flagged sentence. These include 4019:incorrect (accuracy about 0.15) and the near-ceiling trajectories.
   - Part of this is power: binary outcomes near 0.5 have the most room to move. Part is substance: decisions are only open where the model is still undecided.
2. **Every flagged sentence chooses between two competing ways of setting up the problem.** None is plain arithmetic.
   - **Committing to an interpretation or method:** 2238 S26 and S32 (sum-of-coefficients reading), and 2238 S24 by deferring it.
   - **Introducing a key structural fact:** 2238 S25 (terms carry c^k), and 330 S6 (the recursion is 1 + 3·term).
   - **Choosing a number representation:** 3935 S21 introduces the decimal approximation, and S26 corrects it back to the exact fraction.
3. **Negative and positive anchors pair up around the same decision.**
   - In 2238, S25 helps and S26 hurts.
   - In 3935, S21 hurts and S26 helps.
   - The model keeps revisiting the same fork in the road, and the sentence that settles each visit is flagged.
   - No flagged sentence is a pure verification step. 3935 S26 comes closest ("but exactly…"), and it works by changing the representation rather than by checking.
4. **The resampled sentence is usually a paraphrase.** 0–9% of resamples reproduce S_i verbatim.
   - The effect therefore lives in a specific semantic commitment within the sentence: "regardless of the powers of c", "sum of all the coefficients", "approximately 2.744", "but exactly". It does not depend on the sentence's position or form.
   - Where the paraphrases keep that commitment, they reproduce the original sentence's effect, as with the 384 replacements at fork 32 and the decimal replacements at fork 26 of 3935.
5. **There is no sign of a fork-construction artefact.** All boundaries are clean sentence boundaries. The only anomaly found is the r0 vs r1/r2 offset for the two incorrect trajectories, and it does not create the flagged effects.
