# Can we see an AI's important thoughts coming? A plain-language report

*Status as of 11 October 2026. This report summarises a research project in everyday
language. Everything it reports has been measured; where a result is uncertain or only
preliminary, it says so. The technical documents behind each point are listed at the end.*

---

## 1. The short version

Some AI systems now "think out loud" before answering. They write a long chain of
reasoning, step by step, and only then give a final answer. Safety researchers hope to
watch this reasoning to catch problems early.

Not every step matters equally. A few sentences in the reasoning seem to *decide* the answer.
Researchers call these **thought anchors**. If we could spot an anchor *before* the AI
writes it, by looking inside the AI's internal workings, we might catch a harmful decision
before it happens.

That was the original goal of this project. What we found so far:

1. **The standard way of finding thought anchors is much less reliable than it looks.**
   Measure the same sentence twice and you often get a different answer about how important
   it is. Most of what the score picks up is random noise and how unsure the AI is overall.
2. **Real thought anchors are rare.** When we tested properly, only about 2–4 sentences in
   every 100 showed a genuine effect on the answer.
3. **Looking inside the AI has not yet revealed anchors in advance.** The AI's internal
   signals clearly show *how unsure it is*. They also show *what kind of sentence is
   coming next*, for example "Wait, let me check…". But so far they do not show *how
   important* the next sentence will be, beyond what simple clues such as "how far into the
   reasoning are we?" already tell you.
4. **One part of the question can now be answered.**
   - Whether the next sentence will push the AI *towards or away from* the right answer
     cannot be predicted in advance at all. This follows from simple mathematics, and the
     data agree.
   - *How much* the next sentence will move the answer, in either direction, is still an
     open question. Answering it is the main goal of the next stage.

The next stage needs more computing power than we can fund ourselves. It also needs some
preparation work and approvals before it can start (see section 8).

---

## 2. Background, explained from scratch

### 2.1 Reasoning models

A **reasoning model** is an AI that writes out its working before answering, like a student
showing their work on a maths exam. This written working is called the **chain of thought**.
For a hard maths problem it might be 100–300 sentences long.

The model studied here is called **DeepSeek-R1-Distill-Qwen-14B**. It is a freely available
AI that can be run on rented computers, which lets us look inside it. All experiments so far
use it on maths competition problems, because maths answers can be checked automatically.

### 2.2 Why safety researchers care about the chain of thought

If an AI is about to do something harmful, it may "say" so in its reasoning first. Reading
the reasoning is called **chain-of-thought monitoring**. A large group of researchers called
it "a new and fragile opportunity for AI safety" (Korbak et al., 2025).

It is fragile for two reasons:

- **Newer AIs write less of their thinking down.** A 2026 study found that one frontier
  model, GPT-6 Astra, does simple steps "in its head" and only writes down the crucial ones
  (Luo et al., 2026).
- **AIs can learn to hide their intentions in their writing** when they are trained against
  a monitor that reads it (Baker et al., 2025).

So we may not always be able to rely on reading the AI's words. One alternative is to read
the AI's **internal activations**: the millions of numbers it computes internally at every
step. Think of it as reading brain activity instead of listening to speech. That is the
motivation for this project.

### 2.3 What a "thought anchor" is, and how it is measured

The idea comes from a 2025 paper called *Thought Anchors* (Bogdan et al.). To test whether a
sentence matters:

1. Take the AI's reasoning up to just before that sentence.
2. Let the AI continue from there about 100 times, so that each time it writes its own
   version of the rest. This is called **resampling**.
3. Record the final answers it reaches.
4. Do the same from just *after* the sentence.
5. If the mix of final answers changes a lot, the sentence mattered. That makes it a
   candidate thought anchor.

It is a bit like asking: "If this person had not said that particular thing, would the
meeting have reached a different decision?" You find out by replaying the meeting many times.

The original paper found that anchors tend to be **planning sentences** ("Let me try a
different approach…") and **sentences where the model reconsiders** ("Wait, that can't be
right…"). It treated importance as a sliding scale. It did **not** say what fraction of
sentences count as anchors, and it did **not** test whether the measurement gives the same
result when repeated.

### 2.4 The question this project asks

> Just before the AI writes a sentence, does its internal state already show whether that
> sentence will be an anchor?

If yes, a monitor could in principle raise a flag before a decisive step is written. If no,
monitors would have to wait until after the step.

---

## 3. What has been done so far

| When | What | Size |
|---|---|---|
| September 2026 | **First test (the "pilot").** Read the AI's internal state just before each sentence and trained simple predictors to guess whether the sentence would be an anchor. Compared them with simple clues such as position in the reasoning. Final scores were measured on 4 problems kept aside and tested only once. | 20 maths problems, about 6,400 sentences, using anchor labels released with the original paper |
| October 2026 | **Repeat measurement ("Phase 2").** We re-ran the anchor measurement ourselves, twice, on part of the data, to see whether it gives the same answer. The test and its pass mark were written down before the data were generated (a "preregistration"). | 10 problems, 190 sentences, 40,000 AI continuations, about 43.5 hours on one rented GPU |
| October 2026 | **Follow-up analyses**, done without new data, on why the measurement is unreliable and what the AI's internal state does contain | Several analyses, all on computers already paid for |
| October 2026 | **"Answer now" test.** Interrupted the AI mid-reasoning and forced it to give an answer, to see whether that is a cheap substitute for the 100-replay method | 10 reasoning traces, about 15,000 short requests, under one hour of GPU time |
| October 2026 | **Literature review.** Searched for related research, checked what is new, and checked which maths problems the AI could not have seen during training | 8 research notes and 2 reports |

The project author reports about **USD 220** of their own money spent on rented computers. Our
records account for roughly $136–179 of this. The rest covers earlier runs, idle time and
storage, which are not fully logged.

---

## 4. What we found, in detail

### 4.1 The anchor measurement is mostly noise

When we measured the same sentences twice with fresh replays, the two importance rankings
agreed only moderately:

- The rank correlation was **0.44**, where 1.0 means identical rankings and 0 means
  unrelated.
- The pass mark we set in advance was 0.80. **The measurement failed.**

The main reason is a statistical effect we call the **noise floor**.

- **Comparing two batches of answers always shows some difference, purely by chance.** Even
  if a sentence changes nothing, two separate batches of 100 replays will never come out
  exactly the same.
- **The chance difference is bigger when the AI is unsure.** If the AI gives many different
  answers to a problem, two random batches differ more.
- **So the score mostly measures the AI's uncertainty.** It ranks sentences by "how unsure
  the AI was at this point", not by "how much this sentence changed the outcome". That is
  why it partly repeats: uncertainty is stable. But that part has nothing to do with
  anchors.

When we removed the noise floor and measured only the part that reflects real change, the
repeatability dropped to about **0.16** on the original released data. On our own smaller
replay set it was **0.18**, with a wide margin of error. To make that score reliable, each
sentence would need roughly **2,100 replays** instead of 100.

We also checked the original paper's own scoring formula, rather than our simpler version.
Its yes/no "anchor" label was even less repeatable: agreement of only **0.20** on a scale
where 1.0 is perfect.

### 4.2 Real anchors are rare

We then asked, for each sentence, whether its effect was bigger than chance could explain,
using a standard statistical test that corrects for testing many sentences at once.

- **On the full released data:** 131 out of 6,434 sentences (**about 2%**) passed.
- **On our larger replay set:** 7 out of 190 sentences (**about 3.7%**) passed. Because the
  sample is small, the true figure could be anywhere from about 0% to 8%.

These real anchors are concentrated:

- They appear mostly **before the AI has settled on an answer**, and where the AI is still
  uncertain. In uncertain stretches the rate rises to about 4 in 100.
- In our replay set, **4 of the 7 came from a single problem**.
- When we read them, they turned out to be moments where the AI **committed to an
  interpretation of the problem**, often in pairs that pushed the answer one way and then
  back.

### 4.3 What the AI's internal state does and does not contain

We trained simple predictors to read the AI's internal state just before each sentence.

**What it does contain:**

- **How unsure the AI is about the final answer.** This decodes clearly (a correlation of
  about 0.5), especially when comparing *different problems*.
- **What kind of sentence is coming next.** For example, whether the next sentence will be a
  "Wait, let me reconsider…" sentence decodes well, much better than from the visible
  text. This could simply be the AI preparing its next word, since 93% of "reconsidering"
  sentences start with "Wait" or similar. That explanation has not yet been tested.

**What it does not contain, as far as we can tell:**

- **How important the next sentence will be**, beyond simple clues. On the 4 problems kept
  aside, a rule using only "how far into the reasoning are we?" scored about as well as, or
  better than, the predictor that reads the AI's internal state. On the separate validation
  problems the order reversed. With so few test problems, neither side clearly wins.
  - **Measured on held-out problems:** activation predictor 0.094, position alone 0.139.
  - **Base rate:** 0.036, the score a random guess would get.
  - The activation predictor's extra value over simple clues was **not** statistically
    reliable.
- **Whether the AI's final answer will be right.** It looks unreadable, but for a
  surprising reason: the signal points one way in correct reasoning and the opposite way in
  incorrect reasoning, so on average it cancels out. The AI's state could not tell a
  problem's correct reasoning apart from its incorrect reasoning: 49% accuracy on a test
  where guessing gets 50%.

**What we ruled out as explanations, within the limits of these tests:**
- **Problem difficulty.** It was not the cause.
- **How answers were grouped.** Treating "0.5" and "1/2" as the same answer changed nothing.
- **The predictor being too simple.** 108 versions were tried, including more flexible
  ones. None did better.

These tests looked only at one point just before each sentence, on 20 problems. Richer
readings of the AI's state have not been tried yet.

### 4.4 One part of the question is answered by mathematics

Split "importance" into two parts:

- **Direction:** will the next sentence make the right answer *more* or *less* likely?
- **Size:** *how much* will it move the answer, in either direction?

**Direction cannot be predicted in advance, in principle.** Before a sentence is written,
the AI could write many different sentences. Averaged over all of them, the effect on the
answer is exactly zero, because the AI's current chance of being right already includes
everything it might write next. So any "direction" predictor can only predict noise. This
is a known fact from reinforcement learning, but its consequence for this kind of research
had not been pointed out. Our data agree: the measured predictability was about 3%,
essentially zero.

**Size might be predictable, but we cannot yet say how much.** A first estimate suggested
that up to about half of the variation in size could in principle be predicted in advance.
But our checks show the estimating method is **not trustworthy** at the very small effect
sizes seen in real data. On a smaller subset it gave about 0.29, with a margin of error
from 0.13 to 0.88. Getting a trustworthy answer needs a purpose-built experiment, which is
the main goal of the next stage (section 7).

### 4.5 "Just answer now" does not work as a shortcut

Replaying the AI 100 times per sentence is expensive. A cheaper idea is to interrupt it
mid-reasoning and make it state an answer immediately.

- **The answers were mostly premature guesses.** About **78%** were answers the AI never
  reached when allowed to keep reasoning, often small numbers or partial results.
- **They did not track the AI's real chance of being right.**
- **The formal verdict is "inconclusive".** A quality check we set in advance failed, so by
  our own rules the test could not give a clean yes or no. The pattern itself is clear.

The planned fix is to let the AI continue for a short while, about 256 words' worth, before
forcing an answer. This has not been tried yet.

### 4.6 The released data do not match a fresh re-run

When we regenerated part of the original paper's released data with the same settings, the
AI got the answer right **about 11 percentage points more often** than in the released
version. This happened only on the 2 problems where the original reasoning was wrong; on
the others the two matched.

We ruled out the obvious causes: the sampling settings, the length limit, cut-off answers
and chance. One likely suspect is that the released version **removed all line breaks** from
the reasoning before replaying it. The original also used a commercial AI service rather
than running the model directly. **The cause is officially unexplained** until it is tested
directly.

### 4.7 The test problems may be too easy, and partly memorised

- **The model is very strong on this benchmark.** It scores about 94% on the standard
  version of this maths benchmark.
- **Some problems may be memorised.** At least one of the 20 problems is answered
  correctly 80% of the time even with no reasoning at all.
- **Selection was noisy.** The original paper chose "medium difficulty" problems using only
  10 tries each, which is too few to place problems reliably.
- **Easier problems have fewer decisive moments** (Du et al., 2026). So "2% real anchors"
  may partly reflect the easy, familiar problems used, not reasoning in general.

The next stage would use maths competition problems published **after** the AI was built
(2025–2026 contests), which it cannot have seen.

---

## 5. How this fits with other research

The literature search (to October 2026) found:

- **Nobody had checked whether thought-anchor measurements repeat.** We found no published
  reliability test or critique of the method. The noise-floor finding appears to be new.
- **Related work exists for most other parts:**
  - The original authors' follow-up, *Thought Branches* (2025), compared resampling with
    inserting sentences by hand. It did not test deletion, rewording, or the AI's
    randomness setting.
  - A 2026 study, *Legibility is Not Interpretability*, used a similar repeat-measurement
    check on a different question and different models. It must be cited.
  - Another 2026 study (Nikankin et al.) reports that the AI's internal state *does*
    predict which steps are important, at 80% or better accuracy, on the same model. That
    looks like a contradiction, but they read the AI's state *after* the step was written,
    and they define "important" as "can it be removed?". The difference must be explained
    clearly in any write-up.
  - Another 2026 study already applied the "logit lens" technique (decoding the AI's
    intermediate guesses) to this model. Our planned use of it is an extension, not a first.
  - Studies of "forking points" in AI text (Bigelow et al., 2024, and follow-ups) look at
    single words, not sentences, and do not ask whether forks can be predicted from inside.
- **The size-of-effect ceiling appears to be new.** We found no study measuring how
  predictable an anchor's size is in principle.

---

## 6. Honest limits of what we know

- **One AI model, one kind of task.** Only one AI model and only maths problems have been
  tested.
- **Few problems.** The pilot's held-out test had 4 problems and 25 anchors; our repeat
  measurement had 10 problems. One different problem could change some headline numbers.
- **Some results are exploratory.** Only the main repeatability test was written down in
  advance. The noise-floor explanation, the 2% figure and most other findings were worked
  out after seeing the data. That is legitimate, but weaker evidence.
- **The test problems were reused.** After the pilot's one-time test, the same 4 test
  problems were reused for about 25 further analyses. Those later numbers are exploratory.
- **The "Wait" explanation is untested**, as is the "line breaks" explanation for the
  data mismatch.
- **The size-of-effect ceiling estimate is not yet trustworthy.**
- **This is not a safety monitor.** Nothing here is a working tool for spotting dangerous AI
  reasoning. It is research about whether such tools could work.

---

## 7. What we want to do next

The next stage answers the open question: **can the size of a sentence's effect be predicted
before it is written, and how well at best?**

**The core experiment is "nested sampling".**
1. At chosen points in the reasoning, let the AI write several *different* possible next
   sentences.
2. Let it continue about **200 times from each** possible sentence.
3. Separate how much of the effect is due to *where we are in the reasoning*, which can be
   predicted in advance, from how much is due to *which sentence it happened to write*,
   which cannot.

That gives a hard ceiling on what any advance prediction could achieve. It uses 20 problems
the AI cannot have seen.

**Supporting experiments:**
- repeat the reliability checks on 30 more new problems;
- compare three ways of removing a sentence (replaying, deleting, rewording) at two
  randomness settings, to see whether "anchor" depends on how you measure it;
- test cheap substitutes for replaying, such as short-budget answering and reading
  intermediate guesses;
- re-run the released data's exact format, to explain the mismatch;
- optionally, a test on logic puzzles and on a safety-relevant setting.

**Plan and cost.** The work is split into two runs on rented computers, with a check in
between:

| Run | What | Estimated cost |
|---|---|---|
| Run 1 | Choose problems, measure computer speed, small trials. Then stop and decide whether to continue. | about $160–200 |
| Run 2 | The main experiments | about $1,350–2,850, depending on measured speed |

The estimate uses 8 high-end GPUs (H100) at Lambda's current price of $31.92/hour. The
speed on 8 GPUs is an estimate until Run 1 measures it.

---

## 8. What must happen before the next stage can start

These are not research results. They are things a collaborator should know.

1. **Agreement on sharing.** Unpublished results stay within the project team unless
   sharing is agreed, so wider release needs that agreement first.
2. **Approvals under the project's own rules.** The project has strict written rules. The
   next stage changes several things they lock down, and each change needs a written
   record and the owner's approval:
   - the problem set;
   - the use of deletion and rewording;
   - more than 100 replays per point;
   - the type of computer.
3. **Software still to build.** None of the following exists yet:
   - the program that spreads work across 8 GPUs efficiently;
   - the nested-sampling program;
   - tools to record the AI's internal state over whole sentences and its attention;
   - an extended answer checker for answers involving square roots or π;
   - the written plan for Run 2's analysis.

   The tools from the work so far handle one GPU, and the code currently recognises only
   certain GPU models.
4. **Problem licences.** The new competition problems may be used for non-commercial
   research, but their text should not be republished. Results must be shared without
   copying problem text.
5. **Realistic timing.** Given the build list, finishing within 6–8 weeks of starting is optimistic.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| Activations / internal state | The numbers an AI computes internally at each step, its "brain activity" |
| Anchor (thought anchor) | A sentence in the AI's reasoning that strongly changes its final answer |
| Chain of thought | The AI's written step-by-step reasoning before its answer |
| Correlation | A number from −1 to 1 showing how closely two things move together (1 = perfectly, 0 = unrelated) |
| GPU | A specialised computer chip used to run AI models, rented by the hour |
| Noise floor | The difference you see between two batches of replays purely by chance, even when nothing changed |
| Preregistration | Writing down the test and pass mark before collecting the data, so results cannot be cherry-picked |
| Resampling / replay | Letting the AI continue from the same point many times to see the range of outcomes |
| Repeatability (reliability) | Whether measuring the same thing twice gives the same result |
| R²_max (the ceiling) | The largest share of an effect that any advance prediction could possibly explain |

---

## 10. Where the details are

All documents are in the project repository:
https://github.com/cnm13ryan/pre-emission-thought-anchors/tree/study/anchor-reliability-v2/anchor_reliability_v2

| Topic | Document |
|---|---|
| Repeat measurement (Phase 2) | `docs/phase2_results.md`, `docs/phase2_diagnostics.md` |
| What the internal state contains | `docs/representation_rootcause.md`, `docs/probe_capacity.md` |
| Reading the 7 real anchors | `docs/real_effects_reading.md` |
| "Answer now" test | `docs/plan_v3_pilot_results.md` |
| All follow-up analyses in one table | `docs/v3_followup_cpu_results.md` |
| Literature review | `reports/Thought anchors novelty audit.md` |
| Choosing new problems | `reports/Out of distribution math problems.md` |
| Next-stage plan and costs | `docs/v3_followup_gpu_plan.md` |
| The original pilot | `../result_interpretation.md`, `../report.md` |
