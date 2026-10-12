# Where the study stands: a plain-language overview

*Written 2026-10-11. This summarises the study so far, in non-technical terms. The detailed
sources are listed at the end. Everything after the Phase 2 confirmatory test is
exploratory.*

## What this project is about

When a reasoning model works through a maths problem, it writes a chain of sentences before
it gives an answer. A 2025 paper, *Thought Anchors* (Bogdan, Macar, Nanda, Conmy), claimed
that a few of those sentences are pivotal: rewrite one, and the final answer often changes.
The authors measured this by re-running the model from each sentence about 100 times and
comparing how often each answer came out.

Our original goal went one step further: **predict which sentences will be pivotal before
the model writes them**, using the model's internal activations.

## What we found

**1. The importance score mostly does not measure what it claims to.**
- We regenerated the measurements ourselves, twice, on 190 sentences. The two copies agreed
  only weakly: a correlation of 0.44, against the 0.80 we had preregistered as acceptable.
- The part that did repeat was largely an artefact. When the model is unsure of its answer,
  the score comes out high whether or not that particular sentence matters.
- With that artefact removed, the score's reliability is about 0.16. Making it reliable
  would take roughly 2,100 re-runs per sentence instead of 100.

**2. Truly pivotal sentences exist, but they are rare.** About 2% of sentences pass a proper
statistical test. They cluster where the model is still undecided, before it has settled on
an answer.

**3. The earlier claim that activations predict anchors was really predicting uncertainty.**
- The model's internal state tells us how unsure it is, not which sentence will be pivotal.
- We ruled out the obvious excuses: problem difficulty, how answers are grouped, the size and
  type of the probe, and one odd problem.
- One part cannot be predicted even in principle: whether a sentence *helps* or *hurts*.
  Averaged over all the sentences the model might have written, that effect cancels to zero.
  This is a known fact from reinforcement learning. What is new is spelling out what it
  means for this kind of probing.

**4. Asking the model for its answer mid-reasoning does not work.** When forced to answer
part-way through, 78% of its answers are premature guesses that full reasoning never
reaches.

**5. The released data differ from our own regeneration.**
- On the problems where the model's own solution was wrong, the released data show 11
  points lower accuracy than we get.
- The leading suspects:
  - The released prompts squashed every line break into a space.
  - The released data were generated through a third-party hosted service rather than a
    controlled setup.
- This is not yet proven.

## What the literature review added

- **Our main findings appear to be new.** Nobody has published a reliability check or a
  critique of these importance scores.
- **One question is genuinely open:** how predictable is the *size* of a sentence's effect,
  regardless of direction, before it is written? We call this ceiling R²_max.
  - A rough estimate puts it around 0.5. That would mean prediction is possible in
    principle and current methods simply fall short.
  - But the estimate is not trustworthy at the very small effect sizes we actually see.
- **Some planned analyses already exist elsewhere,** so we will present them as extensions or
  comparisons, not as new results:
  - reading the model's intermediate guesses layer by layer (the "logit lens") on this exact
    model;
  - activation probes for step importance;
  - attention-based "receiver heads".
- **Our maths problems are partly too easy and probably seen in training.** The model scores
  94% on this benchmark family, and one of our 20 problems is plainly memorised: it is
  answered correctly 80% of the time with no reasoning at all.

## Which problems are safe to use next

- **Problems first published after the model's release (January 2025) are surely unseen.**
  The best source is about 270 problems from eight maths competitions held in 2025–26
  (AIME, BRUMO, HMMT, SMT, CMIMC), collected by MathArena.
- **Even these need checking.** Some new contest problems closely resemble older ones; 8 of
  the 30 AIME 2025 problems do. Automatic contamination detectors are essentially useless on
  this model. So the publication date, plus a careful duplicate search, is the real
  safeguard.
- **Freshly generated logic puzzles** are a useful second domain. Every instance is new, but
  the model has seen these *kinds* of puzzle in training.

## The plan, in three steps

The full plan is in `docs/v3_followup_gpu_plan.md` (v3). In outline:

**Step 0: preparation, with no GPU.**
- The owner approves a dataset change, because the project's rules freeze the current 20
  problems.
- We preregister the main experiment before collecting any of its data.
- We assemble and de-duplicate the problem pool, extend the answer checker, and build the
  run software.

**Run 1: a short GPU session to screen and test.** About 4.4–7.9 hours on 8 H100 SXM GPUs
(35–63 GPU-hours), roughly $140–250 at Lambda's verified $31.92/h.
- Sample each candidate problem 16 times, then 48 times, and keep those the model solves
  15–85% of the time.
- Run red-flag contamination checks, and screen the puzzles.
- Pilot a cheap "think briefly, then answer" readout.
- Measure the real speed and trace lengths.
- **Then report the yield and a revised cost to the owner, who decides on Run 2.**

**Run 2: the main experiment.** About 42–137 hours on 8 H100 SXM GPUs (336–1,099
GPU-hours), most likely 42–57 hours, which is about $1,340–1,820 at $31.92/h. The upper
end assumes the slow single-GPU rate measured in Phase 2. Run 1 measures the real speed. In priority order:
1. **Nested sampling, the flagship.** On 20 hard, clean problems, sample several candidate
   sentences at each step, then 200 continuations from each. This measures how predictable
   the size of a sentence's effect is.
2. **Repeat the reliability check on the new problems.** If real anchors turn out much more
   common on hard, clean problems than our 2%, that is a headline in itself.
3. **Test whether "anchor" depends on the method.** Resample, delete or paraphrase the same
   sentences, at two temperatures.
4. **Test the line-break explanation** for the released-data gap.
5. **The cheap readout**, only if the pilot works.
6. **Activation, logit-lens and attention analyses, each compared against the ceiling.**
   Attention comes last.
7. **A small token-level analysis, and the puzzle domain.**

## What the outcome could mean

- **Low ceiling.** Pivotal sentences are decided by the sentence itself, so predicting them
  in advance is a dead end. The field should analyse anchors after the fact instead.
- **High ceiling, but our signals fall short.** Better readouts are worth building.
- **"Anchor" depends on how it is measured.** The concept needs redefining before anyone
  builds on it.

Whichever way it falls, the reliability critique of the original method stands on its own as
a contribution.

## Decisions waiting on the owner

**Now:**
1. Approve the dataset change: new competition problems, stored as references only, for
   non-commercial research use.
2. Confirm the hardware: 8 × H100 SXM, recommended at $31.92/h (8 × A100 80 GB at $22.32/h
   is the fallback).
3. Approve the Run 1 budget, about $140–250.

**After Run 1:**
4. How deep the nested sampling goes (4 or 8 candidate sentences per step; 150 or 200
   continuations each).
5. The Run 2 budget.
6. Whether to use an external hosted API for the released-data comparison.

## Sources

| Topic | Location |
|---|---|
| Phase 2 confirmatory result | `docs/phase2_results.md` |
| Why the score fails to reproduce | `docs/phase2_diagnostics.md` |
| Why activations predict uncertainty, not anchoring | `docs/representation_rootcause.md`, `docs/probe_capacity.md` |
| Forced-answer pilot | `docs/plan_v3_pilot_results.md` |
| All follow-up analyses in one table | `docs/v3_followup_cpu_results.md` (details in `analysis/v3_followup/`) |
| Literature review | `reports/Thought anchors novelty audit.md`, `reports/Out of distribution math problems.md` |
| Plan and costs | `docs/v3_followup_gpu_plan.md`, `analysis/v3_followup/gpu_plan_cost/` |
