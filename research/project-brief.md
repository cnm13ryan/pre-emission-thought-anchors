# Predicting High-Leverage Reasoning Transitions Before They Occur

## Purpose

This project asks whether we can predict, **before a reasoning sentence is generated**, whether that sentence will become a high-causal-leverage “thought anchor”: a reasoning step whose content disproportionately influences the model's subsequent reasoning or final behaviour.

The immediate goal is to determine whether information available in the model immediately before generation, such as its hidden state or next-token distribution, contains a detectable precursor of an upcoming consequential reasoning transition. If such a signal exists, it could provide a basis for monitoring models before an important change of plan, reconsideration, or other high-impact reasoning event becomes visible in the chain of thought.

The project will initially study sentence-level reasoning in open-weight reasoning models, where causal importance can be measured using counterfactual resampling and related methods. The longer-term motivation is to understand whether the same idea generalizes from reasoning sentences to consequential tool calls, actions, messages, or other computational events.

## Planned type and degree of involvement

The project will involve substantial independent research by mentees, with mentorship focused on research direction, experimental design, interpretation, troubleshooting, and deciding which hypotheses are worth pursuing.

We will have a weekly project meeting of approximately 30–60 minutes. Individual 1:1s can be used when useful for discussing specific blockers, research decisions, or development goals. Most day-to-day communication will happen asynchronously through Slack.

I expect mentees to make an independent first attempt at research questions and technical problems before escalating them. Failed experiments should be reported clearly rather than hidden or repeatedly extended without evidence that they remain useful.

The main goal of meetings will be to review evidence, identify the largest remaining uncertainty, and choose the cheapest experiment that can meaningfully reduce that uncertainty.

## What I’m asking for

Mentees should commit at least approximately 10 hours per week to the project and communicate early if other commitments make that temporarily unrealistic.

I am especially interested in work that:

- turns broad hypotheses into cheap, discriminating experiments;
- distinguishes competing explanations rather than accumulating suggestive results;
- produces reproducible code and inspectable intermediate results;
- changes direction quickly when evidence undermines the current hypothesis;
- clearly separates empirical findings from interpretation.

The project should move through repeated cycles of:

\[
\text{hypothesis}
\rightarrow
\text{small experiment}
\rightarrow
\text{result}
\rightarrow
\text{updated hypothesis}.
\]

No special confidential dataset is currently required. Unpublished project results, internal discussion, and unfinished analyses should nevertheless stay within the project team unless we agree to share them.

## Research question and sub-questions

### Core research question

**Can we predict, using only information available immediately before a reasoning sentence is generated, whether that sentence will subsequently have unusually high causal influence on the model's reasoning or behaviour?**

Formally, suppose sentence \(S_i\) has not yet been emitted and the model is currently in state \(h_{i^-}\).

We want to learn whether

\[
h_{i^-}
\rightarrow
I(S_i),
\]

where \(I(S_i)\) measures the downstream causal importance of the sentence that will subsequently be generated.

The primary target will initially be the counterfactual-importance measure used in Thought Anchors:

\[
I(S_i)
=
D_{\mathrm{KL}}
\left(
P(A\mid \text{different }S_i)
\;\|\;
P(A\mid S_i)
\right).
\]

High \(I(S_i)\) means that changing the reasoning step substantially changes the model's downstream answer distribution.

### Sub-questions

1. **Does a pre-emission signal exist at all?**

   Before \(S_i\) is generated, can hidden-state information predict its eventual counterfactual importance better than simple baselines?

2. **Where is the predictive information represented?**

   Is the signal concentrated in particular transformer layers, or is it distributed throughout the network?

3. **How early does the signal appear?**

   Is the upcoming anchor predictable only immediately before generation, or can it be detected several tokens or reasoning steps earlier?

4. **What does the predictor detect?**

   Does it primarily identify upcoming planning decisions, uncertainty, backtracking, strategy changes, or something more general?

5. **Does the signal predict causal importance rather than superficial properties?**

   We should test against baselines such as reasoning position, context length, next-token entropy, token probabilities, and other cheap observable features.

6. **Does the signal generalize?**

   Can a predictor trained on some problems predict thought anchors on held-out problems, domains, or model checkpoints?

7. **How does pre-emission predictability relate to the paper's internal measures?**

   Do predicted anchors subsequently become:
   - high-counterfactual-importance sentences;
   - strongly attended-to “broadcasting” sentences;
   - sources of strong downstream causal links under attention suppression?

8. **Is pre-emission prediction operationally early enough to matter?**

   Can the signal identify a consequential transition before the model has already effectively committed to the associated strategy?

### Longer-term question

The broader research question is:

**Can we detect internal precursors of high-causal-leverage computational transitions before they manifest in reasoning, tool use, communication, or actions?**

Sentence-level thought anchors are the first tractable test case rather than the assumed final unit of analysis.

## Scope

### MVP

The minimum successful project would establish whether pre-emission prediction of thought-anchor importance is possible under a controlled setting.

The MVP should:

1. Select one open-weight reasoning model and a manageable reasoning dataset.

2. Reproduce or implement a Thought Anchors-style measure of sentence-level counterfactual importance.

3. For every reasoning sentence \(S_i\), record model information available immediately before \(S_i\) begins, such as:
   - residual-stream or hidden-state activations;
   - final-token representations at different layers;
   - next-token logits;
   - entropy or other uncertainty statistics.

4. Construct a dataset

\[
(h_{i^-}, I(S_i)).
\]

5. Train simple predictors of future anchor importance, beginning with linear probes.

6. Compare them against strong cheap baselines that do not require internal activations.

7. Evaluate on held-out problems rather than randomly splitting individual sentences from the same trajectories across train and test.

8. Determine whether there is convincing evidence that upcoming causal importance is represented before the sentence itself is emitted.

A successful MVP does **not** require building a deployable safety monitor. A clear positive result, clear negative result, or strong characterization of where predictability breaks down would all be useful outcomes.

### Stretch goals

If the MVP produces a real signal, useful extensions include:

- testing multiple model sizes or reasoning-model families;
- predicting the magnitude of causal importance rather than only classifying high-importance sentences;
- predicting several tokens or sentences ahead;
- relating the signal to receiver-head attention;
- predicting specific downstream causal links \(S_i\rightarrow S_j\);
- testing whether the predictor transfers across domains;
- distinguishing planning, uncertainty-management, backtracking, and routine computation;
- studying whether pre-anchor states correspond to unusually high uncertainty over future reasoning trajectories;
- testing whether an anchor corresponds to a transition from multiple plausible reasoning directions into a more committed plan;
- extending the unit of analysis from reasoning sentences to tool calls or actions;
- testing passive monitoring on simple agentic environments.

### Out of scope for the initial project

The MVP will not attempt to:

- solve general mechanistic interpretability;
- monitor arbitrary long-horizon agents in realistic deployment environments;
- build a complete safety intervention system;
- prove that thought anchors correspond to human-interpretable concepts;
- optimize the reasoning model against the predictor;
- study adversarial evasion of the monitor in the first phase;
- require realistic simulation of every possible downstream external environment.

These extensions become meaningful only after establishing whether the basic pre-emission signal exists.

## Methods

The project should proceed in several stages.

### Stage 1: Reproduce the target variable

Implement sentence-level counterfactual importance for a small set of reasoning traces.

For sentence \(S_i\):

1. preserve the reasoning prefix through \(S_i\) and sample downstream continuations;
2. restart immediately before \(S_i\) and sample alternative continuations;
3. filter replacements that are semantically too similar to the original sentence;
4. compare resulting answer distributions;
5. assign \(S_i\) a counterfactual-importance score.

This produces the retrospective ground truth we are trying to predict.

### Stage 2: Capture the model before the sentence occurs

For each sentence boundary, run the original trace through the model only up to the point immediately before the next sentence begins.

Extract candidate pre-emission features such as

\[
h^{(1)}_{i^-},
h^{(2)}_{i^-},
\ldots,
h^{(L)}_{i^-},
\]

where \(h^{(\ell)}_{i^-}\) is the model's representation at layer \(\ell\).

Also record cheap observable predictors such as:

\[
H(P(x_{t+1}\mid x_{\leq t}))
\]

for next-token entropy, context position, and other quantities available without accessing hidden activations.

### Stage 3: Test simple predictors first

Start with deliberately simple models:

\[
\hat I(S_i)=w^\top h_{i^-}+b.
\]

For classification, define anchors using a preregistered threshold such as the highest-importance quantile and fit logistic regression.

The first important comparison is:

\[
\text{activation probe}
\quad\text{vs.}\quad
\text{cheap observable baselines}.
\]

A complex neural predictor is only justified if simple methods establish that there is meaningful signal to exploit.

### Stage 4: Localize and stress-test the signal

If prediction works, test:

- which layers carry the signal;
- how far before the sentence it appears;
- sensitivity to anchor threshold;
- robustness across sampling seeds;
- robustness to problem-level train/test splits;
- whether performance survives controlling for position and uncertainty;
- whether it generalizes to new domains.

### Stage 5: Interpret what the predictor is detecting

Compare highly predicted upcoming anchors against:

- counterfactual importance;
- receiver-head vertical attention;
- attention-suppression causal links;
- semantic categories such as planning, uncertainty, computation, and backtracking.

The objective is to distinguish:

\[
\text{“the model knows an important transition is coming”}
\]

from simpler explanations such as:

\[
\text{“the model is merely uncertain about the next token.”}
\]

### Starting literature

The initial reading should include:

- *Thought Anchors: Which LLM Reasoning Steps Matter?*
- OpenAI, *An Alien Mind*
- OpenAI, *Accidental Chain-of-Thought Grading*
- OpenAI, *Self-Incrimination*
- OpenAI, *Predicting LLM Safety Before Release by Simulating Deployment*

Additional literature on activation probing, chain-of-thought faithfulness, monitoring, and causal interventions should be added as the specific experimental design becomes clearer.

## Output

The preferred final output is a concise empirical research report or public research post accompanied by reproducible code and enough intermediate artifacts for another researcher to reproduce the central result.

The core output should answer:

> **Before a reasoning step is emitted, is there already enough information inside the model to predict that the upcoming step will disproportionately shape what happens next?**

A strong positive result would include:

- a reproducible thought-anchor labeling pipeline;
- a pre-emission activation dataset;
- well-controlled predictive experiments;
- comparison against simple baselines;
- layer and lead-time analyses;
- cross-domain or cross-problem validation;
- a clear account of limitations and alternative explanations.

A strong negative result would also be useful if it convincingly establishes that apparent predictability disappears under proper controls, problem-level generalization, or causal validation.

If results warrant it, the project could become a workshop paper or similar research submission. Mentees who make substantial intellectual, experimental, analytical, or writing contributions should receive corresponding authorship or credit.