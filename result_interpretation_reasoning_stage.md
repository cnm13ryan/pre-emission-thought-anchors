# Do pre-emission activations encode reasoning stage — and does that explain the thought-anchor signal?

Exploratory follow-up to the MVP (see `result_interpretation.md`). All analyses are
post-hoc: nothing here reopens the frozen MVP endpoints. Every number below comes from
`research/reasoning_stage/outputs/analysis_results.json` and `analysis_log.txt`, produced by
the three scripts in `research/reasoning_stage/` from the saved Stage 3/4/7 artifacts and the
Stage 7 activation store. The frozen problem-level splits and leakage controls were reused
unchanged; hyperparameters were chosen only by grouped cross-validation on training problems;
the test problems were scored once per pre-specified model with no test-driven tuning.

---

## 1. The question

The MVP found that a probe on the model's state just before a reasoning sentence weakly
predicts that sentence's counterfactual importance ("anchor" status), but that simple
sentence position predicts anchors at least as well. This follow-up asks: **does the
pre-emission state encode what stage of reasoning the model is in, and can that explain the
apparent anchor signal?** Three specific hypotheses:

1. **Absolute position:** the probe mainly knows how far into the trace it is.
2. **Reasoning phase:** the probe knows what kind of reasoning operation is happening or
   about to happen (planning, computing, verifying, backtracking, concluding).
3. **Transition state:** just before consequential sentences the model is in a distinctive
   "changing mode" state, and anchors coincide with stage transitions.

## 2. Reasoning-stage taxonomy and labeling

Seven stages, grounded in inspection of the actual DeepSeek-R1-distill traces:
**orient** (restating the problem), **plan** (choosing a strategy: "Let me try…",
"Alternatively…"), **compute** (carrying out math; the default for sentences with
mathematical content), **verify** ("let me check", "that seems correct"), **backtrack**
(doubt/error detection: "Wait…", "that can't be right"), **conclude** (stating the final
answer), and **other** (fragments with no signal).

Labels are assigned by a deterministic, ordered rule set over surface markers
(`research/reasoning_stage/label_stages.py`); 30 audited examples per stage are in
`outputs/stage_label_audit.md`. R1-style traces make this unusually tractable — "Wait" alone
opens 733 of 6,474 sentences. These are still noisy, surface-level labels: a sentence like
"Wait, let me check the endpoints" mixes doubt and verification, and the rules resolve such
overlaps by fixed priority. No LLM labeling was used, so the labels are fully reproducible
but only as good as the marker lexicon. Treat stage-level results as robust in their broad
strokes, not in fine distinctions.

Counts over all 6,474 sentences: compute 3,839; backtrack 830; other 655; plan 578;
verify 427; orient 75; conclude 70.

## 3. Data

The merged dataset (`outputs/dataset.parquet` + row-aligned activation matrices for layers
0 and 44) reproduces the MVP exactly: 2,650 / 680 / 689 eligible sentences and 265 / 54 / 25
anchors on train / validation / test, from the Stage 4 canonical labels and the frozen q90
threshold (1.35498). As a further check, my re-fit position-only anchor probe recovers the
MVP's frozen position baseline on test to four decimals (AUCPR 0.1394, AUROC 0.7188).

## 4. Main quantitative results

### A. Position barely predicts reasoning stage

The stage mix is nearly constant across the trace: compute is 54–64% in every position
quintile; backtrack 10–16% everywhere. A multinomial probe from position features gets
validation macro-F1 0.05 and log-loss 1.92 — essentially nothing. The full 12 registered
observables reach macro-F1 0.17 (log-loss 1.80). **Reasoning stage is not determined by
where you are in the trace.** (Only `conclude` and `orient` are position-locked, and they
are rare.)

### B. Activations predict the upcoming stage strongly — and far beyond position

Predicting the stage of the *about-to-be-generated* sentence from the pre-emission boundary
state (multinomial logistic, layer 44):

| Model | Val accuracy | Val macro-F1 | Val log-loss |
|---|---|---|---|
| Majority class (compute) | 0.599 | — | — |
| Position only | 0.076* | 0.053 | 1.92 |
| All 12 observables | 0.266* | 0.175 | 1.80 |
| **Activations (layer 44)** | **0.684** | **0.457** | **1.33** |
| Activations layer 0 | 0.327* | 0.138 | 1.62 |
| Activations + observables | 0.686 | 0.463 | 1.33 |

*Class-balanced training deflates raw accuracy for weak models; compare macro-F1 and
log-loss. Test-split numbers are similar (activations 0.634 accuracy, macro-F1 0.385).

Three things stand out. Activations carry a lot of stage information; position and the
scalar uncertainty observables carry almost none of it; and — unlike the anchor task, where
the layer curve was flat — **stage decoding is strongly layer-dependent** (layer 0 macro-F1
0.14 vs layer 44's 0.46). The pre-emission state at deep layers knows what kind of sentence
is coming.

Important caveat: because the stage labels are surface markers, "predicting the upcoming
stage" is close to "predicting the upcoming words" (the model is often literally about to
emit "Wait"). This is exactly the "richer next-token semantics" channel flagged as
alternative 4.3 in the MVP interpretation, now demonstrated directly: the residual state
contains next-sentence-type information that the registered scalar summaries discard.

### C. Reasoning stage does NOT explain the anchor signal

Anchor prediction (AUCPR; eligible rows; C chosen by grouped CV on train; test prevalence
0.036, validation 0.079):

| Features | Val AP | Test AP |
|---|---|---|
| Position | 0.133 | 0.139 |
| Stage only | 0.088 | **0.035 (= chance)** |
| Position + stage | 0.127 | 0.092 |
| Activations (L44) | 0.171 | 0.094 |
| Activations + position | 0.171 | 0.092 |
| Activations + stage | 0.187 | 0.071 |
| Activations + position + stage | 0.186 | 0.073 |
| Activations residualized on position+stage | 0.155 | 0.086 |
| Observables12 + stage | 0.133 | 0.044 |
| Activations + observables12 + stage | 0.162 | 0.096 |

Paired problem-bootstrap deltas (2,000 reps, percentile 2.5–97.5):

- Activations+position+stage minus position+stage: validation +0.059 (−0.101 to +0.145),
  test −0.019 (−0.164 to +0.098). **Null.**
- Stage added to position: test −0.048 (−0.166 to +0.003) — stage labels *hurt*
  out-of-problem generalization.
- Within compute-stage rows only, activations vs position flips sign across splits
  (validation +0.049 with CI 0.003–0.062; test −0.035 with CI −0.085 to −0.007) —
  unstable, uninterpretable at this sample size.

So the mediation story fails in both directions: stage labels alone predict anchors at
chance on held-out problems, and controlling for stage does not remove the activation
signal (the residualized probe keeps most of its AP). Whatever the weak anchor signal is,
**it is not "the probe knows the reasoning phase, and phase predicts importance."**

### D. Anchors are not concentrated at stage transitions

On train+validation eligible rows, anchor rate is 9.7% at stage transitions vs 9.3% at
same-stage continuations; problem-bootstrap CI for the difference: −0.023 to +0.028.
**Null.** By stage, only `plan` shows enrichment (14.4% vs 9.6% overall; 54 of 375 plan
sentences are anchors), and the most anchor-rich transition pairs are plan-adjacent
(compute→plan 18/100; plan→compute 27/153) — consistent with plan enrichment rather than
with transitions per se. `orient` (1/50) and `conclude` (0/16) are anchor-poor. On the test
split anchors are spread across stages (compute 13/322, backtrack 4/122, plan 3/85) with
no usable concentration.

### E. Stage transitions ARE predictable pre-emission — the one strong positive result

Predicting "the next sentence starts a different stage than the previous sentence"
(base rate: 0.475 validation, 0.601 test):

| Model | Val AP | Test AP |
|---|---|---|
| Position | 0.405 | 0.626 |
| Observables12 | 0.634 | 0.636 |
| **Activations (L44)** | **0.805** | **0.848** |

Paired gain of activations+position over position: validation +0.40 (CI +0.34 to +0.45),
test +0.22 (CI +0.11 to +0.28) — **clearly positive on both held-out splits**, the only
comparison in this entire project with that property.

**Lead-one:** the boundary state *one sentence earlier* also predicts the transition
(AP 0.765 validation / 0.822 test; gain over position CI +0.29 to +0.42 validation,
+0.08 to +0.26 test). But an oracle control deflates this: simply knowing the true stage of
the intervening sentence (plus position) achieves AP 0.762 / 0.792, and activations add
only +0.011 (CI −0.015 to +0.032) on validation and +0.034 (CI +0.001 to +0.081) on test
beyond that oracle. So the lead-one result is almost entirely "the state predicts the next
sentence's stage, and stage sequences have predictable transition statistics" — **no
convincing evidence of genuine two-step anticipation**, mirroring the MVP's lead-one null
for anchors. Lead-one stage identity prediction degrades accordingly (accuracy 0.489 vs
0.684 at lead zero).

## 5. Strongest supported interpretation

**The pre-emission residual state at deep layers strongly encodes what kind of reasoning
sentence is about to be generated — planning, computing, verifying, backtracking — and this
information is almost entirely absent from position and from the registered scalar
observables. However, this reasoning-phase information does not explain the weak
thought-anchor signal: stage labels predict anchors at chance across problems, anchors are
not concentrated at stage transitions, and removing stage+position from the activations
leaves the anchor probe essentially intact.** Of the three hypotheses, the reasoning-phase
hypothesis is *true as a claim about what activations encode* but *false as an explanation
of the anchor result*; the transition-state hypothesis is unsupported for anchors; and the
absolute-position hypothesis survives only in the weak form that position remains the best
single out-of-problem anchor predictor while explaining neither stage nor the activation
content.

The sharpest overall reading: the boundary state is a rich short-horizon predictor of the
*form* of the upcoming reasoning step (highly decodable, layer-dependent, robust across
splits), while the *counterfactual importance* of that step remains barely predictable by
anything — activations, position, or stage — at this sample size.

## 6. Competing explanations that remain open

- **Stage ≈ next-token surface form.** The stage labels are marker-based, so stage
  decodability may reduce to "the model knows its next few words." Distinguishing a real
  phase representation from next-token lookahead needs stage labels that are less tied to
  the opening token (e.g., human labels of sentence function), or probes at boundaries
  several tokens before the marker.
- **The residual anchor signal is still unexplained.** The activation anchor probe survives
  stage+position residualization (val 0.155 / test 0.086) but its incremental value over
  those controls is statistically null. It could be label noise, problem idiosyncrasy, or a
  small real effect; this analysis narrows the candidate mediators (not stage, not
  transitions) without identifying the source.
- **Coarse transition definition.** "Any stage change" happens on roughly half of sentences,
  so transition prediction is a broad discrimination task; rarer, more meaningful
  transitions (e.g., execution→backtracking specifically) were not separately powered.
- **Test-split oddities persist.** The test problems remain unusual (higher transition base
  rate 0.60, lower anchor prevalence 0.036), so validation/test disagreements (e.g., the
  within-compute flip) should be read as small-sample instability.

## 7. Limitations

- **Label quality is the binding constraint on the stage analyses.** The taxonomy is
  rule-based over surface markers; misclassification is certainly present (documented
  examples in the audit file). Reliability was not independently quantified against human
  labels. Results that depend on fine stage distinctions (per-pair transition rates) are
  correspondingly soft; the coarse results (A, B, E) involve large margins and should be
  robust to moderate label noise.
- Anchor-side analyses inherit every MVP limitation: 4 test problems, 25 test anchors,
  noisy counterfactual labels (released-vs-canonical Spearman 0.858), descriptive bootstrap
  over few problems.
- Only layers 0 and 44 were probed here; the stage-layer profile between them was not mapped.
- All probes are linear; a nonlinear readout could change the residual-signal picture.
- Hyperparameter selection for the ladder used grouped CV on train with the MVP's C grid;
  the stage classifiers used a fixed C=0.1 (no tuning), so their absolute numbers are
  conservative.

## 8. Most discriminating follow-ups

1. **Break the marker dependence:** relabel a few hundred sentences by hand (or with an LLM
   judge audited by hand) using functional definitions rather than openers, and re-test B.
   If activation-decodable "stage" survives, it is more than next-token form.
2. **Move the probe boundary back within the previous sentence** (e.g., 5–10 tokens before
   the sentence break). Genuine phase/transition anticipation should decay smoothly;
   next-token surface prediction should collapse at the point where the opening marker
   stops being locally determined.
3. **Target the specific transitions that matter:** collect enough traces to power
   execution→backtracking and plan-commitment events separately, since those are the
   transitions with any hint of anchor enrichment (plan-adjacent pairs at 17–18%).
4. **Use stage posteriors as features in a larger replication:** in the planned
   larger-problem MVP replication, include predicted-stage probabilities alongside position
   in the observable baseline. If the anchor gain over that baseline is still positive,
   the anchor signal is genuinely not phase-mediated.

## 9. Files

- Report: `result_interpretation_reasoning_stage.md` (this file)
- Scripts: `research/reasoning_stage/label_stages.py`, `build_dataset.py`, `analyses.py`
- Artifacts: `research/reasoning_stage/outputs/` — `stage_labels.parquet`,
  `stage_label_audit.md`, `dataset.parquet`, `activations_l0.npy`, `activations_l44.npy`,
  `analysis_results.json`, `analysis_log.txt`
- Inputs (read-only): `data/processed/stage3/sentences.parquet`,
  `stage4/labels.parquet`, `stage7/features.parquet`, and the Stage 7 activation store
  `artifacts/runs/20260915T182306Z-3ab408d80a-7452f1c2/activations/` under
  `~/preanchor-root`.
