# What the Preanchor MVP experiment found

## 1. Question

The experiment asked a narrow prospective question:

**Immediately before a reasoning sentence is generated, does the model's residual-stream state predict how much that subsequently generated sentence will change the model's final-answer distribution under a Thought-Anchors-style counterfactual intervention?**

The more demanding comparison was whether activations provide predictive information **beyond information already available from the visible prefix**, such as reasoning position, context length, and next-token uncertainty.

This is an exploratory construct-validation experiment on one model and one public MATH dataset. It does not directly test deployment monitoring, deception, dangerous actions, or whether the model itself uses an internal representation of "future importance."

---

## 2. Bottom line

The experiment **does not establish that pre-emission activations contain an importance-specific or useful anticipatory signal**.

The selected activation-plus-observables probe achieved test AUCPR 0.094, compared with a test anchor prevalence of 0.036 and AUCPR 0.057 for the registered combined-observables baseline. However, its incremental gain over that baseline was only +0.037, with a problem-bootstrap interval of −0.051 to +0.081. A structured permutation analysis also placed the observed score inside its null distribution (empirical p ≈ 0.17).

Several additional results make a simple "internal precursor" interpretation difficult:

* a position-only baseline had a higher test point estimate than the activation probe (0.139 vs 0.094);
* the activation layer curve was already elevated at the embedding output and was nearly flat across the network;
* performance varied substantially across the four test problems;
* prediction one reasoning sentence earlier fell to approximately the base rate.

These observations rule out a simple strong story in which a robust, layer-localized, one-step-ahead "importance signal" was discovered.

They do **not**, however, establish that no such precursor exists. The test set contains only four problems and 25 positive sentences, and several unresolved measurement and confounding questions could materially change the interpretation of both the positive and negative findings.

The safest current conclusion is therefore:

**There is a raw association between the boundary representation and the later Thought-Anchors-style label, but this MVP does not determine whether that association reflects future importance itself, ordinary trajectory structure, local token/semantic information, or properties of the importance estimator.**

---

## 3. What the experiment directly established

### 3.1 There is an above-baseline raw association

On the frozen test set:

| Predictor                | AUCPR |
| ------------------------ | ----: |
| Prevalence               | 0.036 |
| Combined observables     | 0.057 |
| Activation only          | 0.093 |
| Activation + observables | 0.094 |
| Residualized activation  | 0.127 |
| Position only            | 0.139 |

Thus the residual-stream representation immediately before a sentence contains features that a linear probe can use to rank the subsequently measured anchor label better than prevalence on this particular test set.

This should be described as a **raw predictive association**, not as evidence that the model represents future causal importance.

### 3.2 Incremental value beyond the registered observables was not established

The registered comparison,

$$
\Delta \mathrm{AUCPR}
=
\mathrm{AUCPR}(\text{activation+observables})
-
\mathrm{AUCPR}(\text{observables}),
$$

was +0.0368, with a problem-bootstrap interval of −0.051 to +0.081.

The four test problems also differed substantially:

* +0.133
* +0.077
* +0.009
* −0.066

The estimate therefore depends strongly on which problem is examined.

The structured permutation analysis similarly did not distinguish the observed score from its registered null distribution.

With only four independent test problems, these uncertainty estimates should be treated descriptively rather than as precise population inference.

### 3.3 Position is a strong competing predictor

Position alone obtained AUCPR 0.139, compared with 0.094 for the activation-plus-observables probe.

The paired difference itself is highly uncertain (−0.045, bootstrap interval −0.236 to +0.026), so this does not establish that position is genuinely superior. It does establish that the experiment provides no evidence that the activation probe outperforms this simple positional baseline.

Anchors are also strongly concentrated near the beginning of the eligible test trajectories (15 of 25 test anchors fall in the first position quintile).

This makes trajectory position, reasoning phase, or some quantity correlated with them an important alternative explanation.

### 3.4 The selected layer is not meaningfully localized

Cross-validated performance is nearly flat across the 49 extraction sites (mean CV AUCPR roughly 0.08–0.126), and the embedding output already obtains substantial predictive performance (0.094).

The selected layer 44 should therefore not be interpreted as a privileged "anchor-prediction layer." It is the numerical maximum of an unstable, relatively flat model-selection surface (the winning configuration's four fold scores were 0.033, 0.107, 0.131, and 0.234).

### 3.5 No one-sentence-ahead prediction was demonstrated

When the probe was asked to predict a sentence's importance from the boundary one sentence earlier, test AUCPR fell to approximately the base rate (0.032 vs prevalence 0.032), below even the observables baseline (0.046), while the lead-zero comparison on the same rows kept the usual pattern (0.077 vs 0.051).

This supports the narrow statement:

**The current linear readout provides no evidence of predicting this realized-sentence target one reasoning sentence in advance.**

It does not establish that the model has no earlier internal precursor. The negative result could also reflect limited statistical power, stochastic variation in which future sentence is sampled, label noise, or limitations of the chosen representation and linear decoder.

---

## 4. Why the result cannot yet be given a stronger interpretation

The following are not demonstrated failures. They are unresolved alternative explanations that can generate the current pattern and should be tested before making stronger claims.

### 4.1 The importance label may partly encode remaining reasoning horizon

Earlier interventions have more downstream computation in which their effects can propagate, compound, disappear, or change the final answer.

Therefore the strong relationship between position and measured importance could arise partly from the construction of the target itself:

$$
\text{early position}
\rightarrow
\text{more remaining trajectory}
\rightarrow
\text{larger opportunity for answer distributions to diverge}.
$$

If so, position is not merely a nuisance variable that the activation probe accidentally learns. It is partly built into what the importance metric measures.

The current experiment does not distinguish these possibilities.

A necessary diagnostic is to relate importance to remaining sentences, remaining tokens, distance to convergence, continuation length, and other measures of downstream horizon.

### 4.2 Layer-0 prediction may reflect the boundary token rather than a latent reasoning state

Layer 0 is the embedding of the token immediately preceding the target sentence.

The registered observable baseline includes the preceding token's log-probability but not its identity.

Consequently, layer-0 performance could reflect simple lexical or formatting cues such as punctuation, newline tokens, mathematical notation, digits, or other sentence-boundary tokens.

Because the layer curve is already strong at layer 0, a previous-token identity baseline is an important missing control.

Until that is tested, the flat layer profile should not be interpreted specifically as evidence for reasoning-phase encoding.

### 4.3 The output-only baseline may omit important semantic information

The registered observable model reduces the next-token distribution to entropy, maximum probability, margins, and top-k probability mass.

It does not retain **which tokens** are likely.

A residual state may therefore outperform these scalar summaries simply because it contains semantic information about what the model is about to say.

For example, the model could already be likely to emit language associated with planning, correction, verification, or a new calculation.

A stronger comparison would include next-token identities, expected token embeddings, richer logit features, or prefix/sentence text embeddings.

Until then, "activations add information beyond observables" should not be generalized to "internal monitoring adds information beyond output-side monitoring."

### 4.4 The target is one realized future, not necessarily a stable property of the current state

At the pre-emission boundary, the next sentence has not yet been sampled.

The current target assigns the boundary the importance of the **one sentence that happened to be generated** in the stored trajectory.

But the same prefix could potentially produce multiple different next sentences with very different importance.

The more prospective quantity is arguably:

$$
P(I(S)>\tau\mid h)
$$

or

$$
E[I(S)\mid h],
$$

averaged over possible future sentences.

If within-prefix variation is large, then even a genuinely informative state may have limited ability to predict the importance of one sampled realization.

This issue is particularly relevant to interpreting the lead-one null.

### 4.5 The convergence rule may not measure convergence of answer distributions

The primary eligibility rule marks a trajectory as converged when several consecutive forks have accuracy near either 0 or 1.

Accuracy near 1 plausibly indicates agreement on the correct answer.

Accuracy near 0 does not necessarily imply agreement. All continuations could be wrong while disagreeing strongly with one another.

Therefore the rule can potentially remove later portions of incorrect trajectories even when their answer distributions remain diverse.

The existing K=4 versus K=5 sensitivity changes the number of consecutive forks but does not test this conceptual issue.

The eligibility set should be recomputed using modal-answer concentration, such as:

$$
\max_a P(A=a)\ge0.98,
$$

and the downstream results compared.

### 4.6 `<NO_ANSWER>` may contribute disproportionately to measured importance

The canonical KL includes `<NO_ANSWER>` as an answer category.

If early counterfactual forks are more likely to truncate, fail parsing, or otherwise produce no answer, variation in `<NO_ANSWER>` could contribute substantially to high importance scores.

The KL contribution of `<NO_ANSWER>`, truncation rate, and parse-failure rate should therefore be examined as functions of position and importance.

### 4.7 The binary anchor label may be unstable

The primary canonical KL uses smoothing \(\alpha=10^{-9}\), while the specification also defines \(\alpha=0.5\) and \(1.0\) sensitivities.

With finite resampling, very small smoothing can make categories observed on one side but absent on the other contribute strongly to KL.

The stability of:

* importance ranks;
* q90 anchor membership;
* probe performance

across smoothing choices has not yet been established in the current interpretation.

### 4.8 Eligibility itself may select a non-representative subset of reasoning steps

Of 6,474 sentences, only 4,019 were ultimately label-eligible.

In particular, 608 sentences were excluded because enough semantically dissimilar counterfactual replacements were unavailable.

Difficulty generating a genuinely different continuation may itself reflect something about the stability or resilience of the underlying reasoning step.

The eligible dataset should therefore not automatically be interpreted as representative of all reasoning sentences.

---

## 5. Statistical limitations

The effective number of independent held-out units is four problems, not 689 sentences.

The 2,000 bootstrap repetitions and 100 permutations are useful diagnostics, but they cannot compensate for having only four independent test clusters.

The current effect estimate is highly sensitive to individual test problems, and the bootstrap interval is correspondingly crude.

The test-set prevalence is also substantially below the nominal training quantile at every examined threshold:

* q80 training threshold → approximately 9.4% positive on test;
* q90 → 3.6%;
* q95 → 1.45%.

This suggests a shift in the distribution of the importance target across problems and makes binary classification particularly sparse.

For these reasons, the study should avoid strong claims either that an effect exists or that a practically relevant effect has been ruled out.

---

## 6. Validity checks that did succeed

Several important implementation failures were directly tested and were not found.

* Problem identities were separated between train, validation, and test.
* Preprocessing and residualization were fitted within training folds.
* Selection was frozen before the test set was evaluated (single test attempt, receipt hash matches the freeze hash).
* Calibration used validation only.
* Future-token mutation left the stored pre-emission boundary activations unchanged (bitwise identical at every layer and boundary tested).
* Forbidden target/future fields were absent from the primary feature set.
* Primary metrics were independently recomputed from saved predictions and matched the generated report.

These checks make the empirical observations worth taking seriously.

They do not resolve the construct and confounding questions above.

The label audit also failed its planned agreement threshold: released and canonical importance values correlated at only Spearman 0.858. The prespecified fallback to canonical labels was correctly triggered, but the discrepancy reinforces the need to characterize the target's reliability directly.

The planned fresh Problem 4682 counterfactual-generation pilot could not be located, so the MVP should be described as missing that intended validation component unless the artifact is subsequently recovered.

One numerical discrepancy was investigated and resolved as benign: the residualized model's raw-score and probability columns give AUCPR 0.1267 vs 0.1333. The columns are monotonically identical except that sigmoid saturation maps the two top-ranked positives to exactly probability 1.0; scikit-learn's tie-grouped AP computation then scores the tied pair more favorably by exactly (1/25)·(2/3 − 1/2) = 0.00667, the entire gap. The tie-free value 0.1267 is the correct one.

---

## 7. Current interpretation

The most defensible interpretation is:

**A linear probe of the model state immediately before a reasoning sentence shows an above-baseline association with that sentence's later Thought-Anchors-style counterfactual importance. However, the experiment does not establish that this association is specific to future importance or represents internal anticipation. Its incremental advantage over registered observables is uncertain, simple positional information is highly competitive, the signal is already present at the token-embedding boundary, and no one-sentence-ahead prediction was demonstrated.**

**Several unresolved alternatives could explain these observations, including remaining reasoning horizon, boundary-token identity, richer next-token semantics, stochastic variation in the next generated sentence, and properties of the counterfactual-importance estimator itself.**

The current MVP should therefore be interpreted as identifying the next discriminating experiments rather than as settling whether high-leverage reasoning transitions are internally predictable.

---

## 8. Highest-priority additional analyses

Before substantially extending the probe or attempting mechanistic interpretation, the existing artifacts should be used to test the strongest alternative explanations.

### Priority 1: Audit the target

1. Recompute convergence using modal-answer concentration rather than accuracy.
2. Plot importance against remaining reasoning horizon.
3. Decompose KL contributions, especially `<NO_ANSWER>`.
4. Check importance and q90 membership across KL smoothing choices.
5. Quantify how eligibility differs across position, trajectory type, and other observables.

### Priority 2: Strengthen the nuisance baselines

1. Previous-token identity and token-class baseline.
2. Previous few tokens.
3. Rich next-token/logit baseline.
4. Flexible position controls rather than linear position alone.
5. Test whether later-layer activations add information beyond layer 0.

### Priority 3: Evaluate the actual local-ranking task

Report the already specified per-trajectory:

* Precision@top-10%;
* NDCG@top-10%;
* within-trajectory correlation between predicted and measured importance.

This distinguishes local identification of important transitions from learning between-problem or between-trajectory differences.

### Priority 4: Measure target reliability

For repeated samples from the same prefix, estimate how much subsequent sentence importance varies across possible generated next sentences.

Separately estimate split-half reliability of the counterfactual importance score itself.

### Priority 5: Replicate on genuinely new problems

Only after the preceding issues are understood should the frozen hypothesis be tested on substantially more independent problems and naturally sampled trajectories.

---

## 9. What should and should not be claimed

### Supported

* We implemented a leakage-controlled pre-emission prediction experiment.
* The boundary representation shows an above-prevalence raw association with later Thought-Anchors-style importance in this dataset.
* Incremental performance beyond the registered observable baseline was not established.
* Position is a strong competing predictor.
* The selected layer is not meaningfully localized by the present data.
* The current linear decoder shows no one-sentence-ahead predictive evidence.
* Several unresolved construct and nuisance explanations remain.

### Not supported

* The model "knows" which future thoughts will matter.
* A representation of future causal importance has been identified.
* Position has been shown to explain the activation result.
* The study demonstrates that pre-emission activation monitoring works.
* The study demonstrates that pre-emission activation monitoring cannot work.
* The experiment rules out a useful smaller precursor.
* The current label is equivalent to persistent causal leverage.
* The one-sentence lead result proves that the phenomenon only emerges locally.

---

## 10. One-sentence summary

**This MVP found a weak raw relationship between a model's state immediately before a reasoning sentence and that sentence's later counterfactual answer sensitivity, but the experiment does not yet distinguish an importance-specific internal precursor from simpler properties of the trajectory, local token predictions, stochastic continuation, or the target construction itself.**

---

*Sources: Stage 8 run `20260915T193326Z-57bcb41c84-7ebd0db6` (`cv_results.parquet`, `probe_validation.json`); Stage 9 run `20260915T201107Z-484989d2fe-19f1ab0d` (`metrics/test.json`, `metrics/validation.json`, `frozen_selection.json`, `test_receipt.json`, `analysis_registry.json`, `predictions/test.parquet`, resolved config); Stage 10 run `20260915T211903Z-b230473011-815ca2b7` (`sensitivities.parquet`, `permutations.parquet`, `stages/stage10/summary.json`); Stage 11 report `20260916T000045Z-stage11-report`; `data/processed/stage4/labels_summary.json`; `stage7/feature_audit.json`; Stage 6 `invariance_report.json`; amendment `20260915T170000Z.md`. All headline numbers were independently recomputed from the prediction and permutation tables.*
