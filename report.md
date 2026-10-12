Please conduct a rigorous end-to-end audit of this project as if you were reviewing it for a technical research write-up.

The core research question is whether model-internal activations available before the next reasoning sentence is generated contain useful information for predicting important upcoming reasoning events, especially thought anchors and reasoning-stage transitions.

I want you to inspect the actual code, saved datasets, experiment outputs, analysis files, plots, and reports. Do not rely on the existing write-up's interpretation where the underlying artifacts can be checked directly.

Your goal is to determine:

1. Whether the experimental design actually answers the stated research questions.
2. Whether the implementation faithfully implements the intended design.
3. Whether the reported numerical results are correct and reproducible from the saved artifacts.
4. Whether the baselines and controls are strong enough to distinguish model-internal predictive information from simpler explanations.
5. What conclusions the current evidence supports, what it does not support, and which additional experiments would most improve the evidence.

Please investigate the following areas in detail.

### 1. Reconstruct the experiment from first principles

Explain, based on the implementation:

* What the unit of prediction is.
* Exactly what information is available to each predictor at prediction time.
* What the target labels are and how they are constructed.
* What counts as a thought anchor.
* What counts as a reasoning-stage transition.
* How reasoning stages are assigned.
* Where activations are extracted relative to the sentence being predicted.
* What layer or activation site is used in each experiment and why.
* How train, validation, and test examples are split.
* Whether examples from the same problem, trajectory, resampling family, or closely related source can appear across different splits.
* What preprocessing, normalization, feature construction, dimensionality reduction, model fitting, and hyperparameter selection occur.

Produce a concise schematic of the full data flow from raw reasoning trace to final metric.

### 2. Check for information leakage

Actively search for any way information about the target could leak into the predictor.

In particular, check whether any feature uses:

* tokens from the sentence being predicted;
* future sentences;
* labels or statistics derived using held-out data;
* trace length that would not actually be known at prediction time;
* normalization fitted on validation or test data;
* thresholds, bins, feature selection, layers, or hyperparameters chosen using held-out results;
* examples from the same underlying problem or resampling tree appearing across train and test;
* metadata that acts as a proxy for the target because of the data-generation procedure.

Distinguish genuine leakage from information that would legitimately be available before generation.

### 3. Audit every experimental arm

For each predictor or baseline, determine exactly what information it receives and what hypothesis it tests.

At minimum inspect:

* prevalence / constant baseline;
* absolute or relative position predictors;
* token-count and context-window-position features;
* prefix/context features;
* next-token probability or uncertainty features;
* reasoning-stage features;
* activation-only predictors;
* activation plus observable-feature predictors;
* phase-aware or residualized thought-anchor predictors;
* layer-0 versus layer-44 reasoning-stage comparisons;
* any other saved experimental arms.

For each arm, state:

* input features;
* target;
* model;
* fitting procedure;
* evaluation metric;
* scientific purpose;
* whether it is a fair comparison.

Flag baselines that are unexpectedly weak, misspecified, poorly calibrated, or incapable of exploiting the information they are supposed to represent.

### 4. Verify the reported results directly

Recompute, wherever possible, the headline quantities from the saved predictions and labels rather than trusting summary files.

Check at minimum:

* class prevalence;
* sample counts;
* train/validation/test counts;
* AP;
* AUROC;
* accuracy;
* macro-F1;
* majority-class baselines;
* bootstrap confidence intervals;
* differences between predictors;
* any reported position concentration statistics;
* reasoning-stage distributions;
* anchor rates by reasoning stage;
* layer comparisons;
* phase-transition prediction results.

Confirm whether every important number in the existing reports agrees with the underlying data.

Create a table containing:

reported value | recomputed value | match? | source file | comments

Investigate every discrepancy.

### 5. Assess whether the evaluation metrics are appropriate

Check whether the chosen metrics answer the scientific question, especially under severe class imbalance.

For thought-anchor prediction, assess whether AP, AUROC, calibration, precision-recall curves, and any thresholded metrics are being interpreted appropriately.

For reasoning-stage classification or stage-transition prediction, assess whether accuracy, macro-F1, transition AP, and majority baselines are appropriate.

Identify any metric whose current presentation could give a misleading impression of predictive performance.

### 6. Test the strongest alternative explanations

Treat every positive result as something that needs to survive simpler explanations.

For thought-anchor prediction, investigate at least:

* absolute sentence position;
* relative position within a trace;
* tokens generated so far;
* context-window usage;
* problem length;
* trajectory length;
* current reasoning stage;
* stage-transition proximity;
* lexical or formatting cues already present in the prefix;
* next-token entropy or related uncertainty measures;
* problem identity or problem type;
* trace-specific effects;
* any systematic artifact introduced by the thought-anchor labelling procedure.

For reasoning-stage prediction, investigate whether the activation classifier is detecting:

* genuine higher-level reasoning state;
* simple lexical preparation for the next few tokens;
* mathematical versus non-mathematical text;
* punctuation or formatting patterns;
* sentence-boundary effects;
* local next-token distributions;
* trivial properties of the immediately preceding text.

Where the current experiment cannot distinguish these explanations, say so explicitly.

### 7. Audit model-selection decisions

Determine which choices were made before seeing held-out results and which were made after inspecting them.

Pay particular attention to:

* why layer 44 was selected;
* whether layer 44 was selected using thought-anchor labels and then reused for reasoning-stage prediction;
* whether reasoning-stage results influenced any subsequent choice;
* feature-set selection;
* regularization;
* model family;
* bootstrap procedure;
* threshold choices;
* reported subsets of analyses.

Separate exploratory choices from confirmatory tests.

Assess whether any reported confidence intervals or significance claims ignore the uncertainty introduced by model or layer selection.

### 8. Inspect the thought-anchor labelling procedure

Examine how counterfactual resampling is used to define thought anchors.

Determine:

* the precise mathematical or algorithmic definition;
* how many resamples are used;
* how downstream outcome changes are quantified;
* how the anchor threshold is chosen;
* whether the score is noisy;
* whether labels are stable under repeated resampling;
* whether anchors are disproportionately concentrated at particular positions or stages;
* whether the labelling mechanism itself could induce correlations with position, phase, length, or other observables.

Assess how much label noise may limit the achievable prediction performance.

### 9. Inspect the reasoning-stage taxonomy

Examine the implementation that assigns stages such as orient, plan, compute, verify, reflect, conclude, or other categories used in the project.

Determine:

* exact rules and their ordering;
* how ambiguous sentences are handled;
* class frequencies;
* whether the taxonomy is mutually exclusive and exhaustive;
* likely false positives and false negatives;
* whether labels are mostly determined by obvious lexical markers;
* whether predicting these labels from pre-emission activations is scientifically informative beyond predicting upcoming wording.

Sample actual labelled sentences from each stage and manually inspect whether the assignments appear valid.

### 10. Evaluate the central scientific claims

For each major claim in the current write-up, classify it as:

* strongly supported;
* provisionally supported;
* suggestive only;
* unsupported;
* contradicted by the evidence.

For each classification, explain the exact evidence and the main remaining alternative explanation.

In particular evaluate claims resembling:

* model activations predict upcoming thought anchors;
* activations add information beyond position or other observable baselines;
* thought-anchor predictability is not explained solely by reasoning phase;
* model activations predict upcoming reasoning-stage transitions;
* reasoning phase is encoded before the next sentence is emitted;
* the signal reflects meaningful preparation for a reasoning transition rather than merely short-range lexical preparation.

Do not strengthen the claims beyond what the experiments identify.

### 11. Identify the highest-value missing experiments

After inspecting the actual evidence, propose the smallest set of additional experiments that would most change our confidence in the conclusions.

Prioritize experiments that discriminate between competing explanations.

For every proposed experiment, state:

* question being tested;
* competing hypotheses;
* intervention or comparison;
* expected result under each hypothesis;
* required data or code changes;
* approximate computational cost relative to existing experiments;
* whether it is essential before writing the current results up or a useful follow-up.

Prefer cheap, decisive experiments over broad exploratory sweeps.

Examples worth considering, only if justified by the existing implementation, include:

* controlling for local next-token distributions;
* predicting labels after removing obvious lexical cues;
* testing several activation layers without selecting on the final test set;
* evaluating how early before an event the signal becomes detectable;
* within-stage and within-position matched comparisons;
* problem-held-out or trajectory-family-held-out splits;
* permutation or negative-control labels;
* cross-problem or cross-domain generalization;
* predicting transitions several sentences ahead;
* comparing activations with much stronger prefix-based models;
* testing whether activation information remains after conditioning on all observable features.

### 12. Produce a research-readiness assessment

End with a concise assessment structured around:

**A. What is already solid**
Results or methodological components that appear trustworthy.

**B. What is promising but exploratory**
Findings worth reporting with strong caveats.

**C. What is currently weak**
Implementation, statistical, conceptual, or causal weaknesses.

**D. What could invalidate the main conclusions**
The most important failure modes or confounders.

**E. Minimum experiments before a serious write-up**
The smallest set of additional work you consider necessary.

**F. Stronger follow-up programme**
Experiments that would turn the exploratory result into substantially stronger evidence.

### Important working principles

Be adversarial toward the interpretation while being precise about the evidence.

Do not treat statistically positive results as automatically scientifically meaningful.

Do not equate decodability with faithful representation, causal use, or mechanistic understanding.

Distinguish clearly between:

* prediction;
* representation;
* causal involvement;
* mechanistic explanation.

Prefer inspecting code and raw artifacts over trusting prose summaries.

Whenever possible, quote exact filenames, functions, line locations, experiment IDs, or saved result objects so that every conclusion can be traced back to the implementation.

If a claim cannot be verified from the repository, explicitly identify what information is missing.

Do not rewrite the research report yet. The purpose of this task is to establish what the implementation and evidence actually justify before improving the presentation.

---

⏺ All verification is complete. Let me close out the tasks and write the audit.

⏺ End-to-end audit: pre-emission thought-anchor prediction

  TL;DR. I read the full pipeline code, re-derived the design from the implementation, and recomputed every headline number directly from the saved parquet/npy/json
  artifacts (including refitting the reasoning-stage probes from raw activations). Every reported number reproduces exactly — I found zero numerical discrepancies in
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation
  
  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Reasoning stages (exploratory follow-up): 7 classes assigned by an ordered regex rule set over the sentence's own text
  (research/reasoning_stage/label_stages.py:32-73; priority conclude > backtrack(unless verify) > verify > plan > orient > compute > other). Stage transition =
  stage(S_i) ≠ stage(S_{i−1}) (analyses.py:47-49).

  Eligibility. Char-exact alignment; not the first token; not the final-answer sentence; adjacent fork support; strictly before convergence (first of 5 consecutive
  forks with accuracy in [0,.01] or [.99,1], labels/convergence.py); finite non-negative importance; ≥1 dissimilar replacement. Exclusions: convergence 1,807;
  no-dissimilar 608; missing fork 21; final answer 19.

  Splits. Fixed problem-level 12/4/4 (both trajectories of a problem stay together), asserted disjoint (probes/folds.py:14-17); eligible rows 2,650/680/689; anchors
  265/54/25. Hyperparameter/layer selection: 4-fold StratifiedGroupKFold grouped by problem_id on the 12 train problems only. No sentence, trajectory, or 
  resampling-family crosses splits — but all 20 problems come from the same public dataset the Thought Anchors paper analyzed, which the spec itself flags by naming
  the test an "exploratory held-out engineering test" (SPEC §3.4).

  Preprocessing/fitting. StandardScaler inside each fold-local sklearn pipeline; L2 logistic (class_weight="balanced"), C grid 1e-4…100; residualizer removes
  fold-fitted linear observable projections from activations (probes/models.py:56-73); selection = argmax mean train-CV AP for activation_plus_observables (layer 44,
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Non-leaks that still need honest framing:
  1. Stage features in the anchor ladder are oracle features. stg_* dummies come from S_i's own text (analyses.py:44,187-190) — future information at prediction time.
  As mediation controls this is the right direction (oracle stage is a stronger mediator than decoded stage), and the write-up uses them that way; but "stage-only
  predicts anchors at chance" must not be read as a pre-emission baseline result.
  2. Lead-1 arms carry lead-0 position features. act44_lead1_position, oracle_prevstage_position, and position_lead1_rows all use the target row's position block
  (token counts through S_{i−1}), not the lead-1 boundary's (analyses.py:262-277). At the lead-1 boundary those counts are future information. Empirically immaterial
  (position adds ~0.0001 AP) but the arms are mislabeled. The MVP's Stage 10 lead-1 did this correctly (stage10.py:400-454 uses S_{i−1}'s own observables).
  3. Stable-sort tie-breaking makes "prevalence" a hidden position baseline in the per-trajectory ranking metrics. per_trajectory_ranking sorts with kind="stable"
  (evaluation.py:63) over rows ordered by (trajectory, sentence_idx), so a constant score "selects" the earliest 10% of sentences — which is why prevalence shows P@10%
  = 0.135–0.172, far above chance. Any use of these metrics must fix tie handling.
  4. Teacher-forcing equivalence (activations from re-running stored traces, not from live generation) is fine by construction for this model/precision and was
  spot-checked by Stage 6's logit-identity test — worth one sentence in a write-up, no action needed.

  No genuine leakage found. The forbidden-feature audit, threshold provenance, fold structure, and the freeze/receipt chain all held up under direct inspection.

  ---
  3. Arm-by-arm audit
  
  ┌────────────────────────┬───────────────────────┬───────────────────────────────┬────────────────┬─────────────────┬──────────────────────────────────────────┐
  │          Arm           │        Inputs         │            Target             │  Model / fit   │     Purpose     │                  Fair?                   │
  ├────────────────────────┼──────────────────────┼───────────────────────────────┼──────────────────┼─────────────────┼─────────────────────────────────────────┤
  │ Prevalence             │ none                 │ anchor                        │ constant train   │ AP floor        │ ✓ (but its P@10% is order-contaminated, │
  │                        │                      │                               │ rate             │                 │  §2.3)                                  │
  ├────────────────────────┼──────────────────────┼───────────────────────────────┼──────────────────┼─────────────────┼─────────────────────────────────────────┤
  │                        │ sentence_idx,        │                               │                  │ positional      │ Under-specified: linear-in-position     │
  │ Position               │ generated tokens,    │ anchor                        │ logistic, CV-C   │ confound        │ only; no splines/quantile bins. Still   │
  │ Position              │ generated tokens,   │ anchor                            │ logistic, CV-C   │ confound        │ only; no splines/quantile bins. Still │
  │                       │ context fraction    │                                   │                  │                   │ Still the best test predictor.       │
  ├───────────────────────┼─────────────────────┼───────────────────────────────────┼──────────────────┼───────────────────┼──────────────────────────────────────┤
  Model / fit: logistic, CV-C
  Purpose: positional confound
  Fair?: Under-specified: linear-in-position only; no splines/quantile bins. Still the best test predictor.            
  ────────────────────────────────────────
  Arm: Context / Uncertainty                  
  Inputs: prefix lengths / 6 next-token scalars 
  Target: anchor         
  Model / fit: logistic
  Purpose: cheap observables
  Fair?: ✓ but the uncertainty arm keeps only scalar summaries — no token identities (their own §4.3) 
  ────────────────────────────────────────
  Arm: Observable combined 
  Inputs: all 12         
  Target: anchor         
  Model / fit: logistic
  Purpose: registered baseline
  Fair?: ✓; note test AUROC 0.487 (below chance) — it generalizes poorly across problems, making it a weak bar         
  ────────────────────────────────────────
  Arm: Activation L0…L48 (±observables)
  Inputs: one residual site
  Target: anchor         
  Model / fit: logistic per (layer, C)
  Purpose: main hypothesis + layer curve      
  Fair?: ✓
  ────────────────────────────────────────    
  Arm: Residualized activation
  Inputs: site 44 − fold-fitted linear obs projection                            
  Target: anchor
  Model / fit: logistic  
  Purpose: "beyond observables" control
  Fair?: ✓ but only linear removal of 12 scalars                                 
  ────────────────────────────────────────
  Arm: Ridge continuous  
  Inputs: site 28 + obs
  Target: log1p importance                    
  Model / fit: Ridge α=1e4 
  Purpose: magnitude endpoint
  Fair?: ✓ — test Spearman 0.237, never discussed in either write-up             
  ────────────────────────────────────────
  Arm: Lead-1 (MVP)
  Inputs: boundary before S_{i−1} (correct features)
  Target: anchor of S_i
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Weak/misspecified baselines worth flagging: (a) linear position only; (b) no previous-token identity / previous-K-token lexical baseline anywhere, despite layer 0 ≈
  boundary-token embedding scoring 0.094–0.098 CV AP; (c) uncertainty summaries discard token identity, so "activations beat observables" partially measures
  "vocabulary beats scalars"; (d) obs12's below-chance test AUROC means the "registered baseline" is a low bar on this test set.

  ---
  4. Direct verification of reported results
  
  Method: recomputed from predictions/*.parquet, cv_results.parquet, sensitivities.parquet, permutations.parquet, features.parquet, labels.parquet, and full refits
  from research/reasoning_stage/outputs/{dataset.parquet, activations_l44.npy}. Abbreviated table (every row I checked; "src" = file recomputed from):

  ┌────────────────────────────────────────────┬───────────────────────────────┬─────────────────────────────────────────┬───────┬────────────────────────────────┐
  │                  Quantity                  │           Reported            │               Recomputed                │ Match │             Source             │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ Eligible train/val/test                    │ 2650/680/689                  │ 2650/680/689                            │ ✓     │ stage7/features.parquet        │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ Anchors train/val/test                     │ 265/54/25                     │ 265/54/25                               │ ✓     │ ditto                          │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ Test prevalence                            │ 0.036                         │ 0.036284                                │ ✓     │ ditto                          │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ q90 threshold                              │ 1.35498                       │ 1.3549810893213545                      │ ✓     │ recomputed from train eligible │
  │                                            │                               │                                         │       │  importance                    │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ Test AP:                                   │ .036/.057/.093/.094/.127/.139 │ .0363/.0573/.0925/.0941/.1267/.1394     │ ✓     │ predictions/test.parquet       │
  │ prevalence/obs/act/act+obs/resid/position  │                               │                                         │       │                                │
  ├────────────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────┼───────┼────────────────────────────────┤
  │ ΔAP act+obs − obs                          │ +0.0368 (−0.051, +0.081)      │ +0.03680; CI (−0.0505, +0.0814)         │ ✓     │ test.parquet + test.json       │
  ├────────────────────────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────┼───────────────────────────────┤
  │ Per-problem Δ                              │ +.133/+.077/+.009/−.066          │ 2236:+.1334, 3916:+.0765, 2870:+.0086, │ ✓     │ test.parquet                 │
  │                                            │                                  │  6998:−.0664                           │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ act+obs − position                         │ −0.045 (−0.236, +0.026)          │ −0.0452 (−0.2364, +0.0255)             │ ✓     │ test.json                    │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ 15/25 test anchors in Q1                   │ 15                               │ 15 (Q1–Q4: 15/5/3/2)                   │ ✓     │ features + train quintile    │
  │                                            │                                  │                                        │       │ cuts                         │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │                                            │                                  │                                        │       │ cv_results.parquet; *0.094   │
  │ Layer curve                                │ ~0.08–0.126, flat; L0 "0.094"    │ 0.0853–0.1264, argmax 44; L0 best-C    │ ✓*    │ is L0 at the frozen C,       │
  │                                            │                                  │ 0.0977, L0 at frozen C=0.1: 0.0936     │       │ best-C L0 is 0.098 —         │
  │                                            │                                  │                                        │       │ cosmetic                     │
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  5. Are the metrics appropriate?

  - AP as primary under 3.6% prevalence: appropriate and honestly compared to prevalence. But with 25 positives, AP is dominated by the ranks of a handful of points
  (the residualized 0.1267→0.1333 sigmoid-tie episode — correctly resolved in the write-up — is a symptom). Per-problem APs (+0.133…−0.066) confirm the effective n is
  4, which the write-up states.
  - AUROC: secondary, fine; note obs12 test AUROC 0.487 — the baseline actively mis-generalizes, flattering ΔAP.
  - Calibration (Brier/log-loss/ECE): computed correctly, validation-fitted Platt; not over-claimed anywhere.
  - P@top-10% / NDCG@top-10%: implemented, in metrics/*.json, but never discussed — and currently misleading if quoted, because stable-sort tie-breaking turns the
  constant baseline into an early-position selector (§2.3). Fix ties (random or mid-rank) before using. Substantively: on test, activations do not beat position or
  even the tie-broken constant on within-trajectory ranking (act+obs P@10% 0.135 vs position/prevalence 0.135), which undercuts any "locally identifies important
  sentences" reading.
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  ---
  7. Model-selection audit

  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Implication: the ceiling on any predictor of this label is unknown, and the label's stable component may be substantially "position/horizon + answer-parseability",
  which position-like features capture — consistent with position being the best test predictor.

  ---
  9. The reasoning-stage taxonomy
  
  Rules and ordering: verified (label_stages.py; conclude ≻ backtrack-unless-verify ≻ verify ≻ plan ≻ orient ≻ compute ≻ other). Frequencies: compute 3,839 (59%),
  backtrack 830, other 655, plan 578, verify 427, orient 75, conclude 70. Mutually exclusive and exhaustive by construction; "ambiguity" resolved by fixed priority,
  with marker booleans stored for overlap inspection (good practice).

  Manual inspection of stage_label_audit.md (30/stage): broadly valid for backtrack, plan, verify; systematically loose for orient, conclude, compute. Examples:
  "orient" fires mid-trace on any "the problem says/asks" reference (many samples are re-reads deep in reasoning, not initial orientation); "conclude" fires on
  intermediate "the answer is 1280"-type statements (test split has conclude n=1, so the class is nearly empty off-train); "compute" is a default that fires on any
  digit, capturing sentences like "But that seems too large…" that are functionally verification; "verify" absorbs "Wait, let me double-check…" mixtures by priority.
  The write-up's own characterization ("robust in broad strokes, not fine distinctions") is fair.

  Scientific informativeness: because 56% of assignments trigger within the first 10 characters (93% for backtrack), predicting the label pre-emission is close to
  predicting the next few tokens — something trivially true of a language model's pre-emission state. Decoding stage from activations is therefore only interesting
  relative to (a) a full next-token-distribution baseline and (b) function-based labels not tied to openers. Neither exists yet; claim B should be framed accordingly.

  ---
  10. Central claims, classified
  
  From result_interpretation.md:

  ┌─────────────────────────────────────────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Claim                    │       Verdict       │                                           Basis                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Leakage-controlled pre-emission experiment  │ Strongly supported  │ Code audit + artifact chain (§2, §7)                                                       │
  │ implemented                                 │                     │                                                                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Above-prevalence raw association (test AP   │ Provisionally       │ Numbers verified; but "association with this α=1e-9, NO_ANSWER-inclusive label" — §8       │
  │ 0.094 vs 0.036)                             │ supported           │ instability caps its meaning; 4 test problems                                              │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Incremental value beyond observables not    │ Strongly supported  │ ΔAP +0.037, CI (−0.051,+0.081); permutation p≈0.17 — verified                              │
  │ established                                 │                     │                                                                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Position is a strong competing predictor    │ Strongly supported  │ 0.139 vs 0.094 verified; and §6 shows position is partly in the label                      │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Selected layer not meaningfully localized   │ Strongly supported  │ Curve 0.085–0.126, L0 ≈ 0.094–0.098 — verified                                             │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ No one-sentence-ahead prediction            │ Supported as stated │ Lead-1 0.032 ≈ prevalence 0.032; power/label-noise caveats already in text                 │
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Throughout: every result is decodability by an external linear probe. Nothing here shows the model represents importance/phase in a format it uses, causally relies
  on it, or any mechanism — and to their credit, neither write-up claims otherwise, but a public write-up should state the prediction/representation/causation
  distinction explicitly.

  ---
  11. Highest-value missing experiments
  
  Ordered by (discriminating power ÷ cost). The first four are CPU-only against already-saved artifacts.
  either write-up. The MVP's engineering (split hygiene, leakage controls, selection freeze, single hash-locked test evaluation) is genuinely excellent, and
  result_interpretation.md is already appropriately hedged. The real problems are upstream of the statistics, and I ran four new diagnostics that convert several
  "unresolved alternatives" into demonstrated facts:

  1. The anchor label is unstable under its own smoothing parameter. Across α ∈ {1e-9, 0.5, 1.0}, importance ranks correlate at Spearman only 0.70/0.60, and the q90
  anchor sets overlap at Jaccard 0.19 and 0.13. The binary target is largely an artifact of α=1e-9.
  2. <NO_ANSWER> drives the label. Spearman(importance, NO_ANSWER-fraction of resamples) = 0.45; anchor rate rises from 3.6% (no NO_ANSWER) to 47% (NO_ANSWER-fraction
  > 0.2).
  3. The label mechanically encodes remaining horizon. Spearman(importance, remaining eligible sentences) = +0.30; anchor rate falls monotonically 12.4% → 3.6% across
  relative-position quintiles.
  4. The one "strong positive result" (transition prediction) shrinks ~4× against the correct baseline. The reported gain of activations over position (+0.40 val /
  +0.22 test AP) falls to +0.043 (CI +0.012…+0.077) val / +0.056 (CI +0.024…+0.092) test against a previous-sentence-stage + position baseline — a feature trivially
  computable from the visible prefix with the same regexes that define the labels. The result survives (CI excludes zero on both splits) but at a fraction of the
  advertised magnitude, and the residual is plausibly next-token surface anticipation: 92.8% of "backtrack" sentences literally open with "Wait"-type markers.

  Details below, organized by your twelve sections.

  ---
  1. Experiment reconstruction from the implementation

  Unit of prediction. One eligible sentence S_i inside one of 40 stored DeepSeek-R1-Distill-Qwen-14B trajectories (20 MATH problems × {correct, incorrect} released
  traces from uzaymacar/math-rollouts, pinned revision). 6,474 sentence rows total; 4,019 eligible (labels_summary.json).

  Information at prediction time (lead 0). The residual stream at the boundary token — the last token strictly before S_i's first token (model/boundaries.py:41-52:
  target_token_start = first token whose span end > char_start, boundary_token_idx = target_token_start − 1), extracted by one causal pass over the exact stored trace,
  49 sites (layer=0 embedding output, 1..48 decoder-block outputs pre-final-RMSNorm), FP16, shape [n, 49, 5120]. Plus 12 observables (features/observables.py:22-27):
  sentence index, generated/prefix token counts (+log1p, +context fraction), previous-sentence token length, previous-token log-prob, and 5 next-token-distribution
  scalars (entropy, max-p, margin, top-5/top-10 mass) computed by applying the final RMSNorm + lm_head to the stored layer-48 boundary state. Excluded quantities
  (relative completed position, first-target-token log-prob) exist only as diagnostic__* columns and are structurally barred from model inputs (check_feature_names,
  dataset.py:34-35).

  Target. importance_canonical: smoothed KL between the answer distribution of ~100 resampled continuations without S_i (fork-i samples whose replacement is
  MiniLM-cosine < 0.8 dissimilar; mean 52.7 dissimilar) and ~100 with S_i (all fork-i+1 samples), over the union of normalized answer categories including <NO_ANSWER>,
  α=1e-9 (labels/importance.py:35-97). Anchor = importance ≥ q90 of eligible train importance = 1.3549810893 (labels/threshold.py, fit.py:52 THRESHOLD_QUANTILE=0.90).
  The released labels failed their prespecified gate (Spearman 0.858 < 0.95 vs canonical), triggering the prespecified canonical fallback. Note for my_notes.md: the
  importance score is about the final-answer distribution (KL between answer-category distributions), not trajectory divergence; anchors are by construction the top
  training decile; labels come from finite resampling with no per-sentence uncertainty interval computed anywhere — only the global agreement Spearman and the
  (unanalyzed until now) α-sensitivity columns.

  Not worth doing yet: more layers, nonlinear probes, more permutations, cross-model transfer — all are downstream of the label-validity and lexical-baseline
  questions.

  ---
  12. Research-readiness assessment
  forks with accuracy in [0,.01] or [.99,1], labels/convergence.py); finite non-negative importance; ≥1 dissimilar replacement. Exclusions: convergence 1,807;
  no-dissimilar 608; missing fork 21; final answer 19.
  
  Splits. Fixed problem-level 12/4/4 (both trajectories of a problem stay together), asserted disjoint (probes/folds.py:14-17); eligible rows 2,650/680/689; anchors
  265/54/25. Hyperparameter/layer selection: 4-fold StratifiedGroupKFold grouped by problem_id on the 12 train problems only. No sentence, trajectory, or 
  resampling-family crosses splits — but all 20 problems come from the same public dataset the Thought Anchors paper analyzed, which the spec itself flags by naming
  the test an "exploratory held-out engineering test" (SPEC §3.4).
  
  Preprocessing/fitting. StandardScaler inside each fold-local sklearn pipeline; L2 logistic (class_weight="balanced"), C grid 1e-4…100; residualizer removes
  fold-fitted linear observable projections from activations (probes/models.py:56-73); selection = argmax mean train-CV AP for activation_plus_observables (layer 44,
  C=0.1, CV AP 0.1264); refit on all train; Platt calibration on validation only; freeze → single hash-locked test evaluation (probes/stage9.py, receipt attempt count
  = 1, receipt hash == freeze hash e68f1d82…, verified).
  
  Data-flow schematic.
  
  HF math-rollouts (pinned)                      DeepSeek-R1-14B (pinned)
    │ Stage 2/3: fetch, segment, char-align            │
    ▼                                                  │
  sentences.parquet (6,474 rows) ──────────────┐       │
    │ Stage 4: released resamples (~100/fork)  │       │ Stage 6/7: one causal pass per trace,
    │  similarity filter (<0.8) → canonical KL │       │  hook 49 sites at boundary tokens only
    ▼                                          │       ▼
  labels.parquet (importance, anchor=trainQ90) │  activations [6474 × 49 × 5120] fp16
    │                                          │       │ + RMSNorm∘lm_head on layer-48 states 
    └────────────── join by sentence_id ───────┴──► features.parquet (12 observables + targets)
                                                       │ Stage 8: 4-fold problem-grouped CV on 12 train problems
                                                       │  → select (layer 44, C=0.1) by mean CV AP; residualized control
                                                       │ Stage 9: validation scoring + Platt; freeze; test ONCE (receipt)
                                                       │ Stage 10: thresholds/convergence/subgroups/quintiles/
                                                       │  lead-1/100 stratified permutations
                                                       ▼
                                metrics/{validation,test}.json, sensitivities.parquet
        [post-hoc, untracked] research/reasoning_stage/: regex stage labels → dataset.parquet
         + L0/L44 npy → analyses.py (stage decoding, anchor ladder, transition prediction)
  
  ---
  2. Information leakage
  
  Verified clean (I checked the code paths, not just the docs):
  - No token of S_i or later reaches any feature at lead 0: boundary index construction (boundaries.py:45-46) puts any straddling token in the target, and the Stage 6
  invariance test (16 boundaries, invariance_report.json, primary_pass: true, plus a logit-identity check) empirically confirmed future-token mutation leaves boundary
  states unchanged.
  - Anchor threshold, position/entropy quintile cuts, permutation strata, and all alternate thresholds are train-derived only (labels/threshold.py, stage10.py:357-364,
  624-627 re-verifies the frozen q90 equals the recomputed train quantile).
  - Scaler, residualizer, and calibrator fitting: fold-local / train-only / validation-only respectively (models.py:92-110, calibrate.py).
  - Selection basis is train-CV only; stage9.py re-verifies selection_basis == "train_cv_only" and every artifact hash before test; test evaluated once (receipt
  verified above).
  - The features table is checked against an allow-list; excluded quantities (trajectory length, relative completed position, first-target-token log-prob, correctness)
  can only exist as diagnostic__ columns and are never model inputs.

  Non-leaks that still need honest framing:
  1. Stage features in the anchor ladder are oracle features. stg_* dummies come from S_i's own text (analyses.py:44,187-190) — future information at prediction time.
  As mediation controls this is the right direction (oracle stage is a stronger mediator than decoded stage), and the write-up uses them that way; but "stage-only
  predicts anchors at chance" must not be read as a pre-emission baseline result.
  2. Lead-1 arms carry lead-0 position features. act44_lead1_position, oracle_prevstage_position, and position_lead1_rows all use the target row's position block
  (token counts through S_{i−1}), not the lead-1 boundary's (analyses.py:262-277). At the lead-1 boundary those counts are future information. Empirically immaterial
  (position adds ~0.0001 AP) but the arms are mislabeled. The MVP's Stage 10 lead-1 did this correctly (stage10.py:400-454 uses S_{i−1}'s own observables).
  3. Stable-sort tie-breaking makes "prevalence" a hidden position baseline in the per-trajectory ranking metrics. per_trajectory_ranking sorts with kind="stable"
  (evaluation.py:63) over rows ordered by (trajectory, sentence_idx), so a constant score "selects" the earliest 10% of sentences — which is why prevalence shows P@10%
  = 0.135–0.172, far above chance. Any use of these metrics must fix tie handling.
  4. Teacher-forcing equivalence (activations from re-running stored traces, not from live generation) is fine by construction for this model/precision and was
  spot-checked by Stage 6's logit-identity test — worth one sentence in a write-up, no action needed.

  No genuine leakage found. The forbidden-feature audit, threshold provenance, fold structure, and the freeze/receipt chain all held up under direct inspection.

  ---
  3. Arm-by-arm audit

  Arm: Prevalence
  Inputs: none
  Target: anchor
  Model / fit: constant train rate
  Purpose: AP floor
  Fair?: ✓ (but its P@10% is order-contaminated, §2.3)
  ────────────────────────────────────────
  Arm: Position
  Inputs: sentence_idx, generated tokens, context fraction
  Target: anchor
  Model / fit: logistic, CV-C
  Purpose: positional confound
  Fair?: Under-specified: linear-in-position only; no splines/quantile bins. Still the best test predictor.
  ────────────────────────────────────────
  Arm: Context / Uncertainty
  Inputs: prefix lengths / 6 next-token scalars
  Target: anchor
  Model / fit: logistic
  Purpose: cheap observables
  Fair?: ✓ but the uncertainty arm keeps only scalar summaries — no token identities (their own §4.3)
  ────────────────────────────────────────
  Arm: Observable combined
  Inputs: all 12
  Target: anchor
  Model / fit: logistic
  Purpose: registered baseline
  Fair?: ✓; note test AUROC 0.487 (below chance) — it generalizes poorly across problems, making it a weak bar
  ────────────────────────────────────────
  Arm: Activation L0…L48 (±observables)
  Inputs: one residual site
  Target: anchor
  Model / fit: logistic per (layer, C)
  Purpose: main hypothesis + layer curve
  Fair?: ✓
  ────────────────────────────────────────
  Arm: Residualized activation
  Inputs: site 44 − fold-fitted linear obs projection
  Target: anchor
  Model / fit: logistic
  Purpose: "beyond observables" control
  Fair?: ✓ but only linear removal of 12 scalars
  ────────────────────────────────────────
  Arm: Ridge continuous
  Inputs: site 28 + obs
  Target: log1p importance
  Model / fit: Ridge α=1e4
  Purpose: magnitude endpoint
  Fair?: ✓ — test Spearman 0.237, never discussed in either write-up
  ────────────────────────────────────────
  Arm: Lead-1 (MVP)
  Inputs: boundary before S_{i−1} (correct features)
  Target: anchor of S_i
  Model / fit: refit at frozen (layer, C)
  Purpose: earliness
  Fair?: ✓
  ────────────────────────────────────────
  Arm: Permutations
  Inputs: site 44 + obs
  Target: 100× within-(problem, trace_kind, position-quintile) shuffled anchors
  Model / fit: refit per permutation
  Purpose: structured null
  Fair?: ✓ well designed — nulls out position/problem/trace effects
  ────────────────────────────────────────
  Arm: Stage multiclass (position / obs12 / L0 / L44 / +obs)
  Inputs: pre-emission features
  Target: stage of S_i
  Model / fit: multinomial, fixed C=0.1, no tuning
  Purpose: what does the boundary encode?
  Fair?: ✓ as decodability; majority baseline reported; but see §9 — target ≈ opening tokens
  ────────────────────────────────────────
  Arm: Anchor ladder (stage arms)
  Inputs: oracle stage dummies ± position ± L44
  Target: anchor
  Model / fit: logistic CV-C
  Purpose: mediation
  Fair?: ✓ as mediation, not as baseline
  ────────────────────────────────────────
  Arm: Transition (lead 0)
  Inputs: position / obs12 / L44
  Target: stage(S_i)≠stage(S_{i−1})
  Model / fit: logistic CV-C
  Purpose: "changing-mode" signal
  Fair?: Missing the decisive baseline: prev-stage(+position) — legitimately prefix-computable — was fit only as a lead-1 oracle (analyses.py:270-283), never compared
  at lead 0. I ran it (§6).
  ────────────────────────────────────────
  Arm: Transition lead-1 + prev-stage oracle
  Inputs: L44 at i−1 boundary; oracle stage(S_{i−1})
  Target: transition
  Model / fit: logistic
  Purpose: two-step anticipation
  Fair?: ✓ — honest deflation, correctly reported as null

  Weak/misspecified baselines worth flagging: (a) linear position only; (b) no previous-token identity / previous-K-token lexical baseline anywhere, despite layer 0 ≈
  boundary-token embedding scoring 0.094–0.098 CV AP; (c) uncertainty summaries discard token identity, so "activations beat observables" partially measures
  "vocabulary beats scalars"; (d) obs12's below-chance test AUROC means the "registered baseline" is a low bar on this test set.

  ---
  4. Direct verification of reported results

  Method: recomputed from predictions/*.parquet, cv_results.parquet, sensitivities.parquet, permutations.parquet, features.parquet, labels.parquet, and full refits
  from research/reasoning_stage/outputs/{dataset.parquet, activations_l44.npy}. Abbreviated table (every row I checked; "src" = file recomputed from):

  ┌────────────────────────────────────────────┬──────────────────────────────────┬────────────────────────────────────────┬───────┬──────────────────────────────┐
  │                  Quantity                  │             Reported             │               Recomputed               │ Match │            Source            │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Eligible train/val/test                    │ 2650/680/689                     │ 2650/680/689                           │ ✓     │ stage7/features.parquet      │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Anchors train/val/test                     │ 265/54/25                        │ 265/54/25                              │ ✓     │ ditto                        │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Test prevalence                            │ 0.036                            │ 0.036284                               │ ✓     │ ditto                        │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ q90 threshold                              │ 1.35498                          │ 1.3549810893213545                     │ ✓     │ recomputed from train        │
  │                                            │                                  │                                        │       │ eligible importance          │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Test AP:                                   │ .036/.057/.093/.094/.127/.139    │ .0363/.0573/.0925/.0941/.1267/.1394    │ ✓     │ predictions/test.parquet     │
  │ prevalence/obs/act/act+obs/resid/position  │                                  │                                        │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ ΔAP act+obs − obs                          │ +0.0368 (−0.051, +0.081)         │ +0.03680; CI (−0.0505, +0.0814)        │ ✓     │ test.parquet + test.json     │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Per-problem Δ                              │ +.133/+.077/+.009/−.066          │ 2236:+.1334, 3916:+.0765, 2870:+.0086, │ ✓     │ test.parquet                 │
  │                                            │                                  │  6998:−.0664                           │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ act+obs − position                         │ −0.045 (−0.236, +0.026)          │ −0.0452 (−0.2364, +0.0255)             │ ✓     │ test.json                    │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ 15/25 test anchors in Q1                   │ 15                               │ 15 (Q1–Q4: 15/5/3/2)                   │ ✓     │ features + train quintile    │
  │                                            │                                  │                                        │       │ cuts                         │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │                                            │                                  │                                        │       │ cv_results.parquet; *0.094   │
  │ Layer curve                                │ ~0.08–0.126, flat; L0 "0.094"    │ 0.0853–0.1264, argmax 44; L0 best-C    │ ✓*    │ is L0 at the frozen C,       │
  │                                            │                                  │ 0.0977, L0 at frozen C=0.1: 0.0936     │       │ best-C L0 is 0.098 —         │
  │                                            │                                  │                                        │       │ cosmetic                     │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Winning fold scores                        │ .033/.107/.131/.234              │ .0327/.1314/.1074/.234 (fold order     │ ✓     │ cv_results.parquet           │
  │                                            │                                  │ differs)                               │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Lead-1 test: act+obs / obs / prevalence;   │ .032/.046/.032; .077 vs .051     │ .0319/.0463/.0322; .0774/.0505         │ ✓     │ sensitivities.parquet        │
  │ lead-0 same rows                           │                                  │                                        │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Permutation p (test)                       │ ≈0.17                            │ 0.1683 (n=100; validation 0.099)       │ ✓     │ permutations.parquet         │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Threshold-sensitivity test prevalence      │ 9.4%/3.6%/1.45%                  │ .0943/.0363/.0145                      │ ✓     │ sensitivities.parquet        │
  │ q80/q90/q95                                │                                  │                                        │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Released-vs-canonical Spearman             │ 0.858                            │ 0.85797 (stored); 0.891 on             │ ✓     │ labels_summary.json          │
  │                                            │                                  │ eligible-only                          │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Residualized tie artifact                  │ .1267 vs .1333                   │ .12668 vs .13334                       │ ✓     │ test.parquet                 │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Stage counts                               │ compute 3839 … conclude 70       │ identical                              │ ✓     │ stage_labels.parquet         │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Table B (all 6 rows + lead-1)              │ e.g. act44 .6838/.4571/1.3297    │ identical, incl. my independent refit  │ ✓     │ refit from npy               │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Anchor ladder (all 10 rows)                │ e.g. act44 .1711/.0935           │ identical; refit reproduces to 4 dp    │ ✓     │ analysis_results.json +      │
  │                                            │                                  │                                        │       │ refit                        │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Ladder bootstraps (5 CIs)                  │ e.g. −0.048 (−0.166, +0.003)     │ −0.0475 (−0.1656, +0.0027)             │ ✓     │ analysis_results.json        │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │                                            │ 9.7% vs 9.3%, CI (−.023, +.028); │                                        │       │ analysis_results.json (+ my  │
  │ Transition rates                           │  plan 54/375; c→p 18/100; p→c    │ identical                              │ ✓     │ crosstabs)                   │
  │                                            │ 27/153                           │                                        │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ Table E + gains + lead-1 + oracle          │ e.g. act .805/.848; +.40/+.22;   │ identical; act44/oracle refits         │ ✓     │ refit from npy               │
  │                                            │ oracle .762/.792; +.011/+.034    │ reproduce exactly                      │       │                              │
  ├────────────────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────┼───────┼──────────────────────────────┤
  │ MVP-position reproduction claim            │ AP .1394 / AUROC .7188 to 4 dp   │ confirmed equals frozen stage-9        │ ✓     │ both prediction sets         │
  │                                            │                                  │ position                               │       │                              │
  └────────────────────────────────────────────┴──────────────────────────────────┴────────────────────────────────────────┴───────┴──────────────────────────────┘

  Discrepancies: none. One presentational nit (layer-0 "0.094" is the frozen-C value, not best-C 0.098) and one process nit: research/reasoning_stage/ scripts were
  modified 3 minutes after their outputs were written (mtimes 10:42 vs 10:39 on Sep 17) — my full re-run confirms the current scripts regenerate the saved numbers
  exactly, so this is benign, but nothing under research/ or the reasoning-stage write-up is in git at all (all ?? in status), in stark contrast to the hash-chained
  MVP pipeline.

  ---
  5. Are the metrics appropriate?

  - AP as primary under 3.6% prevalence: appropriate and honestly compared to prevalence. But with 25 positives, AP is dominated by the ranks of a handful of points
  (the residualized 0.1267→0.1333 sigmoid-tie episode — correctly resolved in the write-up — is a symptom). Per-problem APs (+0.133…−0.066) confirm the effective n is
  4, which the write-up states.
  - AUROC: secondary, fine; note obs12 test AUROC 0.487 — the baseline actively mis-generalizes, flattering ΔAP.
  - Calibration (Brier/log-loss/ECE): computed correctly, validation-fitted Platt; not over-claimed anywhere.
  - P@top-10% / NDCG@top-10%: implemented, in metrics/*.json, but never discussed — and currently misleading if quoted, because stable-sort tie-breaking turns the
  constant baseline into an early-position selector (§2.3). Fix ties (random or mid-rank) before using. Substantively: on test, activations do not beat position or
  even the tie-broken constant on within-trajectory ranking (act+obs P@10% 0.135 vs position/prevalence 0.135), which undercuts any "locally identifies important
  sentences" reading.
  - Stage classification: accuracy + macro-F1 + log-loss with a majority baseline is right; the write-up correctly warns that class-balanced training deflates raw
  accuracy for weak models.
  - Transition AP with base rates 0.475/0.601: AP is a weak choice for a ~50–60%-positive class (AP floor ≈ prevalence; test AP 0.848 vs floor 0.601 looks bigger than
  it is). AUROC (0.80 test) or balanced accuracy would be more transparent. The +0.22 "gain over position" partially reflects position's near-floor AP (0.626 ≈ 0.601).
  - Permutation test: well-designed (within problem × trace-kind × position-quintile), refits per permutation, reported descriptively. Only 100 permutations → p
  resolution 0.01; fine for its use.
  - Bootstrap: problem-level, paired, 2000 reps, percentile — correct clustering; with 4 test clusters the write-up rightly calls it descriptive.

  ---
  6. Strongest alternative explanations — with new evidence

  For the anchor result (test AP 0.093–0.094 vs prevalence 0.036, null incremental CI):

  - Position/horizon: Confirmed as strong: position alone 0.139; 15/25 test anchors in Q1. New: importance correlates +0.30 with remaining eligible sentences and −0.31
  with relative position; anchor rate by relative-position quintile: 12.4/13.9/8.5/4.4/3.6%. Because forks earlier in a trace have more remaining generation in which
  distributions can diverge — and more truncation risk — part of this is built into the estimand, exactly as §4.1 hypothesized. Not distinguishable from "model
  anticipates importance" in the current design.
  - <NO_ANSWER> artifact: New: demonstrated. Spearman(importance, NO_ANSWER-fraction) = 0.45; anchor rate 3.6% → 10.2% → 31.0% → 47.3% across NO_ANSWER-fraction bins
  (0, ≤.05, ≤.2, >.2). A large share of "anchors" are sentences whose counterfactual continuations often fail to produce a parseable answer. Any lexical or positional
  feature correlated with "continuations from here tend to truncate/parse-fail" will predict the label.
  - Label instability: New: demonstrated (§8). With Jaccard 0.13–0.19 between q90 anchor sets across α choices, "predicting anchors" means predicting a largely
  α-specific set.
  - Boundary-token identity / lexical prefix cues: untested, and urgent given L0 (the boundary token's embedding) already achieves CV AP ~0.094–0.098 vs 0.126 at the
  best layer. No previous-token-identity baseline exists anywhere.
  - Stage / transition proximity: adequately ruled out as a mediator by the ladder (oracle stage adds nothing across problems, residualized probe keeps most AP) and
  the transition analysis (anchor rate 9.7% vs 9.3%). This part of the follow-up holds up.
  - Problem identity: controlled by grouped splits; the per-problem Δ spread (+0.133 to −0.066) shows how much problem idiosyncrasy remains.

  For the stage/transition results:

  - Stage ≈ upcoming surface form: New: quantified. 92.8% of backtrack sentences open with a "wait/hmm/hold on…" marker; 95.5% have their trigger within 10 characters;
  overall 55.7% of non-other sentences match their assigned stage's regex within the first 10 characters. Decoding "upcoming stage" at layer 44 is therefore
  substantially decoding "the opening token(s) of the next sentence" — knowledge the model must have pre-emission to sample the next token at all. A pure output-side
  baseline (P(next token ∈ marker set) from the full next-token distribution, which the pipeline can compute from stored layer-48 states + head) was never run; the
  scalar summaries (obs12, macro-F1 0.175) don't test this.
  - Transition prediction: New: the decisive control was missing at lead 0. Previous-sentence stage is a deterministic function of prefix text. Refitting with the
  authors' exact protocol: prev-stage+position AP 0.762 val / 0.792 test; activations 0.805/0.848; paired bootstrap gain of activations over the prefix baseline +0.043
  (+0.012…+0.077) val, +0.056 (+0.024…+0.092) test; activations+prev-stage over prev-stage: +0.064/+0.056, CIs excluding zero. So: real residual signal, both splits —
  but ~4–6× smaller than the reported "+0.40/+0.22 over position", and still uncontrolled for next-token marker probability.
  - Punctuation/formatting/math-vs-prose: untested; the compute rule fires on any digit or "=", so "compute vs non-compute" decoding could partly be "math symbols
  upcoming".

  ---
  7. Model-selection audit

  Made before held-out data (confirmatory-grade): layer 44 + C=0.1 (argmax of mean train-CV AP over 49×7 grid for the prespecified primary family,
  probes/select.py:66-95); q90 threshold; feature list; model family; bootstrap design; calibration policy; Stage 10's analysis list (prespecified in
  analysis_registry.json with confirmatory: true/false tags before the test ran). The freeze chain is real: pending selection → frozen_selection.json (SHA e68f1d82…) →
  single test attempt whose receipt records the same hash; verify_freeze re-checks code identity, config hash, split hash, and every artifact hash. I verified attempt
  count = 1 and the hash equality directly.

  Made after seeing results (exploratory): everything in research/reasoning_stage/ — the taxonomy, the ladder, the transition analyses — was designed after the MVP
  test results were known (the write-up says so). Within it, C-selection is train-CV and the protocol is uniform, but ~25 models were scored on the 4 test problems
  with no registration and no multiplicity control; the "no test-driven tuning" claim is accurate at the hyperparameter level but the analysis choices (which arms
  exist, which bootstraps are highlighted) are post hoc. The scripts and outputs are untracked in git — no provenance for this half of the project.

  Layer 44 reuse: chosen on the anchor task's train CV, then reused for stage/transition decoding. Not leakage, but the stage-task layer profile (L0 vs L44 only)
  inherits an anchor-task choice; the flat anchor curve (0.085–0.126) vs claimed steep stage dependence (macro-F1 0.14 → 0.46) makes the contrast credible, but "layer
  44" has no privileged status for either task, and the write-ups say so.

  Selection uncertainty in CIs: the test bootstrap conditions on the selected (layer, C); with a selection surface whose winning cell's fold scores span 0.033–0.234,
  the reported intervals understate total uncertainty. result_interpretation.md §3.4 effectively concedes this; a write-up should say it explicitly.

  Reasoning-stage results influencing later choices: none yet visible in code, but follow-up 4 in the stage write-up proposes feeding stage posteriors into the next
  baseline — fine, since it strengthens the null model.

  ---
  8. The thought-anchor labelling procedure

  Definition and mechanics verified in labels/importance.py (§1 above). Resampling: ~100 released rollouts per fork (sample_count median 100), mean 52.7 dissimilar
  survivors; with side uses all fork-i+1 samples. Threshold: train-only q90, interpolated quantile, frozen. Three label variants stored; released failed its gate
  (0.858 < 0.95, prespecified fallback taken — correct process).

  New findings, all computable from the saved labels.parquet:
  1. Smoothing dominates the estimand. α=1e-9 makes any category observed on one side and absent on the other contribute ~`p·log(p/α_scaled)` — enormous relative to
  distribution shifts among shared categories. Result: Spearman(α=1e-9, α=0.5) = 0.70, (α=1e-9, α=1.0) = 0.60; q90 anchor-set Jaccard 0.187 / 0.130 (with per-α train
  thresholds). The "anchor" construct is not robust to a nuisance parameter with no principled value.
  2. <NO_ANSWER> is a major importance driver (numbers in §6). Since <NO_ANSWER> conflates truncation, parse failure, and genuine non-answering, and its rate plausibly
  depends on remaining generation length, this couples the label to position/horizon mechanically.
  3. Horizon coupling demonstrated (Spearman +0.30 with remaining sentences).
  4. Selection effects: 608 sentences (9.4%) excluded because no sufficiently-dissimilar replacement existed — the "most stable" steps are systematically missing, as
  §4.8 warned. Dissimilar-count itself is uncorrelated with importance (ρ=0.009), which is mildly reassuring about the similarity filter.
  5. No repeated-resampling stability estimate exists (no split-half or bootstrap over the 100 rollouts). Given points 1–2, label noise is plausibly the binding
  constraint on achievable AP; nothing in the repo bounds it.

  Implication: the ceiling on any predictor of this label is unknown, and the label's stable component may be substantially "position/horizon + answer-parseability",
  which position-like features capture — consistent with position being the best test predictor.

  ---
  9. The reasoning-stage taxonomy

  Rules and ordering: verified (label_stages.py; conclude ≻ backtrack-unless-verify ≻ verify ≻ plan ≻ orient ≻ compute ≻ other). Frequencies: compute 3,839 (59%),
  backtrack 830, other 655, plan 578, verify 427, orient 75, conclude 70. Mutually exclusive and exhaustive by construction; "ambiguity" resolved by fixed priority,
  with marker booleans stored for overlap inspection (good practice).

  Manual inspection of stage_label_audit.md (30/stage): broadly valid for backtrack, plan, verify; systematically loose for orient, conclude, compute. Examples:
  "orient" fires mid-trace on any "the problem says/asks" reference (many samples are re-reads deep in reasoning, not initial orientation); "conclude" fires on
  intermediate "the answer is 1280"-type statements (test split has conclude n=1, so the class is nearly empty off-train); "compute" is a default that fires on any
  digit, capturing sentences like "But that seems too large…" that are functionally verification; "verify" absorbs "Wait, let me double-check…" mixtures by priority.
  The write-up's own characterization ("robust in broad strokes, not fine distinctions") is fair.

  Scientific informativeness: because 56% of assignments trigger within the first 10 characters (93% for backtrack), predicting the label pre-emission is close to
  predicting the next few tokens — something trivially true of a language model's pre-emission state. Decoding stage from activations is therefore only interesting
  relative to (a) a full next-token-distribution baseline and (b) function-based labels not tied to openers. Neither exists yet; claim B should be framed accordingly.

  ---
  10. Central claims, classified

  From result_interpretation.md:

  ┌─────────────────────────────────────────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
  │                    Claim                    │       Verdict       │                                           Basis                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Leakage-controlled pre-emission experiment  │ Strongly supported  │ Code audit + artifact chain (§2, §7)                                                       │
  │ implemented                                 │                     │                                                                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Above-prevalence raw association (test AP   │ Provisionally       │ Numbers verified; but "association with this α=1e-9, NO_ANSWER-inclusive label" — §8       │
  │ 0.094 vs 0.036)                             │ supported           │ instability caps its meaning; 4 test problems                                              │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Incremental value beyond observables not    │ Strongly supported  │ ΔAP +0.037, CI (−0.051,+0.081); permutation p≈0.17 — verified                              │
  │ established                                 │                     │                                                                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Position is a strong competing predictor    │ Strongly supported  │ 0.139 vs 0.094 verified; and §6 shows position is partly in the label                      │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Selected layer not meaningfully localized   │ Strongly supported  │ Curve 0.085–0.126, L0 ≈ 0.094–0.098 — verified                                             │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ No one-sentence-ahead prediction            │ Supported as stated │ Lead-1 0.032 ≈ prevalence 0.032; power/label-noise caveats already in text                 │
  │ demonstrated                                │                     │                                                                                            │
  ├─────────────────────────────────────────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ "Not supported" list (§9 of the write-up)   │ Accurate            │ Nothing in my audit licenses strengthening any of them                                     │
  └─────────────────────────────────────────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘

  From result_interpretation_reasoning_stage.md:

  Claim: A: Position barely predicts stage
  Verdict: Supported
  Basis: Verified; stage mix near-constant across quintiles
  ────────────────────────────────────────
  Claim: B: Activations strongly predict upcoming stage, layer-dependent, beyond position/observables
  Verdict: Provisionally supported as a decodability claim — but likely reducible to next-token surface form
  Basis: Numbers verified and independently refit; my marker analysis (93% opener-triggered backtrack) plus absence of a token-identity baseline means "knows what kind

  of sentence is coming" cannot currently be distinguished from "knows the next few tokens". The write-up flags this; the flag deserves top billing, not a caveat.
  ────────────────────────────────────────
  Claim: C: Stage does not explain the anchor signal
  Verdict: Supported, with limited force
  Basis: Oracle-stage mediation fails across problems (verified). But the "anchor signal" being explained is itself statistically null incrementally, so this rules out

  one mediator of a weak effect.
  ────────────────────────────────────────
  Claim: D: Anchors not concentrated at stage transitions
  Verdict: Supported
  Basis: 9.7% vs 9.3%, CI spans zero — verified
  ────────────────────────────────────────
  Claim: E: Stage transitions are predictable pre-emission; "clearly positive on both splits; the only such comparison in the project"
  Verdict: Overstated as presented; existence survives at ~¼ magnitude
  Basis: Vs position: +0.40/+0.22 (verified). Vs the legitimate prefix baseline (prev-stage+position) that the write-up never runs at lead 0: +0.043/+0.056, CIs
  excluding zero (my refits). Still uncontrolled for next-token marker probability, so even this residual may be short-range lexical anticipation.
  ────────────────────────────────────────
  Claim: Lead-1 transition: no convincing two-step anticipation
  Verdict: Supported
  Basis: Their own oracle deflation (+0.011/+0.034) — honest and verified
  ────────────────────────────────────────
  Claim: §5 "reasoning-phase hypothesis true as a claim about what activations encode"
  Verdict: Not yet supported at that strength
  Basis: "Phase" vs "opening tokens" undistinguished (see B)

  Throughout: every result is decodability by an external linear probe. Nothing here shows the model represents importance/phase in a format it uses, causally relies
  on it, or any mechanism — and to their credit, neither write-up claims otherwise, but a public write-up should state the prediction/representation/causation
  distinction explicitly.

  ---
  11. Highest-value missing experiments

  Ordered by (discriminating power ÷ cost). The first four are CPU-only against already-saved artifacts.

  E1. Label-robustness re-analysis (essential; hours, CPU).
  Question: is the anchor result a property of reasoning importance or of α=1e-9 + <NO_ANSWER>?
  Compare: rerun the frozen probes' scoring (and the position baseline) against anchors redefined under (i) α=0.5/1.0 (columns already in labels.parquet), (ii) KL
  excluding <NO_ANSWER> / treating it as its own outcome, (iii) modal-answer-concentration convergence (§4.5). If activation AP tracks the label variants while
  position's does not (or vice versa), that localizes what is being predicted. If everything collapses under (ii), the current result is substantially an
  answer-parseability effect. Expected under "real importance signal": rank stability; under "artifact": Jaccard-0.15-level churn in results mirroring the label churn
  I measured.

  E2. Prefix-lexical baselines (essential; hours, CPU).
  Add to the baseline set: boundary-token identity (one-hot of top-K frequent boundary tokens), previous 5–10 token IDs / hashed n-grams, previous-sentence stage,
  previous-sentence text embedding (MiniLM is already pinned in the repo). Targets: both anchors and transitions. This is the missing control for the flat layer curve
  (L0 ≈ boundary-token embedding) and would either kill or greatly sharpen claims B/E. Expected under "lexical": L44 ≈ lexical baseline; under "deeper state": L44
  margin persists.

  E3. Full next-token-distribution baseline (essential; small GPU or CPU-batch, reuses stored layer-48 states + the Head in observables.py).
  Compute P(next token ∈ marker set per stage), top-K token identities/probabilities at each boundary. Test: does stage/transition decoding from L44 beat this
  output-side predictor? This is the single cleanest way to separate "internal phase state" from "about to say 'Wait'". Expected under lexical-anticipation:
  near-parity; any persistent L44 margin is the first real evidence of a phase representation beyond imminent wording.

  E4. Horizon/position-matched anchor evaluation (essential; hours, CPU).
  Formalize what Stage 10's quintile subsets gesture at (within test Q1: act+obs 0.183 vs position 0.195 vs prevalence 0.080 — n=188, 15 positives):
  within-position-stratum AP/AUROC, plus importance-vs-remaining-tokens regressions, plus <NO_ANSWER>-rate-vs-position decomposition. This quantifies how much of the
  raw association survives removing the mechanical horizon channel.

  E5. Human/LLM functional stage relabel, few hundred sentences (essential before any "reasoning stage encoded" claim; ~days of labeling).
  Label sentence function while masking the first ~5 tokens. Re-test claim B. Expected under surface-form: macro-F1 collapses toward obs12 levels; under real phase
  encoding: degrades gracefully.

  E6. Per-prefix continuation resampling pilot (strong follow-up; moderate GPU).
  For ~100 boundaries, sample N next sentences, measure importance of each → estimates within-prefix variance of the target and the reliability ceiling for lead-0
  prediction (§4.4), and turns the target into E[I|h]. Decisive for interpreting both the positive lead-0 and null lead-1 results.

  E7. Earlier-boundary probes (follow-up; new extraction, moderate GPU).
  Extract activations 5/10/20 tokens before the boundary (within S_{i−1}). Genuine anticipation should decay smoothly; opener-prediction should collapse once the
  opening marker stops being locally determined. Discriminates the two readings of claims B/E without new labels.

  E8. Fresh-problem replication (required before any confirmatory claim; expensive GPU).
  New problems never touched by the Thought Anchors artifacts, trajectories sampled fresh, labels computed once, frozen hypothesis tested once. The current 4-problem
  "test" cannot carry population claims regardless of other fixes.

  Not worth doing yet: more layers, nonlinear probes, more permutations, cross-model transfer — all are downstream of the label-validity and lexical-baseline
  questions.

  ---
  12. Research-readiness assessment

  A. Already solid. The infrastructure and process: problem-level splits with runtime assertions; fold-local preprocessing; train-CV-only selection; validation-only
  calibration; a cryptographically enforced freeze → exactly-once test with a verifiable receipt; prespecified Stage 10 controls including a well-designed stratified
  permutation null; byte-hashed artifact lineage end-to-end. Every reported number reproduces exactly from saved artifacts. The MVP write-up's negative-space claims
  (no incremental value established; position competitive; layer not localized; no lead-1 signal) are all correct and verified. This is a genuinely trustworthy
  measurement; what's uncertain is what it measures.

  B. Promising but exploratory. (i) The raw above-prevalence anchor association (AP 0.093–0.094, AUROC 0.65), and the never-discussed continuous endpoint (test
  Spearman 0.237). (ii) Strong layer-dependent decodability of the upcoming sentence's marker class (macro-F1 0.46 vs 0.17 observables). (iii) The
  transition-prediction residual of +0.04–0.06 AP over the previous-stage prefix baseline, CI > 0 on both splits (my recomputation) — the project's most defensible
  positive, once restated at its true magnitude. (iv) Ruling out stage as the anchor mediator.

  C. Currently weak. (i) The target: α-instability (Jaccard 0.13–0.19), <NO_ANSWER> dominance (anchor rate up to 47%), horizon coupling (ρ=+0.30), no reliability
  estimate, 9.4% eligibility exclusion for un-replaceable sentences. (ii) Missing baselines: no token-identity/lexical-prefix arm, no full next-token-distribution arm,
  linear-position only. (iii) Inference: 4 test problems, ~25 test positives, selection-conditional CIs, ~25 unregistered test-scored models in the follow-up. (iv)
  Governance: the entire reasoning-stage analysis and its write-up are untracked in git. (v) Presentation traps: transition AP against a 0.60 base rate; tie-broken
  P@10%.

  D. What could invalidate the main conclusions. (1) Label redefinition: if E1 shows probe rankings churn like the anchor sets do, "predicting thought anchors" was
  never a stable claim. (2) Lexical reduction: if E2/E3 match L44 on stage/transition tasks, the "reasoning phase is encoded" story reduces to ordinary next-token
  readiness. (3) Horizon mechanics: if position-matched analysis (E4) removes the anchor association, the signal was the estimand's geometry, not anticipation. (4)
  Problem idiosyncrasy: the per-problem Δ range (+0.133 to −0.066) means one different test problem could flip the headline sign.

  E. Minimum before a serious write-up. E1–E4 (all CPU against saved artifacts) plus E5's few hundred relabeled sentences; restate claim E against the prefix-stage
  baseline (+0.04–0.06, not +0.22–0.40); fix the P@10% tie-breaking or drop the metric; commit research/reasoning_stage/ to version control; state the prediction ≠
  representation ≠ causation boundary explicitly. With those, the honest paper is: "a leakage-controlled null-to-weak result on anchor prediction with a demonstrated
  set of label pathologies, plus a small but split-robust pre-emission transition signal that survives prefix-stage control and awaits lexical controls" — a
  legitimately useful contribution.

  F. Stronger follow-up programme. E6 (expected-importance target with reliability ceiling) and E7 (token-distance decay curves) to characterize what and how early; E8
  fresh-problem confirmatory replication of one frozen hypothesis (the transition residual is currently the best candidate); then cross-domain transfer, and only
  after a robust correlational result, intervention-style tests (e.g., does steering the "transition-predictive" direction change transition frequency) to move from
  decodability toward causal involvement.

  ---
  Working-notes answers (from my_notes.md's embedded questions, verified in code): the counterfactual-importance score is KL between final-answer distributions, not
  trajectory divergence (labels/importance.py:35-57); thought anchors are by definition the top-10%-of-train-importance sentences (fit.py:52, labels/threshold.py);
  labels come from ~100 released rollouts per fork (mean 52.7 dissimilar survivors on the "without" side), and no uncertainty interval for individual labels is
  computed anywhere in the repo — the only reliability evidence is the global released-vs-canonical Spearman 0.858 and the α-sensitivity columns whose implications I
  quantified above.
