# Write-up of the project on predicting thought anchors

Simple sentence position contains very little information about the marker-defined reasoning stage.

By simple sentence position, I mean just where a sentence appears in the reasoning trace.

Sentence position is measured using only info about where the model currently is in the reasoning trace, without looking at the sentence semantical meaning or the model’s hidden activations.

—

The project starts from a simple scientific question:

Before a model produces an important reasoning step, is there already something in its internal state that tells us what is about to happen?

This question is then decomposed into two related questions:

Importance: Can the model’s internal state predict whether the next reasoning step will matter a lot for the eventual answer?
Reasoning Dynamics: Can the model’s internal state predict what kind of reasoning step is about to come next, such as planning, computing checking, or backtracking?

—
Imagine a model solving a difficult maths problem:
“Let me try substitution.”
“That gives x=4.”
“Wait, that cannot be right.”
“I should instead use symmetry.”
“Now the solution works.”
Some sentences are routine. Others course-chart the reasoning trajectory.
We originally ask:

Can we predict those consequential moments (thought anchor) in the reasoning traces before that is generated?

Thought anchors are sentences that we identify as consequential to the later reasoning trajectory. The sentences flagged as such are important as perturbing that sentence affects the model’s reasoning trajectory and final response significantly.
The broader motivation is to enable early prevention and ongoing monitoring.
If consequential transitions in the reasoning traces can be detected and predicted in advance, then there is more value in moving CoT monitoring to online-settings where we can identify emerging signs of misalignment earlier to prevent downstream harm.
This in turn can inform how other safety research methods and safety interventions should be developed (i.e., control protocols, kill-switches) in coordination to help reduce misalignment risks.
We start with reasoning traces (instead of model internals or weight signatures) because they provide a more accessible starting point for independent researchers like me to investigate, interpret results, design experiments, and stress-test risky assumptions (and this prediction approach) quickly without the need to pre-load a lot of MechInterp knowledge that I don’t have yet.
We basically rely heavily on re-sampling reasoning traces.
We have a probabilistic estimate of which sentences might matter, where we look at the model internal state via activation probes before the model generates those sentences.
We then train such probes over multiple reasoning trajectories with clear labels.
Then we compare against simple baselines, then we have the data results where we are now interpreting our results and see what we have done right or wrong, and investigate the limitations and the next set of experiments to be run, to resolve the additional uncertainty.
First, we have a rather expensive labeling process. For each sentence S_i, we re-run or re-sample the reasoning trajectory from each sampled sentence, and ask: If this sentence were different, how much would the eventual reasoning trajectory and model output response change?
Based on the results, we give each of the sampled and tested sentences a counterfactual-importance score. 
I was corrected by a chatbot that the counterfactual-importance score is primarily about the downstream model response distribution and not the reasoning trajectory. Tell me whether that is truly the case by verifying with the specific python scripts.
Thought-anchors are correlated with high-scores.
I was corrected by a chatbot that thought anchors are defined from the start to have high counterfactual-importance scores, where we have intentionally labeled sentences with scores within the top 10% of the training-set important scores to be such anchors. Tell me whether that is truly the case by verifying with the specific python scripts.
Once we have obtained those labels, we then stop re-sampling and we take the original reasoning trace and run a forward pass to the model with hooks to extract their hidden internal activations, next-token logits, entropy, sentence position and other observable features. This is where we start creating tuples $(h_{i^-}, I(S_i))$.
I was corrected that the labels used are estimated from finite re-sampling efforts with unknown uncertainty, with disagreement scores of spearman correlation of 0.858. Tell me how exactly the labels are collected and used. And tell me (if any) how the uncertainty interval are computed and measured.
$h_{i^-}$ means that this is the model state immediately before the sentence is generated.
$I(S_i)$ is the importance score of that sentence.
We have gathered many of these tuples and we train a linear probe where we train the linear probe to make predictions based on the internal activation, what’s the importance score of that sentence would be.
We want to see whether this would beat cheaper baselines as opposed to other probes trained on sentence position or next-token logits, or entropy respectively, to predict the importance scores.
I was corrected that the main pre-registered task was: $h_{h^-} \rightarrow P(\text{sentence i is a high-importance anchor).$, as opposed to directly predicting the importance score. 
I am confused. Since if the training data are of tuples of the model internal activation  before generating sentence S_i and the importance score of that sentence S_i. 
Does that not mean that the predictor, in this case the binary classifier (linear probe) should predict the score instead of that sentence ?
I am told that the thing that i need to understand is the difference between the continuous importance score and the binary anchor label derived from that score.
Suppose each training example starts as $(h_{i^-}, I(S_i))$ where $h_{i^-}$ is the activation immediately before sentence $S_i$, and $I(S_i)$ is that sentence’s counterfactual-importance score.
If we trained directly on $I(S_i)$, then the task would be regression: $h_{i^-} -> \hat{I(S_i)}$.
The predictor would then output something like: “I predict the up-coming sentence will have an importance score of 1.7.”.
However, the primary experiment that we have conducted focuses on converting the continuous score into a binary label.
An example of this:
Y_i = 1 if I(S_i) is in the top 10% 0 otherwise.
So the actual training examples for the linear probe is: $(h_{i^-}, Y_i)$.
The classifier then predicts $P(Y_i = 1 \mid h_{i^-})$, which means:
“Given the model’s state right now, how likely is the up-coming sentence to be a high-importance thought anchor?”
I find your point on "trajectories not having good and clear labels" confusing. I don't know much of the underlying implementation and so I want you to provide a clear and concise explanation of what's happening here.
I am told that we are estimating those labels experimentally. For each reasoning sentence S_i, our implementation roughly does something like this.
We stop at each reasoning sentence S_i, then we generate alternative continuations (aka re-sampling reasoning trajectories), then we look at and compare the resulting final answers.
We then use those observables to estimate how much changing $S_i$ changes those answers.
This then yields a numerical importance score $I(S_i)$. That score is based on counter-factual re-sampling, so it is an estimate from a finite number of sampled continuations.
We then turn that distribution of scores in a a discrete distribution of binary labels.
I(S_i) in top 10% => anchor = 1. 
Otherwise => anchor = 0.
The limitations are severe where we have cross-cut the continuities that have removed much of the nuances and underlying data richness that can be used to train a better linear probe.
In particular, I am not aware of this design decision being made at the time as it is mainly all done by the LLM agent.
This design decision is problematic as it not only prevents leveraging the underlying data richness, it creates huge agreement and disagreement problems when it comes to determining borderline cases of whether a particular sentence is an anchor or not, which would then question the original instantiation of the importance score threshold. These kinds of problems would not have happened if we were to continue to proceed with the continuous case.
I have no idea what are exactly the baselines that our experiments rely on and what are the control variables and manipulation variables used and thus the different experimental arms that we are testing, which are the "cheap observable features" that you have mentioned on position, context/prefix information, next-token uncertainty and probability summaries. I would like know much more in detail on what are they, how are they compared against one another, how are they implemented and designed.
I am told that the cleanest way to understand the experiment that we have done is as follows:
Suppose the model is half-way through a reasoning trace and is about to generate sentence S_i.
Before S_i appears, we have something conceptually like:
$h_{i^-}$ = internal activation vector
$p_i$ where we are in the trace
$c_i$ how much context has appeared
$u_i$ how uncertain the next-token distribution is
$Y_i$ whether S_i later turns out to be an anchor
We ask: Which information available before S_i is actually useful for predicting Y_i ?
We have 12 pre-registered observable features, where they roughly fall into three categorical groups:
Feature family
Intuitive question
Examples explicitly supported by the reports
Position
“Where am I in the reasoning?”
sentence index / context position
Context
“How much reasoning has happened already?”
prefix length
Uncertainty / probability
“How concentrated or uncertain is the model about the next token?”
next-token entropy, top-probability mass / token probabilities

We want to see whether the activation probe provides additional prediction value beyond what can be reasoned from looking at reasoning positions, context lengths, next-token entropy and token probabilities.
I am not sure whether this is something that is actually implemented and so we have to fact-check against the python scripts.
I am not familiar with the project or MechInterp stuff and so please re-work your explanation on how the experimental arms and baselines are constructed.
We have seven experimental arms.
The first one is random guessing. There were 25 anchors among the 689 sentences that enters our test (roughly 3.6% of the testing population are anchors).
So assuming there is no additional useful information that can be elicited from the measurement instruments, the probability of predicting whether the next sentence is an anchor is roughly 3.6%.
Next, we have the sentence-position baseline. We want to test whether important sentences simply occur relatively early in the reasoning traces, and so the predictor might just learn this short-cut which is not what we intended.
If we look at the canonical original data tuples that we have curated, we find that this is a possibility where we found that 15 out of 25 test anchors were in the first 20% of their reasoning traces.
A classifier that is trained using only position info achieved AP = 0.139, which is higher than the base-line and (Spoiler alert, higher than the activation probe’s point estimate of 0.093).
The tuple that is used to train the classifier is $(position of S_i, Y_i)$.
where:
Y_i = 1 if sentence is a thought anchor 
Y_i = 0 if otherwise
I am told that we have explicitly discrete position labels for the sentence position [early, middle, late]. But I would have to verify that against actual specific python scripts that contains the canonical actual implementation details.
AP (average precision) essentially answers the question:
“If I rank the likelihood test sentences to be thought anchors in descending order, does it match to the actual data? (aka do the actual anchors tend to appear near the top?)”
I am told that I need to inspect the stage 7 feature and the relevant python scripts and configuration to really tell whether the training data input is “raw sentence index”, “relative trace position”, or some combination of multiple different position-related variables.
The position-info classifier is basically performing logistic regression where the equation is roughly:
$P(thought-anchor) = \sigma (b + w \times position)$ with $w < 0$.
This sentence-position baseline can help us to address: 
Whether there is some apparent signal on predicting where thought-anchors will occur simply from sentence-position information alone?
Next, we have the context baseline.
I am told that this involves a predictor trying to predict thought-anchors on the basis of seeing simple info about how much reasoning has already happened already. But I need more details as I am not involved in the project until now.
I am told that the context baseline is designed to test: whether the amount of reasoning done dictates where thought anchors emerges and arises?
This is to be honest, a little complicated to understand. So basically, we train a probe where the training data tuple is: $(prefix-information, Y_i)$
For example:
“We need to solve for x.”
“Factoring does not immediately work.”
“Let me rewrite the equation.”
[boundary before S_i]
The context baseline records simple information about that preceding prefix. Prefix length is only one of many possible observable features in this category.
So something like the following is possible:
Prefix before Si
Prefix length
Is Si an anchor?
very little reasoning so far
200
1
moderate amount
700
0
long reasoning history
1,500
0

In this baseline, we are interested in knowing: Can the amount of context already accumulated predict whether the next sentence will be a thought-anchor?
Though, we have to check the python scripts and the exact implementation details to know what exactly are the observable features that we have inputted in the training data and what are the other alternatives that are tried and tested.
The training procedures are very similar to that of the sentence-position case and so I will not repeat it here.
The scores: AP = 0.047 and AUROC = 0.518.
Next, we have the uncertainty baseline. It is designed to test:
Whether the uncertainty levels in the model's confidence in coming up with the next token, is a predictor for up-coming thought-anchors?
We take simple summaries of the model’s next-token probability distribution before sentence S_i begins.
Suppose the model is at a sentence boundary and is about to generate S_i.
Before generating the next token, the model assigns probabilities to possible next tokens (assuming it is not greedy decoding and is stochastic generation where the temperature is non-zero), we then have: $P(x_{t+1} \mid x_{\leq t})$.
Exemplar observable features include “next-token entropy” and “top-probability mass”.
One important concept that we need to understand here is entropy: $H(P) = - \sum_x P(x) \log P(x)$ that is the bed-rock concept in information-theory.
Intuitively speaking:
Low entropy H: Model is fairly sure what token comes next
High entropy H: Model is uncertain what token comes next (i.e., many different next tokens look plausible candidates)
We want to know whether entropy is a good predictor of the occurrence of thought-anchors.
Again, the classifier performs logistic regression where it is trained to learn whether some combination of these uncertainty summaries predict the occurrence of thought-anchors.
$P(thought-anchor) = \sigma (b + w_1 (entropy) + w_2 (top-probability feature) + \cdots )$.
The results of this arm seem to be not documented in the results interpretation report and therefore we have to go back and check with the data to ensure that we have this information, to know what the results for this experimental arm are.
I find arm 5 the most confusing. What do you mean by testing all the observables together and see whether in aggregate can help predict thought anchors. Do tell me in detail how do you curate the training examples and how do you make sure that you have each has an equal or uniform weighting or contribution when it comes to predicting the thought anchor and thus the utility of this experimental arm. I am pretty skeptical of the utility of this experimental arm. 
I am told that this involves creating one training row per sentence, and putting all 12 cheap features into that row, and fitting a single regularized logistic-regression model whose coefficients are learned from the training data. I am confused as to how this helps. Is it something to do with Yann LeCun or Randall Balestrio’s self-supervised learning approach?
I am told strongly and bluntly that it is not Yann LeCun / Randall Balestriero–style self-supervised learning. Rather, it is simply ordinary supervised logistic regression on hand-chosen features.
We basically have not a tuple but a tuple vector, where:
X_i = [position-info, prefix-info, entropy, uncertainty-info, …]
Y_i = binary label (either thought-anchor or not)
When we train the classifier, the coefficients w_i, which is the weighting of the feature in X_i, will be learned against the training data.
It is definitively not SSL (self-supervised learning) as that would involve using un-labelled data and constructing a learning objective from that data itself.
However, spoiler alert, there is a loose conceptual similarity between linear probing in SSL and our activation-probe experimental arm.
In SSL research, we usually have raw data, where we then learn representation without any downstream labelling, then we freeze that representation, then train a simple classifier based on that and compare with downstream labels to evaluate prediction accuracy.
Basically this arm to aggregate all the cheap observables and use that as a predictor of thought-anchors. AP = 0.057. Not very high and so this indicates there are some observables that are just not that useful and we should prune them.
My understanding is that the reason that we introduce a scaling factor is to normalise the “order of magnitude” difference across features values, so the gradient does not explode while training the classifier. Is that the correct reason or interpretation or there is something else that I am not aware of.
I am told that my understanding is not entirely correct. I am told that the main reason for introducing a scaling factor in this experimental arm is to ensure that features with very different numerical units, are on comparable scales, such that it makes the logistic-regression optimiser an easier job and be able to apply regularisation penalty equally in proportion, so each feature can be treated equally unless backpropagation pin-point to specific features, as in the usual credit-assignment process.
Simply put, we want to avoid feature units to affect how strongly each feature coefficient is penalized.
Next, we come to arm-6, the activation-only baseline. 
At the boundary immediately before S_i. We know that at every transformer layer, there is one vector of numbers describing its current state.
At layer $\ell$, we call that vector: $h_{i^-}^{(\ell)}$. This is something that can be studied and interpreted in further details if you are already familiar with Anthropic’s early materials on residual stream activations and know that such vectors do not have a basis … Maybe I am wrong as my memory is a bit hazy at this point of time when writing this paragraph. I am deliberating this point here as I am told that the vectors represent the actual coordinates of the model’s internal representation at that point and I strongly feel that needs clarification as that I suspect is not scientifically accurate and needs further clarification, especially on the exact implementation and mathematical details.
Basically these states are extracted at the point one token before the target sentence starts, where it seems that we ran a leakage test.
To be honest, I have no idea what is a leakage test and so please explain to me carefully what that exactly means, why are we running it, and what hope to know after running the test.
Okay, I am told that "leakage" occurs when “future information” (in this case the sentence or any other related information that is associated with what we want to predict) leaks into earlier parts of the training data. This can be commonly known as train-test contamination.
So how does the test run?
Well, this can occur if our implementation is wrong. 
But we basically run a very simple adversarial test where we extract the activation in two cases:
S_1, S_2 \mid S_3, S_4
S_1, S_2 \mid S_3^\ast, S_4^\ast
We expect the activation extracted from both cases should be exactly equal. Otherwise, there is leakage. This is actually also known as a future-token mutation test.
So our training data is simply: $(h_{i^-}^{(\ell)}, Y_i)$.
I think I would like to get into the implementation details wrt pytorch. Like, how are the hooks implemented and thus any scaling or renormalisation, or calibration procedures in the process before using it to train the classifier ?
First, we run the LLM forward pass with PyTorch hooks attached to selected layers to extract the relevant activations. 
We then convert those activations into feature vectors, pair each feature vector with its corresponding label, and split the resulting dataset into training, calibration, and test sets. 
In particular, the train set is further used in grouped cross-validation to choose the layer and regularization strength C. The validation set is used to help fit the Platt calibrator.
Also, the activation vectors are already the features for the activation-only probe, where there is no separate learned feature-construction step beyond selecting the boundary activation from one layer and arranging it as a 5120-dimensional vector.
Next, we fit a StandardScaler on the training features so that each feature is standardized to a comparable numerical scale, then train the regularized logistic-regression probe on these standardized training examples. 
`StandardScaler` is fit only on the relevant training fold during cross-validation, and after model selection the final scaler/probe are re-fit on all training data.
The validation set is then used only for calibration, and the test set remains untouched until everything is frozen.
After the probe has been trained, we use it to produce raw decision scores on the separate calibration set and fit a Platt calibrator that maps those scores to better-calibrated probabilities. 
Finally, we apply the frozen scaler, probe, and calibrator to the test set to obtain the final predicted probabilities.

For each target sentence S_i, we identify the token immediately before that sentence starts.
That is the moment whose internal state we want to use for prediction.
In the code, this is `target_token_start - 1`. We want to measure activation here.
The experiment is asking: “Before S_i is generated, does the model state already tell us whether S_i will later be an anchor?”
We simply run the forward pass and pytorch hooks read out the internal vectors.
The embedding output plus the 48 Transformer-block outputs, giving 49 possible activation sites. Each activation vector has 5120 numbers.
The hook around `boundary_hidden = hidden[0, boundary_indices, :]` is grabbing the vectors.
The scaler independently standardizes every activation dimension using the training data:
$x_j’ = \dfrac{x_j - \mu_j}{\sigma_j}$
So all the 5120 numbers are put onto comparable numerical scales.
We now train the activation probe, where the logistic regression uses L2 regularization to discourage over-fitting, and `class_weight = “balanced”` because anchors are much rarer than non-anchors.
The weighting means mistakes on the rare anchor class count more heavily during training; it does not create duplicate examples.
How do we choose which activation sites (aka layer) that is the most predictive or the optimal regularization strength ?
Note that there are 49 possible activation layers and 7 possible C values.
$C \in \{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100\}$
Recall that C controls regularization inversely:
Small C => Strong Regularization
Large C => Weak Regularization
So C = 1e-4 heavily constrains the logistic-regression weights, while C = 100 lets them fit the training data much more freely.
For every combination (layer i, C_i), we run a 4-fold cross-validation.
A 4-fold cross-validation is a way to test whether a model generalizes without touching the real validation or test-set.
In the work that we have done, we have 12 training problems from the MATH Problem ID: [2238, 1591, 4682, 6596, 3448, 2189, 3550, 4605, 2137, 4019, 3360, 3935].
For each problem, the dataset includes both a released correct trajectory and an incorrect trajectory, so the 12 training problems correspond to 24 trajectories in total.
The fixed split is:
Train: 2238, 1591, 4682, 6596, 3448, 2189, 3550, 4605, 2137, 4019, 3360, 3935
Validation: 330, 4164, 2050, 6481
Test: 6998, 2236, 2870, 3916
For cross-validation, all sentences from both trajectories of one problem stay together. 
For instance, we never put some sentences from Problem 2238 into a training-fold and other sentences from Problem 2238 into its held-out fold.
Given these 12 training problems, we split them into 4 groups or “folds”.
Each time we only train on 3 folds and evaluate on the remaining 1:

Train on 
Evaluate on
2, 3, 4
1
1, 3, 4
2
1, 2, 4
3
1, 2, 3
4

So every training problem gets used for evaluation exactly once. Then we take the mean.
Okay, I now understand where the training problems come from and what does the 4-fold cross-validation mean. But how did we find out the optimal layer and optimal regularisation strength C. Did we just sweep all the different combinations ?
I am told yes. But I am skeptical so we have to really check the python scripts to understand the actual implementation details.
But even if we sweep all the different combinations, with 49 activation sites and 7 candidate regularization strengths: 49 x 7 = 343. We have 343 candidate probes.
For each candidate pair, we run the same 4-fold cross-validation over the 12 training problems.
In each fold, the scaler and the logistic-regression probe were fitted on ¾ of the training problems and evaluated on the remaining 1/4 . We then averaged the four AUCPR scores.
Then we compare all 343 mean cross-validation scores and choose:
$(best-layer, best-C) = \argmax_{\ell, C} mean cross-validation AUCPR$
This is how the “optimal” layer and optimal regularization strength were chosen.
The winning configuration’s four cross-validation AUCPRs were: [0.033, 0.107, 0.131, 0.234], which gives a mean of 0.126.
This configuration is layer 44 with a regularization strength C of (need to check with the data) as it is not documented in the report.


What are the remaining experimental arms that we still have not yet discussed?
Next, we have an experimental arm that uses both activations and the 12 cheap observable features.
The question is: “How well can we predict anchors when we give the classifier both the model’s internal state and everything cheap we already know from the prefix?”
This actually works fairly similar to activation-only. The only change is that, instead of giving the logistic-regression probe with only the activation of a vector with 5120 dimensions, we also give it the 12 cheap observable features.
X_i = [h_{i, 1}, \cdots, h_{i, 5120} ; o_{i, 1}, \cdots , o_{i, 12}]
This gives rise to about 5132 input features per data sample.
Model
Test AUCPR
AUROC
Chance/base rate
0.0363
0.500
Observables only
0.0573
0.487
Activation only
0.0925
0.645
Activation + observables
0.0941
0.650
Position only
0.1394
0.719



Next, we have the experimental arm on residualized activation, where we first statistically remove from the selected activation vector whatever can be linearly predicted from the observable features; then train an anchor probe on what remains.
The question is: “Is there predictive information in the activations that is not merely a reflection of position, prefix-length, uncertainty, and more?”.
This residualized-activation arm can be best understood as a control placed on the activation-only result.
The activation-only arm asks: “Can the raw activation $h$ predict whether the next sentence is an anchor?”
The problem is that $h$ may encode lots of mundane information. For example, it may encode that we are early in the reasoning trace, that the prefix is short, or that the model is uncertain.
Those observable variables themselves can correlate with anchors.
So good activation-only performance does not tell us whether the probe has found anything beyond those simpler cues.
Based on this understanding, the residualized arm asks a stronger question:
After removing the part of the activation that is linearly predictable from our 12 observables, does what remains still predict anchors?
Actually, I wonder to myself: How can we remove the part of the activation that is linearly predictable from our 12 cheap observables?
So basically we use the per-layer activation from the 12 cheap observable based probe and the activation probe then do some basic arithmetic?
Almost. But I am told that the 12-cheap-observables probe does not produce an activation vector.
Residualization uses two things: raw activation $h_i \in \mathbb{R}^{5120}$ and 12 cheap observables $o_i \in \mathbb{R}^12$.
Then we fit a separate linear regression: $o_i \rightarrow \hat{h_i}$, which asks: “Given these 12 observables, what part of the 5120-dimensional activation would I expect?”.
Then the arithmetic is simply: $h_i^{residual} = h_i - \hat{h_i}$, and we train the anchor classifier on that residual activation.
Next, we have an experimental arm on continuous ridge probe, where we instead try to predict the actual continuous counterfactual-importance value, where we use Ridge regression on `log1p(importance)`.
The question is: “Can activations predict how important the next sentence will be?”.
What remains are controls and sensitivity analyses, or maybe ablations (?)
The continuous Ridge probe is basically the same overall probing idea. 
Actually this addresses a problem that I have previously raised before.
$Y_I = log(1 + I(S_i))$
This is actually address given the activation before the sentence begins, can we predict how important that sentence will be.
Ridge regression is basically linear regression with an L2 penalty.
Loss = Squared Prediction Error + \lambda \Vert w \Vert_2^2
