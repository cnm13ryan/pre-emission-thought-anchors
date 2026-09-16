"Plain-language views of saved results; no fitting, selection, or metric changes."

PRIMARY = "activation_plus_observables"
MODEL_NAMES = {
    PRIMARY: "Internal state + simple clues",
    "activation_only": "Internal state only",
    "observable_combined": "Combined simple clues",
    "position": "Sentence position only",
    "context": "Amount of preceding text",
    "uncertainty": "Uncertainty about the next token",
    "prevalence": "Same chance for every sentence",
    "residualized_activation": "Internal state after removing predictable simple clues",
}
TITLES = {
    "roc_pr": ("Can the predictor distinguish high-importance sentences from other sentences?"),
    "calibration": "Do predicted chances match how often high importance actually occurs?",
    "layer_aucpr": "Where inside the model is prediction most useful?",
    "baselines": "Does internal information add anything beyond simple clues?",
    "lead_time": "How do predicted chances change one sentence earlier?",
    "lead_time_performance": "Can we predict high importance one sentence earlier?",
    "threshold_sensitivity": "Does changing the definition of high importance change the result?",
    "position_entropy_controls": (
        "Does the result depend on sentence position or model uncertainty?"
    ),
    "permutation_null": "Could a score this large appear after shuffling the training labels?",
}
DEFINITIONS = (
    "A positive case is a sentence above the high-importance cutoff "
    "set using training data. AUCPR summarizes how well the predictor"
    " finds these cases as its warning threshold changes (here "
    "calculated as average precision; higher is better). The "
    "positive-class base rate is the fraction of sentences that are "
    "positive; it is the random/no-skill reference for "
    "precision-recall performance, not 0.5. ROC-AUC measures how "
    "often a positive case ranks above a negative one; its no-skill "
    "reference is 0.5. A bootstrap interval describes variation when "
    "whole problems are sampled again with replacement. Here the "
    "intervals cover the middle 95% of those resampled results; with "
    "four test problems, they are descriptive and cannot establish "
    "reliable performance on new kinds of problems."
)
IMPORTANCE = (
    "Sentence importance measures how much the distribution of final "
    "answers changes when the model generates a different reasoning "
    "sentence and continues, compared with continuing with the "
    "original sentence. Larger values mean a larger change under this"
    " project's counterfactual measure. They do not mean the sentence"
    " is correct, and do not by themselves establish a general causal"
    " explanation. A high-importance label means the score exceeds a "
    "cutoff fixed on training sentences. The predictor's chance is a "
    "separate quantity: its estimate before the sentence begins, not "
    "the measured importance itself. Missing predictions are "
    "unavailable, not zero."
)
TECHNICAL = (
    "Brier score is the mean squared error of predicted chances "
    "(lower is better). ECE summarizes the gap between predicted "
    "chances and observed frequencies in groups (lower is better). "
    "Log loss penalizes wrong probability forecasts, especially "
    "confident mistakes (lower is better). Platt scaling adjusts "
    "scores into probabilities using validation data; A and B are its"
    " fitted settings. Regularization C controls how strongly large "
    "model weights are discouraged; it is a training setting, not a "
    "scientific finding. q90 is the 90th percentile of training "
    "importance, used as the main cutoff; q80 and q95 use the 80th "
    "and 95th percentiles. Site 44 means the output of decoder layer "
    "44 before the final normalization; site 0 is the embedding "
    "output. Cross-validation tests on held-out groups within the "
    "training problems. A permutation null is the range of scores "
    "after shuffling training labels within the specified groups and "
    "fitting again."
)


def model_name(name):
    return MODEL_NAMES.get(name.removesuffix(":binary"), name.replace("_", " "))


def number(value, digits=4):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def interval(low, high):
    return f"{number(low, 3)} to {number(high, 3)}"


def improvement(low, high):
    if low is None or high is None:
        return "No interval is available, so the strength of evidence for improvement is unknown."
    if low <= 0 <= high:
        return (
            "The interval crosses or includes zero: the data do not clearly "
            "establish an improvement."
        )
    if high < 0:
        return (
            "The entire descriptive interval is below zero, favoring the "
            "simpler comparison in this sample."
        )
    return (
        "The descriptive interval is above zero in this comparison, but "
        "four test problems do not establish a general improvement."
    )


def executive_summary(report):
    row = next(
        r
        for r in report["metric_table"]
        if r["split"] == "test" and r["model"] == f"{PRIMARY}:binary"
    )
    gain, boot = report["gain_over_baseline"], report["bootstrap_test"]
    return [
        "Before a sentence began, internal model information helped rank "
        "high-importance sentences "
        f"in this sample: AUCPR (a score for finding positive cases; higher is better) "
        f"{report['primary']['value']:.4f}, compared with a no-skill base rate "
        f"of {number(row['prevalence'])} ({row['n_positive']} of {row['n']} sentences). "
        f"The score's descriptive interval is {interval(boot['low'], boot['high'])}.",
        f"The observed gain over combined simple clues was {gain['value']:.4f} "
        f"(interval {interval(gain['low'], gain['high'])}). "
        + improvement(gain["low"], gain["high"]),
        report.get("interpretations", {})
        .get("permutation_null", {})
        .get("Observed", "Saved shuffle results are unavailable; chance cannot be assessed here."),
        "This is an exploratory held-out engineering test on four public "
        "problems, within one model, "
        "one dataset and twenty problems overall. It does not establish a"
        " deployable safety monitor, "
        "a causal mechanism inside the predictor, or generalization to "
        "other models or tasks.",
    ]


def interpretations(report, sensitivities=(), permutations=()):
    "Six explicit questions for every view, with observations from its saved source rows."
    test = next(
        r
        for r in report["metric_table"]
        if r["split"] == "test" and r["model"] == f"{PRIMARY}:binary"
    )
    base = number(test["prevalence"])
    gain = report["gain_over_baseline"]
    observations = {
        "roc_pr": (
            f"Test AUCPR is {number(test['average_precision'])}, base rate {base}; "
            f"ROC-AUC is {number(test['auroc'])}. The AUCPR interval is "
            f"{interval(report['bootstrap_test']['low'], report['bootstrap_test']['high'])}."
        ),
        "baselines": (
            f"The gain over combined simple clues is {number(gain['value'])}, interval "
            f"{interval(gain['low'], gain['high'])}; test base rate {base}. "
            + improvement(gain["low"], gain["high"])
        ),
        "calibration": (
            f"Test probability errors: Brier (mean squared chance error) {number(test['brier'])}, "
            f"ECE (average gap between chances and frequencies) {number(test['ece'])}, "
            f"log loss (penalty for wrong probability forecasts) {number(test['log_loss'])}. "
            "Points below the diagonal mean the predictor gives chances that "
            "are too high; "
            "points above it mean chances are too low."
        ),
        "layer_aucpr": (
            f"Layer {report['selected_layer']} was chosen using training problems only. "
            "The line shows the best average training cross-validation score "
            "at each layer."
        ),
        "lead_time": (
            "The boxes show the distribution of predicted chances on "
            "sentences available at both times."
        ),
    }
    position = next(
        (
            r
            for r in report["metric_table"]
            if r["split"] == "test" and r["model"] == "position:binary"
        ),
        None,
    )
    if position:
        observations["baselines"] += (
            f" Sentence position alone scored "
            f"{number(position['average_precision'])} "
            f"(base rate {number(position['prevalence'])}); "
            "the combined baseline is not necessarily the strongest "
            "individual baseline in this sample."
        )
    constant = next(
        (
            r
            for r in report["metric_table"]
            if r["split"] == "test" and r["model"] == "prevalence:binary"
        ),
        None,
    )
    if constant:
        observations["calibration"] += (
            f" Giving every sentence the same chance had Brier error "
            f"{number(constant['brier'])}; lower error is better. "
            "Better ranking does not guarantee better probability forecasts."
        )
    chance = {
        "roc_pr": (
            f"No skill means precision-recall performance near the base rate ({base}) "
            "and ROC-AUC near 0.5."
        ),
        "baselines": (
            "No added information means a gain near zero over combined simple"
            " clues; no ranking skill means AUCPR near each split's base "
            "rate."
        ),
        "calibration": (
            "A constant base-rate forecast can be well calibrated without "
            "identifying which sentences are important. Perfect agreement "
            "follows the diagonal."
        ),
        "layer_aucpr": (
            "With no useful signal, each layer should score near the base "
            "rates of its held-out training folds; chance variation can still"
            " create a best layer."
        ),
        "lead_time": ("Even a predictor with no ranking skill can output a wide range of chances."),
        "lead_time_performance": (
            "No useful early signal means scores near the matching base rate "
            "and no clear gain over simple clues at the same earlier time."
        ),
        "threshold_sensitivity": (
            "With no added information, gains over simple clues stay near "
            "zero; the no-skill base rate changes with the cutoff."
        ),
        "position_entropy_controls": (
            "With no extra internal information, prediction should not "
            "consistently beat simple clues within comparable groups."
        ),
        "permutation_null": (
            "If the observed result is common after shuffling, chance under "
            "this specific shuffle scheme remains a plausible explanation."
        ),
    }
    limits = {
        "roc_pr": (
            "An above-base-rate point estimate does not establish improvement"
            " over simple clues or reliable performance elsewhere."
        ),
        "baselines": (
            "A positive point estimate alone cannot establish that internal "
            "information genuinely improves prediction."
        ),
        "calibration": (
            "No per-bin uncertainty intervals are stored in this view. Small "
            "groups and validation-fitted probability adjustment limit "
            "conclusions about new data."
        ),
        "layer_aucpr": (
            "This view has no paired uncertainty interval comparing layers. "
            "It cannot establish that layer 44 is clearly special, even if it"
            " has the highest training average; no test-layer search was "
            "performed."
        ),
        "lead_time": (
            "These boxes do not measure prediction accuracy. Use the "
            "earlier-prediction performance chart below; a shift in "
            "probabilities alone says nothing about useful warning time."
        ),
        "lead_time_performance": (
            "Only immediately before and one sentence earlier were tested. "
            "The models differ in training rows available at these times; "
            "this does not locate the earliest possible useful signal."
        ),
        "threshold_sensitivity": (
            "Small positive counts, multiple comparisons and only four test "
            "problems limit claims of robustness. An undefined group is not "
            "evidence of zero effect."
        ),
        "position_entropy_controls": (
            "Some groups have few or no positives. These comparisons cannot "
            "rule out position, text length, uncertainty, tokenization "
            "artifacts or label noise as explanations."
        ),
        "permutation_null": (
            "This shuffle preserves problem, trajectory type and position "
            "groups; single-class groups stay fixed. It does not test every "
            "explanation of the score or give the probability that the "
            "hypothesis is true."
        ),
    }
    analyses = {
        "threshold_sensitivity": {
            "threshold",
            "convergence",
            "replacement_quality",
            "trajectory_subgroup",
        },
        "position_entropy_controls": {"position_quintile", "entropy_quintile"},
        "lead_time_performance": {"lead_time"},
    }
    for name, analysis_names in analyses.items():
        pts = [
            r
            for r in sensitivities
            if r.get("analysis") in analysis_names
            and r.get("split") == "test"
            and r.get("probe_name") == PRIMARY
            and r.get("metric") == "average_precision"
            and r.get("comparison") is None
        ]
        if name == "threshold_sensitivity":
            pts = [
                r
                for r in pts
                if r["analysis"] == "threshold"
                and r["variant"] in ("q80", "q95")
                and r["model_source"] == "frozen_stage9"
            ]
        statements = []
        for r in pts:
            rate = r["n_positive"] / r["n_rows"] if r["n_rows"] else None
            paired = next(
                (
                    v
                    for v in sensitivities
                    if all(
                        v.get(k) == r.get(k)
                        for k in (
                            "analysis",
                            "variant",
                            "split",
                            "model_source",
                            "probe_name",
                            "metric",
                        )
                    )
                    and v.get("comparison") == "minus:observable_combined"
                ),
                None,
            )
            comparison = (
                f" Gain over simple clues {number(paired['value'])}, interval "
                f"{interval(paired['boot_low'], paired['boot_high'])}. "
                + improvement(paired["boot_low"], paired["boot_high"])
                if paired
                else ""
            )
            statements.append(
                f"{friendly_variant(r['variant'])} ({source_name(r['model_source'])}): "
                f"AUCPR {number(r['value'])}, base rate {number(rate)}, "
                f"interval {interval(r['boot_low'], r['boot_high'])}, "
                f"{r['n_positive']} positives in {r['n_rows']} sentences." + comparison
            )
        observations[name] = (
            " ".join(statements)
            or "Saved comparison rows are unavailable; no conclusion can be drawn."
        )
    groups = [
        r
        for r in sensitivities
        if r.get("analysis") in ("position_quintile", "entropy_quintile")
        and r.get("split") == "test"
        and r.get("probe_name") == PRIMARY
        and r.get("metric") == "average_precision"
        and r.get("comparison") == "minus:observable_combined"
    ]
    if groups:
        above = sum(r["value"] is not None and r["value"] > 0 for r in groups)
        below = sum(r["value"] is not None and r["value"] < 0 for r in groups)
        missing = sum(r["value"] is None for r in groups)
        observations["position_entropy_controls"] = (
            f"Across position and uncertainty groups, the observed gain is above zero in {above}, "
            f"below zero in {below}, and undefined in {missing}. "
            "The pattern is mixed; the chart shows each group's own no-skill base rate. "
            "The full comparison table retains positive counts and intervals."
        )
    validation_leads = [
        r
        for r in sensitivities
        if r.get("analysis") == "lead_time"
        and r.get("split") == "validation"
        and r.get("probe_name") == PRIMARY
        and r.get("metric") == "average_precision"
        and r.get("comparison") is None
    ]
    for r in validation_leads:
        rate = r["n_positive"] / r["n_rows"] if r["n_rows"] else None
        observations["lead_time_performance"] += (
            f" Validation, {friendly_variant(r['variant']).lower()}: "
            f"AUCPR {number(r['value'])}, base rate {number(rate)}."
        )
    obs = [r for r in permutations if r.get("kind") == "observed" and r.get("split") == "test"]
    null = [
        r["average_precision"]
        for r in permutations
        if r.get("kind") == "permutation"
        and r.get("split") == "test"
        and r.get("average_precision") is not None
    ]
    observations["permutation_null"] = (
        f"{sum(v >= obs[0]['average_precision'] for v in null)} of {len(null)} shuffled-label runs "
        f"matched or exceeded the observed test AUCPR {number(obs[0]['average_precision'])} "
        f"(base rate {base}). Chance under this scheme remains plausible."
        if obs and null
        else "Saved shuffle results are unavailable; chance cannot be assessed here."
    )
    conclusions = {
        "roc_pr": (
            "The observed ranking is better than the no-skill reference in "
            "this sample; the baseline comparison asks the harder question of"
            " added value."
        ),
        "baselines": improvement(gain["low"], gain["high"]),
        "calibration": (
            "Closeness to the diagonal describes probability agreement, not "
            "the ability to rank sentences."
        ),
        "layer_aucpr": (
            "The selected layer is a training-based choice, not evidence of a"
            " uniquely important layer."
        ),
        "lead_time": "These distributions describe the predictor's outputs only.",
        "lead_time_performance": (
            "Read the same-row, same-time baseline comparison; low or "
            "negative gains give no clear support for useful earlier "
            "prediction."
        ),
        "threshold_sensitivity": (
            "Changes in direction across cutoffs or subgroups limit a single "
            "broad claim; a gain interval crossing zero does not clearly "
            "establish improvement."
        ),
        "position_entropy_controls": (
            "Patterns in sparse groups are descriptive, not a demonstration "
            "that simple explanations have been ruled out."
        ),
        "permutation_null": (
            "Use the observed-versus-shuffled comparison to judge how unusual"
            " this result is under the specified shuffle, not as proof of a "
            "real signal."
        ),
    }
    return {
        name: {
            "Question": title,
            "No useful signal": chance[name],
            "Observed": observations[name],
            "Uncertainty": (
                limits[name]
                if name in {"layer_aucpr", "calibration", "lead_time"}
                else "Intervals are descriptive resampling ranges from only four test "
                "problems. "
                "A gain interval crossing zero does not clearly establish improvement."
                if name != "permutation_null"
                else (
                    "Only the saved finite set of shuffles is available; its fraction"
                    " is an estimate, not a precise probability."
                )
            ),
            "Can conclude": conclusions[name],
            "Cannot conclude": limits[name],
        }
        for name, title in TITLES.items()
    }


def source_name(value):
    return {
        "frozen_stage9": "original predictor",
        "refit_frozen_hyperparameters": "retrained with fixed settings",
    }.get(value, value)


def friendly_variant(value):
    return {
        "q80": "Broader definition (top 20% of training importance)",
        "q90": "Main definition (top 10% of training importance)",
        "q95": "Stricter definition (top 5% of training importance)",
        "lead0_frozen_same_rows": "Immediately before, matching sentences",
        "lead1_refit": "One sentence earlier",
        "k4": "Earlier stop after four stable answer steps",
        "k5": "Main stop after five stable answer steps",
    }.get(value, value.replace("_", " "))
