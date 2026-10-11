"""Streamlit commentary adapted from the supplied result_interpretation.md.

These are interpretations and proposed checks, not new analyses or changes to the
frozen experiment. Numerical observations continue to come from saved artifacts.
"""

CONCLUSION = (
    "This experiment does not establish a useful internal warning signal for future "
    "sentence importance. It found a raw association: some features of the model state "
    "helped rank the later importance label in this sample. We cannot yet tell whether "
    "that relationship reflects future importance, ordinary reasoning position, clues "
    "in nearby tokens, or how importance was measured."
)
LIMIT = (
    "This also does not show that such a warning signal is absent. There are only four "
    "independent test problems and 25 positive sentences, and the observed gain varies "
    "across problems. Repeating the resampling "
    "procedure many times does not create more independent problems."
)
FIGURE_NOTES = {
    "roc_pr": (
        "Treat the above-base-rate score as a raw association with the measured label. "
        "It does not show that the model knows which future thoughts will matter, or "
        "that the label measures lasting causal influence."
    ),
    "baselines": (
        "Position is a strong competing predictor, but its higher point estimate does "
        "not establish that it is genuinely better. Nor does it prove that position "
        "explains the activation result. The registered simple clues also omit the "
        "identity of the preceding token and which next tokens are likely. Beating "
        "those clues would not establish an advantage over every output-side predictor."
    ),
    "layer_aucpr": (
        "Prediction is already possible from the embedding output: the representation "
        "of the token immediately before the sentence. The relatively flat curve does "
        "not identify layer 44 as a special importance layer. Punctuation, newlines, "
        "digits or other boundary-token clues could contribute; token identity was "
        "not included in the registered simple-clue baseline. These possibilities "
        "remain untested explanations, not findings about what the predictor uses."
    ),
    "lead_time_performance": (
        "The current linear predictor did not demonstrate useful one-sentence-ahead "
        "prediction of this target. That does not prove information appears only at "
        "the last moment. The label describes the one sentence actually generated; "
        "the same earlier state could lead to different sentences with different "
        "importance. Small samples, noisy labels and the limited predictor could "
        "also explain the negative result."
    ),
    "threshold_sensitivity": (
        "Changing a cutoff is only one check of the label. It does not establish that "
        "importance is stable across the small-probability adjustments used in its "
        "calculation, or across repeated counterfactual samples. Changing the stopping "
        "rule from five steps to four also does not test whether that rule measures "
        "agreement among answers: continuations can all be wrong yet disagree with "
        "one another."
    ),
    "position_entropy_controls": (
        "Earlier sentences leave more reasoning ahead in which changing a sentence "
        "can affect the answer. Position may therefore be related to the construction "
        "of the importance score itself. The current comparisons do not distinguish "
        "that explanation from a predictor learning reasoning phase or other clues."
    ),
    "permutation_null": (
        "The structured shuffle does not separate the observed score clearly from "
        "its chance reference. This is neither evidence of a specific internal "
        "precursor nor proof that a smaller useful precursor cannot exist."
    ),
}
ALTERNATIVES = [
    (
        "More reasoning remains after early sentences",
        "An early replacement has more opportunity to change later answers. Compare "
        "importance with remaining tokens, sentences and distance to the stopping point.",
    ),
    (
        "The preceding token supplies a clue",
        "The embedding output can reveal punctuation, formatting or token identity. "
        "Compare against preceding-token identity and token-category predictors.",
    ),
    (
        "The model is about to say something recognizable",
        "Scalar uncertainty summaries omit which tokens are likely. Compare against "
        "richer next-token and preceding-text information.",
    ),
    (
        "One generated sentence is a noisy target for a state",
        "A prefix can produce different next sentences. Sample multiple next sentences "
        "from the same prefix to measure how much their importance varies.",
    ),
    (
        "Wrong answers are mistaken for agreement",
        "The stopping rule can treat consistently wrong continuations as convergence. "
        "A proposed check would instead examine whether continuations agree on an answer.",
    ),
    (
        "Missing answers affect the score",
        "Truncations or parsing failures contribute a no-answer category. Examine its "
        "contribution to importance across sentence positions.",
    ),
    (
        "The label depends on estimation choices",
        "Compare importance ranks and high-importance membership across smoothing "
        "choices and independent groups of counterfactual samples.",
    ),
    (
        "The included sentences are not representative",
        "Some sentences lack enough different replacements or fail other eligibility "
        "rules. Compare included and excluded sentences before generalizing to all reasoning.",
    ),
]
NEXT_CHECKS = [
    "Audit the target: answer agreement, remaining reasoning length, missing answers, "
    "smoothing and which sentences are included.",
    "Strengthen comparisons: preceding-token identity, richer next-token information, "
    "more flexible position predictors, and later layers versus the embedding output.",
    "Inspect ranking within each reasoning trace using the already specified local "
    "ranking measures; pooled scores can reflect differences between problems or traces.",
    "Measure repeatability of both the importance estimator and future sentences "
    "generated from the same prefix.",
    "Then test the frozen hypothesis on substantially more independent, new problems.",
]
INSPECTOR_NOTE = (
    "This score belongs to the sentence that happened to be generated in this stored "
    "trace. It does not measure the average importance of every sentence that could "
    "have followed the same prefix, or establish that its effect persists through "
    "the rest of the reasoning. Colors or large scores alone cannot settle those questions."
)
VALIDITY = (
    "The supplied interpretation records checks of problem-separated splits, "
    "training-only preprocessing, a frozen selection before test evaluation, "
    "validation-only probability adjustment, future-token invariance, and exclusion "
    "of target/future features. It also records independent metric reproduction. "
    "These checks support the integrity of the measured association; they do not "
    "resolve what the label or predictor represents. The dashboard does not rerun them."
)
LABEL_AUDIT = (
    "The interpretation records that the released-label agreement gate failed and "
    "the planned fallback to canonical labels was used. Correctly following that "
    "fallback does not establish that the importance label is reliable."
)
PILOT_STATUS = (
    "No Stage 5 production summary with fresh labels was found in this root, so the "
    "planned fresh Problem 4682 resampling pilot cannot be verified here. Treat this "
    "validation component as unverified until its artifacts are restored and checked. "
    "Not finding it is not proof it was never run."
)


def pilot_status(pilot: dict | None) -> tuple[bool, str]:
    """(verified, text) for the fresh resampling pilot, from sidecar-verified Stage 5 files.

    ``pilot`` is ``loaders.load_stage5_pilot`` output with its run id under ``run_id``, or
    None when no Stage 5 production artifacts exist in the root.
    """
    if not pilot:
        return False, PILOT_STATUS
    s = pilot["summary"]
    planned = pilot.get("planned_jobs", 0)
    # A resumed run's ``completed`` and ``request_failures`` cover only its final session;
    # completion is judged on the durable, checksum-valid total against the plan.
    total, valid = s.get("completed_total"), s.get("valid_checksums")
    complete = (s.get("namespace") == "production" and planned > 0
                and total == valid == planned and pilot["fresh_label_rows"] > 0)
    if not complete:
        return False, (
            f"Stage 5 run {pilot['run_id']} exists but is incomplete: {total} of {planned} "
            f"planned completions recorded, {valid} checksum-valid, fresh labels for "
            f"{pilot['fresh_label_rows']} sentences. Treat the fresh resampling pilot as "
            "unverified.")
    return True, (
        f"The fresh resampling pilot is present and complete in this root: Stage 5 run "
        f"{pilot['run_id']} holds {total} of {planned} planned completions, all "
        f"checksum-valid, and wrote fresh labels for "
        f"{pilot['fresh_label_rows']} sentences (sidecar-verified). Project records "
        "(docs/stage5_measurements.md) identify it as the Problem 4682 pilot. "
        "result_interpretation.md said it could not be located; these restored artifacts "
        "supersede that statement. This check verifies the files' integrity, not the "
        "pilot's scientific conclusions.")
