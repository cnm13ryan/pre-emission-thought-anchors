"""Read-only Stage 11 dashboard over saved artifacts (SPEC §8.2-8.3).

Run:  uv run streamlit run apps/streamlit_app.py -- --artifacts-root <root>/artifacts
Every page operates entirely from saved Stage 3-10 artifacts; nothing is written, no model
or activation shard is loaded. Full top-10 next-token *lists* were never a canonical
artifact; the inspector shows the preserved summary observables and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from preanchor.report import figures as F
from preanchor.report import language as L
from preanchor.report.loaders import Sources, load_stage8, load_stage9, load_stage10
from preanchor.report.render import _metric_table

PRIMARY = "activation_plus_observables"


@st.cache_data
def evidence(root: str, runs: dict):
    sources = Sources(Path(root))
    s8 = load_stage8(sources, runs["stage8"])
    s9 = load_stage9(sources, runs["stage9"])
    s10 = load_stage10(sources, runs["stage10"])
    test = s9["metrics_test"]
    boot = test["bootstrap"]["binary"][PRIMARY]["average_precision"]
    gain = test["bootstrap"]["aucpr_gain_over_baseline"]
    report = {
        "primary": test["primary"],
        "metric_table": _metric_table({"test": test, "validation": s9["metrics_validation"]}),
        "bootstrap_test": boot,
        "gain_over_baseline": {
            "value": test["aucpr_gain_over_observable_combined"],
            "low": gain["low"],
            "high": gain["high"],
        },
        "selected_layer": s9["frozen_selection"]["binary"]["layer"],
    }
    report["interpretations"] = L.interpretations(
        report, s10["sensitivities"].to_pylist(), s10["permutations"].to_pylist()
    )
    return report, s8, s9, s10, sources.ledger()


def show_figure(name, fig, report):
    st.subheader(L.TITLES[name])
    st.plotly_chart(fig, use_container_width=True)
    for label, text in report["interpretations"][name].items():
        st.markdown(f"**{label}:** {text}")


def reading_guide():
    st.markdown(L.DEFINITIONS)
    with st.expander("Technical terms explained"):
        st.markdown(L.TECHNICAL)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root",
        required=True,
        help="<root>/artifacts of a verified (restored) preanchor root",
    )
    return parser.parse_args(sys.argv[1:])


@st.cache_data
def _runs(artifacts: str) -> dict[str, str]:
    out = {}
    for run in sorted(Path(artifacts, "runs").iterdir()):
        stages = run / "stages"
        if stages.is_dir():
            for stage in stages.iterdir():
                out[stage.name] = run.name
    return out


@st.cache_data
def _parquet(path: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data
def _json(path: str) -> dict:
    return json.loads(Path(path).read_bytes())


def page_overview(root: Path, runs: dict[str, str]) -> None:
    st.header("What did we learn?")
    report, _, s9, _, inputs = evidence(str(root), runs)
    for point in L.executive_summary(report):
        st.markdown(f"- {point}")
    st.subheader("What does sentence importance mean?")
    st.markdown(L.IMPORTANCE)
    reading_guide()
    with st.expander("Run provenance, saved state and frozen settings"):
        st.caption(
            "Read-only: all views use saved artifacts; no model is fitted or evaluated here."
        )
        rows = []
        for stage, run in sorted(runs.items()):
            path = root / "artifacts/runs" / run / "manifest.json"
            manifest = _json(str(path)) if path.exists() else {}
            state = (manifest.get("stages") or {}).get(stage, {}).get("state")
            rows.append(
                {
                    "stage": stage,
                    "run_id": run,
                    "state": state,
                    "git_sha": (manifest.get("identity") or {}).get("git_sha", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.json(s9["frozen_selection"], expanded=False)
        st.json(inputs, expanded=False)


def page_inspector(root: Path, runs: dict[str, str]) -> None:
    st.header("Read a reasoning trace sentence by sentence")
    st.markdown(L.IMPORTANCE)
    labels = _parquet(str(root / "data/processed/stage4/labels.parquet"))
    features = _parquet(
        str(root / "data/processed/stage7/features.parquet"),
        (
            "sentence_id",
            "next_token_entropy",
            "next_token_max_probability",
            "next_token_probability_margin",
            "next_token_top5_mass",
            "next_token_top10_mass",
            "boundary_token_idx",
            "target_token_start",
            "split",
        ),
    )
    preds = []
    for split in ("validation", "test"):
        frame = _parquet(
            str(root / "artifacts/runs" / runs["stage9"] / f"predictions/{split}.parquet")
        )
        preds.append(frame[(frame.probe_name == PRIMARY) & (frame.task == "binary")])
    predictions = pd.concat(preds)[
        ["sentence_id", "probability", "probability_calibrated", "y_true"]
    ]
    problem = st.selectbox("Problem", sorted(labels.problem_id.unique()))
    kind = st.selectbox("Trace", sorted(labels[labels.problem_id == problem].trace_kind.unique()))
    view = (
        labels[(labels.problem_id == problem) & (labels.trace_kind == kind)]
        .sort_values("sentence_idx")
        .merge(features, on="sentence_id", how="left")
        .merge(predictions, on="sentence_id", how="left")
    )
    st.dataframe(
        view[
            [
                "sentence_idx",
                "text",
                "eligible",
                "exclusion_reason",
                "importance_canonical",
                "anchor",
                "probability_calibrated",
            ]
        ].rename(
            columns={
                "sentence_idx": "Sentence number",
                "text": "Exact sentence",
                "eligible": "Included",
                "exclusion_reason": "Why excluded",
                "importance_canonical": "Measured counterfactual importance",
                "anchor": "High importance under stored label rule",
                "probability_calibrated": "Predicted chance before sentence",
            }
        ),
        use_container_width=True,
        height=520,
    )
    st.caption(
        "The measured score and stored high-importance label are shown separately. "
        "The technical view retains released, legacy and canonical label variants and the "
        "actual evaluation label; these must not be assumed interchangeable."
    )
    sentence = st.selectbox("Sentence", view.sentence_idx.tolist())
    selected = view[view.sentence_idx == sentence].iloc[0]
    st.write(selected["text"])
    with st.expander("Sentence details, label variants and token boundaries"):
        st.caption(
            "Full next-token lists were not saved. Only summary measures of the model's "
            "uncertainty and token probabilities are available."
        )
        st.write(selected.to_dict())
        st.dataframe(view, use_container_width=True)


def page_metrics(root: Path, runs: dict[str, str]) -> None:
    st.header("How well did prediction perform?")
    report, _, s9, _, inputs = evidence(str(root), runs)
    reading_guide()
    frame = pd.DataFrame(report["metric_table"])
    frame["model"] = frame["model"].map(L.model_name)
    st.dataframe(
        frame[
            ["split", "model", "average_precision", "prevalence", "auroc", "n", "n_positive"]
        ].rename(
            columns={
                "split": "Set",
                "model": "Predictor",
                "average_precision": "AUCPR",
                "prevalence": "Base rate (no skill)",
                "auroc": "ROC-AUC",
                "n": "Sentences",
                "n_positive": "Positive cases",
            }
        ),
        use_container_width=True,
    )
    by_split = {
        split: F._split_rows(s9[f"predictions_{split}"]) for split in ("validation", "test")
    }
    for name, fig in [
        ("roc_pr", F.roc_pr_figure(by_split, inputs)),
        (
            "baselines",
            F.baselines_figure({split: s9[f"metrics_{split}"] for split in by_split}, inputs),
        ),
        ("calibration", F.calibration_figure(by_split, s9["calibrator"], inputs)),
    ]:
        show_figure(name, fig, report)
    with st.expander("Probability errors and original saved metrics"):
        st.markdown(L.TECHNICAL)
        st.dataframe(frame, use_container_width=True)
        st.json(s9["metrics_test"], expanded=False)
        st.json(s9["metrics_validation"], expanded=False)


def page_layers(root: Path, runs: dict[str, str]) -> None:
    report, s8, _, _, inputs = evidence(str(root), runs)
    reading_guide()
    show_figure(
        "layer_aucpr",
        F.layer_aucpr_figure(s8["cv_results"], s8["pending_selection"], inputs),
        report,
    )
    with st.expander("Training-fold scores and fitting settings"):
        st.markdown(L.TECHNICAL)
        frame = s8["cv_results"].to_pandas()
        frame["Base rate (no skill)"] = frame.val_pos / frame.n_val.replace(0, float("nan"))
        st.dataframe(frame, use_container_width=True)


def page_controls(root: Path, runs: dict[str, str]) -> None:
    st.header("Earlier prediction and checks of alternative explanations")
    report, _, _, s10, inputs = evidence(str(root), runs)
    reading_guide()
    show_figure("lead_time", F.lead_time_figure(s10["lead_time"], inputs), report)
    for name, fig in F.control_figures(s10, inputs).items():
        show_figure(name, fig, report)
    with st.expander("All saved comparisons and reproducibility details"):
        st.markdown(L.TECHNICAL)
        frame = s10["sensitivities"].to_pandas()
        frame["Base rate (no skill)"] = frame["n_positive"] / frame["n_rows"].replace(
            0, float("nan")
        )
        st.dataframe(frame, use_container_width=True, height=400)
        st.dataframe(
            s10["permutations"]
            .to_pandas()
            .assign(base_rate=lambda d: d.n_positive / d.n_rows.replace(0, float("nan"))),
            use_container_width=True,
        )


def main() -> None:
    args = _args()
    artifacts = Path(args.artifacts_root).resolve()
    if not (artifacts / "runs").is_dir() and (artifacts / "artifacts/runs").is_dir():
        artifacts = artifacts / "artifacts"  # accept the root itself as well
    if not (artifacts / "runs").is_dir():
        st.error(
            f"No runs directory under {artifacts}; pass <root>/artifacts of a "
            "verified restored root (e.g. ~/preanchor-root/artifacts)."
        )
        st.stop()
    root = artifacts.parent
    st.set_page_config(page_title="preanchor", layout="wide")
    runs = _runs(str(artifacts))
    page = st.sidebar.radio(
        "Page", ("Overview", "Trajectory inspector", "Metrics", "Layers", "Lead time and controls")
    )
    {
        "Overview": page_overview,
        "Trajectory inspector": page_inspector,
        "Metrics": page_metrics,
        "Layers": page_layers,
        "Lead time and controls": page_controls,
    }[page](root, runs)


if __name__ == "__main__":
    main()
