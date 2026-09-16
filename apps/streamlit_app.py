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

PRIMARY = "activation_plus_observables"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True,
                        help="<root>/artifacts of a verified (restored) preanchor root")
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
    st.header("Overview")
    st.caption("Read-only view of a verified restored root; canonical state lives on "
               "Lambda and in the portable stores.")
    rows = []
    for stage, run in sorted(runs.items()):
        manifest = _json(str(root / "artifacts/runs" / run / "manifest.json"))
        state = (manifest.get("stages") or {}).get(stage, {}).get("state")
        rows.append({"stage": stage, "run_id": run, "state": state,
                     "git_sha": (manifest.get("identity") or {}).get("git_sha", "")[:12]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    s9 = root / "artifacts/runs" / runs["stage9"]
    st.subheader("Frozen selection")
    st.json(_json(str(s9 / "frozen_selection.json")), expanded=False)


def page_inspector(root: Path, runs: dict[str, str]) -> None:
    st.header("Trajectory inspector")
    labels = _parquet(str(root / "data/processed/stage4/labels.parquet"))
    features = _parquet(str(root / "data/processed/stage7/features.parquet"),
                        ("sentence_id", "next_token_entropy", "next_token_max_probability",
                         "next_token_probability_margin", "next_token_top5_mass",
                         "next_token_top10_mass", "boundary_token_idx", "target_token_start",
                         "split"))
    preds = []
    for split in ("validation", "test"):
        frame = _parquet(str(root / "artifacts/runs" / runs["stage9"]
                             / f"predictions/{split}.parquet"))
        preds.append(frame[(frame.probe_name == PRIMARY) & (frame.task == "binary")])
    predictions = pd.concat(preds)[["sentence_id", "probability",
                                    "probability_calibrated", "y_true"]]
    problem = st.selectbox("Problem", sorted(labels.problem_id.unique()))
    kind = st.selectbox("Trace", sorted(labels[labels.problem_id == problem]
                                        .trace_kind.unique()))
    view = labels[(labels.problem_id == problem) & (labels.trace_kind == kind)] \
        .sort_values("sentence_idx") \
        .merge(features, on="sentence_id", how="left") \
        .merge(predictions, on="sentence_id", how="left")
    st.caption("Full top-10 next-token lists are unavailable by design: only the summary "
               "observables (entropy, max probability, margin, top-5/top-10 mass) are part "
               "of the canonical Stage 7 state.")
    st.dataframe(view[["sentence_idx", "text", "eligible", "exclusion_reason",
                       "importance_released", "importance_legacy", "importance_canonical",
                       "anchor", "probability", "probability_calibrated", "y_true",
                       "next_token_entropy", "next_token_max_probability",
                       "next_token_top10_mass", "boundary_token_idx",
                       "target_token_start"]], use_container_width=True, height=520)
    sentence = st.selectbox("Sentence", view.sentence_idx.tolist())
    st.write(view[view.sentence_idx == sentence].iloc[0].to_dict())


def page_metrics(root: Path, runs: dict[str, str]) -> None:
    st.header("Metrics")
    s9 = root / "artifacts/runs" / runs["stage9"]
    report_run = sorted((root / "artifacts/runs").glob("*-stage11-report"))
    for split in ("validation", "test"):
        metrics = _json(str(s9 / f"metrics/{split}.json"))
        st.subheader(f"{split} — binary families")
        frame = pd.DataFrame([{"model": k, **{m: v.get(m) for m in
                               ("average_precision", "auroc", "brier", "ece", "log_loss",
                                "n", "n_positive")}}
                              for k, v in metrics["models"].items()
                              if isinstance(v, dict) and k.endswith(":binary")])
        st.dataframe(frame, use_container_width=True)
    if report_run:
        st.subheader("Report figures")
        for name in ("roc_pr", "calibration", "baselines"):
            html = report_run[-1] / "figures" / f"{name}.html"
            if html.is_file():
                st.components.v1.html(html.read_text(), height=520, scrolling=True)


def page_layers(root: Path, runs: dict[str, str]) -> None:
    st.header("Layers (training-CV diagnostic)")
    st.caption("Training-fold diagnostics only; the held-out test never informed these.")
    report_run = sorted((root / "artifacts/runs").glob("*-stage11-report"))
    if report_run:
        html = report_run[-1] / "figures/layer_aucpr.html"
        if html.is_file():
            st.components.v1.html(html.read_text(), height=560, scrolling=True)
    cv = _parquet(str(root / "artifacts/runs" / runs["stage8"]
                      / "probe_models/cv_results.parquet"))
    st.dataframe(cv[cv.task == "binary"], use_container_width=True, height=400)


def page_controls(root: Path, runs: dict[str, str]) -> None:
    st.header("Lead time and controls (Stage 10)")
    s10 = root / "artifacts/runs" / runs["stage10"]
    for name in ("threshold_sensitivity", "lead_time", "permutation_null",
                 "position_entropy_controls"):
        html = s10 / "figures" / f"{name}.html"
        if html.is_file():
            st.subheader(name)
            st.components.v1.html(html.read_text(), height=520, scrolling=True)
    st.subheader("Sensitivities table")
    st.dataframe(_parquet(str(s10 / "metrics/sensitivities.parquet")),
                 use_container_width=True, height=400)


def main() -> None:
    args = _args()
    artifacts = Path(args.artifacts_root).resolve()
    root = artifacts.parent
    st.set_page_config(page_title="preanchor", layout="wide")
    runs = _runs(str(artifacts))
    page = st.sidebar.radio("Page", ("Overview", "Trajectory inspector", "Metrics",
                                     "Layers", "Lead time and controls"))
    {"Overview": page_overview, "Trajectory inspector": page_inspector,
     "Metrics": page_metrics, "Layers": page_layers,
     "Lead time and controls": page_controls}[page](root, runs)


if __name__ == "__main__":
    main()
