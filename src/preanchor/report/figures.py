"""Stage 11 report figures from saved predictions and metrics alone (SPEC §8.3, §10.7).

Follows the ``viz/stage10`` conventions: each figure is a self-contained ``.html``, its
exact data as sorted-key Plotly figure JSON, and a ``.png`` when a static renderer is
available. Every figure carries the SHA-256 of its input artifacts in ``layout.meta`` and
an annotation naming the evaluation as an exploratory held-out test.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

from preanchor.io import write_artifact
from preanchor.report import language as L

PRIMARY = "activation_plus_observables"
BINARY_PRIMARY = f"{PRIMARY}:binary"
NOTE = "exploratory held-out test (4 problems); descriptive intervals; no significance claim"


def _meta(inputs: dict[str, str], note: str = NOTE) -> dict:
    return {"input_sha256": dict(sorted(inputs.items())), "note": note}


def _split_rows(predictions, probe: str = PRIMARY) -> dict[str, np.ndarray]:
    rows = predictions.to_pylist()
    rows = [r for r in rows if r["probe_name"] == probe and r["task"] == "binary"]
    y = np.array([r["y_true"] for r in rows], dtype=int)
    score = np.array([r["score_raw"] for r in rows], dtype=float)
    prob = np.array(
        [
            r["probability_calibrated"]
            if r.get("probability_calibrated") is not None
            else r["probability"]
            for r in rows
        ],
        dtype=float,
    )
    return {"y": y, "score": score, "probability": prob}


def roc_pr_figure(by_split: dict[str, dict], inputs: dict[str, str]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Finding positives versus false alarms (ROC)",
            "How many warnings are correct? (precision-recall)",
        ),
    )
    for split, data in sorted(by_split.items()):
        fpr, tpr, _ = roc_curve(data["y"], data["score"])
        prec, rec, _ = precision_recall_curve(data["y"], data["score"])
        fig.add_trace(
            go.Scatter(x=fpr.tolist(), y=tpr.tolist(), mode="lines", name=f"{split} ROC"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=rec.tolist(), y=prec.tolist(), mode="lines", name=f"{split} PR"),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=float(data["y"].mean()),
            line_dash="dot",
            row=1,
            col=2,
            annotation_text=f"{split} base rate (no skill) {data['y'].mean():.3f}",
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash", "color": "grey"}
        ),
        row=1,
        col=1,
    )
    fig.update_layout(title="Distinguishing high-importance sentences", meta=_meta(inputs))
    fig.update_xaxes(title_text="Fraction of negatives incorrectly flagged", row=1, col=1)
    fig.update_yaxes(title_text="Fraction of positives found", row=1, col=1)
    fig.update_xaxes(title_text="Fraction of positives found (recall)", row=1, col=2)
    fig.update_yaxes(title_text="Fraction of warnings correct (precision)", row=1, col=2)
    return fig


def calibration_figure(
    by_split: dict[str, dict], calibrator: dict, inputs: dict[str, str], bins: int = 10
):
    import plotly.graph_objects as go

    fig = go.Figure()
    for split, data in sorted(by_split.items()):
        edges = np.linspace(0.0, 1.0, bins + 1)
        which = np.clip(np.digitize(data["probability"], edges) - 1, 0, bins - 1)
        xs, ys, ns = [], [], []
        for b in range(bins):
            mask = which == b
            if mask.any():
                xs.append(float(data["probability"][mask].mean()))
                ys.append(float(data["y"][mask].mean()))
                ns.append(int(mask.sum()))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=split,
                text=[f"Sentences in this group: {n}" for n in ns],
                hovertemplate=(
                    "Predicted chance: %{x:.1%}<br>Observed frequency: "
                    "%{y:.1%}<br>%{text}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="perfect", line={"dash": "dash", "color": "grey"}
        )
    )
    fig.update_layout(
        title="Do predicted chances match observed frequencies?",
        xaxis_title="predicted probability",
        yaxis_title="fraction actually high importance",
        meta=_meta(inputs),
    )
    return fig


def layer_aucpr_figure(cv_results, pending_selection: dict, inputs: dict[str, str]):
    import plotly.graph_objects as go

    rows = [r for r in cv_results.to_pylist() if r["task"] == "binary" and r["layer"] is not None]
    folds: dict[tuple, list[float]] = {}
    for r in rows:
        folds.setdefault((r["probe_name"], int(r["layer"]), r["hyperparameter"]), []).append(
            r["average_precision"]
        )
    best: dict[str, dict[int, float]] = {}
    for (family, layer, _), values in folds.items():
        mean = float(np.mean(values))
        per = best.setdefault(family, {})
        per[layer] = max(per.get(layer, -np.inf), mean)
    fig = go.Figure()
    for family in sorted(best):
        layers = sorted(best[family])
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=[best[family][layer] for layer in layers],
                mode="lines+markers",
                name=L.model_name(family),
            )
        )
    rates = sorted(
        {r["val_pos"] / r["n_val"] for r in rows if r.get("n_val") and r.get("val_pos") is not None}
    )
    fig.add_annotation(
        x=0,
        y=-0.13,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            f"No-skill base rates across held-out training folds: {min(rates):.4f}–{max(rates):.4f}"
            if rates
            else "Training-fold base rates unavailable; no no-skill comparison possible"
        ),
    )
    fig.update_layout(margin={"b": 120})
    selected = pending_selection["binary"]
    fig.add_vline(
        x=int(selected["layer"]),
        line_dash="dot",
        annotation_text=f"chosen layer {selected['layer']}",
    )
    fig.update_layout(
        title="Prediction across layers — training checks only",
        xaxis_title="Layer (0 = embedding output)",
        yaxis_title="Mean training-check AUCPR",
        meta=_meta(inputs, "training-fold diagnostic (SPEC §8.3 Layers)"),
        legend={"orientation": "h", "y": 1.15},
    )
    return fig


def baselines_figure(metrics_by_split: dict[str, dict], inputs: dict[str, str]):
    import plotly.graph_objects as go

    fig = go.Figure()
    names = None
    for split, metrics in sorted(metrics_by_split.items()):
        models = {
            k: v
            for k, v in metrics["models"].items()
            if k.endswith(":binary")
            and isinstance(v, dict)
            and v.get("average_precision") is not None
        }
        names = names or sorted(models, key=lambda k: -models[k]["average_precision"])
        fig.add_trace(
            go.Bar(
                name=split,
                x=[L.model_name(n) for n in names],
                y=[models[n]["average_precision"] for n in names],
                text=[f"Base rate {models[n]['prevalence']:.4f}" for n in names],
                textposition="outside",
                hovertemplate="%{x}<br>AUCPR: %{y:.4f}<br>%{text}<extra>%{fullData.name}</extra>",
            )
        )
        fig.add_hline(
            y=float(models[BINARY_PRIMARY]["prevalence"]),
            line_dash="dot",
        )
    fig.update_layout(
        barmode="group",
        title="Does internal information beat simple clues?",
        yaxis_title="AUCPR (higher is better)",
        meta=_meta(inputs),
    )
    labels = list(fig.data[0].x) if fig.data else []
    fig.update_xaxes(
        tickmode="array",
        tickvals=labels,
        ticktext=["<br>".join(textwrap.wrap(label, 22)) for label in labels],
        tickangle=0,
    )
    return fig


def lead_time_figure(lead_time, inputs: dict[str, str]):
    import plotly.graph_objects as go

    rows = [r for r in lead_time.to_pylist() if r["probe_name"] == PRIMARY and r["available"]]
    fig = go.Figure()
    for split in sorted({r["split"] for r in rows}):
        for lead in sorted({r["lead_sentences"] for r in rows}):
            sel = [r for r in rows if r["split"] == split and r["lead_sentences"] == lead]
            if not sel:
                continue
            timing = "immediately before" if lead == 0 else "one sentence earlier"
            fig.add_trace(
                go.Box(
                    y=[r["probability"] for r in sel],
                    x=[f"{split}: {'immediately before' if lead == 0 else 'one sentence earlier'}"]
                    * len(sel),
                    name=f"{split}: {timing}",
                    boxmean=True,
                )
            )
    fig.update_layout(
        title="Predicted chances at two times — not a measure of accuracy",
        yaxis_title="predicted probability",
        showlegend=False,
        meta=_meta(inputs),
    )
    return fig


def write_figure(
    out: Path, name: str, fig, interpretation: dict | None = None
) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for suffix in (".html", ".embed.html", ".json", ".png"):
        for extra in ("", ".sha256"):
            (out / "figures" / f"{name}{suffix}{extra}").unlink(missing_ok=True)
    # A fixed div id keeps the HTML byte-deterministic across regenerations.
    html_text = fig.to_html(include_plotlyjs=True, full_html=True, div_id=name)
    if interpretation:
        hashes[f"{name}.embed.html"] = write_artifact(
            out / "figures" / f"{name}.embed.html", html_text.encode()
        )
        html_text = attach_interpretation(html_text, interpretation)
    html = html_text.encode()
    hashes[f"{name}.html"] = write_artifact(out / "figures" / f"{name}.html", html)
    data = json.dumps(json.loads(fig.to_json()), sort_keys=True, separators=(",", ":")).encode()
    hashes[f"{name}.json"] = write_artifact(out / "figures" / f"{name}.json", data)
    try:
        png = fig.to_image(format="png", width=1400, height=800)
        hashes[f"{name}.png"] = write_artifact(out / "figures" / f"{name}.png", png)
    except Exception:
        hashes[f"{name}.png"] = None  # static renderer unavailable; html/json are the record
    return hashes


def control_figures(s10: dict, inputs: dict[str, str]) -> dict:
    "Restyle Stage 10 figure builders in memory, never write to their source run."
    from preanchor.viz import stage10

    rows = s10["sensitivities"].to_pylist()
    perms = s10["permutations"].to_pylist()
    if not rows or "metric" not in rows[0] or not perms or "split" not in perms[0]:
        return {}  # Partial fixtures/inputs must not generate invented comparisons.
    figures = {
        "threshold_sensitivity": stage10.threshold_figure(rows),
        "lead_time_performance": stage10.lead_time_figure(rows),
        "position_entropy_controls": stage10.position_entropy_figure(rows),
        "permutation_null": stage10.permutation_figure(perms),
    }
    for name, fig in figures.items():
        fig.update_layout(
            title=L.TITLES[name],
            meta=_meta(inputs),
            margin={"b": 160},
            legend={"orientation": "h", "y": -0.35},
        )
        for trace in fig.data:
            if trace.name:
                if name == "position_entropy_controls":
                    trace.marker.color = {
                        PRIMARY: "#4169e1",
                        "observable_combined": "#eb8c23",
                        "prevalence": "#6b7280",
                    }[trace.name]
                trace.name = plain_label(trace.name)
            if trace.type == "bar":
                trace.x = [plain_label(str(x)) for x in trace.x]
                texts = []
                for text in trace.text:
                    # Upstream text encodes the exact denominator and positive count.
                    n, positives = [int(v.split("=")[1]) for v in text.split(", ")]
                    texts.append(
                        f"Base rate {positives / n:.4f}; {positives}/{n} sentences"
                        if n
                        else "Base rate unavailable: no sentences"
                    )
                trace.text = texts
                trace.textposition = "none"
                trace.hovertemplate = (
                    "%{x}<br>AUCPR: %{y:.4f}<br>%{text}<extra>%{fullData.name}</extra>"
                )
        fig.update_yaxes(title_text="AUCPR (higher is better)")
        if name == "permutation_null":
            fig.update_yaxes(title_text="Number of shuffled-label runs")
            fig.update_xaxes(title_text="AUCPR (higher is better)")
            for col, split in enumerate(("validation", "test"), 1):
                obs = next(
                    (r for r in perms if r["kind"] == "observed" and r["split"] == split), None
                )
                if obs and obs["n_rows"]:
                    rate = obs["n_positive"] / obs["n_rows"]
                    fig.add_vline(
                        x=rate,
                        line_dash="dot",
                        row=1,
                        col=col,
                        annotation_text=f"No-skill base rate {rate:.4f}",
                        annotation_position="bottom right",
                    )
        if name != "permutation_null":
            analysis = {
                "threshold_sensitivity": {"threshold"},
                "lead_time_performance": {"lead_time"},
                "position_entropy_controls": {"position_quintile", "entropy_quintile"},
            }[name]
            rates = {}
            for r in rows:
                if (
                    r["analysis"] in analysis
                    and r["metric"] == "average_precision"
                    and r["comparison"] is None
                    and r["n_rows"]
                    and r["split"] in ("test", "validation")
                ):
                    label = (r["analysis"] + " " if len(analysis) > 1 else "") + r["variant"]
                    rates[(r["split"], label)] = r["n_positive"] / r["n_rows"]
            rate_lines = []
            for split in ("test", "validation"):
                items = [
                    f"{plain_label(label)}: {value:.4f}"
                    for (group, label), value in sorted(rates.items())
                    if group == split
                ]
                # Two lines per split keep the base-rate reference readable in exported images.
                for start in range(0, len(items), 5):
                    rate_lines.append(
                        f"{split} no-skill base rates — " + "; ".join(items[start : start + 5])
                    )
            fig.add_annotation(
                x=0,
                y=-0.17,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                text="<br>".join(rate_lines),
                align="left",
                font={"size": 11},
            )
            fig.update_layout(margin={"b": 270}, legend={"y": -0.48, "font": {"size": 10}})
            if name == "threshold_sensitivity":
                fig.update_xaxes(title_text="Cutoff set using training sentences")
        if name == "lead_time_performance":
            labels = list(dict.fromkeys(x for trace in fig.data for x in trace.x))
            fig.update_xaxes(
                tickmode="array",
                tickvals=labels,
                ticktext=["<br>".join(textwrap.wrap(label, 24)) for label in labels],
                tickangle=0,
            )
            fig.layout.annotations[-1].y = -0.28
        for annotation in fig.layout.annotations:
            annotation.text = plain_label(annotation.text)
    return figures


def plain_label(text: str) -> str:
    text = " / ".join(L.MODEL_NAMES.get(part, part) for part in text.split(" / "))
    for key, value in [
        ("q80", "Top 20%"),
        ("q90", "Top 10%"),
        ("q95", "Top 5%"),
        ("lead0", "Immediately before"),
        ("lead1", "One sentence earlier"),
        ("position quintile", "Sentence position group"),
        ("entropy quintile", "Next-token uncertainty group"),
        ("position_quintile", "Position"),
        ("entropy_quintile", "Uncertainty"),
        ("_frozen_same_rows", ""),
        ("_refit", ""),
        ("frozen scores", "Original predictor"),
        ("refit", "Retrained, fixed settings"),
        ("null", "shuffled labels"),
    ]:
        text = text.replace(key, value)
    return text


def attach_interpretation(content: str, interpretation: dict) -> str:
    "Keep standalone figure explanations accessible without the parent report."
    import html

    prose = "".join(
        f"<p><strong>{html.escape(k)}:</strong> {html.escape(v)}</p>"
        for k, v in interpretation.items()
    )
    block = (
        ("<section style='max-width:1000px;margin:24px auto;font:17px/1.6 system-ui'>")
        + prose
        + "</section>"
    )
    return content.replace("</body>", block + "</body>")
