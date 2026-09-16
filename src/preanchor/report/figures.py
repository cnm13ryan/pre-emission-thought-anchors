"""Stage 11 report figures from saved predictions and metrics alone (SPEC §8.3, §10.7).

Follows the ``viz/stage10`` conventions: each figure is a self-contained ``.html``, its
exact data as sorted-key Plotly figure JSON, and a ``.png`` when a static renderer is
available. Every figure carries the SHA-256 of its input artifacts in ``layout.meta`` and
an annotation naming the evaluation as an exploratory held-out test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from preanchor.io import write_artifact

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
    prob = np.array([r["probability_calibrated"] if r.get("probability_calibrated")
                     is not None else r["probability"] for r in rows], dtype=float)
    return {"y": y, "score": score, "probability": prob}


def roc_pr_figure(by_split: dict[str, dict], inputs: dict[str, str]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Precision-Recall"))
    for split, data in sorted(by_split.items()):
        fpr, tpr, _ = roc_curve(data["y"], data["score"])
        prec, rec, _ = precision_recall_curve(data["y"], data["score"])
        fig.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                                 name=f"{split} ROC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rec.tolist(), y=prec.tolist(), mode="lines",
                                 name=f"{split} PR"), row=1, col=2)
        fig.add_hline(y=float(data["y"].mean()), line_dash="dot", row=1, col=2,
                      annotation_text=f"{split} prevalence {data['y'].mean():.3f}")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line={"dash": "dash", "color": "grey"}), row=1, col=1)
    fig.update_layout(title=f"ROC and PR — {PRIMARY} (binary)", meta=_meta(inputs))
    return fig


def calibration_figure(by_split: dict[str, dict], calibrator: dict, inputs: dict[str, str],
                       bins: int = 10):
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
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=split,
                                 text=[f"n={n}" for n in ns]))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line={"dash": "dash", "color": "grey"}))
    params = calibrator.get("parameters") or {}
    fig.update_layout(
        title=f"Reliability — {BINARY_PRIMARY} ({calibrator.get('method')}, "
              f"A={params.get('A'):.4f}, B={params.get('B'):.4f}, fitted on validation)",
        xaxis_title="predicted probability", yaxis_title="observed anchor rate",
        meta=_meta(inputs))
    return fig


def layer_aucpr_figure(cv_results, pending_selection: dict, inputs: dict[str, str]):
    import plotly.graph_objects as go

    rows = [r for r in cv_results.to_pylist() if r["task"] == "binary"
            and r["layer"] is not None]
    folds: dict[tuple, list[float]] = {}
    for r in rows:
        folds.setdefault((r["probe_name"], int(r["layer"]), r["hyperparameter"]),
                         []).append(r["average_precision"])
    best: dict[str, dict[int, float]] = {}
    for (family, layer, _), values in folds.items():
        mean = float(np.mean(values))
        per = best.setdefault(family, {})
        per[layer] = max(per.get(layer, -np.inf), mean)
    fig = go.Figure()
    for family in sorted(best):
        layers = sorted(best[family])
        fig.add_trace(go.Scatter(x=layers, y=[best[family][layer] for layer in layers],
                                 mode="lines+markers", name=family))
    selected = pending_selection["binary"]
    fig.add_vline(x=int(selected["layer"]), line_dash="dot",
                  annotation_text=f"selected site {selected['layer']} (C={selected['C']})")
    fig.update_layout(title="Training-CV AUCPR by site (best hyperparameter per site) — "
                            "training-fold diagnostic, not held-out evidence",
                      xaxis_title="site (layer)", yaxis_title="mean CV AUCPR",
                      meta=_meta(inputs, "training-fold diagnostic (SPEC §8.3 Layers)"))
    return fig


def baselines_figure(metrics_by_split: dict[str, dict], inputs: dict[str, str]):
    import plotly.graph_objects as go

    fig = go.Figure()
    names = None
    for split, metrics in sorted(metrics_by_split.items()):
        models = {k: v for k, v in metrics["models"].items() if k.endswith(":binary")
                  and isinstance(v, dict) and v.get("average_precision") is not None}
        names = names or sorted(models, key=lambda k: -models[k]["average_precision"])
        fig.add_trace(go.Bar(
            name=split, x=[n.split(":")[0] for n in names],
            y=[models[n]["average_precision"] for n in names],
            text=[f"n={models[n]['n']}, pos={models[n]['n_positive']}" for n in names],
            textposition="outside"))
        fig.add_hline(y=float(models[BINARY_PRIMARY]["prevalence"]), line_dash="dot",
                      annotation_text=f"{split} prevalence")
    fig.update_layout(barmode="group", title="AUCPR by model family (binary)",
                      yaxis_title="average precision", meta=_meta(inputs))
    return fig


def lead_time_figure(lead_time, inputs: dict[str, str]):
    import plotly.graph_objects as go

    rows = [r for r in lead_time.to_pylist() if r["probe_name"] == PRIMARY
            and r["available"]]
    fig = go.Figure()
    for split in sorted({r["split"] for r in rows}):
        for lead in sorted({r["lead_sentences"] for r in rows}):
            sel = [r for r in rows if r["split"] == split and r["lead_sentences"] == lead]
            if not sel:
                continue
            fig.add_trace(go.Box(y=[r["probability"] for r in sel],
                                 x=[f"{split} lead {lead}"] * len(sel),
                                 name=f"{split} lead {lead}", boxmean=True))
    fig.update_layout(title=f"Predicted anchor probability at lead 0 vs lead 1 — {PRIMARY} "
                            "(same mappable rows)",
                      yaxis_title="predicted probability", showlegend=False,
                      meta=_meta(inputs))
    return fig


def write_figure(out: Path, name: str, fig) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for suffix in (".html", ".json", ".png"):
        for extra in ("", ".sha256"):
            (out / "figures" / f"{name}{suffix}{extra}").unlink(missing_ok=True)
    # A fixed div id keeps the HTML byte-deterministic across regenerations.
    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id=name).encode()
    hashes[f"{name}.html"] = write_artifact(out / "figures" / f"{name}.html", html)
    data = json.dumps(json.loads(fig.to_json()), sort_keys=True, separators=(",", ":")).encode()
    hashes[f"{name}.json"] = write_artifact(out / "figures" / f"{name}.json", data)
    try:
        png = fig.to_image(format="png", width=1400, height=800)
        hashes[f"{name}.png"] = write_artifact(out / "figures" / f"{name}.png", png)
    except Exception:
        hashes[f"{name}.png"] = None  # static renderer unavailable; html/json are the record
    return hashes
