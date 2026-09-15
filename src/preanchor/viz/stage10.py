"""Stage 10 figures from ``metrics/sensitivities.parquet``, ``metrics/permutations.parquet``
and ``predictions/lead_time.parquet`` alone (no model, no activations). Each figure is written
as a self-contained ``.html`` and its exact data as ``.json`` (Plotly figure JSON); a ``.png``
is added when a static renderer is available. Every figure is labelled as a sensitivity /
control view of an exploratory held-out test; none is a primary result."""

from __future__ import annotations

import json
from pathlib import Path

from preanchor.io import write_artifact

PRIMARY = "activation_plus_observables"
BASELINE = "observable_combined"
NOTE = "sensitivity/control view; exploratory held-out test (4 problems); descriptive intervals"


def _rows(table) -> list[dict]:
    return table.to_pylist() if hasattr(table, "to_pylist") else list(table)


def _points(rows: list[dict], analysis: str, split: str, source: str, model: str,
            metric: str = "average_precision") -> list[dict]:
    out = [r for r in rows if r["analysis"] == analysis and r["split"] == split
           and r["model_source"] == source and r["probe_name"] == model
           and r["metric"] == metric and r["comparison"] is None]
    return sorted(out, key=lambda r: r["variant"])


def _bar(fig_rows: list[dict], name: str, x: list[str]):
    import plotly.graph_objects as go

    y = [r["value"] for r in fig_rows]
    low = [None if r["boot_low"] is None or r["value"] is None else r["value"] - r["boot_low"]
           for r in fig_rows]
    high = [None if r["boot_high"] is None or r["value"] is None else r["boot_high"] - r["value"]
            for r in fig_rows]
    return go.Bar(name=name, x=x, y=y,
                  error_y={"type": "data", "symmetric": False, "array": high, "arrayminus": low,
                           "visible": True},
                  text=[f"n={r['n_rows']}, pos={r['n_positive']}" for r in fig_rows],
                  textposition="outside")


def threshold_figure(rows: list[dict]):
    import plotly.graph_objects as go

    fig = go.Figure()
    for split in ("validation", "test"):
        for source, label in (("refit_frozen_hyperparameters", "refit"),
                              ("frozen_stage9", "frozen scores")):
            for model in (PRIMARY, BASELINE):
                pts = _points(rows, "threshold", split, source, model)
                if pts:
                    fig.add_trace(_bar(pts, f"{split} / {label} / {model}",
                                       [p["variant"] for p in pts]))
    fig.update_layout(title=f"Threshold sensitivity (train quantiles q80/q90/q95) — {NOTE}",
                      barmode="group", yaxis_title="AUCPR", xaxis_title="train quantile")
    return fig


def lead_time_figure(rows: list[dict]):
    import plotly.graph_objects as go

    fig = go.Figure()
    for split in ("validation", "test"):
        pts = []
        for variant, source in (("lead0_frozen_same_rows", "frozen_stage9"),
                                ("lead1_refit", "refit_frozen_hyperparameters")):
            for model in (PRIMARY, BASELINE, "activation_only"):
                p = [r for r in _points(rows, "lead_time", split, source, model)
                     if r["variant"] == variant]
                pts += [(f"{variant.split('_')[0]} / {model}", q) for q in p]
        if pts:
            fig.add_trace(_bar([q for _, q in pts], split, [x for x, _ in pts]))
    fig.update_layout(title=f"Lead zero versus lead one at the frozen site — {NOTE}",
                      barmode="group", yaxis_title="AUCPR")
    return fig


def permutation_figure(perms: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("validation", "test"))
    for col, split in enumerate(("validation", "test"), start=1):
        null = [r["average_precision"] for r in perms
                if r["kind"] == "permutation" and r["split"] == split
                and r["average_precision"] is not None]
        obs = [r["average_precision"] for r in perms
               if r["kind"] == "observed" and r["split"] == split]
        fig.add_trace(go.Histogram(x=null, name=f"{split} null (n={len(null)})", nbinsx=30),
                      row=1, col=col)
        if obs and obs[0] is not None:
            fig.add_vline(x=obs[0], line_color="red", row=1, col=col,
                          annotation_text=f"observed {obs[0]:.3f}")
    fig.update_layout(title="Permutation null of the selected-layer AUCPR (100 stratified "
                            f"label permutations) — {NOTE}", xaxis_title="AUCPR")
    return fig


def position_entropy_figure(rows: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "position quintile / validation", "position quintile / test",
        "entropy quintile / validation", "entropy quintile / test"))
    for r_i, analysis in enumerate(("position_quintile", "entropy_quintile"), start=1):
        for c_i, split in enumerate(("validation", "test"), start=1):
            for model in (PRIMARY, BASELINE, "prevalence"):
                pts = _points(rows, analysis, split, "frozen_stage9", model)
                if pts:
                    fig.add_trace(go.Bar(
                        name=model, x=[p["variant"] for p in pts], y=[p["value"] for p in pts],
                        text=[f"n={p['n_rows']}, pos={p['n_positive']}" for p in pts],
                        showlegend=(r_i == 1 and c_i == 1)), row=r_i, col=c_i)
    fig.update_layout(title=f"AUCPR by position and entropy quintile (train-derived cuts) — {NOTE}",
                      barmode="group")
    return fig


def _write(run: Path, name: str, fig) -> dict[str, str]:
    out = {}
    for suffix in (".html", ".json", ".png"):
        for extra in ("", ".sha256"):
            (run / "figures" / f"{name}{suffix}{extra}").unlink(missing_ok=True)
    # A fixed div id keeps the HTML byte-deterministic across regenerations.
    html = fig.to_html(include_plotlyjs=True, full_html=True, div_id=name).encode()
    out[f"{name}.html"] = write_artifact(run / "figures" / f"{name}.html", html)
    data = json.dumps(json.loads(fig.to_json()), sort_keys=True, separators=(",", ":")).encode()
    out[f"{name}.json"] = write_artifact(run / "figures" / f"{name}.json", data)
    try:
        png = fig.to_image(format="png", width=1400, height=800)
        out[f"{name}.png"] = write_artifact(run / "figures" / f"{name}.png", png)
    except Exception:
        out[f"{name}.png"] = None  # static renderer unavailable; html/json are the record
    return out


def write_figures(run: Path, sensitivities, permutations, leads) -> dict[str, str | None]:
    rows, perms = _rows(sensitivities), _rows(permutations)
    hashes: dict[str, str | None] = {}
    hashes.update(_write(run, "threshold_sensitivity", threshold_figure(rows)))
    hashes.update(_write(run, "lead_time", lead_time_figure(rows)))
    hashes.update(_write(run, "permutation_null", permutation_figure(perms)))
    hashes.update(_write(run, "position_entropy_controls", position_entropy_figure(rows)))
    return hashes
