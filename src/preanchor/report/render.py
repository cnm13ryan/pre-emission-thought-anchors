"""Stage 11 report assembly: figures, ``reports/mvp_result.{md,html}`` and a deterministic
``report.json`` whose tables derive only from saved predictions and metrics.

Outputs land in a fresh Stage 11 output directory; nothing under the Stage 3-10 state is
written. ``report.json``, the markdown, the HTML and every figure's data JSON are
byte-deterministic for fixed inputs (SPEC §10.7); ``manifest.json`` alone carries the
timestamp and is excluded from determinism claims.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from preanchor.io import write_artifact, write_json
from preanchor.report import figures as F
from preanchor.report import language as L
from preanchor.report.loaders import Sources, load_stage8, load_stage9, load_stage10

MARKDOWN_NOTE = (
    "The primary endpoint is an exploratory engineering evaluation on"
    " four public test problems; intervals are descriptive, no "
    "significance is claimed, and nothing here reopens the frozen "
    "Stage 8/9 selection."
)


def _metric_table(metrics_by_split: dict[str, dict]) -> list[dict]:
    rows = []
    for split, metrics in sorted(metrics_by_split.items()):
        for name, m in sorted(metrics["models"].items()):
            if not isinstance(m, dict) or not name.endswith(":binary"):
                continue
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "average_precision": m.get("average_precision"),
                    "auroc": m.get("auroc"),
                    "brier": m.get("brier"),
                    "ece": m.get("ece"),
                    "log_loss": m.get("log_loss"),
                    "n": m.get("n"),
                    "n_positive": m.get("n_positive"),
                    "prevalence": m.get("prevalence"),
                    "source": f"metrics/{split}.json",
                }
            )
    return rows


def _markdown(report: dict) -> str:
    lines = [
        "# Predicting important reasoning sentences before they begin",
        "",
        "## What did we learn?",
        "",
    ]
    lines += [f"- {point}" for point in L.executive_summary(report)]
    lines += [
        "",
        "## How to read these results",
        "",
        L.IMPORTANCE,
        "",
        L.DEFINITIONS,
        "",
        "Training problems were used to choose the predictor; validation "
        "problems were "
        "used to adjust its probability forecasts. Test problems were "
        "held aside until those "
        "choices were frozen. Validation is not an independent "
        "confirmation of probability accuracy.",
        "",
        "## Prediction scores and their no-skill reference",
        "",
        "Each row describes the same saved prediction results. Higher "
        "AUCPR and ROC-AUC "
        "mean better ranking, not necessarily more trustworthy probabilities.",
        "",
        (
            "| Set | Predictor | AUCPR | Base rate (no skill) | ROC-AUC | "
            "Sentences | Positive cases |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["metric_table"]:
        lines.append(
            f"| {r['split']} | {L.model_name(r['model'])} | "
            f"{L.number(r['average_precision'])} | {L.number(r['prevalence'])} | "
            f"{L.number(r['auroc'])} | {r['n']} | {r['n_positive']} |"
        )
    for name in report["figures"]:
        lines += [
            "",
            f"## {L.TITLES[name]}",
            "",
            f'<iframe title="{L.TITLES[name]}" src="../figures/{name}.embed.html" '
            'width="100%" height="620" loading="lazy"></iframe>',
            "",
            f"[Open interactive figure](../figures/{name}.html) · "
            f"[Exact figure data](../figures/{name}.json)",
            "",
        ]
        for label, text in report["interpretations"][name].items():
            lines += [f"**{label}:** {text}", ""]
    lines += [
        "<details><summary>Technical details and reproducibility</summary>",
        "",
        L.TECHNICAL,
        "",
        f"Selected layer: {report['selected_layer']}; "
        f"C: {report['selected_hyperparameter']}; training cutoff: {report['anchor_threshold']!r}.",
        f"Calibration: {report['calibration']!r}.",
        "",
        "Probability error metrics (lower is better):",
        "",
        "| Set | Predictor | Brier score | ECE | Log loss |",
        "|---|---|---|---|---|",
    ]
    for r in report["metric_table"]:
        lines.append(
            f"| {r['split']} | {L.model_name(r['model'])} | {L.number(r['brier'])} | "
            f"{L.number(r['ece'])} | {L.number(r['log_loss'])} |"
        )
    lines += [
        "",
        f"Stage 8 run: `{report['stage8_run_id']}`; Stage 9 run: "
        f"`{report['stage9_run_id']}`; Stage 10 run: `{report['stage10_run_id']}`.",
        f"Freeze SHA-256: `{report['freeze_sha256']}`.",
        "",
        "Every consumed artifact's full SHA-256 (also retained in report.json):",
        "",
    ]
    lines += [f"- `{path}` `{digest}`" for path, digest in report["input_sha256"].items()]
    lines += [
        "",
        "All saved AUCPR control comparisons (a gain is a difference in AUCPR):",
        "",
        "| Analysis | Variant | Set | Fitting | Predictor | Comparison | Score or gain | "
        "Base rate (no skill) | Interval | Positive / total | Reading of gain |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.get("control_results", []):
        rate = r["n_positive"] / r["n_rows"] if r["n_rows"] else None
        comparison = r["comparison"] or "score"
        reading = L.improvement(r["boot_low"], r["boot_high"]) if r["comparison"] else "—"
        lines.append(
            f"| {r['analysis']} | {L.friendly_variant(r['variant'])} | {r['split']} | "
            f"{L.source_name(r['model_source'])} | {L.model_name(r['probe_name'])} | "
            f"{comparison} | {L.number(r['value'])} | {L.number(rate)} | "
            f"{L.interval(r['boot_low'], r['boot_high'])} | "
            f"{r['n_positive']} / {r['n_rows']} | {reading} |"
        )
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def _html(markdown: str) -> str:
    from markdown_it import MarkdownIt

    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(markdown)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>What did we learn about important reasoning sentences?</title>"
        "<style>body{font:18px/1.6 "
        "system-ui,sans-serif;max-width:1100px;margin:auto;"
        "padding:24px;color:#182536}h1,h2{line-height:1.25}h2{margin-top:2em}"
        "table{display:block;overflow:auto;border-collapse:collapse;font-size:15px}"
        "td,th{padding:8px;border-bottom:1px solid #ccd3dc;text-align:left}"
        "iframe{border:0}details{margin:2em 0}code{overflow-wrap:anywhere}"
        "summary{cursor:pointer;font-weight:bold}</style></head>"
        f"<body>{body}</body></html>"
    )


def render_report(
    root: Path, out: Path, *, stage9_run_id: str, stage10_run_id: str, stage8_run_id: str
) -> dict:
    sources = Sources(Path(root))
    s9 = load_stage9(sources, stage9_run_id)
    s10 = load_stage10(sources, stage10_run_id)
    s8 = load_stage8(sources, stage8_run_id)
    inputs = sources.ledger()

    by_split = {
        "validation": F._split_rows(s9["predictions_validation"]),
        "test": F._split_rows(s9["predictions_test"]),
    }
    metrics_by_split = {"validation": s9["metrics_validation"], "test": s9["metrics_test"]}
    prediction_inputs = {k: v for k, v in inputs.items() if "predictions/" in k}

    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    figure_hashes: dict[str, str | None] = {}
    figures = {}

    def queue_figure(out, name, figure):
        figures[name] = figure
        return {}

    figure_hashes.update(queue_figure(out, "roc_pr", F.roc_pr_figure(by_split, prediction_inputs)))
    figure_hashes.update(
        queue_figure(
            out,
            "calibration",
            F.calibration_figure(
                by_split,
                s9["calibrator"],
                {
                    **prediction_inputs,
                    **{k: v for k, v in inputs.items() if k.endswith("calibrator.json")},
                },
            ),
        )
    )
    figure_hashes.update(
        queue_figure(
            out,
            "layer_aucpr",
            F.layer_aucpr_figure(
                s8["cv_results"],
                s8["pending_selection"],
                {k: v for k, v in inputs.items() if "probe_models/" in k},
            ),
        )
    )
    figure_hashes.update(
        queue_figure(
            out,
            "baselines",
            F.baselines_figure(
                metrics_by_split, {k: v for k, v in inputs.items() if "metrics/" in k}
            ),
        )
    )
    figure_hashes.update(
        queue_figure(
            out,
            "lead_time",
            F.lead_time_figure(
                s10["lead_time"],
                {k: v for k, v in inputs.items() if k.endswith("lead_time.parquet")},
            ),
        )
    )

    control_figures = F.control_figures(s10, inputs)
    for name, figure in control_figures.items():
        figure_hashes.update(queue_figure(out, name, figure))

    frozen = s9["frozen_selection"]
    test = s9["metrics_test"]
    boot = test["bootstrap"]["binary"][F.PRIMARY]["average_precision"]
    gain_interval = test["bootstrap"]["aucpr_gain_over_baseline"]
    gain = {
        "point": test["aucpr_gain_over_observable_combined"],
        "low": gain_interval["low"],
        "high": gain_interval["high"],
    }
    calibration = s9["calibrator"]["parameters"]
    report = {
        "schema_version": 1,
        "stage": "stage11",
        "stage8_run_id": stage8_run_id,
        "stage9_run_id": stage9_run_id,
        "stage10_run_id": stage10_run_id,
        "freeze_sha256": test["freeze_sha256"],
        "selected_layer": frozen["binary"]["layer"],
        "selected_hyperparameter": frozen["binary"]["C"],
        "anchor_threshold": frozen["anchor_threshold"],
        "primary": test["primary"],
        "bootstrap_test": {
            "low": boot["low"],
            "high": boot["high"],
            "replicates": test["bootstrap"]["replicates_requested"],
        },
        "gain_over_baseline": {"value": gain["point"], "low": gain["low"], "high": gain["high"]},
        "calibration": {
            "method": s9["calibrator"]["method"],
            "A": calibration["A"],
            "B": calibration["B"],
            "fitted_on": s9["calibrator"]["fitted_on"],
        },
        "metric_table": _metric_table(metrics_by_split),
        "figures": ["roc_pr", "calibration", "layer_aucpr", "baselines", "lead_time"],
        "figure_sha256": figure_hashes,
        "input_sha256": inputs,
        "note": MARKDOWN_NOTE,
    }
    report["control_results"] = [
        r for r in s10["sensitivities"].to_pylist() if r.get("metric") == "average_precision"
    ]
    report["figures"] += list(control_figures)
    report["interpretations"] = L.interpretations(
        report, s10["sensitivities"].to_pylist(), s10["permutations"].to_pylist()
    )
    # Attach the same interpretation directly below every standalone interactive figure.
    for name in report["figures"]:
        figure_hashes.update(
            F.write_figure(out, name, figures[name], report["interpretations"][name])
        )
    markdown = _markdown(report)
    hashes = {
        "report.json": write_artifact(
            out / "report.json",
            (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        ),
        "reports/mvp_result.md": write_artifact(out / "reports/mvp_result.md", markdown.encode()),
        "reports/mvp_result.html": write_artifact(
            out / "reports/mvp_result.html", _html(markdown).encode()
        ),
    }
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "stage": "stage11",
            "command": "render-report",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "root": str(root),
            "input_sha256": inputs,
            "output_sha256": {**hashes, **{f"figures/{k}": v for k, v in figure_hashes.items()}},
        },
    )
    return {"out": str(out), "report": report, "outputs": hashes}
