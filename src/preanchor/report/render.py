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
from preanchor.report.loaders import Sources, load_stage8, load_stage9, load_stage10

MARKDOWN_NOTE = (
    "The primary endpoint is an exploratory engineering evaluation on four public test "
    "problems; intervals are descriptive, no significance is claimed, and nothing here "
    "reopens the frozen Stage 8/9 selection.")


def _metric_table(metrics_by_split: dict[str, dict]) -> list[dict]:
    rows = []
    for split, metrics in sorted(metrics_by_split.items()):
        for name, m in sorted(metrics["models"].items()):
            if not isinstance(m, dict) or not name.endswith(":binary"):
                continue
            rows.append({"split": split, "model": name,
                         "average_precision": m.get("average_precision"),
                         "auroc": m.get("auroc"), "brier": m.get("brier"),
                         "ece": m.get("ece"), "log_loss": m.get("log_loss"),
                         "n": m.get("n"), "n_positive": m.get("n_positive"),
                         "prevalence": m.get("prevalence"),
                         "source": f"metrics/{split}.json"})
    return rows


def _markdown(report: dict) -> str:
    s9, primary = report["stage9_run_id"], report["primary"]
    boot = report["bootstrap_test"]
    lines = [
        "# Pre-emission thought anchors — MVP result",
        "",
        f"Stage 9 run `{s9}` (freeze `{report['freeze_sha256'][:12]}…`), "
        f"Stage 10 run `{report['stage10_run_id']}`, "
        f"Stage 8 selection run `{report['stage8_run_id']}`.",
        "",
        f"> {MARKDOWN_NOTE}",
        "",
        "## Primary endpoint",
        "",
        f"- Selected probe: `{primary['model']}` at site {report['selected_layer']} "
        f"(C={report['selected_hyperparameter']}), threshold q90 = "
        f"{report['anchor_threshold']!r}.",
        f"- Test AUCPR: **{primary['value']:.4f}** "
        f"(bootstrap {boot['low']:.3f} … {boot['high']:.3f}, "
        f"{boot['replicates']} clustered replicates).",
        f"- Gain over combined observables: {report['gain_over_baseline']['value']:.4f} "
        f"({report['gain_over_baseline']['low']:.3f} … "
        f"{report['gain_over_baseline']['high']:.3f}); the interval includes zero.",
        f"- Calibration: {report['calibration']['method']} fitted on validation only "
        f"(A={report['calibration']['A']:.5f}, B={report['calibration']['B']:.5f}).",
        "",
        "## Binary metrics by model family",
        "",
        "| split | model | AUCPR | AUROC | Brier | ECE | log loss | n | positives |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    for r in report["metric_table"]:
        lines.append(f"| {r['split']} | {r['model']} | {fmt(r['average_precision'])} | "
                     f"{fmt(r['auroc'])} | {fmt(r['brier'])} | {fmt(r['ece'])} | "
                     f"{fmt(r['log_loss'])} | {r['n']} | {r['n_positive']} |")
    lines += [
        "",
        "## Figures",
        "",
    ]
    for name in report["figures"]:
        lines.append(f"- `figures/{name}.html` (data: `figures/{name}.json`)")
    lines += [
        "",
        "## Sensitivities and controls (Stage 10)",
        "",
        "The primary gain keeps its sign at q80, under K=4 convergence and on supported "
        "rows, vanishes at q95 and at lead one on test, and lies inside the stratified "
        "permutation null (see `figures/lead_time.html` and the Stage 10 run's own "
        "figures). Details: `docs/stage10_validation.md`.",
        "",
        "## Input artifacts",
        "",
        "Every consumed Stage 3-10 artifact, by SHA-256 (also in `report.json`):",
        "",
    ]
    for path, digest in report["input_sha256"].items():
        lines.append(f"- `{path}` `{digest[:16]}…`")
    return "\n".join(lines) + "\n"


def _html(markdown: str) -> str:
    import html as html_module

    body = html_module.escape(markdown)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>preanchor MVP result</title></head>"
            f"<body><pre style='max-width:100ch;white-space:pre-wrap'>{body}</pre>"
            "</body></html>")


def render_report(root: Path, out: Path, *, stage9_run_id: str, stage10_run_id: str,
                  stage8_run_id: str) -> dict:
    sources = Sources(Path(root))
    s9 = load_stage9(sources, stage9_run_id)
    s10 = load_stage10(sources, stage10_run_id)
    s8 = load_stage8(sources, stage8_run_id)
    inputs = sources.ledger()

    by_split = {"validation": F._split_rows(s9["predictions_validation"]),
                "test": F._split_rows(s9["predictions_test"])}
    metrics_by_split = {"validation": s9["metrics_validation"], "test": s9["metrics_test"]}
    prediction_inputs = {k: v for k, v in inputs.items() if "predictions/" in k}

    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    figure_hashes: dict[str, str | None] = {}
    figure_hashes.update(F.write_figure(out, "roc_pr", F.roc_pr_figure(
        by_split, prediction_inputs)))
    figure_hashes.update(F.write_figure(out, "calibration", F.calibration_figure(
        by_split, s9["calibrator"],
        {**prediction_inputs,
         **{k: v for k, v in inputs.items() if k.endswith("calibrator.json")}})))
    figure_hashes.update(F.write_figure(out, "layer_aucpr", F.layer_aucpr_figure(
        s8["cv_results"], s8["pending_selection"],
        {k: v for k, v in inputs.items() if "probe_models/" in k})))
    figure_hashes.update(F.write_figure(out, "baselines", F.baselines_figure(
        metrics_by_split, {k: v for k, v in inputs.items() if "metrics/" in k})))
    figure_hashes.update(F.write_figure(out, "lead_time", F.lead_time_figure(
        s10["lead_time"],
        {k: v for k, v in inputs.items() if k.endswith("lead_time.parquet")})))

    frozen = s9["frozen_selection"]
    test = s9["metrics_test"]
    boot = test["bootstrap"]["binary"][F.PRIMARY]["average_precision"]
    gain_interval = test["bootstrap"]["aucpr_gain_over_baseline"]
    gain = {"point": test["aucpr_gain_over_observable_combined"],
            "low": gain_interval["low"], "high": gain_interval["high"]}
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
        "bootstrap_test": {"low": boot["low"], "high": boot["high"],
                           "replicates": test["bootstrap"]["replicates_requested"]},
        "gain_over_baseline": {"value": gain["point"], "low": gain["low"],
                               "high": gain["high"]},
        "calibration": {"method": s9["calibrator"]["method"], "A": calibration["A"],
                        "B": calibration["B"], "fitted_on": s9["calibrator"]["fitted_on"]},
        "metric_table": _metric_table(metrics_by_split),
        "figures": ["roc_pr", "calibration", "layer_aucpr", "baselines", "lead_time"],
        "figure_sha256": figure_hashes,
        "input_sha256": inputs,
        "note": MARKDOWN_NOTE,
    }
    markdown = _markdown(report)
    hashes = {
        "report.json": write_artifact(
            out / "report.json",
            (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()),
        "reports/mvp_result.md": write_artifact(out / "reports/mvp_result.md",
                                                markdown.encode()),
        "reports/mvp_result.html": write_artifact(out / "reports/mvp_result.html",
                                                  _html(markdown).encode()),
    }
    write_json(out / "manifest.json", {
        "schema_version": 1, "stage": "stage11", "command": "render-report",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "root": str(root), "input_sha256": inputs,
        "output_sha256": {**hashes, **{f"figures/{k}": v for k, v in figure_hashes.items()}}})
    return {"out": str(out), "report": report, "outputs": hashes}
