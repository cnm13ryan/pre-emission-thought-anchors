"""Read-only reproduction of blog evidence; never reruns the locked test command.

Run with CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/preanchor-mpl python -m
preanchor.report.blog_evidence --root ~/preanchor-root --output research/blog
Figures use frozen predictions. Transition refits reproduce already reported models
at their saved training-selected C; no new selection or experimental labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from preanchor.ids import derived_seed
from preanchor.io import write_parquet

S8 = "20260915T193326Z-57bcb41c84-7ebd0db6"
S9 = "20260915T201107Z-484989d2fe-19f1ab0d"
S10 = "20260915T211903Z-b230473011-815ca2b7"
PRIMARY = "activation_plus_observables"
SPLITS = ["validation", "test"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bootstrap(frame, scores, seed):
    """Independent paired problem bootstrap, all rows and cluster multiplicities kept."""
    rng = np.random.default_rng(seed)
    problems = np.sort(frame.problem_id.unique())
    y = frame.y_true.to_numpy()
    draws = []
    for _ in range(2000):
        sampled = rng.choice(problems, len(problems), replace=True)
        ix = np.concatenate([np.flatnonzero(frame.problem_id.to_numpy() == p) for p in sampled])
        draws.append({name: ap(y[ix], np.asarray(s)[ix]) for name, s in scores.items()})
    return pd.DataFrame(draws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    (out / "figures").mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    runs = args.root / "artifacts/runs"
    sources = {}

    def read(path):
        path = Path(path)
        sha = digest(path)
        sources[str(path)] = sha
        sidecar = Path(str(path) + ".sha256")
        if sidecar.exists():
            assert sha == sidecar.read_text().split()[0], path
        return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())

    labels = read(args.root / "data/processed/stage4/labels.parquet")
    features = read(args.root / "data/processed/stage7/features.parquet")
    cv = read(runs / S8 / "probe_models/cv_results.parquet")
    freeze = read(runs / S9 / "frozen_selection.json")
    assert freeze["binary"]["layer"] == 44 and freeze["binary"]["C"] == 0.1
    read(runs / S9 / "metrics/test_receipt.json")
    read(runs / S10 / "metrics/sensitivities.parquet")
    perm = read(runs / S10 / "metrics/permutations.parquet")
    lead = read(runs / S10 / "predictions/lead_time.parquet")
    stage_dir = repo / "research/reasoning_stage/outputs"
    stage_results = read(stage_dir / "analysis_results.json")
    d = read(stage_dir / "dataset.parquet")
    for site in [0, 44]:
        p = stage_dir / f"activations_l{site}.npy"
        sources[str(p)] = digest(p)
    e = labels[labels.eligible].merge(
        features[["sentence_id", "split"]], on="sentence_id", validate="one_to_one"
    )
    assert e.sentence_id.is_unique and len(e) == 4019
    sets = {sp: set(e[e.split == sp].problem_id) for sp in ["train", *SPLITS]}
    assert not (
        sets["train"] & sets["test"]
        or sets["train"] & sets["validation"]
        or sets["validation"] & sets["test"]
    )
    assert np.isclose(
        e.loc[e.split == "train", "importance_primary"].quantile(0.9), 1.3549810893213545
    )
    evidence = {
        "stage": "exploratory",
        "preregistered": False,
        "scope": "Read-only blog verification; original endpoint remains frozen",
        "counts": e.groupby("split").anchor.agg(["size", "sum", "mean"]).to_dict("index"),
        "anchor": {},
        "transition": {},
        "smoothing": {},
        "lead": {},
    }
    predictions = {}
    bootrows = []
    for sp in SPLITS:
        allp = read(runs / S9 / f"predictions/{sp}.parquet")
        m = read(runs / S9 / f"metrics/{sp}.json")
        p = allp[allp.task == "binary"].copy()
        predictions[sp] = p
        ref = p[p.probe_name == PRIMARY].sort_values("sentence_id")
        merged = ref.merge(e[["sentence_id", "anchor"]], on="sentence_id", validate="one_to_one")
        assert np.array_equal(merged.y_true, merged.anchor)
        scores = {
            name: g.set_index("sentence_id").loc[ref.sentence_id, "score_raw"].to_numpy()
            for name, g in p.groupby("probe_name")
        }
        b = bootstrap(ref, scores, m["bootstrap"]["seed"])
        for name, score in scores.items():
            assert np.isclose(
                ap(ref.y_true, score), m["models"][f"{name}:binary"]["average_precision"]
            )
        comparisons = {}
        for name in ["observable_combined", "position"]:
            delta = b[PRIMARY] - b[name]
            ci = np.percentile(delta, [2.5, 97.5])
            saved = m["bootstrap"]["paired_differences"][f"{PRIMARY}-minus-{name}"][
                "average_precision"
            ]
            assert np.allclose(ci, [saved["low"], saved["high"]], atol=1e-12)
            comparisons[name] = {
                "delta": ap(ref.y_true, scores[PRIMARY]) - ap(ref.y_true, scores[name]),
                "low": ci[0],
                "high": ci[1],
            }
        b["split"] = sp
        b["task"] = "anchor"
        bootrows.append(b)
        null = perm[(perm.split == sp) & (perm.kind == "permutation")].average_precision
        observed = ap(ref.y_true, scores[PRIMARY])
        evidence["anchor"][sp] = {
            "ap": {k: ap(ref.y_true, v) for k, v in scores.items()},
            "comparisons": comparisons,
            "permutation_p": (1 + int((null >= observed).sum())) / (len(null) + 1),
            "problem_counts": e[e.split == sp]
            .groupby("problem_id")
            .anchor.agg(["size", "sum"])
            .to_dict("index"),
        }
        lead_rows = lead[(lead.split == sp) & lead.available & (lead.probe_name == PRIMARY)]
        matched_ids = set(lead_rows.loc[lead_rows.lead_sentences == 1, "sentence_id"])
        lead_rows = lead_rows[lead_rows.sentence_id.isin(matched_ids)]
        assert lead_rows.groupby("lead_sentences").size().nunique() == 1
        evidence["lead"][sp] = {
            str(k): {
                "ap": ap(g.y_true, g.score_raw),
                "n": len(g),
                "positives": int(g.y_true.sum()),
                "prevalence": g.y_true.mean(),
            }
            for k, g in lead_rows.groupby("lead_sentences")
        }
        cont = allp[(allp.task == "continuous") & (allp.probe_name == PRIMARY)]
        evidence["anchor"][sp]["continuous_spearman"] = spearmanr(
            cont.importance_true, cont.importance_predicted
        ).statistic

    cuts = e.loc[e.split == "train", "sentence_idx"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    e["position_bin"] = np.searchsorted(cuts, e.sentence_idx, side="left")
    e["relative_eligible_rank"] = e.groupby("trajectory_id").sentence_idx.rank(pct=True)
    evidence["position_cuts"] = cuts.tolist()
    evidence["early_test"] = {
        "absolute_index_0_to_25": int(e[(e.split == "test") & (e.sentence_idx <= 25)].anchor.sum()),
        "first_fifth_of_eligible_rows": int(
            e[(e.split == "test") & (e.relative_eligible_rank <= 0.2)].anchor.sum()
        ),
    }
    for alpha in ["0_5", "1_0"]:
        col = f"importance_canonical_alpha_{alpha}"
        threshold = e.loc[e.split == "train", col].quantile(0.9)
        alternate = e[col] >= threshold
        evidence["smoothing"][alpha] = {
            "threshold": threshold,
            "spearman": spearmanr(e.importance_primary, e[col]).statistic,
            "intersection": int((e.anchor & alternate).sum()),
            "union": int((e.anchor | alternate).sum()),
            "jaccard": float((e.anchor & alternate).sum() / (e.anchor | alternate).sum()),
        }
    e["no_answer_fraction"] = e.no_answer_dissimilar_count / e.dissimilar_count
    e["no_answer_bin"] = pd.cut(
        e.no_answer_fraction, [-1, 0, 0.05, 0.2, 1], labels=["0", "(0, 5%]", "(5%, 20%]", ">20%"]
    )
    evidence["no_answer"] = (
        e.groupby("no_answer_bin", observed=True)
        .anchor.agg(["size", "sum", "mean"])
        .to_dict("index")
    )
    evidence["no_answer_spearman"] = spearmanr(e.importance_primary, e.no_answer_fraction).statistic

    # Reproduce the two existing transition models, including the stronger prefix baseline.
    assert d.groupby("trajectory_id").sentence_idx.apply(lambda x: x.is_monotonic_increasing).all()
    d["prev_stage"] = d.groupby("trajectory_id").stage.shift(1)
    mask = d.prev_stage.notna()
    d["y_true"] = (d.stage != d.prev_stage).astype(int)
    pos = ["sentence_idx", "generated_token_count", "prefix_context_fraction"]
    stages = ["orient", "plan", "compute", "verify", "backtrack", "conclude", "other"]
    pstg = (
        pd.get_dummies(d.prev_stage).reindex(columns=stages, fill_value=0).astype(float).to_numpy()
    )
    designs = {
        "activation": np.load(stage_dir / "activations_l44.npy"),
        "prefix_baseline": np.column_stack([pstg, d[pos].to_numpy(float)]),
    }
    train = mask & (d.split == "train")
    fitted = {}
    for name, X in designs.items():
        key = "act44" if name == "activation" else "oracle_prevstage_position"
        C = stage_results["transition_pred"][key]["C"]
        sc = StandardScaler().fit(X[train])
        model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(
            sc.transform(X[train]), d.loc[train, "y_true"]
        )
        assert model.n_iter_.max() < 3000
        fitted[name] = (sc, model)
    transition_rows = []
    for sp in SPLITS:
        ix = mask & (d.split == sp)
        part = d.loc[ix, ["sentence_id", "problem_id", "trajectory_id", "split", "y_true"]].copy()
        scs = {}
        for name, X in designs.items():
            sc, model = fitted[name]
            scs[name] = model.predict_proba(sc.transform(X[ix]))[:, 1]
            part[name] = scs[name]
            key = "act44" if name == "activation" else "oracle_prevstage_position"
            assert (
                round(ap(part.y_true, scs[name]), 4)
                == stage_results["transition_pred"][key][sp]["ap"]
            )
        seed = derived_seed(20260914, "blog_transition_bootstrap", sp)
        b = bootstrap(part, scs, seed)
        delta = b.activation - b.prefix_baseline
        ci = np.percentile(delta, [2.5, 97.5])
        evidence["transition"][sp] = {
            "n": len(part),
            "positives": int(part.y_true.sum()),
            "prevalence": part.y_true.mean(),
            "activation_ap": ap(part.y_true, scs["activation"]),
            "prefix_baseline_ap": ap(part.y_true, scs["prefix_baseline"]),
            "delta": ap(part.y_true, scs["activation"]) - ap(part.y_true, scs["prefix_baseline"]),
            "low": ci[0],
            "high": ci[1],
            "bootstrap_seed": seed,
            "C": {k: float(v[1].C) for k, v in fitted.items()},
        }
        b["split"], b["task"] = sp, "transition"
        bootrows.append(b)
        transition_rows.append(part)
    write_parquet(
        out / "transition_predictions.parquet",
        pa.Table.from_pandas(pd.concat(transition_rows), preserve_index=False),
    )
    write_parquet(
        out / "bootstrap_draws.parquet",
        pa.Table.from_pandas(pd.concat(bootrows), preserve_index=False),
    )
    write_parquet(out / "target_audit.parquet", pa.Table.from_pandas(e, preserve_index=False))
    evidence["source_sha256"] = sources
    evidence["script_sha256"] = digest(__file__)
    (out / "evidence.json").write_text(
        json.dumps(
            evidence, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)
        )
        + "\n"
    )
    figures(out, predictions, e, perm, cv, evidence, stage_results)
    print(
        json.dumps(
            {k: v for k, v in evidence.items() if k not in ["source_sha256", "script_sha256"]},
            indent=2,
        )
    )


def figures(out, predictions, e, perm, cv, ev, rs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    blue, orange, green, gray = "#2467aa", "#cc6b20", "#27806c", "#747474"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(out / "figures" / f"{name}.png", dpi=190, bbox_inches="tight")
        fig.savefig(out / "figures" / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, sp in zip(axes, SPLITS, strict=True):
        p = predictions[sp]
        for name, c, ls, label in [
            (PRIMARY, blue, "-", "Activations + observables"),
            ("position", green, "--", "Position"),
            ("observable_combined", orange, "-.", "Combined observables"),
        ]:
            g = p[p.probe_name == name].sort_values("score_raw", ascending=False, kind="stable")
            ax.step(
                np.r_[0, np.arange(1, len(g) + 1) / len(g)],
                np.r_[0, np.cumsum(g.y_true)],
                where="post",
                color=c,
                ls=ls,
                label=label,
            )
        ax.plot([0, 1], [0, g.y_true.sum()], ":", color=gray, label="Random-ranking expectation")
        ax.set(
            xlabel="Fraction of sentences inspected, highest scores first",
            ylabel="Actual anchors retrieved",
            title=f"{sp.title()}: {len(g)} sentences, {int(g.y_true.sum())} anchors",
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Can the ranking bring anchors to the top?")
    save(fig, "01_anchor_retrieval")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for sp, c, m in [("train", gray, "s"), ("validation", orange, "D"), ("test", blue, "o")]:
        tab = e[e.split == sp].groupby("position_bin").anchor.agg(["mean", "size", "sum"])
        axes[0].plot(tab.index, tab["mean"], marker=m, color=c, label=sp.title())
    axes[0].set(
        xticks=range(5),
        xticklabels=["0–25", "26–53", "54–86", "87–129", "130+"],
        xlabel="Sentence index (bins fixed on training data)",
        ylabel="Fraction labelled as anchors",
        ylim=(0, 0.23),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    axes[0].annotate(
        "Test: 15 anchors / 188 sentences",
        xy=(0, 15 / 188),
        xytext=(0.1, 0.2),
        arrowprops={"arrowstyle": "->", "color": gray},
        fontsize=8,
    )
    de = e[e.split.isin(SPLITS)]
    tab = de.groupby(["split", "problem_id"]).anchor.agg(["sum", "size"])
    for i, ((sp, _p), r) in enumerate(tab.iterrows()):
        axes[1].barh(i, r["sum"], color=blue if sp == "test" else orange)
        axes[1].text(
            r["sum"] + 0.35, i, f"{int(r['sum'])} / {int(r['size'])}", va="center", fontsize=9
        )
    axes[1].set(
        yticks=range(len(tab)),
        yticklabels=[f"{sp[:3]} · {p}" for sp, p in tab.index],
        xlabel="Number of anchors (annotation: anchors / sentences)",
        xlim=(0, 39),
    )
    fig.suptitle("Position predicts a sparse, unevenly distributed target")
    save(fig, "02_position_and_sample")

    fig, axes = plt.subplots(
        1, 3, figsize=(13.3, 4.1), layout="constrained", gridspec_kw={"width_ratios": [1, 1.35, 1]}
    )
    for sp, m in [("validation", "D"), ("test", "o")]:
        p = predictions[sp]
        for pid, g in p.groupby("problem_id"):
            a = g[g.probe_name == PRIMARY]
            b = g[g.probe_name == "observable_combined"]
            x, y = ap(b.y_true, b.score_raw), ap(a.y_true, a.score_raw)
            axes[0].scatter(x, y, marker=m, color=blue, facecolors="none" if sp == "test" else blue)
            axes[0].annotate(
                str(pid), (x, y), xytext=(3, 4), textcoords="offset points", fontsize=7
            )
    axes[0].plot([0, 0.45], [0, 0.45], ":", color=gray)
    axes[0].set(
        xlim=(0, 0.45),
        ylim=(0, 0.45),
        xlabel="AP: combined observables",
        ylabel="AP: activations + observables",
        title="Each point is one problem\nFilled diamonds: validation; open circles: test",
    )
    labs = []
    for i, (sp, base) in enumerate(
        [(sp, b) for sp in SPLITS for b in ["observable_combined", "position"]]
    ):
        v = ev["anchor"][sp]["comparisons"][base]
        c = blue if base == "observable_combined" else green
        axes[1].plot([v["low"], v["high"]], [i, i], color=c, lw=2)
        axes[1].scatter(v["delta"], i, color=c)
        labs.append(f"{sp[:3]} vs {'observables' if base == 'observable_combined' else 'position'}")
    axes[1].axvline(0, color=gray, ls=":")
    axes[1].set(
        yticks=range(4),
        yticklabels=labs,
        xlabel="Paired AP difference, 95% problem bootstrap",
        title="All four intervals include zero",
        xlim=(-0.26, 0.12),
    )
    n = perm[(perm.split == "test") & (perm.kind == "permutation")].average_precision
    axes[2].hist(n, bins=16, color="#aacbe8", edgecolor="white")
    axes[2].axvline(
        ev["anchor"]["test"]["ap"][PRIMARY], color=blue, lw=2, label="Observed AP = 0.0941"
    )
    axes[2].set(
        xlabel="Test AP after stratified training-label shuffle",
        ylabel="Number of refits",
        title="100 permutations; descriptive p = 0.168",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle("The anchor experiment does not establish incremental activation information")
    save(fig, "03_incremental_information")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    vals = [1, ev["smoothing"]["0_5"]["jaccard"], ev["smoothing"]["1_0"]["jaccard"]]
    axes[0].bar(range(3), vals, color=[gray, blue, blue])
    axes[0].set(
        xticks=range(3),
        xticklabels=["Original\nα = 10⁻⁹", "α = 0.5", "α = 1"],
        ylim=(0, 1.14),
        ylabel="Anchor-set overlap (intersection / union)",
        title="Changing smoothing changes anchor membership",
    )
    axes[0].axhline(1, color=gray, ls=":")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center")
    tab = e.groupby("no_answer_bin", observed=True).anchor.agg(["size", "sum", "mean"])
    axes[1].bar(range(4), tab["mean"], color=orange)
    for i, (_, r) in enumerate(tab.iterrows()):
        axes[1].text(
            i, r["mean"] + 0.012, f"{int(r['sum'])}/{int(r['size'])}", ha="center", fontsize=9
        )
    axes[1].axhline(e.anchor.mean(), color=gray, ls=":", label="All eligible rows: 8.6%")
    axes[1].set(
        xticks=range(4),
        xticklabels=tab.index,
        xlabel="Fraction of dissimilar continuations with no parsed answer",
        ylabel="Fraction labelled as anchors",
        ylim=(0, 0.56),
        title="Missing answers are strongly associated with anchors",
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8)
    fig.suptitle("The measured target depends strongly on estimator choices and missing answers")
    save(fig, "04_target_validity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for i, sp in enumerate(SPLITS):
        v = ev["transition"][sp]
        axes[0].scatter(
            v["prevalence"],
            i,
            marker="|",
            s=190,
            color="black",
            label="Prevalence" if i == 0 else None,
        )
        axes[0].scatter(
            v["prefix_baseline_ap"],
            i,
            marker="s",
            s=60,
            color=green,
            label="Previous stage + position" if i == 0 else None,
        )
        axes[0].scatter(
            v["activation_ap"],
            i,
            marker="o",
            s=60,
            color=blue,
            label="Activations, site 44" if i == 0 else None,
        )
        for key, dy in [("prefix_baseline_ap", -0.17), ("activation_ap", 0.17)]:
            axes[0].text(v[key], i + dy, f"{v[key]:.4f}", ha="center", fontsize=9)
        axes[1].plot([v["low"], v["high"]], [i, i], color=blue, lw=2)
        axes[1].scatter(v["delta"], i, color=blue)
        axes[1].text(
            v["delta"],
            i + 0.18,
            f"+{v['delta']:.4f} [{v['low']:.4f}, {v['high']:.4f}]",
            ha="center",
            fontsize=9,
        )
    axes[0].set(
        yticks=[0, 1],
        yticklabels=["Validation", "Test"],
        xlim=(0.4, 0.96),
        ylim=(-0.5, 1.8),
        xlabel="Transition average precision",
        title="The strong visible-prefix baseline explains much of the score",
    )
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].axvline(0, color=gray, ls=":")
    axes[1].set(
        yticks=[0, 1],
        yticklabels=["Validation", "Test"],
        ylim=(-0.5, 1.8),
        xlim=(-0.015, 0.11),
        xlabel="Activation AP − prefix-baseline AP",
        title="A smaller positive difference remains on both splits",
    )
    fig.suptitle(
        "Upcoming marker-defined transitions are predictable beyond previous stage and position"
    )
    save(fig, "05_transition_control")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    for i, sp in enumerate(SPLITS):
        ax = axes[i]
        v = ev["lead"][sp]
        ax.scatter(
            [0, 1],
            [v["0"]["ap"], v["1"]["ap"]],
            color=blue,
            s=65,
            label="Activations + observables",
        )
        ax.axhline(v["0"]["prevalence"], ls=":", color=gray, label="Matched-row prevalence")
        for k in ["0", "1"]:
            ax.annotate(
                f"AP {v[k]['ap']:.4f}",
                (int(k), v[k]["ap"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Immediately before Sᵢ", "Before Sᵢ₋₁"],
            xlim=(-0.4, 1.4),
            ylim=(0, 0.215),
            ylabel="Anchor average precision",
            title=f"{sp.title()}: {v['0']['n']} matched sentences, {v['0']['positives']} anchors",
        )
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Moving one sentence earlier gives conflicting results across splits")
    save(fig, "06_anchor_lead_time")

    fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
    a = cv[(cv.probe_name == PRIMARY) & (cv.task == "binary")]
    best = (
        a.groupby(["layer", "hyperparameter"])
        .average_precision.mean()
        .reset_index()
        .sort_values("average_precision")
        .groupby("layer")
        .tail(1)
        .sort_values("layer")
    )
    for _, r in best.iterrows():
        fs = a[(a.layer == r.layer) & (a.hyperparameter == r.hyperparameter)].average_precision
        ax.scatter(np.repeat(r.layer, len(fs)), fs, s=10, color=blue, alpha=0.4)
    ax.plot(
        best.layer,
        best.average_precision,
        color=blue,
        label="Mean of 4 training folds, best C at each site",
    )
    ax.axhline(0.1, color=gray, ls=":", label="Pooled training prevalence (fold prevalence varies)")
    ax.axvline(44, color=orange, ls="--", label="Selected site 44")
    ax.set(
        xlabel="Activation site: 0 = token embedding; 1–48 = decoder outputs",
        ylabel="Anchor average precision in training CV",
        title="Fold variation is much larger than the layer trend",
    )
    ax.legend(fontsize=8)
    save(fig, "S1_anchor_layers")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
