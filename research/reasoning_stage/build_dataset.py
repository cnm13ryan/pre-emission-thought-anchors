# ruff: noqa: E501
"""Build the merged sentence-level dataset for the reasoning-stage analysis.

Joins: stage3 sentences (text, importance) + stage labels + stage7 observable
features + pre-emission activations (layers 0 and 44) + split assignment from
the resolved MVP config. Read-only against the artifact store.

Writes research/reasoning_stage/outputs/dataset.parquet (metadata + features)
and activations_l0.npy / activations_l44.npy row-aligned with it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = "/Users/cnm13ryan/preanchor-root"
ACT = f"{ROOT}/artifacts/runs/20260915T182306Z-3ab408d80a-7452f1c2/activations"
OUT = Path(__file__).parent / "outputs"

SPLITS = {
    "train": [2238, 1591, 4682, 6596, 3448, 2189, 3550, 4605, 2137, 4019, 3360, 3935],
    "validation": [330, 4164, 2050, 6481],
    "test": [6998, 2236, 2870, 3916],
}
ANCHOR_THRESHOLD = 1.3549810893213545  # frozen_selection.json, q90 of train eligible


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # stage4 labels.parquet is the pipeline's label source: importance_primary is the
    # canonical importance (canonical_fallback) and `anchor` the frozen q90 label.
    s = pd.read_parquet(f"{ROOT}/data/processed/stage4/labels.parquet")[
        ["sentence_id", "trajectory_id", "problem_id", "trace_kind", "sentence_idx",
         "text", "eligible", "exclusion_reason", "importance_primary", "importance_log1p",
         "label_source", "anchor", "sample_count", "dissimilar_count", "fork_accuracy"]
    ]
    stages = pd.read_parquet(OUT / "stage_labels.parquet")[["sentence_id", "stage"]]
    feats = pd.read_parquet(f"{ROOT}/data/processed/stage7/features.parquet")
    ix = pd.read_parquet(f"{ACT}/index.parquet")[["sentence_id", "shard_path", "row_idx"]]
    df = (
        s.merge(stages, on="sentence_id")
        .merge(feats, on="sentence_id", suffixes=("", "_f"))
        .merge(ix, on="sentence_id", suffixes=("", "_ix"))
    )
    assert len(df) == len(s) == 6474, len(df)
    prob2split = {p: sp for sp, ps in SPLITS.items() for p in ps}
    df["split"] = df["problem_id"].map(prob2split)
    assert df["split"].notna().all()
    df["anchor"] = df["anchor"].astype(int)

    n = len(df)
    acts0 = np.empty((n, 5120), dtype=np.float32)
    acts44 = np.empty((n, 5120), dtype=np.float32)
    for shard, g in df.groupby("shard_path"):
        with safe_open(f"{ACT}/{shard}", framework="numpy") as f:
            r = f.get_tensor("resid")  # (rows, 49, 5120) float16
        acts0[g.index.values] = r[g["row_idx"].values, 0, :].astype(np.float32)
        acts44[g.index.values] = r[g["row_idx"].values, 44, :].astype(np.float32)

    keep = [
        "sentence_id", "trajectory_id", "problem_id", "trace_kind", "sentence_idx",
        "text", "stage", "split", "eligible", "exclusion_reason", "anchor",
        "importance_primary", "importance_log1p", "label_source", "sample_count", "dissimilar_count",
        "generated_token_count", "prefix_token_count", "log1p_prefix_token_count",
        "prefix_context_fraction", "previous_sentence_tokens",
        "previous_emitted_token_logprob", "next_token_entropy",
        "next_token_max_probability", "next_token_probability_margin",
        "next_token_top5_mass", "next_token_top10_mass",
    ]
    df[keep].to_parquet(OUT / "dataset.parquet", index=False)
    np.save(OUT / "activations_l0.npy", acts0)
    np.save(OUT / "activations_l44.npy", acts44)
    print(df.groupby("split").agg(n=("anchor", "size"), eligible=("eligible", "sum"),
                                  anchors=("anchor", lambda x: int(x[df.loc[x.index, "eligible"]].sum()))))
    print("saved", len(df), "rows")


if __name__ == "__main__":
    main()
