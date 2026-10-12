"""Replacement similarity per rollout: cosine(MiniLM(chunk_removed), MiniLM(chunk_resampled)).

Same comparison and pinned model as the MVP (sentence-transformers/all-MiniLM-L6-v2 at
1110a243fdf4706b3f48f1d95db1a4f5529b4d41; dissimilar iff cosine < 0.8). Adapted from
``research/redesign/similarity.py`` (this repository, uncommitted at f11823f).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arv2.paths import DERIVED  # noqa: E402

COLS = ["trajectory_id", "fork_idx", "sample_idx", "chunk_removed", "chunk_resampled"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    files = sorted(glob.glob(str(DERIVED / "rollouts/*/fork_*.parquet")))
    if len(files) != 6474:
        raise SystemExit(f"expected 6474 fork files, found {len(files)}")
    f = pd.concat([pd.read_parquet(x, columns=COLS) for x in files], ignore_index=True)
    ok = f.chunk_removed.notna() & f.chunk_resampled.notna()
    texts = sorted(set(f.loc[ok, "chunk_removed"]) | set(f.loc[ok, "chunk_resampled"]))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                device=args.device)
    vec = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    pos = {t: i for i, t in enumerate(texts)}
    sim = np.full(len(f), np.nan)
    a = f.loc[ok, "chunk_removed"].map(pos).to_numpy()
    b = f.loc[ok, "chunk_resampled"].map(pos).to_numpy()
    sim[ok.to_numpy()] = np.einsum("ij,ij->i", vec[a].astype(np.float64),
                                   vec[b].astype(np.float64))
    out = f[["trajectory_id", "fork_idx", "sample_idx"]].copy()
    out["similarity"] = sim
    out.to_parquet(DERIVED / "similarity.parquet", index=False)
    print({"samples": len(out), "nan": int(np.isnan(sim).sum()), "unique_texts": len(texts)})


if __name__ == "__main__":
    main()
