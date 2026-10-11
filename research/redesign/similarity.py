"""Phase 1 (redesign): replacement similarity per reacquired rollout with the pinned MiniLM.

Same comparison as ``preanchor.labels.similarity`` (cosine of normalised embeddings of
``chunk_removed`` and ``chunk_resampled``), run on this machine's CPU/MPS rather than the
CUDA device of Stage 4. Borderline samples near 0.8 may therefore flip; the reproduction
gate in ``phase1_labels.py`` measures how many.

Usage: python research/redesign/similarity.py --root DIR [--device mps]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
COLUMNS = ["trajectory_id", "fork_idx", "sample_idx", "chunk_removed", "chunk_resampled"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    files = sorted(glob.glob(str(args.root / "rollouts/*/fork_*.parquet")))
    frame = pd.concat([pd.read_parquet(f, columns=COLUMNS) for f in files], ignore_index=True)
    valid = frame.chunk_removed.notna() & frame.chunk_resampled.notna()
    texts = sorted(set(frame.loc[valid, "chunk_removed"]) | set(frame.loc[valid, "chunk_resampled"]))
    print(f"{len(frame)} samples, {len(texts)} unique texts", flush=True)
    model = SentenceTransformer(MODEL, revision=REVISION, device=args.device)
    vectors = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    index = {t: i for i, t in enumerate(texts)}
    sim = np.full(len(frame), np.nan)
    a = frame.loc[valid, "chunk_removed"].map(index).to_numpy()
    b = frame.loc[valid, "chunk_resampled"].map(index).to_numpy()
    sim[valid.to_numpy()] = np.einsum("ij,ij->i", vectors[a].astype(np.float64),
                                      vectors[b].astype(np.float64))
    out = frame[["trajectory_id", "fork_idx", "sample_idx"]].copy()
    out["similarity"] = sim
    out.to_parquet(args.root / "similarity.parquet", index=False)
    print({"samples": len(out), "nan": int(np.isnan(sim).sum()), "device": args.device})


if __name__ == "__main__":
    main()
