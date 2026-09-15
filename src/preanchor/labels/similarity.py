"""Replacement-sentence dissimilarity with the pinned MiniLM model (SPEC §3.6).

For every fork-i resample the comparison is between the original sentence
(``chunk_removed``) and the model's replacement (``chunk_resampled``); a sample is
dissimilar when cosine similarity is strictly below the threshold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
THRESHOLD = 0.8


def replacement_pairs(rows: Sequence[dict]) -> list[tuple[str, str]]:
    pairs = []
    for index, row in enumerate(rows):
        removed, resampled = row.get("chunk_removed"), row.get("chunk_resampled")
        if not isinstance(removed, str) or not isinstance(resampled, str):
            raise ValueError(f"Sample {index} lacks string chunk_removed/chunk_resampled")
        pairs.append((removed, resampled))
    return pairs


class ReplacementSimilarity:
    """Batched cosine similarity between original and replacement sentences."""

    def __init__(self, *, device: str = "cpu", model_id: str = SIMILARITY_MODEL,
                 revision: str = SIMILARITY_REVISION, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model_id, self.revision, self.device = model_id, revision, device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, revision=revision, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            vectors = self.model.encode(missing, batch_size=self.batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True,
                                        show_progress_bar=False)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = np.asarray(vector, dtype=np.float32)
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1))

    def clear(self) -> None:
        self._cache.clear()

    def scores(self, rows: Sequence[dict]) -> list[float]:
        """Cosine similarity per sample, aligned with ``rows``."""
        pairs = replacement_pairs(rows)
        if not pairs:
            return []
        removed = self.embed([p[0] for p in pairs]).astype(np.float64)
        resampled = self.embed([p[1] for p in pairs]).astype(np.float64)
        values = np.einsum("ij,ij->i", removed, resampled)
        return [float(v) for v in values]

    def identity(self) -> dict:
        return {"model_id": self.model_id, "model_revision": self.revision,
                "device": self.device, "threshold": THRESHOLD,
                "comparison": "cosine(chunk_removed, chunk_resampled) < threshold"}


def similarity_fingerprint(scores: Sequence[float]) -> str:
    """Stable digest of a score vector for provenance records."""
    return hashlib.sha256(np.asarray(scores, dtype=np.float64).tobytes()).hexdigest()
