"""Structural validation for Stage 4 label tables."""

from __future__ import annotations

import math

PRIMARY_SOURCES = {"released", "canonical_fallback"}
REQUIRED = {"sentence_id", "problem_id", "trace_kind", "sentence_idx", "eligible",
            "exclusion_reason", "importance_released", "importance_legacy",
            "importance_canonical", "importance_primary", "label_source"}


def _finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value >= 0


def validate_label_rows(rows: list[dict], *, primary_source: str) -> dict:
    """Every row carries one uniform primary source; only eligible rows carry labels."""
    if primary_source not in PRIMARY_SOURCES:
        raise ValueError("invalid primary source")
    if not rows:
        raise ValueError("empty label table")
    eligible = 0
    seen = set()
    for row in rows:
        if not REQUIRED <= row.keys():
            raise ValueError(f"label schema missing fields: {sorted(REQUIRED - row.keys())}")
        if row["sentence_id"] in seen:
            raise ValueError("duplicate sentence_id")
        seen.add(row["sentence_id"])
        if row["label_source"] != primary_source:
            raise ValueError("mixed primary sources")
        if row["eligible"]:
            if row["exclusion_reason"] is not None:
                raise ValueError("eligible row carries an exclusion reason")
            if not _finite_nonnegative(row["importance_primary"]):
                raise ValueError("invalid primary label")
            if not _finite_nonnegative(row["importance_canonical"]):
                raise ValueError("eligible row lacks a finite canonical label")
            if primary_source == "canonical_fallback" \
                    and row["importance_primary"] != row["importance_canonical"]:
                raise ValueError("primary label must equal the canonical label")
            if primary_source == "released" \
                    and row["importance_primary"] != row["importance_released"]:
                raise ValueError("primary label must equal the released label")
            eligible += 1
        else:
            if not row["exclusion_reason"]:
                raise ValueError("ineligible row lacks an exclusion reason")
            if row["importance_primary"] is not None:
                raise ValueError("ineligible row must not carry a primary label")
    return {"schema_version": 1, "rows": len(rows), "eligible": eligible,
            "primary_source": primary_source}
