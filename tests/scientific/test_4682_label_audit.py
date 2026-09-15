"""Stage 4 label-table gate: uniform primary source, masked exclusions, validated rows."""

import pytest

from preanchor.labels.validation import validate_label_rows


def row(**overrides):
    base = {"sentence_id": "1:correct:0", "problem_id": 1, "trace_kind": "correct",
            "sentence_idx": 0, "eligible": True, "exclusion_reason": None,
            "importance_released": 0.5, "importance_legacy": 0.4, "importance_canonical": 0.3,
            "importance_primary": 0.3, "label_source": "canonical_fallback"}
    base.update(overrides)
    return base


def test_primary_fallback_schema():
    assert validate_label_rows([row()], primary_source="canonical_fallback")["eligible"] == 1


def test_released_primary_must_equal_released_value():
    ok = row(label_source="released", importance_primary=0.5)
    assert validate_label_rows([ok], primary_source="released")["rows"] == 1
    with pytest.raises(ValueError, match="must equal the released"):
        validate_label_rows([row(label_source="released", importance_primary=0.3)],
                            primary_source="released")


def test_mixed_sources_and_unmasked_exclusions_are_rejected():
    with pytest.raises(ValueError, match="mixed primary sources"):
        validate_label_rows([row(), row(sentence_id="1:correct:1", label_source="released")],
                            primary_source="canonical_fallback")
    with pytest.raises(ValueError, match="ineligible row must not carry"):
        validate_label_rows([row(eligible=False, exclusion_reason="missing_adjacent_fork")],
                            primary_source="canonical_fallback")
    with pytest.raises(ValueError, match="lacks an exclusion reason"):
        validate_label_rows([row(eligible=False, importance_primary=None)],
                            primary_source="canonical_fallback")
    with pytest.raises(ValueError, match="invalid primary label"):
        bad = row(importance_primary=float("nan"), importance_canonical=float("nan"))
        validate_label_rows([bad], primary_source="canonical_fallback")
    with pytest.raises(ValueError, match="duplicate"):
        validate_label_rows([row(), row()], primary_source="canonical_fallback")
