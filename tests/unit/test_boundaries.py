"""SPEC §10.2 structural boundary checks on synthetic offsets and the tiny tokenizer."""

import sys
from pathlib import Path

import pytest

from preanchor.model.boundaries import (
    audit_boundaries,
    boundary_for,
    boundary_mapping_sha256,
    boundary_table,
    tokenize_trace,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from tiny_qwen2 import tiny_tokenizer  # noqa: E402

# "ab cd" -> tokens "ab", " cd" (a leading-space token overlaps a sentence start at 3)
OFFSETS = [(0, 2), (2, 5)]


def test_overlapping_token_belongs_to_target():
    start, boundary, end, overlap = boundary_for(OFFSETS, 3, 5)
    assert (start, boundary, end, overlap) == (1, 0, 2, True)


def test_exact_start_is_not_overlapping():
    start, boundary, end, overlap = boundary_for([(0, 2), (2, 3), (3, 5)], 3, 5)
    assert (start, boundary, end, overlap) == (2, 1, 3, False)


def test_first_token_sentence_rejected():
    with pytest.raises(ValueError, match="boundary_token_idx < 0"):
        boundary_for(OFFSETS, 0, 2)


def test_span_outside_trace_rejected():
    with pytest.raises(ValueError):
        boundary_for(OFFSETS, 4, 9)


def test_table_and_audit_with_character_tokenizer():
    tok = tiny_tokenizer()
    trace = "Q: 1\n<think>\nOkay. Let me. Done."
    ids, offsets = tokenize_trace(tok, trace)
    assert tok.decode(ids) == trace and len(ids) == len(trace)
    rows = [{"sentence_id": f"t:{i}", "sentence_idx": i, "char_start": trace.index(s),
             "char_end": trace.index(s) + len(s)}
            for i, s in enumerate(["Okay.", "Let me.", "Done."])]
    table = boundary_table(offsets, rows)
    assert [b.boundary_token_idx for b in table] == [trace.index("Okay.") - 1,
                                                     trace.index("Let me.") - 1,
                                                     trace.index("Done.") - 1]
    assert all(b.prefix_token_count == b.boundary_token_idx + 1 for b in table)
    audit = audit_boundaries(ids, offsets, table)
    assert audit["n_sentences"] == 3 and audit["overlapping_tokens"] == 0
    assert audit["boundary_mapping_sha256"] == boundary_mapping_sha256(table)
    with pytest.raises(ValueError, match="contiguous"):
        boundary_table(offsets, rows[1:])


def test_audit_detects_prefix_token_crossing_boundary():
    from preanchor.model.boundaries import Boundary

    bad = [Boundary("t:0", 0, 3, 5, 2, 1, 2, False, 2)]  # token 1 (2,5) ends after char 3
    with pytest.raises(ValueError, match="Prefix token ends after"):
        audit_boundaries([1, 2, 3], [(0, 2), (2, 5), (5, 6)], bad)
