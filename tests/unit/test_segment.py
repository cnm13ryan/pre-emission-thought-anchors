import pytest

from preanchor.data.segment import align_sequential, span_v2


def test_sequential_alignment_repeated_text():
    text = "<think>\nCheck it.\nCheck it.\nDone."
    spans = align_sequential(text, ["Check it.", "Check it.", "Done."], 7)
    assert [(s.start, s.end) for s in spans] == [(8, 17), (18, 27), (28, 33)]


def test_alignment_accepts_whitespace_only_difference():
    spans = align_sequential("A  value is here.", ["A value is here."])
    assert spans[0].method == "whitespace_tolerant"
    assert spans[0].text == "A  value is here."


def test_alignment_rejects_unresolved_chunk():
    with pytest.raises(ValueError, match="Unresolved"):
        align_sequential("abc", ["missing"])


def test_span_v2_retains_terminal_fragment():
    assert [s.text for s in span_v2("One. Two")] == ["One.", "Two"]
