from preanchor.labels.importance import normalize_answer
def test_answer_normalization():
    assert normalize_answer(r' \boxed{ 42 } ') == '42'
    assert normalize_answer(None) == '<NO_ANSWER>'
