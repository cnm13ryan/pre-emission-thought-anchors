from preanchor.labels.importance import smoothed_kl
def test_hand_kl_is_nonnegative():
    assert smoothed_kl(['a','a'],['a','a']) == 0
    assert smoothed_kl(['a','a'],['b','b']) > 0
