from preanchor.labels.importance import canonical_from_forks
def test_adjacent_fork_direction():
    assert canonical_from_forks({0:[{'chunk_removed':True,'answer':'a'}],1:[{'chunk_removed':True,'answer':'a'}]},0) == 0
