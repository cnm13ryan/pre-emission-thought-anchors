import json
from preanchor.labels.recompute import recompute_sentence

def test_recompute_and_missing(tmp_path):
    rows=[{'chunk_removed':'x','answer':'1'},{'chunk_removed':'x','answer':'2'}]
    (tmp_path/'a.json').write_text(json.dumps(rows)); (tmp_path/'b.json').write_text(json.dumps(rows))
    out=recompute_sentence(0,{0:tmp_path/'a.json',1:tmp_path/'b.json'})
    assert out['eligible'] and out['importance_primary'] >= 0
    assert recompute_sentence(1,{0:tmp_path/'a.json'})['exclusion_reason']=='missing_adjacent_fork'
