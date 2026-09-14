import json
from pathlib import Path
from preanchor.io import UnitStore

def test_sentence_unit_resume_preserves_checksum(tmp_path):
 s=UnitStore(tmp_path); key='a'*64
 def w(d): (d/'result.json').write_text('{"sentence_id":"x"}')
 p,created=s.produce(key,w); before=(p/'result.json').read_bytes(); p2,created2=s.produce(key,w)
 assert not created2 and before==(p2/'result.json').read_bytes()
