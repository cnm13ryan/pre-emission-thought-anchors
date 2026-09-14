import json
from pathlib import Path
import subprocess
import sys
from preanchor.io import UnitStore

def test_sentence_unit_resume_preserves_checksum(tmp_path):
 s=UnitStore(tmp_path); key='a'*64
 def w(d): (d/'result.json').write_text('{"sentence_id":"x"}')
 p,created=s.produce(key,w); before=(p/'result.json').read_bytes(); p2,created2=s.produce(key,w)
 assert not created2 and before==(p2/'result.json').read_bytes()

def test_sentence_unit_kill_resume_preserves_completed_and_recomputes_partial(tmp_path):
    script = '''
import sys,time
from pathlib import Path
from preanchor.io import UnitStore
s=UnitStore(Path(sys.argv[1]))
for i in range(4):
    k=f"{i+1:064x}"
    s.produce(k, lambda d,i=i: (d/'result.json').write_text('{"i":%d}'%i))
    time.sleep(.25)
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)])
    import time
    time.sleep(.75)
    child.terminate()
    child.wait(timeout=10)
    # The interruption occurs after the first work-unit boundaries; completed
    # units remain byte-identical on resume.
    before = {f"{i+1:064x}": (tmp_path/f"{i+1:064x}"/'result.json').read_bytes()
              for i in range(2)}
    store = UnitStore(tmp_path)
    for i in range(4):
        k=f"{i+1:064x}"
        _, created = store.produce(k, lambda d,i=i: (d/'result.json').write_text('{"i":%d}'%i))
        if i < 2:
            assert not created
    assert all((tmp_path/k/'result.json').read_bytes() == v for k,v in before.items())

    # Corruption is quarantined and only that unit is regenerated.
    target = tmp_path/'0000000000000000000000000000000000000000000000000000000000000003'/'result.json'
    target.write_text('{corrupt')
    _, created = store.produce('0000000000000000000000000000000000000000000000000000000000000003',
                               lambda d: (d/'result.json').write_text('{"i":2}'))
    assert created
    assert (tmp_path/'0000000000000000000000000000000000000000000000000000000000000003'/'result.json').read_text() == '{"i":2}'
