import hashlib
import os
import select
import signal
import subprocess
import sys

import pytest

from preanchor.io import UnitStore, write_json
from preanchor.runtime import Runtime, StageInterrupted


def test_sigkill_releases_lock_and_preserves_completed_bytes(tmp_path):
    key_done = hashlib.sha256(b'done').hexdigest()
    key_partial = hashlib.sha256(b'partial').hexdigest()
    store = UnitStore(tmp_path)
    done, _ = store.produce(key_done, lambda p: write_json(p / 'value.json', {'done': True}))
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in done.iterdir()}
    code = '''
import sys, time
from pathlib import Path
from preanchor.io import UnitStore, write_json

def writer(path):
    write_json(path / 'value.json', {'partial': True})
    print('partial-ready', flush=True)
    time.sleep(30)
UnitStore(Path(sys.argv[1])).produce(sys.argv[2], writer)
'''
    process = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), key_partial],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 10)[0], 'Worker failed to become ready'
        assert process.stdout.readline().strip() == 'partial-ready'
        process.kill()
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    _, created = store.produce(key_done, lambda _: pytest.fail('completed unit rewritten'))
    assert not created
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in done.iterdir()}
    _, created = store.produce(
        key_partial, lambda p: write_json(p / 'value.json', {'complete': True}))
    assert created
    assert len(list((tmp_path / 'quarantine').iterdir())) == 1


def test_sigterm_cooperates_with_main_loop_and_restores_handler(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(StageInterrupted, match='SIGTERM'), Runtime(tmp_path, 'test', 1) as runtime:
        os.kill(os.getpid(), signal.SIGTERM)
        runtime.pause(0.1)
    assert signal.getsignal(signal.SIGTERM) == before
    assert (tmp_path / 'heartbeat.json').exists()
