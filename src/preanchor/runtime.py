"""Main-loop heartbeat and graceful signal handling for resumable stages."""

import signal
import time
from pathlib import Path

from preanchor.io import atomic_bytes, json_bytes
from preanchor.logging import redact
from preanchor.manifest import utc_now


class StageInterrupted(Exception):
    pass


class Runtime:
    def __init__(self, directory: Path, stage: str, total: int, tracking=None):
        self.directory, self.stage, self.total = directory, stage, total
        self.tracking = tracking
        self.started = time.monotonic()
        self.last_pulse = -float("inf")
        self.completed = 0
        self.last_output = None
        self.last_error = None
        self.stop_signal = None
        self.old_handlers = {}

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal)
        self.pulse(force=True)
        return self

    def _signal(self, signum, frame):
        self.stop_signal = signal.Signals(signum).name

    def check_stop(self):
        if self.stop_signal:
            raise StageInterrupted(self.stop_signal)

    def pulse(self, *, force=False, state="running"):
        self.check_stop()
        now = time.monotonic()
        if not force and now - self.last_pulse < 30:
            return
        elapsed = now - self.started
        rate = self.completed / elapsed if elapsed > 0 else 0
        payload = {"schema_version": 1, "updated_at": utc_now(), "stage": self.stage,
                   "state": state, "completed": self.completed, "total": self.total,
                   "last_durable_output": self.last_output, "rate_per_second": rate,
                   "eta_seconds": (self.total - self.completed) / rate if rate else None,
                   "last_error": self.last_error, "elapsed_seconds": elapsed}
        atomic_bytes(self.directory / "heartbeat.json", json_bytes(redact(payload)), replace=True)
        if self.tracking:
            try:
                self.tracking.log({"stage/completed": self.completed, "stage/total": self.total,
                                   "stage/rate": rate, "stage/elapsed_seconds": elapsed})
            except Exception:
                pass  # Local heartbeat remains authoritative if a telemetry send fails.
        self.last_pulse = now

    def pause(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pulse()
            time.sleep(min(0.1, max(0, end - time.monotonic())))
        self.check_stop()

    def __exit__(self, exc_type, exc, tb):
        for signum, handler in self.old_handlers.items():
            signal.signal(signum, handler)
