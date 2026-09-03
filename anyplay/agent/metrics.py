"""Per-stage latency + drop tracking for the C++ core (P0 instrumentation).

Every stage of the capture -> decision -> input loop appends a
``{ts, stage, duration_ms}`` record to a single ``metrics.jsonl`` file.
This is the unified observability sink required by the implementation plan
(§5.2 P0: "every stage logs stage, ts, duration_ms to metrics.jsonl"), and it
is what the later latency budget (plan §6) and the P5 "metrics.jsonl clean"
exit criterion are measured against.

Stages recorded
---------------
From the shm read path (``native_capture.NativeCapture.get_frame``):

* ``capture_wait``    producer-side wait for a source sample (ns stamp in the
                      ring) — i.e. the source's frame cadence.
* ``ring_residence``  time a published frame sat in the ring before being
                      read (consumer lag; ~0 when the consumer keeps up).
* ``frame_read``      wall time the consumer spent in ``get_frame``.

From the decision loop (``agent.Agent.step``):

* ``obs_fetch``       temporal observation acquisition.
* ``inference``       VL model decide.
* ``input``           validated action dispatch to the input controller.
* ``step_total``      full decision-step wall time.

Rule 2 (no disk I/O on the real-time frame path) is respected: records are
buffered in memory and flushed to disk at most once per ``flush_interval``
(default 1 s) and on close — never per frame.

The file path is resolved (in order) from an explicit argument, the
``SDAI_METRICS`` environment variable, then ``metrics.jsonl`` in the CWD.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

DEFAULT_WINDOW = 2000


def default_metrics_path() -> Path:
    """Shared default metrics file, overridable via ``SDAI_METRICS``."""
    return Path(os.environ.get("SDAI_METRICS") or "metrics.jsonl")


def _percentile(sorted_vals: list, q: float) -> float:
    """Linear-interpolation percentile of an ascending-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class MetricsRecorder:
    """Thread-safe, time-gated per-stage latency + drop-rate recorder.

    Safe to share across the capture thread and the decision thread; every
    call is lock-protected and the disk flush is bounded to at most one per
    ``flush_interval`` so the real-time frame path never blocks on I/O.
    """

    def __init__(
        self,
        path: Optional[Any] = None,
        flush_interval: float = 1.0,
        window: int = DEFAULT_WINDOW,
    ):
        self.path = Path(path) if path is not None else default_metrics_path()
        self.flush_interval = max(0.05, float(flush_interval))
        self.window = max(16, int(window))

        self._lock = threading.Lock()
        self._buf: list = []            # pending records awaiting flush
        self._windows: dict = {}        # stage -> deque of duration_ms
        self._last_flush = time.monotonic()
        self._record_count = 0
        self._closed = False

        # Cumulative daemon counters (frames, drops) for drop-rate tracking.
        self._stats_first: Optional[tuple] = None   # (frames, drops, t_mono)
        self._stats_last: Optional[tuple] = None    # (frames, drops, t_mono, fps)

        # Best-effort: make sure the parent directory exists.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, stage: str, duration_ms: float, **extra: Any) -> None:
        """Append one latency sample for ``stage`` (milliseconds)."""
        if self._closed:
            return
        rec = {
            "ts": round(time.time(), 6),
            "stage": stage,
            "duration_ms": round(float(duration_ms), 3),
        }
        if extra:
            rec.update(extra)
        due = False
        with self._lock:
            self._buf.append(rec)
            self._record_count += 1
            dq = self._windows.get(stage)
            if dq is None:
                dq = deque(maxlen=self.window)
                self._windows[stage] = dq
            dq.append(float(duration_ms))
            due = (time.monotonic() - self._last_flush) >= self.flush_interval
        if due:
            self._flush()

    def observe_stats(self, frames: int, fps: float, drops: int) -> None:
        """Store a snapshot of cumulative daemon counters for drop-rate.

        Updates the in-memory window (see ``summary``) and also appends a
        lightweight ``{"type": "stats"}`` sample so an offline ``core-status``
        can reconstruct a session drop window from the file alone.
        """
        now = time.monotonic()
        with self._lock:
            if self._stats_first is None:
                self._stats_first = (frames, drops, now)
            self._stats_last = (frames, drops, now, float(fps))
            self._buf.append({
                "ts": round(time.time(), 6),
                "type": "stats",
                "frames": int(frames),
                "fps": round(float(fps), 3),
                "drops": int(drops),
            })

    # ------------------------------------------------------------------
    # Flush (bounded, off the hot path)
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        with self._lock:
            if not self._buf:
                self._last_flush = time.monotonic()
                return
            batch = self._buf
            self._buf = []
            self._last_flush = time.monotonic()
        try:
            with open(self.path, "a", buffering=1) as f:
                for r in batch:
                    f.write(json.dumps(r) + "\n")
        except Exception:
            # Never crash the capture/decision path on a metrics write;
            # requeue the batch so the samples are not lost.
            with self._lock:
                self._buf = batch + self._buf

    # ------------------------------------------------------------------
    # Summary / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """In-memory summary: per-stage percentiles + drop-rate window."""
        with self._lock:
            stages = {}
            for stage, dq in self._windows.items():
                vals = sorted(dq)
                stages[stage] = {
                    "n": len(vals),
                    "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
                    "p50": round(_percentile(vals, 0.50), 3),
                    "p95": round(_percentile(vals, 0.95), 3),
                    "p99": round(_percentile(vals, 0.99), 3),
                    "max": round(vals[-1], 3) if vals else 0.0,
                }
            drop = self._drop_info_locked()
            total = self._record_count
        return {
            "path": str(self.path),
            "record_count": total,
            "stages": stages,
            "drop": drop,
        }

    def _drop_info_locked(self) -> Optional[dict]:
        if self._stats_first is None or self._stats_last is None:
            return None
        f0, d0, t0 = self._stats_first
        f1, d1, t1, fps = self._stats_last
        df = f1 - f0
        dd = d1 - d0
        return {
            "frames": f1,
            "fps": round(fps, 2),
            "drops": d1,
            "window_frames": df,
            "window_drops": dd,
            "drop_rate": round((dd / df), 6) if df > 0 else 0.0,
            "window_s": round(t1 - t0, 3),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self._flush()

    def __enter__(self) -> "MetricsRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
