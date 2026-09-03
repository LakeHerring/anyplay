"""Temporal observation buffer.

Sits on top of a zero-copy frame source (``NativeCapture``) and builds the
temporal observations the rest of the system consumes:

    observation(t) = { frame(t), frame(t-200ms), frame(t-400ms) }

The capture daemon keeps publishing 60 FPS into shared memory regardless of
how slow the AI is (Rule 1: capture FPS is decoupled from inference FPS).
This buffer samples the latest frame on a background thread at a fixed
interval and keeps a short timestamped history in RAM (Rule 2: no disk I/O
for real-time frames).

Typical use:

    buf = ObservationBuffer(capture, sample_interval=0.016, history=2.0)
    buf.start()

    obs = buf.observation()           # 3-frame temporal observation
    obs = buf.observation(spacings=(0.0, 0.2, 0.4))

Each ``TemporalObservation`` carries:

* ``frames``: list of RGB uint8 arrays, oldest first, each ``.copy()``-safe
  (they are independent copies, not live SHM views)
* ``timestamps``: wall-clock (time.monotonic) seconds per frame
* ``age_ms``: age of the newest frame at sample time
* ``input_state`` / ``previous_action``: filled in by the agent, not here

The number of frames and temporal spacing are configurable, per plan §7.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Frame:
    """A single timestamped capture frame (plan \u00a7P1)."""

    ts: float          # time.monotonic() when the frame was sampled
    id: int            # capture/daemon frame id (0 if unknown)
    rgb: np.ndarray    # (H, W, 3) uint8


@dataclass
class TemporalObservation:
    """A timestamped set of frames + metadata handed to the decision model."""

    timestamp: float                      # time.monotonic() of the newest frame
    frames: List[np.ndarray] = field(default_factory=list)   # oldest first
    timestamps: List[float] = field(default_factory=list)    # per frame
    frame_ids: List[int] = field(default_factory=list)       # per frame (capture id)
    age_ms: float = 0.0                   # age of newest frame at sample time
    input_state: dict = field(default_factory=dict)
    previous_action: Optional[str] = None
    game_state: Optional[dict] = None     # optional reliable game state (P6+)

    @property
    def current(self) -> Optional[np.ndarray]:
        return self.frames[-1] if self.frames else None

    @property
    def last_action(self) -> Optional[str]:
        """Alias of :attr:`previous_action` (plan \u00a7P1 naming)."""
        return self.previous_action

    def as_frames(self) -> List[Frame]:
        """Return the frames as plan \u00a7P1 :class:`Frame` records (oldest first)."""
        return [Frame(ts=ts, id=(self.frame_ids[i] if i < len(self.frame_ids) else 0),
                      rgb=frm)
                for i, (ts, frm) in enumerate(zip(self.timestamps, self.frames))]


class ObservationBuffer:
    """Renders the newest frame into a bounded timestamped history.

    Parameters
    ----------
    capture:
        Object with ``get_frame(timeout=float) -> np.ndarray | None``
        (e.g. ``NativeCapture``) and ``frame_id() -> int``.
    sample_interval:
        Seconds between samples on the background thread.
        1/60 ~= capture rate; lower it to save CPU if the AI runs slower.
    history:
        Seconds of frames to keep. Must cover the largest temporal spacing
        the agent will request.
    """

    def __init__(self, capture, sample_interval: float = 0.016,
                 history: float = 2.0, spacings: Sequence[float] = (0.0, 0.2, 0.4),
                 obs_cfg=None):
        self.capture = capture
        # Rates come from ObsConfig when supplied (plan \u00a7P1: nothing
        # hardcoded), otherwise from the explicit legacy arguments.
        if obs_cfg is not None:
            sample_interval = obs_cfg.sample_interval
            history = obs_cfg.history
            spacings = obs_cfg.spacings
        self.sample_interval = sample_interval
        self.history = history
        self._default_spacings = tuple(spacings)

        self._frames: Deque[Tuple[float, np.ndarray, int]] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Diagnostics
        self.samples = 0
        self.misses = 0

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="obs-buffer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Background sampling
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        interval = self.sample_interval
        while not self._stop.is_set():
            t0 = time.monotonic()
            frame = None
            try:
                frame = self.capture.get_frame(timeout=0.1)
            except Exception:
                pass
            if frame is not None:
                ts = time.monotonic()
                fid = 0
                get_fid = getattr(self.capture, "frame_id", None)
                if get_fid is not None:
                    try:
                        fid = int(get_fid())
                    except Exception:
                        fid = 0
                with self._lock:
                    self._frames.append((ts, np.asarray(frame).copy(), fid))
                    self._trim_locked()
                self.samples += 1
            else:
                self.misses += 1
            # sleep the remainder of the interval
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                self._stop.wait(interval - elapsed)

    def _trim_locked(self) -> None:
        cutoff = time.monotonic() - self.history
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    # ------------------------------------------------------------------
    # Observation extraction
    # ------------------------------------------------------------------

    def observation(self,
                    spacings: Optional[Sequence[float]] = None
                    ) -> Optional[TemporalObservation]:
        """Build a temporal observation.

        ``spacings`` are ages in seconds, newest first (0.0 = now,
        0.2 = 200 ms ago, ...). Defaults to the buffer's configured
        spacings (from :class:`ObsConfig` or the constructor). Returned
        frames are oldest first. Returns None if no frames have been
        sampled yet.
        """
        if spacings is None:
            spacings = self._default_spacings
        now = time.monotonic()
        with self._lock:
            if not self._frames:
                return None
            newest_ts = self._frames[-1][0]
            # Copy what we need while holding the lock.
            picked = []
            for age in spacings:
                target = newest_ts - age
                # find the newest sampled frame at or before target
                chosen = None
                for ts, frame, fid in reversed(self._frames):
                    if ts <= target + 1e-9:
                        chosen = (ts, frame, fid)
                        break
                if chosen is None:
                    # no frame that old; fall back to the oldest available
                    chosen = self._frames[0]
                picked.append(chosen)

        # de-duplicate while preserving the requested order (oldest first)
        seen = set()
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        frame_ids: List[int] = []
        for ts, frame, fid in sorted(picked, key=lambda p: p[0]):
            if id(frame) in seen:
                continue
            seen.add(id(frame))
            frames.append(frame.copy())
            timestamps.append(ts)
            frame_ids.append(fid)

        if not frames:
            return None

        return TemporalObservation(
            timestamp=newest_ts,
            frames=frames,
            timestamps=timestamps,
            frame_ids=frame_ids,
            age_ms=(now - newest_ts) * 1000.0,
        )

    def sample(self, ts: Optional[float] = None,
               spacings: Optional[Sequence[float]] = None
               ) -> Optional[TemporalObservation]:
        """Plan \u00a7P1 name for :meth:`observation`.

        ``ts`` is accepted for API compatibility (a precise-time sampling
        hook); the newest sampled frame is the anchor.
        """
        return self.observation(spacings=spacings)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def depth(self) -> int:
        with self._lock:
            return len(self._frames)

    def health(self) -> dict:
        with self._lock:
            n = len(self._frames)
            span = (self._frames[-1][0] - self._frames[0][0]
                    if n else 0.0)
        return {
            "samples": self.samples,
            "misses": self.misses,
            "depth": n,
            "span_s": round(span, 3),
        }
