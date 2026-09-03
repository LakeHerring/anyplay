"""A deterministic, display-free frame source for headless observation tests.

Mirrors the ``NativeCapture`` contract the :class:`ObservationBuffer` needs:

* ``get_frame(timeout=float) -> np.ndarray``  -- (H, W, 3) uint8
* ``frame_id() -> int``                        -- newest published frame id

``frame_id`` advances with wall-clock time at ``fps`` (independent of the pull
rate), exactly like the shared-memory capture daemon: the frame buffer keeps
producing at the source rate no matter how slow the consumer is (plan Rule 1).
That is what makes inter-frame gaps and the temporal offsets measurable with
no display, compositor, or Wayland session attached.

Each rendered frame is visually distinct (a moving bar, a vertical gradient,
and the frame id baked into the blue channel) so a temporal observation's
frames can be told apart both numerically and in the side-by-side image.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class SyntheticCapture:
    """Pure-Python synthetic capture source (headless, no display needed)."""

    def __init__(self, fps: int = 60, width: int = 320, height: int = 240):
        self.fps = max(1, int(fps))
        self.width = int(width)
        self.height = int(height)
        self._t0 = time.monotonic()
        self._fid = 0
        self._x = np.arange(self.width, dtype=np.float32)
        self._y = np.arange(self.height, dtype=np.float32)
        self._sigma2 = 2.0 * (self.width / 12.0) ** 2

    def _render(self, fid: int) -> np.ndarray:
        # Moving vertical bar (red), vertical gradient (green), id mod 256 (blue).
        pos = (fid * 5) % self.width
        col = np.exp(-((self._x - pos) ** 2) / self._sigma2)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[..., 0] = (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)  # R: bar
        img[..., 1] = (self._y[:, None].repeat(self.width, axis=1)
                       * (255.0 / max(self.height - 1, 1))).astype(np.uint8)
        img[..., 2] = fid % 256
        return img

    def frame_id(self) -> int:
        return self._fid

    def get_frame(self, timeout: Optional[float] = None) -> np.ndarray:
        # Advance the published frame id by wall-clock time at ``fps``.
        self._fid = int((time.monotonic() - self._t0) * self.fps)
        return self._render(self._fid)

    def close(self) -> None:
        pass
