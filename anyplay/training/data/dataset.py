"""Temporal video dataset for imitation learning.

Each sample is a window of consecutive frames plus the action state of the
last frame in the window:

    frames   (T, 3, H, W) float32 in [0, 1]
    buttons  (n_buttons,) float32 in {0, 1}
    axes     (n_axes,)    float32 in [-1, 1]

Frame cache
-----------
``cache_frames=True`` (the default) decodes the session's video ONCE and
keeps the base frames in RAM. Every epoch then yields zero-copy views into
that cache, so epochs 2..N never touch GStreamer again. For a 300 s
session at 128x96 the cache is ~1.6 GB (float32); a 600 s session is
~3.3 GB. If the cache cannot be built (MemoryError) the dataset falls back
to streaming and re-decodes every epoch.

Note: in cache mode each sample's tensors are views into the shared cache.
Do not mutate them in place (``div_``, ``add_`` ...). The training loop
only copies (``to(device)``, ``torch.stack``), which is safe.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .preprocessor import Session, build_action_space


class VideoDataset(IterableDataset):
    """Yield temporal (frame_window -> action) samples from a session."""

    def __init__(self, session_dir, train_fps=30, width=128, height=96, window=4,
                 action_space=None, cache_frames=True):
        self.session = Session(session_dir)
        self.train_fps = train_fps
        self.width = width
        self.height = height
        self.window = max(1, int(window))

        if action_space is None:
            path = Path(session_dir) / "action_space.json"
            if path.exists():
                action_space = json.loads(path.read_text())
            else:
                action_space = build_action_space(session_dir)
        self.action_space = action_space

        bounds = self.action_space.get("axis_bounds")
        if bounds is None:
            bounds = self.session.axis_bounds(self.action_space["axes"])
        self.axis_bounds = {int(k): (float(lo), float(hi))
                            for k, (lo, hi) in bounds.items()}
        self.motion_codes = [int(c) for c in self.action_space.get("motion", [])]
        self.motion_scale = {int(k): float(v)
                             for k, v in self.action_space.get("motion_scale", {}).items()}

        # (base (N,3,H,W) f32, buttons (N,B), axes (N,A), motion (N,M)) or None
        self._cache = None
        if cache_frames:
            self._build_cache()

    # -- cache -------------------------------------------------------------

    @property
    def cached_frames(self):
        """Number of base frames in the RAM cache, or None if streaming."""
        return self._cache[0].shape[0] if self._cache is not None else None

    def _build_cache(self):
        """Decode the session once; subsequent epochs iterate the cache."""
        button_codes = self.action_space["buttons"]
        axis_codes = self.action_space["axes"]
        frames = []
        buttons, axes, motion = [], [], []
        for frame, action in self.session.iter_frames_with_actions(
            fps=self.train_fps,
            width=self.width,
            height=self.height,
            button_codes=button_codes,
            axis_codes=axis_codes,
            motion_codes=self.motion_codes,
        ):
            frames.append(frame)  # (H, W, 3) uint8
            buttons.append([action["buttons"][c] for c in button_codes])
            axes.append([self._norm_axis(c, action["axes"][c]) for c in axis_codes])
            motion.append([self._norm_motion(c, action["motion"][c])
                           for c in self.motion_codes])
        if not frames:
            self._cache = None
            return
        try:
            # One batched conversion: uint8 (N,H,W,3) -> float32 (N,3,H,W) [0,1]
            base = (torch.from_numpy(np.stack(frames, axis=0))
                    .permute(0, 3, 1, 2).contiguous().float().div_(255.0))
            n = base.shape[0]
            self._cache = (
                base,
                torch.tensor(buttons, dtype=torch.float32) if buttons
                else torch.zeros((n, 0), dtype=torch.float32),
                torch.tensor(axes, dtype=torch.float32) if axes
                else torch.zeros((n, 0), dtype=torch.float32),
                torch.tensor(motion, dtype=torch.float32) if motion
                else torch.zeros((n, 0), dtype=torch.float32),
            )
        except MemoryError:
            # Streaming fallback: re-decode every epoch (slow but safe).
            self._cache = None

    # -- normalization ------------------------------------------------------

    def _norm_axis(self, code, value):
        lo, hi = self.axis_bounds.get(code, (-32768, 32768))
        if hi <= lo:
            return 0.0
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return float(np.clip((value - mid) / half, -1.0, 1.0))

    def _norm_motion(self, code, value):
        return float(np.clip(value / self.motion_scale.get(code, 1.0), -1.0, 1.0))

    # -- iteration ----------------------------------------------------------

    def __iter__(self):
        if self._cache is not None:
            yield from self._iter_cached()
        else:
            yield from self._iter_stream()

    def _iter_cached(self):
        base, buttons, axes, motion = self._cache
        n = base.shape[0]
        w = self.window
        for i in range(w - 1, n):
            yield {
                "frames": base[i - w + 1: i + 1],   # (w,3,H,W) zero-copy view
                "buttons": buttons[i],
                "axes": axes[i],
                "motion": motion[i],
            }

    def _iter_stream(self):
        button_codes = self.action_space["buttons"]
        axis_codes = self.action_space["axes"]
        buffer = deque(maxlen=self.window)

        for frame, action in self.session.iter_frames_with_actions(
            fps=self.train_fps,
            width=self.width,
            height=self.height,
            button_codes=button_codes,
            axis_codes=axis_codes,
            motion_codes=self.motion_codes,
        ):
            buffer.append(frame)
            if len(buffer) < self.window:
                continue
            frames = (
                torch.from_numpy(np.stack(list(buffer), axis=0))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            buttons = torch.tensor(
                [action["buttons"][c] for c in button_codes], dtype=torch.float32
            )
            axes = torch.tensor(
                [self._norm_axis(c, action["axes"][c]) for c in axis_codes],
                dtype=torch.float32,
            )
            motion = torch.tensor(
                [self._norm_motion(c, action["motion"][c]) for c in self.motion_codes],
                dtype=torch.float32,
            )
            yield {"frames": frames, "buttons": buttons, "axes": axes,
                   "motion": motion}
