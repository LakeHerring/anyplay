"""Decode a capture session into training-rate frames and aligned actions.

A session directory contains ``video.mp4`` (60 FPS master), ``inputs.jsonl``
(raw evdev events) and ``meta.json`` (clock alignment). This module:

* discovers the action space (which key codes / axes appear in the recording)
* decodes the video at the training rate (30 FPS) and resolution — in-process
  via GStreamer (filesrc ! decodebin ! appsink) when available, falling back
  to an ffmpeg subprocess
* replays the input events to produce the action state for each frame
"""

import json
import subprocess
import threading
from pathlib import Path

import numpy as np

from evdev.ecodes import EV_ABS, EV_KEY, EV_REL

_GST = None  # resolved lazily: True/False once probed

# Hi-res wheel variants are redundant with the coarse wheel; exclude them so
# the action space doesn't model the same scroll gesture twice.
_HIRES_REL = {11, 12}  # REL_WHEEL_HI_RES, REL_HWHEEL_HI_RES


def _gst_decode_available():
    """True when PyGObject + a GStreamer build with decodebin are usable."""
    global _GST
    if _GST is not None:
        return _GST
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        _GST = Gst.ElementFactory.find("decodebin") is not None
    except Exception:
        _GST = False
    return _GST


class Session:
    """A capture session directory."""

    def __init__(self, session_dir):
        self.dir = Path(session_dir)
        self.video = self.dir / "video.mp4"
        self.inputs = self.dir / "inputs.jsonl"
        meta_path = self.dir / "meta.json"
        if not self.video.exists():
            raise FileNotFoundError(f"no video in session: {self.dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"no meta.json in session: {self.dir}")
        self.meta = json.loads(meta_path.read_text())
        self.events = self._load_events()
        self.pointer_devs, self._rel_strict = self._pointer_devs()

    def _load_events(self):
        """(t, type, code, value, dev) tuples, sorted by t. dev='' on old files."""
        if not self.inputs.exists():
            return []
        events = []
        with open(self.inputs) as f:
            for line in f:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    events.append((ev["t"], ev["type"], ev["code"], ev["value"],
                                   ev.get("dev", "")))
        events.sort(key=lambda e: e[0])
        return events

    def _pointer_devs(self):
        """(pointer device paths, strict) from meta.

        ``strict`` is False for old single-device sessions whose meta has no
        per-device roles; then every event's dev counts as the pointer, which
        preserves the legacy single-device behavior.
        """
        inp = self.meta.get("input_device")
        if isinstance(inp, list) and inp and any("role" in d for d in inp):
            return ({d.get("path", "") for d in inp if d.get("role") == "pointer"},
                    True)
        if isinstance(inp, dict) and "role" in inp:
            if inp.get("role") == "pointer":
                return {inp.get("path", "")}, True
            return set(), True
        return set(), False

    def _is_rel_dev(self, dev):
        """True if EV_REL events on ``dev`` count as mouse motion."""
        return (not self._rel_strict) or (dev in self.pointer_devs)

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    def discover_action_space(self):
        """Key/axis/motion codes used in the recording, sorted.

        ``motion`` = relative (EV_REL) codes from the *pointer* device only, so
        a keyboard's stray REL events are not mistaken for mouse movement.
        """
        buttons, axes, pointer_rel = set(), set(), set()
        for _t, typ, code, _v, dev in self.events:
            if typ == EV_KEY:
                buttons.add(code)
            elif typ == EV_ABS:
                axes.add(code)
            elif typ == EV_REL and self._is_rel_dev(dev):
                pointer_rel.add(code)
        motion = sorted(c for c in pointer_rel if c not in _HIRES_REL)
        return {"buttons": sorted(buttons), "axes": sorted(axes), "motion": motion}

    def axis_bounds(self, axes):
        """Observed (min, max) per axis code, for normalization."""
        bounds = {}
        for code in axes:
            vals = [v for _t, typ, c, v, _d in self.events if typ == EV_ABS and c == code]
            bounds[code] = (min(vals), max(vals)) if vals else (-32768, 32768)
        return bounds

    def motion_scale(self, motion_codes, fps=30):
        """Per-motion-code normalization scale = 99th pct of per-frame |delta|.

        Relative deltas are summed per frame (matching the labeling) so the
        scale matches the per-frame values the network is trained to predict.
        """
        offset = self.meta.get("video_start_offset", 0.0)
        scales = {}
        for code in motion_codes:
            per_frame = {}
            for t, typ, c, v, dev in self.events:
                if typ != EV_REL or c != code or not self._is_rel_dev(dev):
                    continue
                ft = int((t - offset) * fps)
                per_frame[ft] = per_frame.get(ft, 0) + v
            deltas = [abs(x) for x in per_frame.values() if x != 0]
            scales[code] = max(1.0, float(np.percentile(deltas, 99)) if deltas else 1.0)
        return scales

    # ------------------------------------------------------------------
    # Label statistics (no video decode needed)
    # ------------------------------------------------------------------
    def button_duty(self, button_codes):
        """Fraction of the session each key code was held down (0..1).

        Computed straight from the input log; used as BCE pos_weight so the
        model doesn't collapse to "always zero" on sparse button labels.
        """
        duration = float(self.meta.get("duration", 0.0)) or 1.0
        offset = self.meta.get("video_start_offset", 0.0)
        end = offset + duration
        duty = {}
        for code in button_codes:
            held, down_at, down = 0.0, None, False
            for t, typ, c, value, _dev in self.events:
                if typ != EV_KEY or c != code:
                    continue
                if value and not down:
                    down_at, down = t, True
                elif not value and down:
                    held += t - down_at
                    down = False
            if down:
                held += end - down_at
            duty[code] = min(1.0, max(0.0, held / duration))
        return duty

    # ------------------------------------------------------------------
    # Video decoding
    # ------------------------------------------------------------------
    def iter_frames(self, fps=30, width=128, height=96):
        """Yield uint8 (H, W, 3) RGB frames at ``fps``.

        Uses an in-process GStreamer pipeline when available; otherwise falls
        back to an ffmpeg subprocess. Both resample to ``fps`` (duplicate/
        drop) and stretch to ``width x height``.
        """
        if _gst_decode_available():
            return self._iter_frames_gst(fps, width, height)
        return self._iter_frames_ffmpeg(fps, width, height)

    def _iter_frames_gst(self, fps, width, height):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        desc = (
            f"filesrc location={self.video} ! decodebin ! videorate ! "
            f"video/x-raw,framerate={fps}/1 ! videoscale ! videoconvert ! "
            f"video/x-raw,width={width},height={height},format=RGB ! "
            f"appsink name=sink max-buffers=16 drop=false sync=false"
        )
        loop = GLib.MainLoop()
        threading.Thread(target=loop.run, daemon=True).start()
        pipeline = Gst.parse_launch(desc)
        pipeline.set_state(Gst.State.PLAYING)
        try:
            sink = pipeline.get_by_name("sink")
            while True:
                sample = sink.emit("pull-sample")
                if sample is None:  # EOS
                    break
                buf = sample.get_buffer()
                ok, m = buf.map(Gst.MapFlags.READ)
                if not ok:
                    break
                try:
                    frame = np.frombuffer(m.data, dtype=np.uint8) \
                        .reshape(height, width, 3)
                finally:
                    buf.unmap(m)
                yield frame
        finally:
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    def _iter_frames_ffmpeg(self, fps, width, height):
        """Yield uint8 (H, W, 3) RGB frames at ``fps`` via ffmpeg."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(self.video),
            "-vf", f"fps={fps},scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = width * height * 3
        try:
            while True:
                data = proc.stdout.read(frame_size)
                if len(data) < frame_size:
                    break
                yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        finally:
            proc.stdout.close()
            proc.wait()

    # ------------------------------------------------------------------
    # Action alignment
    # ------------------------------------------------------------------
    def iter_frames_with_actions(self, fps=30, width=128, height=96,
                                 button_codes=(), axis_codes=(), motion_codes=()):
        """Yield ``(frame_hwc_uint8, action)`` for each frame at ``fps``.

        ``action`` = {"buttons": {code: 0/1}, "axes": {code: last_value},
        "motion": {code: per-frame REL delta}}. Buttons/axes are held state;
        motion is the summed relative delta since the previous frame.
        """
        offset = self.meta.get("video_start_offset", 0.0)
        keys = {}
        axes = {}
        motion_acc = {}
        events = self.events
        n_events = len(events)
        ei = 0

        for frame_index, frame in enumerate(self.iter_frames(fps, width, height)):
            t_video = frame_index / fps
            # Apply all events that happened by this frame's time.
            while ei < n_events and events[ei][0] - offset <= t_video:
                _t, typ, code, value, dev = events[ei]
                if typ == EV_KEY:
                    keys[code] = 1 if value else 0
                elif typ == EV_ABS:
                    axes[code] = value
                elif typ == EV_REL and self._is_rel_dev(dev):
                    motion_acc[code] = motion_acc.get(code, 0) + value
                ei += 1
            action = {
                "buttons": {c: keys.get(c, 0) for c in button_codes},
                "axes": {c: axes.get(c, 0) for c in axis_codes},
                "motion": {c: motion_acc.get(c, 0) for c in motion_codes},
            }
            motion_acc = {}  # deltas are per-frame; reset for the next frame
            yield frame, action


def build_action_space(session_dir, overwrite=False):
    """Discover the action space from recorded inputs and cache it.

    Returns the space dict (also written to ``action_space.json``).
    """
    session_dir = Path(session_dir)
    path = session_dir / "action_space.json"
    if path.exists() and not overwrite:
        return _normalize_space(json.loads(path.read_text()))

    session = Session(session_dir)
    space = session.discover_action_space()
    bounds = session.axis_bounds(space["axes"])
    space["axis_bounds"] = {str(code): list(b) for code, b in bounds.items()}
    # Motion scale assumes the 30 FPS training rate (DatasetConfig default).
    space["motion_scale"] = ({str(c): s for c, s in
                              session.motion_scale(space["motion"], 30).items()}
                             if space["motion"] else {})
    space["n_frames_hint"] = None  # filled lazily; not required
    path.write_text(json.dumps(space, indent=2))
    return _normalize_space(space)


def _normalize_space(space):
    """JSON round-trip makes codes/keys strings; coerce back to ints."""
    space["axes"] = [int(c) for c in space.get("axes", [])]
    space["motion"] = [int(c) for c in space.get("motion", [])]
    bounds = {}
    for code, (lo, hi) in space.get("axis_bounds", {}).items():
        bounds[int(code)] = (float(lo), float(hi))
    space["axis_bounds"] = bounds
    space["motion_scale"] = {int(k): float(v)
                             for k, v in space.get("motion_scale", {}).items()}
    return space
