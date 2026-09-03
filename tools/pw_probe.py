#!/usr/bin/env python3
"""One-shot PipeWire screencast probe.

Runs as:  tools/portal-pw-fd -- .venv/bin/python tools/pw_probe.py [seconds]

Flow:
  1. Verify the portal gave us a PipeWire remote fd (PW_FD env).
  2. Poll `pw-cli list-objects` until the KWin screencast node appears.
  3. Capture frames IN-PROCESS with GStreamer (pipewiresrc → appsink);
     the negotiated size is learned from the first sample, so no
     separate size-probe pass is needed.
  4. Save the first frame as PNG and print a black-frame verdict.
"""
import os
import re
import subprocess
import sys
import threading
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import numpy as np

POLL_TIMEOUT_S = 45


def log(msg: str) -> None:
    print(f"[probe] {msg}", file=sys.stderr, flush=True)


def node_fd_ok() -> bool:
    fd = os.environ.get("PW_FD")
    if not fd:
        log("ERROR: PW_FD not set — portal flow did not complete")
        return False
    try:
        os.fstat(int(fd))
        log(f"PipeWire remote fd={fd} is open")
        return True
    except OSError as e:
        log(f"ERROR: PW_FD={fd} not usable: {e}")
        return False


def find_screencast_node(deadline: float) -> tuple[int, str] | None:
    """Return (node_id, node_name) of the KWin screencast video node."""
    seen = set()
    while time.time() < deadline:
        out = subprocess.run(
            ["pw-cli", "list-objects"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for m in re.finditer(r"^\s*(\d+):\s+Node \[\s*(\d+)\s*\]", out, re.M):
            idx, node_id = int(m.group(1)), int(m.group(2))
            block = out[m.end():]
            nxt = re.search(r"^\s*\d+:\s", block, re.M)
            if nxt:
                block = block[: nxt.start()]
            name_m = re.search(r'"Node\.name"\s*=\s*"([^"]*)"', block)
            name = name_m.group(1) if name_m else "?"
            if "screencast" in name.lower() or "kwin" in name.lower():
                return node_id, name
            seen.add((node_id, name))
        time.sleep(0.5)
    log(f"no screencast node found; all nodes seen: {sorted(seen)}")
    return None


class PwScreen:
    """In-process PipeWire screencast capture via GStreamer.

    ``frames()`` yields uint8 (H, W, 3) RGB frames from a
    ``pipewiresrc → appsink`` pipeline at the requested fps. The
    negotiated size is learned from the first sample (``width``/``height``
    are set before the first frame is yielded).
    """

    def __init__(self, node_id: int, fps: int = 30):
        Gst.init(None)
        self.node_id = node_id
        self.fps = fps
        self.width = None
        self.height = None
        # A GLib main loop is needed for state changes and bus messages.
        self._loop = GLib.MainLoop()
        threading.Thread(target=self._loop.run, daemon=True).start()
        self._pipeline = Gst.parse_launch(
            f"pipewiresrc select-node-id={node_id} ! videoconvert ! "
            f"video/x-raw,format=RGB ! videorate ! "
            f"video/x-raw,framerate={fps}/1 ! "
            f"appsink name=sink max-buffers=16 drop=true")
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"pipewiresrc pipeline for node {node_id} failed to start")

    def _pull(self):
        sample = self._pipeline.get_by_name("sink").emit("pull-sample")
        if sample is None:
            return None
        caps = sample.get_caps()
        if caps is not None and self.width is None:
            st = caps.get_structure(0)
            self.width = int(st.get_value("width"))
            self.height = int(st.get_value("height"))
        return sample

    def frames(self):
        """Yield RGB frames until the node goes away / EOS."""
        try:
            while True:
                sample = self._pull()
                if sample is None:
                    break
                buf = sample.get_buffer()
                ok, m = buf.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                try:
                    data = m.data  # bytes copy; valid after unmap
                finally:
                    buf.unmap(m)
                yield np.frombuffer(data, dtype=np.uint8) \
                    .reshape(self.height, self.width, 3)
        finally:
            self.close()

    def close(self):
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._loop.quit()


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

    if not node_fd_ok():
        return 1

    log("waiting for KWin screencast node ...")
    found = find_screencast_node(time.time() + POLL_TIMEOUT_S)
    if not found:
        log("FATAL: screencast node never appeared")
        return 2
    node_id, name = found
    log(f"screencast node: id={node_id} name={name}")

    screen = PwScreen(node_id, fps=10)
    frames: list[np.ndarray] = []
    deadline = time.time() + seconds
    try:
        for frame in screen.frames():
            frames.append(frame)
            if time.time() >= deadline:
                break
    finally:
        screen.close()

    if not frames:
        log("FATAL: no frames captured")
        return 3
    log(f"captured {len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]} RGB")

    os.makedirs("/tmp/pw_probe", exist_ok=True)
    first = frames[0]

    nonblack = -1.0
    means = None
    try:
        from PIL import Image
        img = Image.fromarray(first)
        img.save("/tmp/pw_probe/first.png")
        arr = np.asarray(img.resize((160, 90)))
        nonblack = float((arr.max(axis=2) > 24).mean())
        means = [round(float(c), 1) for c in arr.reshape(-1, 3).mean(axis=0)]
    except ImportError:
        log("PIL missing; skipping stats")

    if nonblack >= 0:
        verdict = "OK (not black)" if nonblack > 0.02 else "BLACK — capture failed"
        log(f"frame0: means={means} nonblack={nonblack:.3f} (160x90 sample)")
        log(f"VERDICT: {verdict}")
        return 0 if nonblack > 0.02 else 4
    log("VERDICT: unknown (no stats library)")
    return 5


if __name__ == "__main__":
    sys.exit(main())
