"""Capture X11 screen video to MP4 with ffmpeg at a fixed frame rate.

Uses x11grab for the capture source. The start time is recorded with
CLOCK_MONOTONIC right before ffmpeg launches; ``meta.json`` stores it so the
input log (recorded on the same clock) can be aligned to video frames.
"""

import shutil
import subprocess
import time
from pathlib import Path


class VideoCapturer:
    def __init__(self, out_path, display=":0.0", region="", fps=60, crf=18, preset="veryfast", duration=0.0):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH (apt install ffmpeg)")
        self.out_path = Path(out_path)
        self.display = display
        self.region = region
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.duration = duration
        self.start_time = None
        self.proc = None

    def _ffmpeg_args(self):
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "x11grab", "-framerate", str(self.fps),
        ]
        if self.region:
            x, y, w, h = (int(v) for v in self.region.split(","))
            args += ["-video_size", f"{w}x{h}", "-i", f"{self.display}+{x},{y}"]
        else:
            args += ["-i", self.display]
        if self.duration > 0:
            args += ["-t", f"{self.duration:.3f}"]
        args += [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-an", str(self.out_path),
        ]
        return args

    def start(self):
        self.start_time = time.monotonic()
        self.proc = subprocess.Popen(
            self._ffmpeg_args(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Give ffmpeg a moment to fail loudly (bad display, bad region, ...).
        time.sleep(0.5)
        if self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed to start: {err.strip()[:500]}")

    def stop(self):
        """Stop ffmpeg and return the capture duration in seconds."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return time.monotonic() - self.start_time

    def running(self):
        return self.proc is not None and self.proc.poll() is None
