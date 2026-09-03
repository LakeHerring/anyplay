"""Capture video from a portal/PipeWire window to MP4 (Wayland).

The game in question is a native Wayland client, so the X11 sources
(``x11grab`` / ``ximagesrc``) used by :class:`VideoCapturer` and
:class:`GstVideoCapturer` cannot see it. This capturer reuses the portal
shared-memory reader (:class:`NativeCapture`), which performs the
``xdg-desktop-portal`` handshake (window picker) and streams the selected
window's frames into shared memory at the window's **native resolution**.

Frames are sampled at a fixed ``fps`` and encoded to an H.264 MP4 through an
``ffmpeg`` ``rawvideo``-stdin subprocess, matching the output contract of the
other backends (same ``video.mp4`` path, same ``start_time`` semantics, so
``video_start_offset`` in ``meta.json`` still aligns the input stream).

The public interface is identical to the other capturers::

    capturer = PortalVideoCapturer(video_path, fps=60, crf=18, ...)
    capturer.start()
    video_start = capturer.start_time   # time.monotonic() at first frame
    while capturer.running():
        ...
    duration = capturer.stop()
"""

import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

# Human-readable meaning of common ffmpeg exit codes (see man ffmpeg):
# 0 = clean exit, 1 = generic error, 137 = SIGKILL (often OOM),
# 143 = SIGTERM, 130/-2 = SIGINT (Ctrl-C).
_RC_MEANING = {
    0: "clean exit",
    1: "generic ffmpeg error",
    137: "SIGKILL (often OOM)",
    -9: "SIGKILL (often OOM)",
    143: "SIGTERM",
    -15: "SIGTERM",
    130: "SIGINT (Ctrl-C)",
    -2: "SIGINT (Ctrl-C)",
}


class PortalVideoCapturer:
    """Record a portal-captured window to an H.264 MP4.

    Parameters
    ----------
    out_path :
        Output ``.mp4`` path.
    fps : int
        Master framerate. The portal stream is damage-driven and can arrive at
        the game's uncapped render rate, so we sample the latest frame at this
        fixed rate to produce a steady master video.
    crf : int
        libx264 quality (0-51, lower = better).
    preset : str
        libx264 speed preset.
    duration : float
        Record this many seconds, then stop (0 = until :meth:`stop` is called).
    portal_types : int
        Portal capture type: ``2`` = interactive window picker (any game).
    portal_timeout : float
        Seconds to wait for the portal consent dialog.
    slots : int
        Shared-memory ring size (unused for recording, kept for parity).
    """

    def __init__(
        self,
        out_path,
        fps: int = 60,
        crf: int = 18,
        preset: str = "veryfast",
        duration: float = 0.0,
        portal_types: int = 2,
        portal_timeout: float = 180.0,
        slots: int = 8,
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found in PATH (apt install ffmpeg)"
            )
        self.out_path = Path(out_path)
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.duration = duration
        self.portal_types = portal_types
        self.portal_timeout = portal_timeout
        self.slots = slots

        self.start_time = None
        self.width = 0
        self.height = 0

        self._ncap = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._frames = 0
        self._err = None
        # Diagnostics: bounded tail of ffmpeg stderr (drained in a thread so
        # the pipe can never fill and deadlock the encoder), mirrored to
        # <video>.ffmpeg.log next to the output file.
        self._stderr_lines: deque = deque(maxlen=200)
        self._stderr_file = None

    def start(self):
        """Open the portal capture and begin encoding in the background.

        Blocks until the portal handshake completes, the first frame is
        captured, and ``start_time`` is set.
        """
        from .native_capture import NativeCapture

        print(
            f"portal video: opening window picker -- select the game window "
            f"(source=portal, types={self.portal_types}) ...",
            flush=True,
        )
        # NativeCapture.__init__ performs the D-Bus portal handshake, the
        # window picker, auto-detects the resolution, and opens the shm. It
        # returns with self.width / self.height known.
        self._ncap = NativeCapture(
            source="portal",
            fps=self.fps,
            slots=self.slots,
            portal_timeout=self.portal_timeout,
            portal_types=self.portal_types,
        )
        self.width = self._ncap.width
        self.height = self._ncap.height
        print(
            f"portal video: capturing window at {self.width}x{self.height}",
            flush=True,
        )

        self._thread = threading.Thread(
            target=self._record,
            name="portal-video-record",
            daemon=True,
        )
        self._thread.start()

        # Wait until the first frame is encoded and start_time is set.
        t0 = time.monotonic()
        while self.start_time is None and not self._stop.is_set():
            if self._err is not None:
                raise RuntimeError(
                    f"portal video capture failed: {self._err}"
                )
            if time.monotonic() - t0 > 15.0:
                raise RuntimeError(
                    "portal video capture: no frames within 15s "
                    "(the window may not be rendering)"
                )
            time.sleep(0.02)
        print(
            f"portal video: encoding -> {self.out_path}",
            flush=True,
        )

    def _start_ffmpeg(self):
        w, h = self.width, self.height
        # yuv420p needs even dimensions; pad if the window is odd-sized.
        pad = (
            ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            if (w % 2 or h % 2)
            else []
        )
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.fps),
            "-i", "-",
            *pad,
            "-c:v", "libx264",
            "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-an",
            str(self.out_path),
        ]
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            self._stderr_file = open(
                str(self.out_path) + ".ffmpeg.log",
                "w",
                buffering=1,
            )
        except OSError:
            self._stderr_file = None
        threading.Thread(
            target=self._drain_ffmpeg_stderr,
            name="ffmpeg-stderr-drain",
            daemon=True,
        ).start()

    def _drain_ffmpeg_stderr(self):
        """Drain ffmpeg's stderr pipe into a tail + log file.

        Without this, >64 KB of stderr output (repeated warnings) would
        fill the pipe and block ffmpeg's own write(2), which stalls the
        encoder and backpressures the record thread.
        """
        assert self._proc is not None
        try:
            for raw in self._proc.stderr:
                line = raw.decode(errors="replace").rstrip()
                self._stderr_lines.append(line)
                if self._stderr_file is not None:
                    self._stderr_file.write(line + "\n")
        except (OSError, ValueError):
            pass

    def _stderr_tail(self, n: int = 10) -> str:
        return " | ".join(list(self._stderr_lines)[-n:])

    def _record(self):
        period = 1.0 / self.fps
        t0 = None
        n = 0
        try:
            while not self._stop.is_set():
                frame = self._ncap.get_frame(timeout=2.0)
                if frame is None:
                    continue
                if t0 is None:
                    self._start_ffmpeg()
                    t0 = time.monotonic()
                    self.start_time = t0
                # tobytes() copies the frame out of the shm ring into a
                # Python bytes object, so the ring can be recycled freely.
                self._proc.stdin.write(frame.tobytes())
                n += 1
                self._frames = n
                # Pace log: these lines stop exactly when the write above
                # starts blocking (encoder backpressure), which is the
                # diagnostic signature we need.
                if n % 60 == 0:
                    print(
                        f"portal encode: t={time.monotonic() - t0:7.1f}s "
                        f"frames={n}",
                        flush=True,
                    )
                # Fail fast if the encoder dies: a dead ffmpeg means a
                # truncated video and a useless session, so stop recording
                # immediately instead of blocking on the closed pipe.
                rc = self._proc.poll()
                if rc is not None:
                    self._err = (
                        f"ffmpeg exited early (rc={rc}: "
                        f"{_RC_MEANING.get(rc, 'unknown')}); stderr tail: "
                        f"{self._stderr_tail(5)!r}"
                    )
                    break
                if (
                    self.duration > 0
                    and (time.monotonic() - t0) >= self.duration
                ):
                    break
                target = t0 + n * period
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except (BrokenPipeError, ValueError) as e:
            # ffmpeg died (bad args / disk full).
            self._err = f"encoder pipe closed: {e}"
        except Exception as e:  # noqa: BLE001 - record thread must not die
            self._err = f"record thread: {e!r}"
        finally:
            self._stop.set()

    def running(self):
        if self._stop.is_set():
            return False
        if self._thread is None or not self._thread.is_alive():
            return False
        return True

    def stop(self):
        """Stop recording, finalize the MP4, and release the portal capture.

        Returns the recorded duration in seconds.
        """
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            rc = self._proc.returncode
            print(
                f"ffmpeg exit: rc={rc} "
                f"({_RC_MEANING.get(rc, 'unknown')})",
                flush=True,
            )
            tail = self._stderr_tail(10)
            if tail:
                print(f"ffmpeg stderr tail: {tail}", flush=True)
            log = self.out_path.__str__() + ".ffmpeg.log"
            print(f"ffmpeg stderr log: {log}", flush=True)

        if self._ncap is not None:
            try:
                # Daemon's own counters (frames/fps/drops) -- the key
                # diagnostic: drops > 0 means the shm ring overflowed,
                # i.e. the reader (this process) fell behind.
                daemon_stats = self._ncap.stats()
                print(
                    f"capture daemon stats: {daemon_stats}",
                    flush=True,
                )
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            try:
                daemon_tail = self._ncap.log_tail(3)
                if daemon_tail:
                    print(f"capture daemon tail: {daemon_tail}", flush=True)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            self._ncap.close()

        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except OSError:
                pass

        if self.start_time is not None:
            dur = time.monotonic() - self.start_time
            expected = int(dur * self.fps)
            if expected > 60 and self._frames < 0.5 * expected:
                print(
                    f"WARNING: video has only {self._frames} frames "
                    f"(~{self._frames / self.fps:.1f}s) for a "
                    f"{dur:.1f}s session (expected ~{expected} frames). "
                    "The video is TRUNCATED; this session is unusable for "
                    "training. See the pace log above and the ffmpeg stderr "
                    "log for when and why writes stalled.",
                    flush=True,
                )

        if self.start_time is not None:
            return time.monotonic() - self.start_time
        return 0.0

    @property
    def frames(self):
        return self._frames
