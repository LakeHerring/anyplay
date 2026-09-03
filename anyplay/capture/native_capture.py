"""Zero-copy client for the native GStreamer capture daemon.

The daemon owns the GStreamer capture pipeline and publishes RGB frames
into a POSIX shared-memory ring.

Two capture sources are supported:

    source="daemon"
        GStreamer ximagesrc / X11 capture.

    source="portal"
        xdg-desktop-portal + PipeWire capture.
        This is the appropriate path for a Wayland desktop.

The returned frame is a zero-copy NumPy view:

    cap = NativeCapture(
        "0,0,640,480",
        width=640,
        height=480,
        fps=60,
        source="portal",
    )

    frame = cap.get_frame()

    # frame:
    #   shape = (480, 640, 3)
    #   dtype = uint8
    #   RGB

Do not hold the returned frame for long because the daemon may overwrite
the shared-memory slot. Copy it immediately if it needs to survive.
"""

from __future__ import annotations

import mmap
import os
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Environment override (regression testing: point the client at the
# C++ capture daemon, core/build/anyplay-capture, which implements the
# identical protocol). Unset -> the proven C daemon.
DAEMON = Path(
    os.environ.get(
        "SDAI_DAEMON_BIN",
        str(PROJECT_ROOT / "native" / "capture-daemon"),
    )
)
PORTAL_FD = PROJECT_ROOT / "tools" / "portal-pw-fd"
# Environment override (regression / migration testing): point the client
# at the C++ portal daemon, core/build/anyplay-portal-capture, which
# implements the identical protocol. Unset -> the proven C daemon.
PORTAL_CAP = Path(
    os.environ.get(
        "SDAI_PORTAL_BIN",
        str(PROJECT_ROOT / "native" / "portal-capture"),
    )
)
# True when the caller explicitly overrode the portal binary (e.g. the C++
# build). A custom binary must already exist; it is NOT built by the C
# native/build.sh.
PORTAL_CAP_IS_CUSTOM = "SDAI_PORTAL_BIN" in os.environ
BUILD = PROJECT_ROOT / "native" / "build.sh"


# ---------------------------------------------------------------------------
# Shared-memory layout
# ---------------------------------------------------------------------------

# Header:
#
#   magic
#   version
#   width
#   height
#   slot_bytes
#   n_slots
#   alive
#   padding
#
_HDR32 = struct.Struct("<8I")


# Statistics:
#
#   idx
#   frames
#   drops
#
_HDR64 = struct.Struct("<3Q")


_MAGIC = 0x49445321

_HDR_BASE = 64
_SLOT_META = 32


# ---------------------------------------------------------------------------
# Binary availability
# ---------------------------------------------------------------------------

def _ensure_daemon() -> Path:
    """Ensure the X11 daemon binary exists."""

    if DAEMON.exists():
        return DAEMON

    if BUILD.exists():
        subprocess.run(
            ["bash", str(BUILD)],
            check=True,
        )

        if DAEMON.exists():
            return DAEMON

    raise FileNotFoundError(
        f"capture daemon binary missing: {DAEMON}\n"
        "Run native/build.sh"
    )


def _daemon_supports_input(bin_path: Path) -> bool:
    """Feature-probe a daemon binary for the C++ input-ring flags.

    The proven C daemons exit on unknown args after printing usage; the
    C++ daemons (core/build/*) print usage containing --keyboard. Both
    print usage on --help (an unknown arg), so one probe covers both.
    """

    try:

        proc = subprocess.run(
            [str(bin_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

    except (OSError, subprocess.TimeoutExpired):

        return False

    out = (proc.stdout or "") + (proc.stderr or "")

    return "--keyboard" in out


def _ensure_portal() -> None:
    """Ensure the portal capture binaries exist."""

    # The D-Bus handshake helper is always the native C tool.
    if not PORTAL_FD.exists():
        if BUILD.exists():
            subprocess.run(["bash", str(BUILD)], check=True)
        if not PORTAL_FD.exists():
            raise FileNotFoundError(
                f"portal capture binary missing: {PORTAL_FD}\n"
                "Run native/build.sh"
            )

    # The capture daemon: either the proven C binary (auto-built by
    # native/build.sh) or a caller-provided custom binary (e.g. the C++
    # build), which must already exist.
    if not PORTAL_CAP.exists():
        if PORTAL_CAP_IS_CUSTOM:
            raise FileNotFoundError(
                f"custom portal binary missing (SDAI_PORTAL_BIN): {PORTAL_CAP}\n"
                "Build it first: make -C core"
            )
        if BUILD.exists():
            subprocess.run(["bash", str(BUILD)], check=True)
        if not PORTAL_CAP.exists():
            raise FileNotFoundError(
                f"portal capture binary missing: {PORTAL_CAP}\n"
                "Run native/build.sh"
            )


# ---------------------------------------------------------------------------
# NativeCapture
# ---------------------------------------------------------------------------

class NativeCapture:
    """Latest-frame access to the native capture daemon.

    Parameters
    ----------
    region:
        Capture region in ``x,y,w,h`` format.

        Used by the X11 daemon source.

    width, height:
        Output frame dimensions.

        The native pipeline performs scaling before the frame reaches
        shared memory.

        If zero, the source dimensions are used.

    fps:
        Capture rate.

        The portal path is ultimately compositor-driven.

    display:
        X11 display used by daemon mode.

        Default is ``:0.0``.

    slots:
        Number of shared-memory ring slots.

    source:
        Capture backend.

        ``"daemon"``
            X11/GStreamer ximagesrc.

        ``"portal"``
            xdg-desktop-portal + PipeWire.

    portal_timeout:
        Maximum time to wait for portal setup and user consent.

    portal_types:
        Portal capture source type (portal path only).

        ``1``
            Whole desktop.

        ``2`` (default)
            Interactive window picker -- click any window, which makes
            this a general (any-game) capture.

        ``3``
            Both.

    keyboard:
        Comma-separated /dev/input event nodes to record as keyboard
        (C++ daemon only, P-core step 2). Events land in an input-event
        ring exposed as ``self.input_ring`` (InputEventRing).

    pointer:
        Comma-separated /dev/input event nodes to record as pointer.

        Device ids in the ring follow CLI order: keyboard device(s)
        first, then pointer device(s).
    """

    def __init__(
        self,
        region: str = "",
        width: int = 0,
        height: int = 0,
        fps: int = 60,
        display: str = ":0.0",
        slots: int = 8,
        source: str = "daemon",
        portal_timeout: float = 180.0,
        portal_types: int = 2,
        keyboard: str = "",
        pointer: str = "",
    ):

        if source not in ("daemon", "portal"):
            raise ValueError(
                f"Unknown capture source: {source!r}. "
                "Expected 'daemon' or 'portal'."
            )

        self.source = source
        self.region = region
        self.requested_width = width
        self.requested_height = height
        self.fps = fps
        self.display = display
        self.slots = slots
        self.portal_timeout = portal_timeout
        self.portal_types = portal_types
        self.keyboard = keyboard
        self.pointer = pointer
        self.input_ring = None

        # ---------------------------------------------------------------
        # Per-process IPC paths
        # ---------------------------------------------------------------

        tag = os.getpid()

        self._shm_path = (
            f"/dev/shm/anyplay_cap_{tag}.bin"
        )

        self._sock_path = (
            f"/tmp/anyplay_cap_{tag}.sock"
        )

        self._input_shm_path = (
            f"/dev/shm/anyplay_inp_{tag}.bin"
        )

        # Remove stale paths from a previous crashed process.

        for path in (
            self._shm_path,
            self._sock_path,
            self._input_shm_path,
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        # ---------------------------------------------------------------
        # Build command
        # ---------------------------------------------------------------

        if source == "portal":

            _ensure_portal()

            # portal-pw-fd performs the D-Bus portal handshake and
            # passes the resulting PipeWire information to portal-capture.

            cmd = [
                str(PORTAL_FD),
                "--types",
                str(self.portal_types),
                "--timeout",
                str(int(portal_timeout)),
                "--",
                str(PORTAL_CAP),
                "--slots",
                str(slots),
                "--shm",
                self._shm_path,
                "--sock",
                self._sock_path,
            ]

            if width and height:
                cmd += [
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                ]
            else:
                print(
                    "[native-capture] no width/height set; "
                    "portal-capture will auto-detect the window's "
                    "native resolution",
                    flush=True,
                )
            if self.portal_types == 2:
                print(
                    "[native-capture] portal window picker will open -- "
                    "select the game window in the KDE 'Share Screen' "
                    "dialog (any game)",
                    flush=True,
                )

            if self.keyboard or self.pointer:

                if not _daemon_supports_input(PORTAL_CAP):
                    raise RuntimeError(
                        "keyboard/pointer input recording needs the "
                        "C++ portal daemon; build with 'make -C core' "
                        "and set SDAI_PORTAL_BIN to "
                        f"{PROJECT_ROOT / 'core' / 'build' / 'anyplay-portal-capture'}"
                    )

                if self.keyboard:
                    cmd += ["--keyboard", self.keyboard]
                if self.pointer:
                    cmd += ["--pointer", self.pointer]
                cmd += ["--input-shm", self._input_shm_path]

            ready_timeout = (
                portal_timeout + 30.0
            )

        else:

            daemon = _ensure_daemon()

            cmd = [
                str(daemon),
                "--region",
                region,
                "--fps",
                str(fps),
                "--slots",
                str(slots),
                "--display",
                display,
                "--shm",
                self._shm_path,
                "--sock",
                self._sock_path,
            ]

            if width and height:
                cmd += [
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                ]

            if self.keyboard or self.pointer:

                if not _daemon_supports_input(daemon):
                    raise RuntimeError(
                        "keyboard/pointer input recording needs the "
                        "C++ capture daemon; build with 'make -C core' "
                        "and set SDAI_DAEMON_BIN to "
                        f"{PROJECT_ROOT / 'core' / 'build' / 'anyplay-capture'}"
                    )

                if self.keyboard:
                    cmd += ["--keyboard", self.keyboard]
                if self.pointer:
                    cmd += ["--pointer", self.pointer]
                cmd += ["--input-shm", self._input_shm_path]

            ready_timeout = 10.0

        # ---------------------------------------------------------------
        # Start capture process
        # ---------------------------------------------------------------

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._cmd = cmd

        # ---------------------------------------------------------------
        # Wait for READY
        # ---------------------------------------------------------------

        self._wait_ready(
            ready_timeout
        )

        # ---------------------------------------------------------------
        # Open the input-event ring, if the daemon was given devices.
        # (C++ daemons only; the probe above guarantees support.)
        # ---------------------------------------------------------------

        if self.keyboard or self.pointer:

            from .input_ring import InputEventRing

            self.input_ring = InputEventRing(self._input_shm_path)

        # ---------------------------------------------------------------
        # Drain the daemon's stdout (post-READY stats / warning lines).
        # Without a drain, >64 KB of daemon output would fill the pipe
        # and block the daemon's write(2), stalling the whole capture.
        # ---------------------------------------------------------------

        # 4096 lines ~= 7 min at 1 stats line/s, so the tail covers a
        # full-length session for post-mortem.
        self._log_lines = deque(maxlen=4096)

        threading.Thread(
            target=self._drain_stdout,
            name="capture-stdout-drain",
            daemon=True,
        ).start()

        # ---------------------------------------------------------------
        # Open shared memory
        # ---------------------------------------------------------------

        self._fd = os.open(
            self._shm_path,
            os.O_RDWR,
        )

        self._mm = mmap.mmap(
            self._fd,
            0,
        )

        (
            magic,
            version,
            width_actual,
            height_actual,
            slot_bytes,
            n_slots,
            alive,
            _pad,
        ) = _HDR32.unpack_from(
            self._mm,
            0,
        )

        # ---------------------------------------------------------------
        # Validate shared-memory header
        # ---------------------------------------------------------------

        if magic != _MAGIC:

            self.close()

            raise RuntimeError(
                "Invalid shared-memory magic: "
                f"{magic:#x} "
                f"(expected {_MAGIC:#x})"
            )

        if alive != 1:

            self.close()

            raise RuntimeError(
                "Capture daemon reported itself as not alive."
            )

        if width_actual <= 0 or height_actual <= 0:

            self.close()

            raise RuntimeError(
                "Capture daemon reported invalid dimensions: "
                f"{width_actual}x{height_actual}"
            )

        if slot_bytes != (
            width_actual
            * height_actual
            * 3
        ):

            self.close()

            raise RuntimeError(
                "Unexpected shared-memory frame size: "
                f"slot_bytes={slot_bytes}, "
                f"expected={width_actual * height_actual * 3}"
            )

        # ---------------------------------------------------------------
        # Store layout
        # ---------------------------------------------------------------

        self.version = version

        self.width = width_actual
        self.height = height_actual

        self.slot_bytes = slot_bytes
        self.n_slots = n_slots

        self._slot_stride = (
            _SLOT_META
            + slot_bytes
        )

        # NumPy view of the entire mmap.
        #
        # No copy is performed here.
        self._buf = np.frombuffer(
            self._mm,
            dtype=np.uint8,
        )

        # Optional P0 metrics sink (set by a consumer, e.g. Agent). When
        # non-None, get_frame() records per-frame timing stages to it.
        self.metrics = None

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def _drain_stdout(self) -> None:
        """Consume daemon stdout lines into a bounded tail."""

        try:

            for line in self._proc.stdout:

                self._log_lines.append(line.rstrip())

        except (OSError, ValueError):

            pass

    def log_tail(self, n: int = 5) -> str:
        """Return the last n lines of daemon output (diagnostics)."""

        return " | ".join(list(self._log_lines)[-n:])

    # -------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------

    def _wait_ready(self, timeout: float = 10.0):
        """Wait until the capture process prints READY."""

        deadline = (
            time.monotonic()
            + timeout
        )

        while True:

            line = self._proc.stdout.readline()

            if line.startswith("READY "):
                return

            if (
                line == ""
                and self._proc.poll() is not None
            ):

                raise RuntimeError(
                    "Capture process exited "
                    f"(rc={self._proc.returncode})."
                )

            if time.monotonic() > deadline:

                self.close()

                raise RuntimeError(
                    "Capture process did not become "
                    f"ready within {timeout:.0f}s."
                )

    # -------------------------------------------------------------------
    # Shared-memory frame index
    # -------------------------------------------------------------------

    def _idx(self) -> int:
        """Return newest published frame index + 1.

        0 means no frame has been published yet.
        """

        return _HDR64.unpack_from(
            self._mm,
            32,
        )[0]

    def frame_id(self) -> int:
        """Return newest published frame ID.

        0 means no frame has been published yet.
        """

        idx = self._idx()

        return (
            idx - 1
            if idx > 0
            else 0
        )

    # -------------------------------------------------------------------
    # Frame access
    # -------------------------------------------------------------------

    def get_frame(
        self,
        timeout: float = 2.0,
    ):
        """Return newest frame as a zero-copy NumPy RGB view.

        Returns
        -------
        numpy.ndarray | None

            Shape:

                (height, width, 3)

            dtype:

                uint8

            Format:

                RGB

        The returned array points directly into shared memory.

        If the caller needs to keep the frame while inference or another
        operation is running, make a copy immediately:

            frame = cap.get_frame()

            if frame is not None:
                frame = frame.copy()

        """

        deadline = (
            time.monotonic()
            + timeout
        )

        t_start = time.monotonic()

        idx = self._idx()

        # ---------------------------------------------------------------
        # Wait for first frame
        # ---------------------------------------------------------------

        while idx == 0:

            if time.monotonic() > deadline:
                return None

            time.sleep(0.005)

            idx = self._idx()

        # ---------------------------------------------------------------
        # Locate newest ring slot
        # ---------------------------------------------------------------

        off = (
            _HDR_BASE
            + (
                (idx - 1)
                % self.n_slots
            )
            * self._slot_stride
        )

        data_off = (
            off
            + _SLOT_META
        )

        # ---------------------------------------------------------------
        # Seqlock read
        # ---------------------------------------------------------------

        for _ in range(1000):

            seq1 = struct.unpack_from(
                "<I",
                self._mm,
                off,
            )[0]

            # Producer currently writing this slot.
            if seq1 & 1:

                time.sleep(
                    0.0005
                )

                continue

            frame = self._buf[
                data_off:
                data_off + self.slot_bytes
            ].reshape(
                self.height,
                self.width,
                3,
            )

            # Publish-time stamps, read in the same snapshot and validated
            # by the seq2 check below (P0 latency metrics; zero cost when no
            # metrics sink is attached).
            ts_ns = wait_ns = 0
            if self.metrics is not None:
                ts_ns = struct.unpack_from("<Q", self._mm, off + 16)[0]
                wait_ns = struct.unpack_from("<Q", self._mm, off + 24)[0]

            seq2 = struct.unpack_from(
                "<I",
                self._mm,
                off,
            )[0]

            # Producer didn't modify it while we read.
            if seq1 == seq2:
                if self.metrics is not None:
                    self._record_stamps(ts_ns, wait_ns, t_start)
                return frame

        raise RuntimeError(
            "Frame slot seqlock did not settle."
        )

    def frame_stamps(self):
        """Return ``(frame_id, ts_ns, wait_ns)`` for the newest ring slot.

        Cheap metadata read (no frame data). ``ts_ns`` is CLOCK_MONOTONIC at
        publish; ``wait_ns`` is the producer's wait for a source sample (the
        source frame cadence). Both live in previously-unused slot bytes that
        older consumers ignore. Returns ``None`` before the first frame.
        """

        idx = self._idx()

        if idx == 0:
            return None

        off = (
            _HDR_BASE
            + ((idx - 1) % self.n_slots) * self._slot_stride
        )

        frame_id = struct.unpack_from("<Q", self._mm, off + 8)[0]
        ts_ns = struct.unpack_from("<Q", self._mm, off + 16)[0]
        wait_ns = struct.unpack_from("<Q", self._mm, off + 24)[0]

        return frame_id, ts_ns, wait_ns

    def _record_stamps(
        self,
        ts_ns: int,
        wait_ns: int,
        t_start: float,
    ) -> None:
        """Record P0 timing stages for the frame just read.

        In-memory only (``MetricsRecorder`` flushes to disk at most once per
        interval), so this adds no per-frame I/O to the frame path.

        * ``capture_wait``    producer wait for a source sample (wait_ns).
        * ``ring_residence``  now - ts_ns: how long the frame sat in the ring.
        * ``frame_read``      wall time spent in get_frame.
        """

        if self.metrics is None:
            return

        now = time.monotonic_ns()
        residence_ms = max(0.0, (now - ts_ns) / 1e6)
        read_ms = (time.monotonic() - t_start) * 1000.0

        m = self.metrics
        m.record("capture_wait", wait_ns / 1e6)
        m.record("ring_residence", residence_ms)
        m.record("frame_read", read_ms)

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def frame_info(self):
        """Return useful information about the capture."""

        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "dtype": "uint8",
            "format": "RGB",
            "slot_bytes": self.slot_bytes,
            "slots": self.n_slots,
            "frame_id": self.frame_id(),
        }

    def frame_health(self):
        """Analyze the newest frame.

        This is intended as a quick sanity check to catch the exact
        failure mode where a capture pipeline delivers buffers containing
        only black pixels.
        """

        frame = self.get_frame()

        if frame is None:
            return {
                "valid": False,
                "source": self.source,
                "width": self.width,
                "height": self.height,
                "nonblack_pct": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
            }

        frame = np.asarray(frame)

        gray = (
            0.299 * frame[..., 0]
            + 0.587 * frame[..., 1]
            + 0.114 * frame[..., 2]
        )

        nonblack = np.count_nonzero(
            gray > 8
        )

        return {
            "valid": True,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "nonblack_pct": (
                nonblack
                / gray.size
                * 100.0
            ),
            "mean": float(
                gray.mean()
            ),
            "std": float(
                gray.std()
            ),
            "min": int(
                frame.min()
            ),
            "max": int(
                frame.max()
            ),
        }

    # -------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------

    def stats(self) -> str:
        """Get daemon statistics through the Unix socket."""

        with socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        ) as sock:

            sock.settimeout(2.0)

            sock.connect(
                self._sock_path
            )

            sock.sendall(b"s")

            return (
                sock.recv(128)
                .decode()
                .strip()
            )

    def stats_dict(self) -> dict:
        """Parse the daemon stats string into ``{frames, fps, drops}``.

        Returns zeros on any error (e.g. daemon not running / socket gone),
        so callers can poll it unconditionally.
        """

        import re

        out = {"frames": 0, "fps": 0.0, "drops": 0}

        try:
            s = self.stats()
        except Exception:
            return out

        m = re.search(r"frames=(\d+)", s)

        if m:
            out["frames"] = int(m.group(1))

        m = re.search(r"fps=([\d.]+)", s)

        if m:
            out["fps"] = float(m.group(1))

        m = re.search(r"drops=(\d+)", s)

        if m:
            out["drops"] = int(m.group(1))

        return out

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------

    def close(
        self,
        timeout: float = 5.0,
    ):
        """Stop capture and release resources."""

        # ---------------------------------------------------------------
        # Tell daemon to quit
        # ---------------------------------------------------------------

        try:

            with socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM,
            ) as sock:

                sock.settimeout(
                    timeout
                )

                sock.connect(
                    self._sock_path
                )

                sock.sendall(b"q")

                sock.recv(4)

        except OSError:
            pass

        # ---------------------------------------------------------------
        # Wait for process
        # ---------------------------------------------------------------

        proc = getattr(
            self,
            "_proc",
            None,
        )

        if proc is not None:

            try:

                proc.wait(
                    timeout=timeout
                )

            except subprocess.TimeoutExpired:

                proc.kill()

                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

        # ---------------------------------------------------------------
        # Release NumPy buffer
        # ---------------------------------------------------------------

        self._buf = None

        # ---------------------------------------------------------------
        # Close mmap
        # ---------------------------------------------------------------

        mm = getattr(
            self,
            "_mm",
            None,
        )

        if mm is not None:

            try:
                mm.close()

            except (
                OSError,
                ValueError,
                BufferError,
            ):
                pass

        # ---------------------------------------------------------------
        # Close file descriptor
        # ---------------------------------------------------------------

        fd = getattr(
            self,
            "_fd",
            None,
        )

        if fd is not None:

            try:
                os.close(fd)

            except OSError:
                pass

        # ---------------------------------------------------------------
        # Remove IPC files
        # ---------------------------------------------------------------

        for path in (
            self._shm_path,
            self._sock_path,
            self._input_shm_path,
        ):

            try:
                os.unlink(path)

            except FileNotFoundError:
                pass

        # ---------------------------------------------------------------
        # Close the input-event ring
        # ---------------------------------------------------------------

        ring = getattr(
            self,
            "input_ring",
            None,
        )

        if ring is not None:

            try:
                ring.close()

            except (OSError, ValueError, BufferError):
                pass

            self.input_ring = None

    # -------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(
        self,
        *exc,
    ):
        self.close()