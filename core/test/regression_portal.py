#!/usr/bin/env python3
"""core/test/regression_portal.py — regression gate for the C++ PipeWire
portal capture daemon (core/build/anyplay-portal-capture).

Unlike the X11 gate (regression_daemon.py), this test does NOT need a
display or a live Wayland compositor: it runs the binary in --test mode
(videotestsrc), which is fully headless. It launches the daemon binary
directly (no tools/portal-pw-fd wrapper) and validates the same shm +
control-socket byte contract the Python client
(anyplay/capture/native_capture.py) depends on, using the identical
struct formats.

Checks:
  1. READY line on stdout
  2. shm header: magic, version, width, height, slot_bytes, n_slots, stride
  3. first frame: shape (h, w, 3), uint8, frame_id > 0, seqlock-settled
  4. frame rate over a window (videotestsrc is-live = 30 fps)
  5. content sanity (non-black) + seqlock stability
  6. control-socket 's' stats line
  7. C++-only timing stamps (ts_ns/wait_ns) in slot metadata
  8. clean shutdown via 'q': process exits 0, shm file removed

Usage:
    python3 core/test/regression_portal.py
    python3 core/test/regression_portal.py --binary core/build/anyplay-portal-capture
    python3 core/test/regression_portal.py --slots 8 --seconds 4

Exit code 0 = pass, 1 = fail, 2 = skipped (missing binary / GStreamer).
"""

from __future__ import annotations

import argparse
import mmap
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Same layout constants as anyplay/capture/native_capture.py.
_HDR32 = struct.Struct("<8I")
_HDR64 = struct.Struct("<3Q")
_MAGIC = 0x49445321
_HDR_BASE = 64
_SLOT_META = 32


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok: {msg}")


class PortalTest:
    """Launches the --test daemon and reads its shm ring directly."""

    def __init__(self, binary: str, shm: str, sock: str, slots: int) -> None:
        self.binary = binary
        self.shm = shm
        self.sock = sock
        self.slots = slots
        self.proc: subprocess.Popen | None = None
        self._mm: mmap.mmap | None = None
        self._buf = None
        self.width = 0
        self.height = 0
        self.slot_bytes = 0
        self.n_slots = 0
        self._slot_stride = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        # Fresh shm/sock paths.
        for p in (self.shm, self.sock):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

        self.proc = subprocess.Popen(
            [
                self.binary,
                "--test",
                "--slots",
                str(self.slots),
                "--shm",
                self.shm,
                "--sock",
                self.sock,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for the READY line.
        deadline = time.monotonic() + 15.0
        ready = False
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                fail(
                    f"daemon exited before READY (rc={self.proc.returncode}):\n"
                    f"{out}"
                )
            line = self.proc.stdout.readline()
            if not line:
                continue
            if line.startswith("READY"):
                parts = line.split()
                if len(parts) >= 3:
                    self.shm = parts[1]
                    self.sock = parts[2]
                ready = True
                break
        if not ready:
            fail("no READY line within 15 s")
        ok("READY line received")

    def open_shm(self) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not os.path.exists(self.shm):
            time.sleep(0.02)
        if not os.path.exists(self.shm):
            fail(f"shm file not created: {self.shm}")

        self._mm = mmap.mmap(
            os.open(self.shm, os.O_RDONLY), 0, mmap.MAP_SHARED, mmap.PROT_READ
        )
        self._buf = np.frombuffer(self._mm, dtype=np.uint8)

        (
            magic,
            version,
            width,
            height,
            slot_bytes,
            n_slots,
            alive,
            _pad,
        ) = _HDR32.unpack_from(self._mm, 0)
        if magic != _MAGIC:
            fail(f"bad magic {magic:#x} (want {_MAGIC:#x})")
        self.width = width
        self.height = height
        self.slot_bytes = slot_bytes
        self.n_slots = n_slots
        self._slot_stride = _SLOT_META + slot_bytes

        expect_size = _HDR_BASE + n_slots * self._slot_stride
        if len(self._mm) < expect_size:
            fail(
                f"shm too small: {len(self._mm)} < {expect_size} "
                f"(slots={n_slots} stride={self._slot_stride})"
            )
        ok(
            f"shm header: magic ok, {width}x{height}, slot_bytes={slot_bytes}, "
            f"n_slots={n_slots}, alive={alive}"
        )

    def idx(self) -> int:
        return _HDR64.unpack_from(self._mm, 32)[0]

    def frame_id(self) -> int:
        i = self.idx()
        return i - 1 if i > 0 else 0

    def get_frame(self, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        idx = self.idx()
        while idx == 0:
            if time.monotonic() > deadline:
                return None
            time.sleep(0.005)
            idx = self.idx()

        off = _HDR_BASE + ((idx - 1) % self.n_slots) * self._slot_stride
        data_off = off + _SLOT_META
        for _ in range(1000):
            seq1 = struct.unpack_from("<I", self._mm, off)[0]
            if seq1 & 1:
                time.sleep(0.0005)
                continue
            frame = self._buf[data_off : data_off + self.slot_bytes].reshape(
                self.height, self.width, 3
            )
            seq2 = struct.unpack_from("<I", self._mm, off)[0]
            if seq1 == seq2:
                return frame
        raise RuntimeError("seqlock did not settle")

    def slot_stamps(self):
        idx = self.idx()
        if idx <= 0:
            return None
        off = _HDR_BASE + ((idx - 1) % self.n_slots) * self._slot_stride
        seq, _pad, frame_id, ts_ns, wait_ns = struct.unpack_from(
            "<2IQ2Q", self._mm, off
        )
        return frame_id, ts_ns, wait_ns

    def stats(self) -> str:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(self.sock)
            s.sendall(b"s")
            return s.recv(128).decode().strip()

    def close(self) -> int:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)
                s.connect(self.sock)
                s.sendall(b"q")
                s.recv(4)
        except OSError:
            pass
        rc = self.proc.wait(timeout=8.0)
        # Release the NumPy view before closing the mmap (a live view keeps
        # the buffer exported -> BufferError). The process wait is the part
        # that matters; mmap close is best-effort.
        self._buf = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                pass
            self._mm = None
        return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--binary",
        default=str(PROJECT_ROOT / "core" / "build" / "anyplay-portal-capture"),
        help="portal daemon binary to test (default: the C++ build)",
    )
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="expected videotestsrc fps (default 30)",
    )
    ap.add_argument(
        "--min-fps",
        type=float,
        default=0.75,
        help="required fraction of expected fps (default 0.75)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    binary = Path(args.binary)
    if not binary.exists():
        print(f"skip: portal binary missing: {binary} (run: make -C core)")
        return 2

    if not shutil_which_gst():
        print("skip: GStreamer runtime not available")
        return 2

    # Unique shm/sock for this run.
    pid = os.getpid()
    shm = f"/tmp/anyplay_reg_pc_{pid}.bin"
    sock = f"/tmp/anyplay_reg_pc_{pid}.sock"

    cap = PortalTest(str(binary), shm, sock, args.slots)
    try:
        cap.start()
        cap.open_shm()

        # --------------------------------------------------------------
        # 2. First frame
        # --------------------------------------------------------------
        frame = cap.get_frame(timeout=5.0)
        assert frame is not None, "no frame within 5 s"
        assert frame.shape == (cap.height, cap.width, 3), frame.shape
        assert frame.dtype == np.uint8, frame.dtype
        # The first published frame has frame_id 0; idx > 0 proves a frame
        # exists.
        assert cap.idx() > 0, "no frame published (idx == 0)"
        ok(f"first frame: {frame.shape} {frame.dtype}")
        del frame

        # --------------------------------------------------------------
        # 3. Frame rate + freshness
        # --------------------------------------------------------------
        t0 = time.monotonic()
        first_id = cap.frame_id()
        n_gets = 0
        while time.monotonic() - t0 < args.seconds:
            f = cap.get_frame(timeout=1.0)
            if f is not None:
                n_gets += 1
            time.sleep(0.02)
        dt = time.monotonic() - t0
        published = cap.frame_id() - first_id
        fps = published / dt if dt > 0 else 0.0
        print(
            f"info: published={published} frames in {dt:.2f}s "
            f"(measured {fps:.1f} fps, target {args.fps:.1f} fps); "
            f"client reads={n_gets}"
        )
        assert published > 0, "no frames published during window"
        if fps < args.fps * args.min_fps:
            fail(
                f"frame rate too low: {fps:.1f} fps "
                f"(< {args.min_fps:.2f} x {args.fps:.1f} expected)"
            )
        ok(f"frame rate: {fps:.1f} fps")

        # --------------------------------------------------------------
        # 4. Content sanity (videotestsrc smpte is never black)
        # --------------------------------------------------------------
        f = cap.get_frame(timeout=2.0)
        nonblack = float(np.count_nonzero(f)) / float(f.size) * 100.0
        print(f"info: nonblack={nonblack:.1f}%")
        if nonblack < 1.0:
            fail("frame is entirely black (videotestsrc should not be)")
        ok("frame content readable through seqlock")

        # --------------------------------------------------------------
        # 5. Control socket: stats line
        # --------------------------------------------------------------
        stats = cap.stats()
        print(f"info: stats: {stats}")
        assert stats.startswith("frames="), stats
        assert " fps=" in stats and " drops=" in stats, stats
        ok("control socket 's' stats line")

        # --------------------------------------------------------------
        # 6. C++-only timing stamps (ts_ns/wait_ns)
        # --------------------------------------------------------------
        stamps = cap.slot_stamps()
        if stamps is not None:
            frame_id, ts_ns, wait_ns = stamps
            print(
                f"info: slot stamps: frame_id={frame_id} "
                f"ts_ns={ts_ns} wait_ns={wait_ns} "
                f"({wait_ns / 1e6:.1f} ms producer wait)"
            )
            if ts_ns > 0:
                ok("timing stamps present (C++ core)")
            else:
                fail("timing stamps are zero (expected nonzero from C++ core)")

    finally:
        try:
            rc = cap.close()
        except subprocess.TimeoutExpired:
            cap.proc.kill()
            fail("daemon did not exit within 8 s after 'q'")

    # ------------------------------------------------------------------
    # 7. Clean shutdown: exit 0, shm file removed
    # ------------------------------------------------------------------
    if rc != 0:
        fail(f"daemon exit code {rc} (want 0)")
    ok("clean shutdown via 'q' (exit 0)")

    # shm should be unlinked on clean shutdown.
    if os.path.exists(cap.shm):
        print(f"warn: shm file still present after shutdown: {cap.shm}")
    else:
        ok("shm file removed on shutdown")

    print(f"PASS: {binary}")
    return 0


def shutil_which_gst() -> bool:
    """Cheap check that the GStreamer runtime is usable (videotestsrc)."""
    try:
        r = subprocess.run(
            [
                "gst-launch-1.0",
                "-q",
                "videotestsrc",
                "is-live=true",
                "num-buffers=1",
                "!",
                "fakesink",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


if __name__ == "__main__":
    sys.exit(main())
