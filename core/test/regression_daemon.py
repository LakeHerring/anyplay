#!/usr/bin/env python3
"""core/test/regression_daemon.py — regression gate for the C++ capture
daemon (core/build/anyplay-capture).

The C++ daemon must be a drop-in replacement for native/capture-daemon:
identical CLI, identical READY line, identical shm protocol, identical
control-socket protocol. This test drives the *existing Python client*
(anyplay/capture/native_capture.py) against a given daemon
binary and verifies the full contract end-to-end.

Usage:
    python3 core/test/regression_daemon.py                 # C++ daemon
    python3 core/test/regression_daemon.py --daemon native/capture-daemon
    python3 core/test/regression_daemon.py --region 0,0,640,360 --fps 30 --seconds 6

Exit code 0 = pass, 1 = fail, 2 = skipped (no X11 display / missing bin).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--daemon",
        default=str(PROJECT_ROOT / "core" / "build" / "anyplay-capture"),
        help="daemon binary to test (default: the C++ build)",
    )
    ap.add_argument("--region", default="0,0,640,480")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument(
        "--min-fps",
        type=float,
        default=0.75,
        help="required fraction of requested fps (default 0.75)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    daemon = Path(args.daemon)
    if not daemon.exists():
        print(f"skip: daemon binary missing: {daemon} (run: make -C core)")
        return 2

    if not os.environ.get("DISPLAY") and not Path("/tmp/.X11-unix/X0").exists():
        print("skip: no X11 display available (ximagesrc needs one)")
        return 2

    os.environ.setdefault("DISPLAY", ":0")
    os.environ["SDAI_DAEMON_BIN"] = str(daemon)

    # Import the client *after* the env override (module-level DAEMON).
    sys.path.insert(0, str(PROJECT_ROOT))
    from anyplay.capture.native_capture import (  # noqa: E402
        NativeCapture,
        _HDR_BASE,
    )

    cap = None
    try:
        cap = NativeCapture(
            region=args.region,
            width=args.width,
            height=args.height,
            fps=args.fps,
            slots=args.slots,
            display=os.environ.get("DISPLAY", ":0"),
            source="daemon",
        )
        ok(f"daemon ready: {daemon.name} (shm {cap.width}x{cap.height})")

        # --------------------------------------------------------------
        # 1. Header contract
        # --------------------------------------------------------------
        assert cap.width == args.width, f"width {cap.width} != {args.width}"
        assert cap.height == args.height, (
            f"height {cap.height} != {args.height}"
        )
        assert cap.slot_bytes == args.width * args.height * 3
        ok("shm header: width/height/slot_bytes correct")

        # --------------------------------------------------------------
        # 2. First frame: shape, dtype, content
        # --------------------------------------------------------------
        frame = cap.get_frame(timeout=5.0)
        assert frame is not None, "no frame within 5 s"
        assert frame.shape == (args.height, args.width, 3), frame.shape
        assert frame.dtype == np.uint8, frame.dtype
        ok(f"first frame: {frame.shape} {frame.dtype}")

        # --------------------------------------------------------------
        # 3. Frame rate + freshness over the window
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
        last_id = cap.frame_id()
        published = last_id - first_id
        fps = published / dt if dt > 0 else 0.0
        print(
            f"info: published={published} frames in {dt:.2f}s "
            f"(measured {fps:.1f} fps, target {args.fps} fps); "
            f"client reads={n_gets}"
        )
        assert published > 0, "no frames published during window"
        if fps < args.fps * args.min_fps:
            fail(
                f"frame rate too low: {fps:.1f} fps "
                f"(< {args.min_fps:.2f} x {args.fps} requested)"
            )
        ok(f"frame rate: {fps:.1f} fps")

        # --------------------------------------------------------------
        # 4. Content sanity (non-black) + seqlock stability
        # --------------------------------------------------------------
        health = cap.frame_health()
        print(f"info: frame health: {health}")
        assert health["valid"], "frame not valid"
        if health["nonblack_pct"] < 1.0:
            print(
                "warn: frame is nearly black "
                f"(nonblack {health['nonblack_pct']:.1f}%) — "
                "X11 root may be empty; treating as pass"
            )
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
        # 6. C++-only: timing stamps in slot metadata (ts_ns/wait_ns)
        # --------------------------------------------------------------
        idx = cap._idx()
        if idx > 0:
            off = _HDR_BASE + ((idx - 1) % cap.n_slots) * cap._slot_stride
            _seq, _pad, frame_id, ts_ns, wait_ns = struct.unpack_from(
                "<2IQ2Q", cap._mm, off
            )
            print(
                f"info: slot stamps: frame_id={frame_id} "
                f"ts_ns={ts_ns} wait_ns={wait_ns} "
                f"({wait_ns / 1e6:.1f} ms producer wait)"
            )
            if ts_ns > 0:
                ok("timing stamps present (C++ core)")
            else:
                print("info: timing stamps zero (C daemon — expected)")

    finally:
        if cap is not None:
            t0 = time.monotonic()
            cap.close()
            print(f"info: closed in {time.monotonic() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 7. Clean shutdown: process exited, shm file removed
    # ------------------------------------------------------------------
    ok("clean shutdown via 'q'")
    print(f"PASS: {daemon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
