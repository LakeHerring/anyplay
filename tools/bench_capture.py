#!/usr/bin/env python3

"""
Benchmark screen-capture backends: MSS vs native GStreamer daemon.

This benchmark measures both:

  1. Capture performance
  2. Capture validity

Capture validation checks:

  - Did we actually receive frames?
  - Resolution
  - Percentage of non-black pixels
  - Mean brightness
  - Brightness standard deviation
  - Minimum/maximum pixel values
  - Frame-to-frame variation
  - Whether native frame IDs advance
  - Preview PNGs for manual inspection

This is specifically intended to catch situations where a capture
pipeline successfully delivers buffers but those buffers contain
nothing but black pixels.

Example:

    python tools/bench_capture.py \
        --region 0,0,640,480 \
        --play-fps 30 \
        --duration 5

With a different output size:

    python tools/bench_capture.py \
        --region 0,0,1920,1080 \
        --size 640x480 \
        --play-fps 30 \
        --duration 5

Preview images are written to:

    capture_check/
"""


import argparse
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from mss import MSS


# ---------------------------------------------------------------------------
# Repository import support
# ---------------------------------------------------------------------------

# Allows:
#
#     python tools/bench_capture.py
#
# from the repository root.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def pct(sorted_vals, q):
    """Return a percentile from an already sorted list."""

    if not sorted_vals:
        return float("nan")

    index = min(
        len(sorted_vals) - 1,
        int(len(sorted_vals) * q)
    )

    return sorted_vals[index]


def frame_health(frame, black_threshold=8):
    """
    Analyze a captured frame.

    black_threshold:
        Pixels with grayscale brightness <= this value are considered
        effectively black.

    Returns a dictionary containing capture-health statistics.
    """

    if frame is None:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "channels": 0,
            "nonblack_pct": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }

    arr = np.asarray(frame)

    if arr.size == 0:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "channels": 0,
            "nonblack_pct": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }

    if arr.ndim == 2:
        height, width = arr.shape
        channels = 1
        gray = arr

    elif arr.ndim == 3:
        height, width, channels = arr.shape

        # OpenCV convention: BGR.
        #
        # If this is RGB the brightness statistics are still effectively
        # equivalent for this sanity check.
        if channels >= 3:
            gray = cv.cvtColor(
                arr[..., :3],
                cv.COLOR_BGR2GRAY
            )
        else:
            gray = arr[..., 0]

    else:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "channels": 0,
            "nonblack_pct": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
        }

    nonblack_pixels = np.count_nonzero(
        gray > black_threshold
    )

    total_pixels = gray.size

    return {
        "valid": True,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "nonblack_pct": (
            nonblack_pixels / total_pixels * 100.0
            if total_pixels
            else 0.0
        ),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": int(gray.min()),
        "max": int(gray.max()),
    }


def frame_difference(a, b):
    """
    Calculate the mean absolute pixel difference between two frames.

    A value near 0 means the frames are effectively identical.

    This is useful for detecting a frozen/black capture stream.
    """

    if a is None or b is None:
        return 0.0

    if a.shape != b.shape:
        return 0.0

    try:
        diff = cv.absdiff(
            np.asarray(a),
            np.asarray(b)
        )

        if diff.ndim == 3:
            diff = cv.cvtColor(
                diff[..., :3],
                cv.COLOR_BGR2GRAY
            )

        return float(diff.mean())

    except Exception:
        return 0.0


def aggregate_health(samples):
    """
    Aggregate multiple frame-health samples.

    min/max are preserved as integer extrema rather than being averaged.
    """

    if not samples:
        return frame_health(None)

    valid_samples = [
        sample
        for sample in samples
        if sample.get("valid", False)
    ]

    if not valid_samples:
        return frame_health(None)

    return {
        "valid": True,

        "width": int(
            valid_samples[-1]["width"]
        ),

        "height": int(
            valid_samples[-1]["height"]
        ),

        "channels": int(
            valid_samples[-1]["channels"]
        ),

        "nonblack_pct": float(
            np.mean([
                sample["nonblack_pct"]
                for sample in valid_samples
            ])
        ),

        "mean": float(
            np.mean([
                sample["mean"]
                for sample in valid_samples
            ])
        ),

        "std": float(
            np.mean([
                sample["std"]
                for sample in valid_samples
            ])
        ),

        # Preserve actual observed range.
        "min": int(
            min(
                sample["min"]
                for sample in valid_samples
            )
        ),

        "max": int(
            max(
                sample["max"]
                for sample in valid_samples
            )
        ),
    }


def evaluate_health(health, changed):
    """
    Produce a PASS/WARNING/FAIL status.

    These thresholds are deliberately aimed at detecting catastrophic
    capture failures, not judging whether a particular game scene looks
    visually good.
    """

    if not health["valid"]:
        return (
            "FAIL",
            "no valid frames received"
        )

    if (
        health["max"] <= 2
        or health["nonblack_pct"] < 0.5
    ):
        return (
            "FAIL",
            "capture is effectively completely black"
        )

    if (
        health["nonblack_pct"] < 2.0
        or health["mean"] < 3.0
    ):
        return (
            "WARNING",
            "capture is extremely dark"
        )

    if changed < 0.05:
        return (
            "WARNING",
            "frames appear almost completely static"
        )

    return (
        "PASS",
        "capture contains usable pixel data"
    )


def print_health(label, health, changed):
    """Print a capture-health report."""

    status, reason = evaluate_health(
        health,
        changed
    )

    print(f"{label}: {status}")

    print(
        f"  frame received   : "
        f"{'YES' if health['valid'] else 'NO'}"
    )

    print(
        f"  resolution       : "
        f"{health['width']}x{health['height']}"
    )

    print(
        f"  channels         : "
        f"{health['channels']}"
    )

    print(
        f"  non-black pixels : "
        f"{health['nonblack_pct']:8.3f}%"
    )

    print(
        f"  mean brightness  : "
        f"{health['mean']:8.3f}"
    )

    print(
        f"  std deviation    : "
        f"{health['std']:8.3f}"
    )

    print(
        f"  min pixel        : "
        f"{int(health['min']):8d}"
    )

    print(
        f"  max pixel        : "
        f"{int(health['max']):8d}"
    )

    print(
        f"  frame difference : "
        f"{changed:8.4f}"
    )

    print(
        f"  result            : "
        f"{reason}"
    )

    print()

    return status


def save_preview(frame, path, source_format="bgr"):
    """
    Save a captured frame as PNG.

    source_format:
        'bgr' - OpenCV/MSS-style BGR
        'rgb' - RGB frame
    """

    if frame is None:
        return False

    image = np.asarray(frame)

    if image.size == 0:
        return False

    image = image.copy()

    if (
        source_format.lower() == "rgb"
        and image.ndim == 3
        and image.shape[2] >= 3
    ):
        image = cv.cvtColor(
            image[..., :3],
            cv.COLOR_RGB2BGR
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return bool(
        cv.imwrite(
            str(path),
            image
        )
    )


# ---------------------------------------------------------------------------
# MSS benchmark
# ---------------------------------------------------------------------------

def bench_mss(
    region,
    duration: float,
    play_fps: int,
    out_w: int,
    out_h: int,
    preview_dir=None,
):
    x, y, w, h = region

    mon = {
        "left": x,
        "top": y,
        "width": w,
        "height": h,
    }

    # Capture the training-size output region.
    mon_out = {
        "left": x,
        "top": y,
        "width": out_w,
        "height": out_h,
    }

    with MSS() as screen:

        # ---------------------------------------------------------------
        # Capture validation
        # ---------------------------------------------------------------

        first_frame = None
        last_frame = None

        health_samples = []
        differences = []

        health_duration = min(
            duration,
            3.0
        )

        health_end = (
            time.monotonic()
            + health_duration
        )

        while time.monotonic() < health_end:

            arr = np.asarray(
                screen.grab(mon_out),
                dtype=np.uint8
            )

            # MSS returns BGRA.
            frame = arr[..., :3]

            if first_frame is None:
                first_frame = frame.copy()

            if last_frame is not None:
                differences.append(
                    frame_difference(
                        last_frame,
                        frame
                    )
                )

            last_frame = frame.copy()

            health_samples.append(
                frame_health(frame)
            )

        health = aggregate_health(
            health_samples
        )

        changed = (
            float(np.mean(differences))
            if differences
            else 0.0
        )

        # ---------------------------------------------------------------
        # Save preview images
        # ---------------------------------------------------------------

        if preview_dir:

            save_preview(
                first_frame,
                Path(preview_dir)
                / "mss_first.png",
                source_format="bgr"
            )

            save_preview(
                last_frame,
                Path(preview_dir)
                / "mss_last.png",
                source_format="bgr"
            )

        # ---------------------------------------------------------------
        # Raw polling benchmark
        # ---------------------------------------------------------------

        latencies = []
        frame_count = 0
        bytes_read = 0

        end = (
            time.monotonic()
            + duration
        )

        while time.monotonic() < end:

            t0 = time.monotonic()

            arr = np.asarray(
                screen.grab(mon),
                dtype=np.uint8
            )

            latencies.append(
                time.monotonic() - t0
            )

            frame_count += 1
            bytes_read += arr.nbytes

        latencies.sort()

        raw_fps = (
            frame_count / duration
            if duration > 0
            else 0.0
        )

        # ---------------------------------------------------------------
        # Play-loop cost
        # ---------------------------------------------------------------

        steps = int(
            duration * play_fps
        )

        costs = []

        for _ in range(steps):

            t0 = time.monotonic()

            arr = np.asarray(
                screen.grab(mon_out),
                dtype=np.uint8
            )

            # BGRA -> RGB.
            rgb = np.ascontiguousarray(
                arr[..., [2, 1, 0]][..., :3]
            )

            # Keep reference alive.
            _ = rgb

            costs.append(
                (time.monotonic() - t0)
                * 1000.0
            )

        costs.sort()

    return {
        "raw_fps": raw_fps,

        "raw_mb": (
            bytes_read / frame_count / 1e6
            if frame_count
            else 0.0
        ),

        "play_p50": pct(
            costs,
            0.50
        ),

        "play_p95": pct(
            costs,
            0.95
        ),

        "play_mb": (
            out_w
            * out_h
            * 3
            / 1e6
        ),

        "health": health,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# Native GStreamer benchmark
# ---------------------------------------------------------------------------

def bench_native(
    region,
    duration: float,
    play_fps: int,
    out_w: int,
    out_h: int,
    preview_dir=None,
    source: str = "daemon",
):
    from anyplay.capture.native_capture import (
        NativeCapture
    )

    x, y, w, h = region

    cap = NativeCapture(
        region=f"{x},{y},{w},{h}",
        width=out_w,
        height=out_h,
        fps=max(60, play_fps),
        source=source,
    )

    try:

        # ---------------------------------------------------------------
        # Capture validation
        # ---------------------------------------------------------------

        first_frame = None
        last_frame = None

        health_samples = []
        differences = []

        health_duration = min(
            duration,
            3.0
        )

        health_end = (
            time.monotonic()
            + health_duration
        )

        while time.monotonic() < health_end:

            fr = cap.get_frame()

            if fr is None:
                continue

            frame = np.asarray(fr)

            if first_frame is None:
                first_frame = frame.copy()

            if last_frame is not None:
                differences.append(
                    frame_difference(
                        last_frame,
                        frame
                    )
                )

            last_frame = frame.copy()

            health_samples.append(
                frame_health(frame)
            )

        health = aggregate_health(
            health_samples
        )

        changed = (
            float(np.mean(differences))
            if differences
            else 0.0
        )

        # ---------------------------------------------------------------
        # Save previews
        # ---------------------------------------------------------------

        if preview_dir:

            save_preview(
                first_frame,
                Path(preview_dir)
                / "native_first.png",
                source_format="rgb"
            )

            save_preview(
                last_frame,
                Path(preview_dir)
                / "native_last.png",
                source_format="rgb"
            )

        # ---------------------------------------------------------------
        # Native frame delivery rate
        # ---------------------------------------------------------------

        seen = 0
        last_id = None

        end = (
            time.monotonic()
            + duration
        )

        while time.monotonic() < end:

            fr = cap.get_frame()

            if fr is not None:

                fid = cap.frame_id()

                if fid != last_id:

                    seen += 1
                    last_id = fid

        raw_fps = (
            seen / duration
            if duration > 0
            else 0.0
        )

        stats = cap.stats()

        # ---------------------------------------------------------------
        # Play-loop cost
        # ---------------------------------------------------------------

        steps = int(
            duration * play_fps
        )

        costs = []

        for _ in range(steps):

            t0 = time.monotonic()

            fr = cap.get_frame()

            if fr is not None:

                # NativeCapture already gives us the frame buffer.
                #
                # This creates the contiguous array that the ML
                # pipeline would actually consume.
                rgb = np.ascontiguousarray(fr)

                _ = rgb

            costs.append(
                (time.monotonic() - t0)
                * 1000.0
            )

        costs.sort()

        return {
            "raw_fps": raw_fps,

            "raw_mb": (
                out_w
                * out_h
                * 3
                / 1e6
            ),

            "play_p50": pct(
                costs,
                0.50
            ),

            "play_p95": pct(
                costs,
                0.95
            ),

            "play_mb": (
                out_w
                * out_h
                * 3
                / 1e6
            ),

            "stats": stats,

            "health": health,
            "changed": changed,
        }

    finally:
        cap.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--region",
        default="0,0,640,480",
        help=(
            "capture region 'x,y,w,h' "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help=(
            "seconds per measurement "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--play-fps",
        type=int,
        default=30,
        help=(
            "play-loop rate "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--size",
        default="",
        help=(
            "native output size 'WxH' "
            "(default: region size)"
        ),
    )

    parser.add_argument(
        "--source",
        choices=("daemon", "portal"),
        default="daemon",
        help=(
            "native capture source: "
            "daemon (X11) or portal (Wayland/PipeWire)"
        ),
    )

    parser.add_argument(
        "--preview-dir",
        default="capture_check",
        help=(
            "directory for preview PNGs "
            "(default: %(default)s)"
        ),
    )

    args = parser.parse_args()
    # ---------------------------------------------------------------
    # Parse region
    # ---------------------------------------------------------------

    try:

        region = tuple(
            int(v.strip())
            for v in args.region.split(",")
        )

    except ValueError as exc:

        raise SystemExit(
            "--region must be x,y,w,h"
        ) from exc

    if len(region) != 4:
        raise SystemExit(
            "--region must be x,y,w,h"
        )

    x, y, w, h = region

    if w <= 0 or h <= 0:
        raise SystemExit(
            "region width/height must be greater than zero"
        )

    # ---------------------------------------------------------------
    # Parse output size
    # ---------------------------------------------------------------

    if args.size:

        try:

            out_w, out_h = (
                int(v.strip())
                for v in args.size.lower().split("x")
            )

        except ValueError as exc:

            raise SystemExit(
                "--size must be WxH, e.g. 640x480"
            ) from exc

        if out_w <= 0 or out_h <= 0:
            raise SystemExit(
                "--size dimensions must be greater than zero"
            )

    else:

        out_w = w
        out_h = h

    # ---------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("SCREEN CAPTURE BENCHMARK")
    print("=" * 72)

    print(
        f"region        : {region}"
    )

    print(
        f"duration      : {args.duration}s"
    )

    print(
        f"play FPS      : {args.play_fps}"
    )

    print(
        f"native output : {out_w}x{out_h}"
    )

    print(
        f"previews      : {args.preview_dir}"
    )

    print()

    # ---------------------------------------------------------------
    # MSS
    # ---------------------------------------------------------------

    print("-" * 72)
    print("MSS")
    print("-" * 72)

    try:

        m = bench_mss(
            region,
            args.duration,
            args.play_fps,
            out_w,
            out_h,
            args.preview_dir,
        )

        m_status = print_health(
            "MSS capture health",
            m["health"],
            m["changed"],
        )

    except Exception as exc:

        print(
            f"MSS: ERROR: {exc}"
        )

        m = {
            "raw_fps": 0.0,
            "raw_mb": 0.0,
            "play_p50": float("nan"),
            "play_p95": float("nan"),
            "play_mb": 0.0,
            "health": frame_health(None),
            "changed": 0.0,
        }

        m_status = "FAIL"

    # ---------------------------------------------------------------
    # Native GStreamer
    # ---------------------------------------------------------------

    print("-" * 72)
    print(f"NATIVE GSTREAMER ({args.source.upper()})")
    print("-" * 72)

    try:

        n = bench_native(
            region,
            args.duration,
            args.play_fps,
            out_w,
            out_h,
            args.preview_dir,
            args.source,
        )

        n_status = print_health(
            "Native capture health",
            n["health"],
            n["changed"],
        )

    except Exception as exc:

        print(
            f"Native: ERROR: {exc}"
        )

        n = {
            "raw_fps": 0.0,
            "raw_mb": 0.0,
            "play_p50": float("nan"),
            "play_p95": float("nan"),
            "play_mb": 0.0,
            "stats": "unavailable",
            "health": frame_health(None),
            "changed": 0.0,
        }

        n_status = "FAIL"

    # ---------------------------------------------------------------
    # Performance
    # ---------------------------------------------------------------

    print("-" * 72)
    print("PERFORMANCE")
    print("-" * 72)

    print(
        f"{'':10s}"
        f"{'raw FPS':>12s}"
        f"{'play p50':>14s}"
        f"{'play p95':>14s}"
        f"{'bytes/step':>14s}"
    )

    print(
        f"{'mss':10s}"
        f"{m['raw_fps']:10.0f}"
        f"{m['play_p50']:11.3f} ms"
        f"{m['play_p95']:11.3f} ms"
        f"{m['play_mb']:11.2f} MB"
    )

    print(
        f"{'native':10s}"
        f"{n['raw_fps']:10.0f}"
        f"{n['play_p50']:11.4f} ms"
        f"{n['play_p95']:11.4f} ms"
        f"{n['play_mb']:11.2f} MB"
    )

    print()

    print(
        "Raw capture:"
    )

    print(
        f"  MSS    : "
        f"{m['raw_fps']:.0f} grabs/s"
    )

    print(
        f"  Native : "
        f"{n['raw_fps']:.0f} delivered fps"
    )

    print()

    print(f"Native {args.source}: {n['stats']}")

    print(
        f"  {n.get('stats', 'unavailable')}"
    )

    print()

    # ---------------------------------------------------------------
    # Preview files
    # ---------------------------------------------------------------

    print("-" * 72)
    print("PREVIEW FILES")
    print("-" * 72)

    print(
        f"MSS first  : "
        f"{args.preview_dir}/mss_first.png"
    )

    print(
        f"MSS last   : "
        f"{args.preview_dir}/mss_last.png"
    )

    print(
        f"Native first: "
        f"{args.preview_dir}/native_first.png"
    )

    print(
        f"Native last : "
        f"{args.preview_dir}/native_last.png"
    )

    print()

    # ---------------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------------

    print("=" * 72)
    print("CAPTURE VERDICT")
    print("=" * 72)

    print(
        f"MSS    : {m_status}"
    )

    print(
        f"Native : {n_status}"
    )

    print()

    if (
        m_status == "FAIL"
        and n_status == "FAIL"
    ):

        print(
            "CRITICAL: Both capture backends failed "
            "the capture sanity check."
        )

        print(
            "DO NOT use this capture configuration "
            "for training data."
        )

    elif n_status == "FAIL":

        print(
            "CRITICAL: Native GStreamer capture failed."
        )

        print(
            "The native pipeline should NOT be used "
            "for training data until fixed."
        )

        if m_status == "PASS":

            print(
                "MSS appears to contain valid pixels."
            )

    elif m_status == "FAIL" and n_status == "PASS":

        print(
            "Native GStreamer capture is producing "
            "valid pixel data."
        )

        print(
            "MSS appears to be unusable/black."
        )

        print(
            "This is consistent with an X11/XWayland "
            "capture limitation under Wayland."
        )

    else:

        print(
            "Both capture paths passed the basic "
            "pixel sanity check."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The PNG previews are the final manual sanity check."
    )

    print(
        "Open them before recording long training sessions."
    )

    print()


if __name__ == "__main__":
    main()