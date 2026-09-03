"""Portal capture smoke test.

Runs the full portal -> PipeWire -> shm pipeline and checks that we get
real (non-black) pixels. A KDE consent dialog appears; approve "Full
Screen" (or the game window) at the prompt.

    .venv/bin/python tools/portal_smoke.py [--width 128 --height 96]
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from anyplay.capture.native_capture import NativeCapture  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--frames", type=int, default=30,
                    help="frames to collect before analyzing")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds to wait for consent/handshake")
    ap.add_argument("--types", type=int, default=2,
                    help="portal source types: 1=monitor 2=window 4=region "
                         "(default 2, window picker)")
    args = ap.parse_args()

    print(f"starting portal capture (types={args.types}); approve the KDE "
          "consent dialog (Full Screen)...")
    t0 = time.monotonic()
    cap = NativeCapture(
        width=args.width, height=args.height,
        slots=args.slots, source="portal", portal_timeout=args.timeout,
        portal_types=args.types,
    )
    print(f"portal ready in {time.monotonic() - t0:.1f}s "
          f"({cap.width}x{cap.height}, {cap.n_slots} slots)")

    frames = []
    t1 = time.monotonic()
    while len(frames) < args.frames:
        f = cap.get_frame(timeout=5.0)
        if f is None:
            break
        frames.append(f.copy())  # copy; daemon overwrites slots
        if time.monotonic() - t1 > 30:
            break

    print(f"stats: {cap.stats()}")
    if not frames:
        print("FAIL: no frames received")
        cap.close()
        return 1

    arr = np.stack(frames).astype(np.int16)
    mean = arr.mean()
    std = arr.std()
    mx = arr.max()
    nonblack = (arr > 8).mean()
    print(f"frames={len(frames)}  mean={mean:.1f}  std={std:.1f}  "
          f"max={mx}  nonblack={nonblack:.1%}")
    if mean < 2.0:
        print("FAIL: frames are (nearly) all black")
        cap.close()
        return 2
    print("PASS: portal capture is producing real pixels")
    cap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
