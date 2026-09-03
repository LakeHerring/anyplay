#!/usr/bin/env python3
"""core/test/regression_input.py — regression gate for the C++ evdev
input ring (P-core step 2).

Headless and deterministic: creates a virtual keyboard and a virtual
mouse with UInput, launches the daemon in --input-only mode reading
them (no X11 display, no PipeWire required), injects a scripted
event sequence, and verifies the Python ring client
(anyplay/capture/input_ring.py) sees exactly the right per-device
subsequences: order, (type, code, value) payloads, dev_id tagging,
kernel timestamps, zero drops.

Phase 2 checks the stale-instance sentinel: a second daemon reopened
with O_TRUNC on the same input shm path must kill the first daemon
(one-shot SIGBUS re-raise) while itself staying alive.

Usage:
    python3 core/test/regression_input.py
    python3 core/test/regression_input.py --daemon core/build/anyplay-portal-capture

Exit code 0 = pass, 1 = fail, 2 = skipped (missing bin / no /dev/uinput).
"""

from __future__ import annotations

import argparse
import glob
import os
import resource
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The sentinel phase intentionally crashes daemon A with SIGBUS
# (truncated shm). Suppress core dumps for this process and its daemon
# children so each run does not leave a coredump on disk.
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok: {msg}")


# ---------------------------------------------------------------------------
# UInput helpers
# ---------------------------------------------------------------------------

def create_uinput(capabilities, timeout: float = 5.0):
    """Create a UInput device and wait for its /dev/input/event* node.

    Returns (UInput, path). Creation is one-at-a-time so the new node
    can be identified unambiguously by diffing the glob.
    """
    import evdev
    from evdev import UInput

    before = set(glob.glob("/dev/input/event*"))
    dev = UInput(capabilities)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = set(glob.glob("/dev/input/event*")) - before
        if len(new) == 1:
            path = next(iter(new))
            time.sleep(0.1)  # let udev settle before we read from it
            return dev, path
        time.sleep(0.05)
    dev.close()
    raise RuntimeError("UInput node did not appear within %.1f s" % timeout)


# ---------------------------------------------------------------------------
# Daemon lifecycle helpers
# ---------------------------------------------------------------------------

def launch_daemon(bin_path: str, kb: str, ms: str, shm: str, sock: str):
    args = [
        bin_path,
        "--input-only",
        "--keyboard", kb,
        "--pointer", ms,
        "--input-shm", shm,
        "--sock", sock,
    ]
    return subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def read_ready_line(p: subprocess.Popen, timeout: float = 10.0) -> str:
    """Read daemon stdout until a READY line arrives.

    Scans line by line (the daemon may emit other lines first, e.g. the
    [evdev] open messages), and reports the merged output if the daemon
    dies before becoming ready.
    """
    fd = p.stdout.fileno()
    buf = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode(errors="replace")
                if text.startswith("READY"):
                    return text
        if p.poll() is not None:
            err = p.stderr.read().decode(errors="replace")
            out = buf.decode(errors="replace")
            detail = " ".join(x for x in (out.strip(), err.strip()) if x)
            raise RuntimeError(f"daemon exited early: {detail}")
    raise RuntimeError("no READY line within %.0f s" % timeout)


def _dump_daemon_tail(p: subprocess.Popen) -> None:
    """On test failure: stop a still-running daemon and print its output
    so hangs and early crashes are diagnosable from the log."""
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            print("  (daemon ignored SIGTERM and was killed)")
    try:
        out = p.stdout.read().decode(errors="replace").strip()
    except Exception:
        out = ""
    try:
        err = p.stderr.read().decode(errors="replace").strip()
    except Exception:
        err = ""
    for label, text in (("daemon stdout", out), ("daemon stderr", err)):
        if text:
            print(f"  --- {label} (exit={p.returncode}):")
            for line in text.split("\n")[-15:]:
                print(f"      {line}")


def socket_stats(sock_path: str) -> str:
    """Send 's' (one command per connection) and read the stats line."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(b"s")
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.decode(errors="replace").strip()
    finally:
        s.close()


def socket_quit(sock_path: str) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(b"q")
        s.recv(4)
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Test body
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--daemon",
        default=str(PROJECT_ROOT / "core" / "build" / "anyplay-capture"),
        help="daemon binary to test (default: the C++ X11 build)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    daemon = Path(args.daemon)
    if not daemon.exists():
        print(f"skip: daemon binary missing: {daemon} (run: make -C core)")
        return 2
    if not os.path.exists("/dev/uinput"):
        print("skip: /dev/uinput missing (uinput module not loaded)")
        return 2
    if not glob.glob("/dev/input/event*"):
        print("skip: no /dev/input/event* nodes (no udev)")
        return 2
    try:
        import evdev  # noqa: F401
        from evdev import ecodes
    except ImportError:
        print("skip: python-evdev not installed (pip install evdev)")
        return 2

    EV_KEY, EV_REL = ecodes.EV_KEY, ecodes.EV_REL
    KEY_A, KEY_D, KEY_E = ecodes.KEY_A, ecodes.KEY_D, ecodes.KEY_E
    BTN_LEFT, REL_X, REL_Y = ecodes.BTN_LEFT, ecodes.REL_X, ecodes.REL_Y

    # Scripted sequence, interleaved across devices. Expected per-device
    # subsequences (EV_SYN is filtered by the daemon, so syns are absent).
    # dev_id 0 = keyboard (CLI order), dev_id 1 = pointer.
    KB_EXPECTED = [
        (EV_KEY, KEY_A, 1), (EV_KEY, KEY_A, 0),
        (EV_KEY, KEY_D, 1), (EV_KEY, KEY_D, 0),
        (EV_KEY, KEY_E, 1), (EV_KEY, KEY_E, 0),
    ]
    MS_EXPECTED = [
        (EV_KEY, BTN_LEFT, 1),
        (EV_REL, REL_X, 7), (EV_REL, REL_X, -3), (EV_REL, REL_Y, -2),
        (EV_KEY, BTN_LEFT, 0),
    ]

    kb = ms = pA = pB = None
    ring = None
    shm_a = sock_a = sock_b = ""
    try:
        # ----------------------------------------------------------------
        # Phase 1: virtual devices
        # ----------------------------------------------------------------
        kb, kb_path = create_uinput({EV_KEY: [KEY_A, KEY_D, KEY_E]})
        ok(f"UInput keyboard at {kb_path}")
        ms, ms_path = create_uinput(
            {EV_KEY: [BTN_LEFT], EV_REL: [REL_X, REL_Y]}
        )
        ok(f"UInput mouse at {ms_path}")

        # ----------------------------------------------------------------
        # Phase 2: launch daemon A (--input-only, headless)
        # ----------------------------------------------------------------
        shm_a = tempfile.mktemp(prefix="anyplay_regress_inp_", dir="/tmp")
        sock_a = tempfile.mktemp(prefix="anyplay_regress_inp_")
        pA = launch_daemon(str(daemon), kb_path, ms_path, shm_a, sock_a)
        ready = read_ready_line(pA)
        parts = ready.split()
        # Layout: READY <frame_shm> <sock> input=<input_shm>
        if len(parts) < 3 or parts[2] != sock_a:
            fail(f"unexpected READY line: {ready!r}")
        in_field = [p for p in parts if p.startswith("input=")]
        if len(in_field) != 1 or in_field[0] != f"input={shm_a}":
            fail(f"READY line missing input=<shm>: {ready!r}")
        ok(f"daemon A ready: {ready.strip()}")

        # ----------------------------------------------------------------
        # Phase 3: scripted injection + drain
        # ----------------------------------------------------------------
        sys.path.insert(0, str(PROJECT_ROOT))
        from anyplay.capture.input_ring import InputEventRing

        ring = InputEventRing(shm_a)
        got: list = []
        stop = threading.Event()

        def drainer() -> None:
            while not stop.is_set():
                got.extend(ring.drain())
                time.sleep(0.05)

        t = threading.Thread(target=drainer, daemon=True)
        t.start()
        time.sleep(0.2)  # let the drain loop settle

        t0 = time.time()
        kb.write(EV_KEY, KEY_A, 1); kb.syn()
        ms.write(EV_KEY, BTN_LEFT, 1); ms.syn()
        kb.write(EV_KEY, KEY_A, 0); kb.syn()
        ms.write(EV_REL, REL_X, 7); ms.syn()
        kb.write(EV_KEY, KEY_D, 1); kb.syn()
        ms.write(EV_REL, REL_X, -3); ms.syn()
        ms.write(EV_REL, REL_Y, -2); ms.syn()
        kb.write(EV_KEY, KEY_D, 0); kb.syn()
        ms.write(EV_KEY, BTN_LEFT, 0); ms.syn()
        kb.write(EV_KEY, KEY_E, 1); kb.syn()
        kb.write(EV_KEY, KEY_E, 0); kb.syn()
        time.sleep(0.5)  # let the ring drain fully

        stop.set()
        t.join()
        got.extend(ring.drain())
        n_events = ring.events
        n_drops = ring.drops
        was_alive = ring.alive
        ring_idx = ring.idx
        ring.close()
        ring = None

        kb_seq = [(e.type, e.code, e.value) for e in got if e.dev_id == 0]
        ms_seq = [(e.type, e.code, e.value) for e in got if e.dev_id == 1]
        other = [e for e in got if e.dev_id not in (0, 1)]

        ring_state = (
            f"ring: idx={ring_idx} events={n_events} drops={n_drops} "
            f"alive={was_alive} drained_total={len(got)}"
        )
        assert not other, (
            f"unexpected dev_id in {[(e.dev_id,) for e in other]} | {ring_state}"
        )
        assert kb_seq == KB_EXPECTED, (
            f"keyboard subsequence mismatch:\n"
            f"  expected={KB_EXPECTED}\n  got     ={kb_seq}\n  {ring_state}"
        )
        assert ms_seq == MS_EXPECTED, (
            f"mouse subsequence mismatch:\n"
            f"  expected={MS_EXPECTED}\n  got     ={ms_seq}\n  {ring_state}"
        )
        ok(f"per-device subsequences exact "
           f"(kb={len(kb_seq)} ms={len(ms_seq)}, {len(got)} total)")

        assert n_drops == 0, f"ring drops={n_drops}"
        assert n_events == len(KB_EXPECTED) + len(MS_EXPECTED), (
            f"ring.events={n_events}, expected "
            f"{len(KB_EXPECTED) + len(MS_EXPECTED)}"
        )
        assert was_alive, "ring not alive after injection"
        ok(f"ring counters: events={n_events} drops={n_drops}")

        # Timestamps must be kernel-stamped and within the injection
        # window (allowing for scheduling skew).
        ts = [e.ts for e in got]
        assert all(t0 - 2.0 <= x <= time.time() + 0.5 for x in ts), (
            f"event timestamps outside injection window: {min(ts)}..{max(ts)}"
        )
        ok(f"timestamps within window: {min(ts) - t0:+.3f}..{max(ts) - t0:+.3f} s")

        # ----------------------------------------------------------------
        # Phase 4: control-socket stats line
        # ------------------------------------------------------------------
        stats = socket_stats(sock_a)
        print(f"info: stats: {stats}")
        assert stats.startswith("frames=0"), stats
        assert f"in_events={len(KB_EXPECTED) + len(MS_EXPECTED)}" in stats, stats
        assert "in_drops=0" in stats, stats
        ok("control socket stats: in_events/in_drops")

        # ----------------------------------------------------------------
        # Phase 5: stale-instance sentinel
        #   A's publish path reads n_slots/idx from the shared header, so
        #   a same-size O_TRUNC re-extension leaves A alive (constrained
        #   to the new file's geometry). The sentinel's real trigger is a
        #   file *destroyed under the mapping*: the new instance's
        #   constructor truncates the path to zero before re-extending, and
        #   a constructor that crashes mid-flight leaves a zero-byte file.
        #   Either way, the next ring touch by the stale daemon must die
        #   by one-shot SIGBUS re-raise, and a fresh daemon must then be
        #   able to take over the path cleanly.
        # ------------------------------------------------------------------
        # NOTE: the ring client was closed before this phase — its mapping
        # is stale once the file is destroyed and must not be touched.
        os.truncate(shm_a, 0)  # the destroyed-under-mapping state

        kb.write(EV_KEY, KEY_E, 1); kb.syn()
        kb.write(EV_KEY, KEY_E, 0); kb.syn()

        deadline = time.monotonic() + 5.0
        while pA.poll() is None:
            if time.monotonic() > deadline:
                fail("daemon A did not die after its shm was destroyed "
                     "(expected SIGBUS)")
            time.sleep(0.1)
        assert pA.returncode < 0, (
            f"daemon A exited normally ({pA.returncode}); expected death "
            f"by signal"
        )
        assert pA.returncode == -signal.SIGBUS, (
            f"daemon A killed by signal {-pA.returncode}, expected SIGBUS "
            f"({signal.SIGBUS})"
        )
        ok("stale-instance sentinel: daemon A died by SIGBUS, no fault loop")

        # A fresh daemon takes over the path cleanly.
        sock_b = tempfile.mktemp(prefix="anyplay_regress_inp_")
        pB = launch_daemon(str(daemon), kb_path, ms_path, shm_a, sock_b)
        try:
            ready_b = read_ready_line(pB)
            ok(f"daemon B took over the path: {ready_b.strip()}")

            kb.write(EV_KEY, KEY_E, 1); kb.syn()
            kb.write(EV_KEY, KEY_E, 0); kb.syn()
            time.sleep(0.3)
            stats_b = socket_stats(sock_b)
            print(f"info: stats B: {stats_b}")
            assert f"in_events=2" in stats_b and "in_drops=0" in stats_b, stats_b
            ok("daemon B publishing on the reclaimed path")
        finally:
            if pB.poll() is None:
                try:
                    socket_quit(sock_b)
                    pB.wait(timeout=5)
                except Exception:
                    pB.terminate()
                    pB.wait(timeout=5)
            ok("daemon B shut down cleanly")

    except AssertionError as e:
        for _p in (pA, pB):
            if _p is not None:
                _dump_daemon_tail(_p)
        fail(str(e))
    except Exception as e:  # noqa: BLE001
        for _p in (pA, pB):
            if _p is not None:
                _dump_daemon_tail(_p)
        fail(f"{type(e).__name__}: {e}")
    finally:
        for p in (pA, pB):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        if ring is not None:
            ring.close()
        for d in (kb, ms):
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass
        for f in (shm_a, sock_a, sock_b):
            try:
                os.unlink(f)
            except OSError:
                pass

    print(f"PASS: {daemon.name} input ring")
    return 0


if __name__ == "__main__":
    sys.exit(main())
