"""Python consumer for the C++ evdev input-event ring.

The C++ capture daemon (core/build/anyplay-capture or
core/build/anyplay-portal-capture, P-core step 2) normalizes EV_KEY /
EV_ABS / EV_REL events from the configured /dev/input nodes and publishes
them into a POSIX shared-memory ring (layout:
core/include/anyplay/input_ring_protocol.h).

This module is the zero-copy reader:

    ring = InputEventRing("/dev/shm/anyplay_inp_<pid>.bin")
    for ev in ring.drain():
        # ev: InputEvent(ts_sec, ts_usec, type, code, value)

Events are byte-identical to kernel struct input_event records, i.e. the
same tuples py-evdev would have produced from the same device — which is
what makes this a drop-in replacement for InputRecorder's output.

The daemon unlinks the shm file on exit; hold the open fd (we do) to keep
reading until the daemon is gone.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from typing import Iterator, List, NamedTuple

# ---------------------------------------------------------------------------
# Shared-memory layout (must match anyplay/input_ring_protocol.h)
# ---------------------------------------------------------------------------

# Header, first 32 bytes (8 x u32). The u64 counters follow at offset 32
# (idx, events, drops) — same trick as the frame ring header.
_HDR32 = struct.Struct("<8I")
_HDR64 = struct.Struct("<3Q")

_MAGIC = 0x49505449  # "IPTI"
_VERSION = 1

_HDR_BASE = 64
_SLOT_META = 32
_EVENTS_PER_SLOT = 32
_SLOT_STRIDE = _SLOT_META + _EVENTS_PER_SLOT * 24  # 800

# One kernel input_event: {i64 tv_sec, i64 tv_usec, u16 type,
# u16 code, i32 value} == 24 bytes (identical to py-evdev's layout).
# NOTE: 'q' is the 8-byte signed long long — the kernel's struct timeval
# fields are each 8 bytes on 64-bit, so 'l' (4 bytes) would misalign the
# rest of the record.
_EV = struct.Struct("<qqHHi")


class InputEvent(NamedTuple):
    """One normalized evdev event.

    Fields match kernel struct input_event (the same tuples py-evdev
    would produce), plus dev_id: the device's index in the daemon's
    CLI order (keyboard device(s) first, then pointer device(s)).
    """

    ts_sec: int
    ts_usec: int
    type: int
    code: int
    value: int
    dev_id: int = 0

    @property
    def ts(self) -> float:
        """Event timestamp in seconds (timeval -> float)."""
        return self.ts_sec + self.ts_usec / 1e6


class InputEventRing:
    """Reader for the daemon's input-event ring.

    Parameters
    ----------
    path:
        The input shm path the daemon printed after READY
        (``READY <shm> <sock> input=<path>``).
    """

    def __init__(self, path: str):
        self._fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, 0)
        size = os.fstat(self._fd).st_size

        (
            magic,
            version,
            n_slots,
            events_per_slot,
            alive,
            _p1,
            _p2,
            _p3,
        ) = _HDR32.unpack_from(self._mm, 0)

        if magic != _MAGIC:
            raise RuntimeError(
                f"Invalid input-ring magic: {magic:#x} "
                f"(expected {_MAGIC:#x})"
            )
        if version != _VERSION:
            raise RuntimeError(
                f"Unsupported input-ring version: {version}"
            )
        if events_per_slot != _EVENTS_PER_SLOT:
            raise RuntimeError(
                f"Unexpected events_per_slot: {events_per_slot}"
            )
        expected = _HDR_BASE + n_slots * _SLOT_STRIDE
        if size != expected:
            raise RuntimeError(
                f"Unexpected input-ring size: {size}, "
                f"expected={expected}"
            )

        self._n_slots = n_slots
        self._alive = alive

        # Consumer state: how many slots have been handed out so far.
        self._consumed = 0

    # -------------------------------------------------------------------
    # Header counters
    # -------------------------------------------------------------------

    def _hdr64(self) -> tuple:
        return _HDR64.unpack_from(self._mm, 32)

    @property
    def idx(self) -> int:
        """Number of slots published so far (next slot index)."""
        return self._hdr64()[0]

    @property
    def events(self) -> int:
        """Total events published (daemon-side counter)."""
        return self._hdr64()[1]

    @property
    def drops(self) -> int:
        """Total events dropped by the daemon (no consumer / overrun)."""
        return self._hdr64()[2]

    @property
    def alive(self) -> bool:
        return bool(
            _HDR32.unpack_from(self._mm, 0)[4]
        )

    # -------------------------------------------------------------------
    # Event drain
    # -------------------------------------------------------------------

    def drain(self, max_events: int = 0) -> List[InputEvent]:
        """Return all events published since the last drain, in order.

        max_events: 0 (default) = no limit.
        """
        out: List[InputEvent] = []
        consumed = self._consumed
        n = self._n_slots

        while consumed < self.idx:
            s = consumed % n
            base = _HDR_BASE + s * _SLOT_STRIDE

            seq0 = struct.unpack_from("<I", self._mm, base)[0]
            if seq0 & 1:
                # Slot mid-write; give the producer a moment.
                time.sleep(0.001)
                continue

            # Meta layout: seq(u32@0), dev_id(u32@4), slot_id(u64@8),
            # ts_ns(u64@16), count(u64@24). count is a u64, so it must be
            # unpacked separately from the leading u32 pair.
            seq1 = struct.unpack_from("<I", self._mm, base)[0]
            dev_id = struct.unpack_from("<I", self._mm, base + 4)[0]
            slot_id, ts_ns, count = struct.unpack_from(
                "<3Q", self._mm, base + 8
            )
            count = int(count)
            if count > _EVENTS_PER_SLOT:
                # Corrupt / mid-write slot: skip it rather than reading
                # past the slot boundary.
                consumed += 1
                continue

            evs: List[InputEvent] = []
            for i in range(count):
                (
                    ts_sec,
                    ts_usec,
                    etype,
                    ecode,
                    evalue,
                ) = _EV.unpack_from(
                    self._mm,
                    base + _SLOT_META + i * 24,
                )
                evs.append(
                    InputEvent(
                        ts_sec,
                        ts_usec,
                        etype,
                        ecode,
                        evalue,
                        dev_id,
                    )
                )

            seq2 = struct.unpack_from("<I", self._mm, base)[0]
            if seq2 != seq1:
                # Torn read (producer rewrote the slot mid-drain); retry
                # this slot from the top.
                continue

            if slot_id != consumed:
                # The slot was overwritten by a later lap before we got
                # to it: the events for this idx are gone (overrun).
                # Skip forward; the data for slot_id will be read when
                # `consumed` catches up to it (if it survives that
                # long).
                consumed += 1
                continue

            out.extend(evs)
            consumed += 1
            if max_events and len(out) >= max_events:
                break

        self._consumed = consumed
        return out

    def iter_drain(self) -> Iterator[InputEvent]:
        """Drain and yield events one at a time (lazy view)."""
        for ev in self.drain():
            yield ev

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
