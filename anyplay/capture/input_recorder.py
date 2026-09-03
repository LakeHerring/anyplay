"""Record raw evdev input events to JSONL, synchronized with the video clock.

Each event line is::

    {"t": <sec>, "type": int, "code": int, "value": int, "dev": "/dev/input/eventN"}

where ``t`` is seconds since the recorder started (CLOCK_MONOTONIC) and ``dev``
identifies the source device. The video capturer records its own start time;
``meta.json`` stores the offset between the two so events can be mapped onto
video frames during build.

By default the recorder captures the **keyboard and the mouse** together, since
games are usually driven by both (aiming/clicking on the mouse, movement/
abilities on the keyboard). Events are read passively (no ``grab()``), so the
game still receives the player's real input while recording. Old single-device
sessions (lines without a ``dev`` field) still load fine.
"""

import json
import threading
import time
from pathlib import Path

import evdev
from evdev.ecodes import EV_ABS, EV_KEY, EV_REL

# An X/Y axis (code 0 or 1) in REL or ABS marks a pointer (mouse).
_XY = (0, 1)


def list_devices() -> list[evdev.InputDevice]:
    """Return every readable input device (used by ``main.py devices``)."""
    try:
        try:
            paths = evdev.list_devices(writable=False)
        except TypeError:
            # Older evdev without the `writable` kwarg.
            paths = evdev.list_devices()
    except PermissionError as e:
        raise PermissionError(
            f"cannot read /dev/input: {e}. Add your user to the 'input' "
            "group (sudo usermod -aG input $USER) and re-login."
        ) from e
    devices = []
    for path in paths:
        try:
            dev = evdev.InputDevice(path)
            dev.capabilities()
            devices.append(dev)
        except PermissionError:
            continue
    return devices


def _has_xy_motion(caps) -> bool:
    return (
        (EV_REL in caps and any(c in caps[EV_REL] for c in _XY))
        or (EV_ABS in caps and any(c in caps[EV_ABS] for c in _XY))
    )


def _role_of(caps) -> str:
    """Label a device's capabilities as 'pointer', 'keyboard', or 'other'.

    A multi-part mouse shows up as several evdev nodes: X/Y motion (REL or
    ABS), buttons (as EV_KEY on their own node), and sometimes a wheel node.
    All of them are labelled 'pointer'; a keyboard (>=20 keys, e.g. with
    media/RGB axes) stays 'keyboard'.
    """
    n_keys = len(caps.get(EV_KEY, ()))
    if _has_xy_motion(caps):
        return "pointer"
    # Axis-only nodes (wheel/scroll, misc) belong to the mouse, not a keyboard.
    if (EV_REL in caps or EV_ABS in caps) and n_keys < 20:
        return "pointer"
    if n_keys >= 20:
        return "keyboard"
    return "other"


class InputRecorder:
    """Record EV_KEY/EV_ABS/EV_REL events from one or more input devices.

    With an empty query every keyboard and every pointer/mouse node are opened
    and recorded in parallel (mouse buttons arrive as EV_KEY on their own
    node, and the wheel may live on yet another); each line is tagged with the
    source device path.
    """

    WATCHED_TYPES = (EV_KEY, EV_ABS, EV_REL)

    def __init__(self, device_query, out_path):
        self.out_path = Path(out_path)
        self._devices = self._find_devices(device_query)
        self.start_time = None
        self._stop = threading.Event()
        self._threads = []
        self._out = None
        self._lock = threading.Lock()

    @staticmethod
    def _find_devices(query):
        devices = list_devices()
        if not query:
            # Auto: every keyboard and every pointer/mouse node (motion,
            # buttons, wheel). Keyboard first for a stable ordering.
            picked = []
            for role in ("keyboard", "pointer"):
                picked.extend(
                    d for d in devices if _role_of(d.capabilities()) == role)
            if not picked:
                if not devices:
                    raise ValueError("no input devices found")
                picked.append(devices[0])
            return picked
        for d in devices:
            if d.path == query or query in d.path or query in d.name:
                return [d]
        raise ValueError(
            f"no input device matching {query!r}. Run 'main.py devices' to list."
        )

    @property
    def info(self):
        """List of {"role","name","path"}, one per recorded device."""
        return [
            {"role": _role_of(d.capabilities()), "name": d.name, "path": d.path}
            for d in self._devices
        ]

    def start(self):
        self.start_time = time.monotonic()
        self._out = open(self.out_path, "w", buffering=1)
        for d in self._devices:
            t = threading.Thread(target=self._loop, args=(d,), daemon=True)
            t.start()
            self._threads.append(t)

    def _loop(self, device):
        try:
            for ev in device.read_loop():
                if self._stop.is_set():
                    break
                if ev.type in self.WATCHED_TYPES:
                    line = json.dumps(
                        {
                            "t": time.monotonic() - self.start_time,
                            "type": ev.type,
                            "code": ev.code,
                            "value": ev.value,
                            "dev": device.path,
                        }
                    ) + "\n"
                    with self._lock:
                        if self._out is not None:
                            self._out.write(line)
        except OSError:
            # Device closed by stop().
            pass

    def stop(self):
        self._stop.set()
        for d in self._devices:
            try:
                d.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
        with self._lock:
            if self._out is not None:
                self._out.close()
                self._out = None


__all__ = ["InputRecorder", "list_devices"]
