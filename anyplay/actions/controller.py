"""Input backend + action controller.

Never connects a model to raw keyboard events (plan §12). The
``InputController`` is the *only* component that touches the input
backend, and it only ever executes validated ``Action`` objects from the
``ActionSpace``.

Backend: a Linux **uinput virtual keyboard** (python-evdev). A uinput
device behaves like any real keyboard to X11/Proton/Wayland clients, so
the game receives the keys without any window-manager trickery and
without stealing focus from the real keyboard.

Responsibilities (plan §12):

* validating actions (done upstream by the parser; re-checked here)
* applying duration (hold/release timing on a worker thread)
* preventing conflicting inputs (all previous keys released before a new
  action's keys are pressed)
* rate limiting (minimum gap between action starts)
* releasing keys correctly (``release_all`` on WAIT, stop, and errors)
* maintaining action state (``state()`` feeds the observation buffer)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import evdev
from evdev import InputDevice, UInput, ecodes as e

from .action_space import Action

# Default hold time for tap actions (duration_ms == 0).
TAP_MS = 30

# Old py-evdev (Debian/Ubuntu system package) uses uppercase KEY_*
# constants and has no Key enum / ecodes_to_names; normalize both.
if hasattr(e, "Key"):
    _KEYS = {int(k.value): k.name for k in e.Key if k.name.startswith("KEY_")}
else:
    _KEYS = {v: n[4:] for n, v in vars(e).items()
             if isinstance(n, str) and n.startswith("KEY_") and isinstance(v, int)}


def key_name(code: int) -> str:
    return _KEYS.get(code, str(code))


def _all_key_codes():
    """Every real KEY_* code (uinput capability set).

    Excludes sentinels: the kernel rejects codes > 0x2ff (KEY_MAX),
    which old py-evdev exposes as a constant.
    """
    return sorted(c for c in _KEYS if 0 < c <= 0x2ff)


class UInputBackend:
    """A virtual USB-style keyboard created via /dev/uinput."""

    def __init__(self, name: str = "anyplay-vkb"):
        # Full KEY_* capability set so any action key is pressable.
        # (UInput(events, name=...) is the same call in old and new
        # py-evdev.)
        events = {e.EV_KEY: _all_key_codes()}
        self.ui = UInput(
            events, name=name, vendor=0x1234, product=0x5678,
            version=0x0100)
        self.name = getattr(self.ui, "name", name)

    def press(self, key: int) -> None:
        self.ui.write(e.EV_KEY, key, 1)
        self.ui.syn()

    def release(self, key: int) -> None:
        self.ui.write(e.EV_KEY, key, 0)
        self.ui.syn()

    def close(self) -> None:
        try:
            self.ui.close()
        except Exception:
            pass
        try:
            self.ui.unload_kernel_module()
        except Exception:
            pass


def find_backend_device() -> Optional[InputDevice]:
    """Find our virtual keyboard among /dev/input/event* (diagnostics)."""
    for dev in evdev.list_devices():
        try:
            d = InputDevice(dev)
            if "anyplay-vkb" in d.name:
                return d
        except OSError:
            continue
    return None


class InputController:
    """Executes validated actions on a uinput keyboard.

    Parameters
    ----------
    backend:
        ``UInputBackend`` (default: create one). Pass ``None`` for a
        dry-run controller that logs but sends nothing (unit testing /
        no-uinput environments).
    min_interval_s:
        Rate limit: minimum seconds between the *starts* of two actions.
        0.025 rejects a runaway 60 Hz loop (~16.7 ms) while allowing
        deliberate quick sequences (Qwen decisions arrive 200+ ms apart).
    """

    def __init__(self, backend: Optional[UInputBackend] = None,
                 min_interval_s: float = 0.05):
        self.backend = backend
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._worker_loop, name="input-controller", daemon=True)
        self._queue = []          # (action, deadline)
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._last_start = 0.0
        self._held: List[int] = []
        self._current: Optional[str] = None
        self.executed = 0
        self.rejected = 0
        self._worker.start()

    # ------------------------------------------------------------------

    def execute(self, action: Action) -> bool:
        """Queue an action. Returns True if accepted, False if rate-limited."""
        with self._cv:
            if self._stop:
                return False
            now = time.monotonic()
            if now - self._last_start < self.min_interval_s:
                self.rejected += 1
                return False
            self._last_start = now
            self._queue.append(action)
            self._cv.notify()
            return True

    def release_all(self) -> None:
        """Immediately release every held key and drop queued actions."""
        with self._cv:
            self._queue.clear()
            self._release_held()
            self._current = None

    def state(self) -> dict:
        """Current key state (feeds observation.input_state)."""
        with self._lock:
            return {
                "held": [key_name(k) for k in self._held],
                "current_action": self._current,
                "executed": self.executed,
                "rejected": self.rejected,
            }

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._queue.clear()
            self._cv.notify()
        self._worker.join(timeout=2.0)
        self.release_all()
        if self.backend is not None:
            self.backend.close()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            action = None
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=0.2)
                if self._stop and not self._queue:
                    return
                if self._queue:
                    action = self._queue.pop(0)
            if action is None:
                continue
            self._run(action)

    def _run(self, action: Action) -> None:
        if not action.keys:          # WAIT
            with self._lock:
                self._release_held()
                self._current = action.name
            self.executed += 1
            return
        with self._lock:
            # Conflict rule: release everything before pressing new keys.
            self._release_held()
            for k in action.keys:
                if self.backend is not None:
                    self.backend.press(k)
                self._held.append(k)
            self._current = action.name
        self.executed += 1
        hold_ms = action.duration_ms if action.duration_ms > 0 else TAP_MS
        time.sleep(hold_ms / 1000.0)
        with self._lock:
            self._release_held()

    def _release_held(self) -> None:
        """Caller must hold the lock."""
        for k in reversed(self._held):
            if self.backend is not None:
                self.backend.release(k)
        self._held.clear()
