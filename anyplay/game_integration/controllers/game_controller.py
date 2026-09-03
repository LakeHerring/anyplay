"""Ingest the policy's outputs and replay them via a UInput virtual device.

The virtual device is a hybrid built from the cloned physical input device's
capabilities plus EV_REL caps for the motion codes, so the game receives both
keyboard keys/mouse buttons (EV_KEY), axis state (EV_ABS), and relative mouse
movement (EV_REL).
"""

from evdev import UInput
from evdev.ecodes import EV_ABS, EV_KEY, EV_REL

from ...capture.input_recorder import InputRecorder


class GameController:
    def __init__(self, source_query: str = "", action_space: dict | None = None):
        self.action_space = action_space or {
            "buttons": [], "axes": [], "motion": [],
            "axis_bounds": {}, "motion_scale": {},
        }
        bounds = self.action_space.get("axis_bounds", {})
        self.axis_bounds = {int(k): (float(lo), float(hi))
                            for k, (lo, hi) in bounds.items()}
        self.motion_codes = [int(c) for c in self.action_space.get("motion", [])]
        self.motion_scale = {int(k): float(v)
                             for k, v in self.action_space.get("motion_scale", {}).items()}

        # Clone the physical device so the virtual one matches what the player
        # used; add EV_REL caps for mouse-motion injection.
        source = InputRecorder._find_device(source_query)
        self.source_name = source.name
        self.ui = self._build_uinput(source)
        self._key_state = {}
        self._axis_state = {}

    @staticmethod
    def _build_uinput(source):
        caps = dict(source.capabilities())
        rel_cap = caps.get(EV_REL)
        rel = set(rel_cap) if not isinstance(rel_cap, dict) else set(rel_cap.keys())
        # ensure EV_REL is a flat code set UInput understands
        caps[EV_REL] = sorted(rel)
        return UInput(
            caps,
            name="AnyPlay",
            vendor=0x0001, product=0x0001, version=0x0001,
        )

    def _denorm_axis(self, code: int, norm: float) -> int:
        lo, hi = self.axis_bounds.get(code, (0, 1))
        return int(lo + (max(-1.0, min(1.0, float(norm))) + 1.0) * 0.5 * (hi - lo))

    def _denorm_motion(self, code: int, norm: float) -> int:
        scale = self.motion_scale.get(code, 1.0)
        return int(round(max(-1.0, min(1.0, float(norm))) * scale))

    def write_action(self, buttons: dict, axes: dict, motion: dict | None = None):
        """Emit a full controller state (delta-optimized: only changed values).

        ``motion`` is a {EV_REL code: normalized [-1,1] per-frame delta}; it is
        denormalized and written as a relative delta (skipped when zero).
        """
        motion = motion or {}
        for code in self.action_space.get("buttons", []):
            target = 1 if buttons.get(code, 0) else 0
            if self._key_state.get(code, 0) != target:
                self.ui.write(EV_KEY, code, target)
                self._key_state[code] = target
        for code in self.action_space.get("axes", []):
            raw = self._denorm_axis(code, axes.get(code, 0.0))
            if self._axis_state.get(code) != raw:
                self.ui.write(EV_ABS, code, raw)
                self._axis_state[code] = raw
        for code in self.motion_codes:
            delta = self._denorm_motion(code, motion.get(code, 0.0))
            if delta != 0:
                self.ui.write(EV_REL, code, delta)
        self.ui.syn()

    def reset(self):
        for code, state in list(self._key_state.items()):
            if state:
                self.ui.write(EV_KEY, code, 0)
                self._key_state[code] = 0
        for code in self._axis_state:
            raw = self._denorm_axis(code, 0.0)
            self.ui.write(EV_ABS, code, raw)
            self._axis_state[code] = raw
        # EV_REL is relative (deltas) -- nothing to reset.
        self.ui.syn()

    def close(self):
        self.reset()
        self.ui.close()
