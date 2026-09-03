"""Constrained action space for the game agent.

Rule 3 of the engineering rules: *unrestricted LLM output must never reach
the input controller.* The model emits a name from this fixed vocabulary;
anything else is rejected by the parser.

Actions are game-specific and configurable (plan §10). The default space is
tuned for Shadow Dungeon (side-view action game, WASD-ish keyboard).

Each action maps to one or more keycodes with an optional hold duration.
The ``InputController`` is responsible for press/release sequencing,
duration, and conflict resolution (plan §12) — this module only declares
*what* an action is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# evdev keycodes (input-event-codes / linux/input-event-codes.h)
KEY_A = 30
KEY_D = 32
KEY_S = 31
KEY_W = 13
KEY_SPACE = 57
KEY_E = 18
KEY_F = 21
KEY_J = 42
KEY_K = 43
KEY_UP = 103
KEY_DOWN = 108
KEY_LEFT = 105
KEY_RIGHT = 106
KEY_ESC = 1


@dataclass(frozen=True)
class Action:
    """One discrete decision the policy can make."""

    name: str
    keys: Tuple[int, ...]      # keycodes pressed together
    duration_ms: int = 0       # 0 = tap (press+release in one event burst)
    description: str = ""

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Default Shadow Dungeon action space.
#
# Shadow Dungeon is a side-view action game: left/right movement, jump,
# melee/weapon attack, interact, and block/dodge. The exact keybinds should
# be verified against the in-game settings before the first real run; the
# defaults below follow common conventions (WASD + space + E).
# ---------------------------------------------------------------------------

DEFAULT_ACTIONS: Tuple[Action, ...] = (
    Action("WAIT", (), 0, "do nothing / stand still"),
    Action("MOVE_LEFT", (KEY_A,), 200, "hold left"),
    Action("MOVE_RIGHT", (KEY_D,), 200, "hold right"),
    Action("MOVE_DOWN", (KEY_S,), 200, "crouch / move down"),
    Action("JUMP", (KEY_W,), 0, "jump"),
    Action("ATTACK", (KEY_SPACE,), 0, "primary attack"),
    Action("DEFEND", (KEY_E,), 0, "block / parry / dodge-roll"),
    Action("INTERACT", (KEY_F,), 0, "pick up / talk / activate"),
    # Combinations (plan §10)
    Action("FORWARD_LEFT", (KEY_A, KEY_W), 200, "move left while airborne/up"),
    Action("FORWARD_RIGHT", (KEY_D, KEY_W), 200, "move right while airborne/up"),
    Action("DODGE_LEFT", (KEY_A, KEY_E), 0, "dodge backward"),
    Action("DODGE_RIGHT", (KEY_D, KEY_E), 0, "dodge forward"),
    Action("JUMP_FORWARD", (KEY_D, KEY_W), 0, "jump right"),
)


class ActionSpace:
    """Registry of valid actions + lookup helpers."""

    def __init__(self, actions: Optional[Tuple[Action, ...]] = None):
        self.actions: Dict[str, Action] = {}
        for a in actions or DEFAULT_ACTIONS:
            self.actions[a.name] = a
        if "WAIT" not in self.actions:
            raise ValueError("action space must include WAIT")

    def __contains__(self, name: str) -> bool:
        return name in self.actions

    def __iter__(self):
        return iter(self.actions)

    def names(self) -> list:
        return list(self.actions)

    def get(self, name: str) -> Optional[Action]:
        return self.actions.get(name)

    def vocabulary_text(self) -> str:
        """Human-readable list for model prompts."""
        lines = []
        for a in self.actions.values():
            lines.append(f"  {a.name}: {a.description}")
        return "\n".join(lines)

    def release_keys(self) -> Tuple[int, ...]:
        """All keycodes any action can use — released on WAIT/switch."""
        out = set()
        for a in self.actions.values():
            out.update(a.keys)
        return tuple(sorted(out))
