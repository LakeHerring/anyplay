"""Parse Qwen's decision output into a validated action.

The model is prompted to answer with strict JSON:

    {"action": "DODGE_LEFT", "confidence": 0.87, "reason": "..."}

This parser is defensive: it extracts the first JSON object it can find in
the model text (models sometimes wrap it in prose or markdown fences),
normalizes the action name (uppercase, spaces/hyphens -> underscore), and
rejects anything that is not in the ``ActionSpace``. An unparseable answer
yields a *rejected* decision, never a free-form input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .action_space import Action, ActionSpace

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)
_NAME_FIX = re.compile(r"[\s\-]+")


@dataclass
class Decision:
    """Result of parsing one model answer."""

    ok: bool
    action: Optional[Action] = None
    raw_name: Optional[str] = None
    confidence: Optional[float] = None
    reason: str = ""
    error: str = ""


def _normalize(name: str) -> str:
    return _NAME_FIX.sub("_", name.strip().upper()).strip("_")


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    # strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        j = json.loads(text)
        return j if isinstance(j, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_OBJ.search(text)
    if m:
        try:
            j = json.loads(m.group(0))
            return j if isinstance(j, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_decision(text: str, space: ActionSpace) -> Decision:
    """Parse a model answer; always returns a Decision (never raises)."""
    if not text or not text.strip():
        return Decision(ok=False, error="empty model output")

    j = _extract_json(text)
    if j is None:
        return Decision(ok=False, error="no JSON object in model output")

    raw = j.get("action")
    if not isinstance(raw, str) or not raw.strip():
        return Decision(ok=False, raw_name=str(raw),
                        error="'action' missing or not a string")

    name = _normalize(raw)
    action = space.get(name)
    if action is None:
        return Decision(
            ok=False, raw_name=name,
            error=f"'{name}' not in action space "
                  f"({', '.join(space.names())})")

    conf = j.get("confidence")
    if conf is not None:
        try:
            conf = float(conf)
            if not 0.0 <= conf <= 1.0:
                conf = None
        except (TypeError, ValueError):
            conf = None

    reason = j.get("reason")
    reason = reason if isinstance(reason, str) else ""

    return Decision(ok=True, action=action, raw_name=name,
                    confidence=conf, reason=reason)
