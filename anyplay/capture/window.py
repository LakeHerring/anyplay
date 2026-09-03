"""Locate the Shadow Dungeon game window on X11 (no external tools needed).

Notes:
- Proton/Wine games only set ``_NET_WM_NAME`` (UTF8_STRING) and do not set
  the legacy ``WM_NAME`` property — python-xlib's ``get_wm_name()`` only
  reads the legacy one and returns None for them, so we read the property
  directly (``_NET_WM_NAME`` first, ``WM_NAME`` as fallback).
- The window tree is walked a few levels deep because a window manager may
  reparent top-levels into frame windows; absolute screen coordinates are
  computed by walking the parent chain up to the root and summing offsets.
"""

from __future__ import annotations

from dataclasses import dataclass

from Xlib import Xatom
from Xlib.display import Display

DEFAULT_GAME_HINTS = ("shadow dungeon", "shadowdungeon")
_MAX_DEPTH = 4


@dataclass
class GameWindow:
    wid: int
    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def region(self) -> str:
        """'x,y,w,h' for ffmpeg x11grab / mss."""
        return f"{self.x},{self.y},{self.width},{self.height}"


def _window_title(w, net_wm_name, utf8, wm_name, string) -> str:
    """Read the window title, handling Proton's UTF8-only naming."""
    for prop, typ in ((net_wm_name, utf8), (wm_name, string)):
        try:
            p = w.get_full_property(prop, typ)
        except Exception:
            continue
        if p is not None and p.value:
            return p.value.decode("utf-8", "replace").strip()
    return ""


def _absolute_position(w, root):
    """Absolute (x, y) of ``w``: sum of geometry offsets up to the root."""
    x = y = 0
    for _ in range(16):  # sanity guard against cycles
        if w.id == root.id:
            break
        g = w.get_geometry()
        x += g.x
        y += g.y
        parent = w.query_tree().parent
        if parent is None or parent.id == root.id:
            break
        w = parent
    return x, y


def find_game_window(display: str = ":0.0", hints=DEFAULT_GAME_HINTS):
    """Return the best-matching game window, or None if not found.

    ``hints`` is a sequence of substrings matched case-insensitively against
    the window title. If several windows match (e.g. a launcher splash and
    the game), the one with the largest area wins.
    """
    d = Display(display)
    root = d.screen().root
    net_wm_name = d.intern_atom("_NET_WM_NAME")
    utf8 = d.intern_atom("UTF8_STRING")
    matches: list[GameWindow] = []

    def walk(w, depth):
        if depth > _MAX_DEPTH:
            return
        try:
            name = _window_title(w, net_wm_name, utf8, Xatom.WM_NAME, Xatom.STRING)
            if name:
                low = name.lower()
                if any(h.lower() in low for h in hints):
                    g = w.get_geometry()
                    x, y = _absolute_position(w, root)
                    matches.append(GameWindow(w.id, name, x, y, g.width, g.height))
            for child in w.query_tree().children:
                walk(child, depth + 1)
        except Exception:
            return

    for child in root.query_tree().children:
        walk(child, 1)

    if not matches:
        return None
    return max(matches, key=lambda m: m.width * m.height)
