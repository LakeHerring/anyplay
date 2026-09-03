#!/usr/bin/env python3
"""Async xdg-desktop-portal ScreenCast probe using Gio (GLib) D-Bus.

Purpose: empirically map the portal protocol on this Wayland/KDE system and
produce a PipeWire fd + stream info the capture daemon can attach to.

Protocol (confirmed from xdg-desktop-portal router source, screen-cast.c):
  - CreateSession / SelectSources / Start are REQUEST-based. The sync D-Bus
    return is a request handle; the *real* result arrives via the
    org.freedesktop.portal.Request::Response signal on that request object.
  - OpenPipeWireRemote is synchronous: (h fd, a{sv} results) direct return.

Run:  PW_PROBE=1 .venv/bin/python tools/portal_probe.py
"""
import os
import sys
import time
import signal

import gi
gi.require_version("Gio", "2.0")
from gi.repository import Gio, GLib

DEST = "org.freedesktop.portal.Desktop"
PATH = "/org/freedesktop/portal/desktop"
SC = "org.freedesktop.portal.ScreenCast"
REQ = "org.freedesktop.portal.Request"

loop = GLib.MainLoop()


def log(*a):
    print("[probe]", *a, flush=True)


class Portal:
    def __init__(self):
        self.bus = Gio.SessionBus.sync(None)
        self._pending = {}  # request_path -> list[callable(response, results)]
        self._match = Gio.DBusSignalRule(
            sender=DEST, name=REQ, member="Response", path=None
        )
        self.bus.signal_subscribe(
            DEST,
            REQ,
            None,  # any member
            "/org/freedesktop/portal/desktop/request/",
            None,
            Gio.DBusSignalFlags.NONE,
            self._on_signal,
        )
        # Also watch Request Closed signal
        self.bus.signal_subscribe(
            DEST,
            REQ,
            "Closed",
            "/org/freedesktop/portal/desktop/request/",
            None,
            Gio.DBusSignalFlags.NONE,
            self._on_signal,
        )

    def _on_signal(self, _conn, sender, object_path, interface, member, params):
        log(f"  SIGNAL {member} on {object_path} sender={sender} params={params!r}")
        if member != "Response":
            return
        response = params[0]
        results = params[1]
        cbs = self._pending.pop(str(object_path), [])
        for cb in cbs:
            try:
                cb(response, results)
            except Exception as e:  # noqa: BLE001
                log("  callback error:", e)

    def call(self, method, params, expected, timeout=15000):
        """Synchronous method call. Returns the sync reply variant."""
        variant = Gio.Variant(expected, params)
        log(f"-> {SC}.{method} {params!r} :: {expected}")
        reply = self.bus.call_sync(
            DEST,
            PATH,
            SC,
            method,
            variant,
            None,
            Gio.DBusCallFlags.NONE,
            timeout,
            None,
        )
        log(f"<- {method} sync reply: {reply!r}")
        return reply

    def wait_response(self, request_path, cb, timeout=60.0):
        """Register cb to fire when Request.Response arrives for request_path."""
        self._pending.setdefault(str(request_path), []).append(cb)
        # Arm a timeout so we don't hang forever.
        GLib.timeout_add(int(timeout * 1000), self._on_timeout, request_path)

    def _on_timeout(self, request_path):
        if request_path in self._pending:
            log(f"!! timeout waiting for Response on {request_path}")
            self._pending.pop(request_path)
            loop.quit()
        return False


def main():
    token = f"anyplay_{int(time.time())}"
    p = Portal()
    log("token:", token, " pid:", os.getpid())

    # --- 1. CreateSession (request-based) ---
    # options: session_handle_token
    opts = Gio.Variant("a{sv}", [("session_handle_token", Gio.Variant("s", token))])
    reply = p.call("CreateSession", [opts], "(a{sv})")
    # sync reply is (o session) per the interface; capture it
    sync_session = None
    try:
        sync_session = reply.unpack()[0]
    except Exception:
        pass
    log("sync session from CreateSession:", sync_session)

    def on_session(response, results):
        log(f"  CreateSession Response: response={response} results={dict(results)!r}")
        results_d = dict(results)
        session = results_d.get("session") or sync_session
        log("  effective SESSION:", session)
        state["session"] = session
        if response != 0 or not session:
            log("  CreateSession failed; aborting")
            loop.quit()
            return
        step_select(p, state, session)

    # The request path for the async Response: derived from sync reply if it is
    # a request path, else we must find it. In the portal the sync return of a
    # request method is the request object path.
    req_path = sync_session
    if req_path and str(req_path).startswith("/org/freedesktop/portal/desktop/request/"):
        p.wait_response(req_path, on_session)
        log("waiting for CreateSession Response on", req_path)
    else:
        # No request path in sync reply -> treat sync session as final.
        on_session(0, {"session": sync_session})

    state = {}
    loop.run()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *a: (loop.quit(), sys.exit(130)))
    main()
