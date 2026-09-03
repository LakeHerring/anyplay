#!/usr/bin/env python3
"""AnyPlay command line.

Pipeline:
    python main.py devices                      # list readable input devices
    python main.py capture --session s1         # 60 FPS video + input log
    python main.py build --session s1           # discover action space
    python main.py train --session s1           # imitation learning
    python main.py play --checkpoint s1/checkpoints/policy_best.pt
    python main.py play --checkpoint ... --smoke   # no screen/UInput, self-test
    python main.py input-check                     # C++ input ring check (no video)
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

from anyplay.utils.config.config import ProjectConfig, ObsConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def cmd_devices(_args):
    from evdev.ecodes import EV_ABS, EV_KEY

    from anyplay.capture.input_recorder import list_devices

    print(f"{'path':24} {'keys':>5} {'axes':>5}  name")
    for d in list_devices():
        caps = d.capabilities()
        print(f"{d.path:24} {len(caps.get(EV_KEY, [])):5d} "
              f"{len(caps.get(EV_ABS, [])):5d}  {d.name}")


def _game_hints(args):
    name = getattr(args, "game_name", None) or "shadow dungeon"
    return (name, name.replace(" ", ""))


def _find_launcher(hints):
    """Find a .desktop launcher whose Name matches the hints.

    Returns (path, argv) or (None, None).
    """
    import glob
    import shlex
    from pathlib import Path

    dirs = [Path.home() / "Desktop", Path.home() / ".local" / "share" / "applications"]
    for base in dirs:
        for path in sorted(glob.glob(str(base / "*.desktop"))):
            name = ""
            exe = ""
            try:
                for line in Path(path).read_text().splitlines():
                    line = line.strip()
                    if not name and line.startswith("Name="):
                        name = line[5:].strip()
                    elif line.startswith("Exec="):
                        exe = line[5:].strip()
            except OSError:
                continue
            if not exe or not any(h.lower() in name.lower() for h in hints):
                continue
            return path, shlex.split(exe)
    return None, None


def resolve_region(explicit, display, hints, wait_seconds=60.0, allow_launch=False):
    """Resolve the screen region to record/watch.

    An explicit --region always wins. Otherwise the game window is located,
    polling every 2 s for up to ``wait_seconds``; with ``allow_launch`` the
    game is started via its .desktop launcher on the first miss.
    """
    if explicit:
        return explicit

    from anyplay.capture.window import find_game_window

    launcher = None
    if allow_launch:
        launcher_path, launcher_argv = _find_launcher(hints)
        if launcher_argv:
            launcher = (launcher_path, launcher_argv)
        else:
            print("note: --launch set but no matching .desktop launcher found")

    deadline = time.monotonic() + wait_seconds
    launched = False
    while True:
        win = find_game_window(display)
        if win is not None:
            print(f"game window: {win.name!r} -> {win.region} ({win.width}x{win.height})")
            return win.region
        if time.monotonic() >= deadline:
            break
        if launcher is not None and not launched:
            import subprocess
            print(f"game window not found - launching via {launcher[0]} ...")
            subprocess.Popen(launcher[1])
            launched = True
        print("waiting for the game window ...")
        time.sleep(2)
    raise SystemExit(
        f"game window not found (looked for {hints[0]!r} on {display} for "
        f"{wait_seconds:.0f}s). Launch the game, use --launch, or pass --region x,y,w,h."
    )


def cmd_capture(args):
    from anyplay.capture.capture import run_capture

    cfg = ProjectConfig()
    cfg.capture.fps = args.fps
    cfg.capture.backend = args.backend
    cfg.capture.duration = args.duration
    cfg.capture.portal_types = args.portal_types
    if args.backend == "portal":
        # Wayland portal capture: the window picker selects the game window at
        # its native resolution, so skip the X11 region resolution entirely.
        cfg.capture.region = ""
        print("portal capture: the 'Share Screen' window picker will open -- "
              "select the game window")
    else:
        cfg.capture.region = resolve_region(
            args.region, cfg.capture.display, _game_hints(args),
            args.wait_window, args.launch)
    cfg.capture.input_device = args.device
    name = args.session or time.strftime("%Y%m%d-%H%M%S")
    run_capture(cfg.session_dir(name), cfg.capture)


def cmd_build(args):
    from anyplay.training.data.preprocessor import build_action_space

    cfg = ProjectConfig()
    space = build_action_space(cfg.session_dir(args.session), overwrite=args.overwrite)
    print(f"buttons ({len(space['buttons'])}): {space['buttons']}")
    print(f"axes ({len(space['axes'])}): {space['axes']}")
    motion = space.get("motion", [])
    if motion:
        scales = {k: round(v, 1) for k, v in space.get("motion_scale", {}).items()}
        print(f"motion ({len(motion)}): {motion}  scale={scales}")
    if not space["buttons"] and not space["axes"] and not motion:
        print("warning: no actions recorded - train will fail on this session")


def cmd_train(args):
    from anyplay.training.training import train_session

    cfg = ProjectConfig()
    train_session(
        cfg.session_dir(args.session),
        epochs=args.epochs,
        window=args.window,
        batch_size=args.batch_size,
        lr=args.lr,
        train_fps=cfg.dataset.train_fps,
        width=cfg.dataset.width,
        height=cfg.dataset.height,
        button_weight=cfg.train.button_weight,
        axis_weight=cfg.train.axis_weight,
        motion_weight=cfg.train.motion_weight,
        device=args.device,
    )


class _NullController:
    """Prints actions instead of writing them (smoke mode)."""

    def __init__(self):
        self.n = 0

    def write_action(self, buttons, axes, motion=None):
        self.n += 1
        if self.n == 1 or self.n % 25 == 0:
            pressed = [c for c, v in buttons.items() if v]
            axis_str = ", ".join(f"{c}={a:+.2f}" for c, a in axes.items())
            motion = motion or {}
            motion_str = ", ".join(f"m{c}={a:+.2f}" for c, a in motion.items())
            tail = ("  " + motion_str) if motion_str else ""
            print(f"  step {self.n:4d}: keys={pressed} [{axis_str}]{tail}")

    def reset(self):
        pass

    def close(self):
        pass


def cmd_vl(args):
    from anyplay.vl import VLModel

    cfg = ProjectConfig()
    cfg.vl.port = args.port
    if args.model:
        cfg.vl.model = args.model
    if args.mmproj:
        cfg.vl.mmproj = args.mmproj
    cfg.vl.max_tokens = args.max_tokens

    model = VLModel(cfg.vl)
    try:
        model.ensure_server()
        t0 = time.time()
        res = model.ask(args.prompt, image=args.image,
                        system=args.system, thinking=args.thinking)
        dt = time.time() - t0
        if res["reasoning"] and args.thinking:
            print(f"reasoning:\n{res['reasoning']}\n")
        print(res["content"])
        print(f"\n[{dt:.1f}s | server left running at "
              f"http://{cfg.vl.host}:{cfg.vl.port} | stop: python main.py vl-stop]")
    finally:
        if args.stop_after:
            model.stop_server()


def cmd_vl_serve(args):
    from anyplay.vl import VLModel

    cfg = ProjectConfig()
    cfg.vl.port = args.port
    if args.model:
        cfg.vl.model = args.model
    if args.mmproj:
        cfg.vl.mmproj = args.mmproj
    cfg.vl.ctx_size = args.ctx_size

    model = VLModel(cfg.vl)
    if model.is_running():
        print(f"server already running at http://{cfg.vl.host}:{cfg.vl.port}")
        return
    print(f"starting llama.cpp server (model: {cfg.vl.resolved_model()})")
    model.start_server(wait=False)
    print(f"ready when /health returns ok; log: {cfg.vl.log_path()} "
          f"(pid {model._proc.pid})")
    try:
        while model._proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        model.stop_server()


def cmd_vl_stop(args):
    from anyplay.vl import VLModel

    cfg = ProjectConfig()
    cfg.vl.port = args.port
    ok = VLModel(cfg.vl).stop_server()
    print("stopped" if ok else "not running (or still shutting down)")


def cmd_play(args):
    import torch

    from anyplay.game_integration.game_integration import GameIntegration

    cfg = ProjectConfig()
    cfg.play.region = "" if args.smoke else resolve_region(
        args.region, cfg.play.display, _game_hints(args),
        args.wait_window, args.launch)
    cfg.play.input_device = args.device_in
    cfg.play.threshold = args.threshold
    cfg.play.fps = args.fps
    cfg.play.source = args.source

    gi = GameIntegration(
        args.checkpoint, cfg.play, cfg.dataset, device=args.device, window=args.window
    )
    if args.smoke:
        print("smoke mode: synthetic frames, no screen, no UInput")
        controller = _NullController()
        for _ in range(args.steps):
            frame = torch.rand(3, gi.height, gi.width)
            action = gi.predict(frame)
            if action is not None:
                controller.write_action(action["buttons"], action["axes"],
                                        action.get("motion"))
        print(f"smoke OK: {controller.n} actions produced")
    else:
        gi.run(max_steps=args.steps)


def cmd_agent(args):
    from anyplay.agent import Agent, AgentConfig

    import os

    source = args.source
    if args.width is None:
        width = 0 if source == "portal" else 320
    else:
        width = args.width
    if args.height is None:
        height = 0 if source == "portal" else 240
    else:
        height = args.height

    cfg = AgentConfig(
        source=source,
        region=args.region or "0,0,320,240",
        width=width,
        height=height,
        fps=args.fps,
        display=args.display or os.environ.get("DISPLAY", ":0"),
        daemon_bin=args.daemon,
        portal_types=args.portal_types,
        portal_timeout=args.portal_timeout,
        decision_interval=args.decision_interval,
        obs=ObsConfig(frame_count=args.frame_count,
                      offsets_ms=tuple(int(x) for x in args.offsets.split(",")),
                      cap_fps=args.fps,
                      policy_fps=1.0 / max(args.decision_interval, 1e-6)),
        dry_run=not args.real_input,
        use_model=args.model,
        dataset_path=args.dataset,
        record_frames=args.record_frames,
        frames_dir=args.frames_dir,
        metrics_path=args.metrics,
        max_steps=args.steps,
    )
    agent = Agent(cfg)
    mode = []
    if source == "portal":
        mode.append("PORTAL capture (PipeWire; real Wayland pixels)")
    else:
        mode.append(f"daemon capture ({cfg.region})")
    if cfg.use_model:
        mode.append("Qwen model ON")
    else:
        mode.append("no model (WAIT)")
    if cfg.dry_run:
        mode.append("dry-run input (no uinput)")
    else:
        mode.append("REAL uinput input")
    if cfg.dataset_path:
        mode.append(f"dataset -> {cfg.dataset_path}")
    print(f"agent: {', '.join(mode)} | decision every "
          f"{cfg.decision_interval*1000:.0f} ms")
    if source == "portal":
        print("a 'Share Screen' window picker will open -- select the game window")
    print(f"action space: {agent.space.names()}")
    try:
        agent.start()
        t0 = time.time()
        agent.run()
        dt = time.time() - t0
        h = agent.health()
        print(f"\nstopped after {dt:.1f}s: steps={h['steps']} "
              f"rejections={h['rejections']}")
        if h["last"]:
            r = h["last"]
            print(f"last step {r.step}: action={r.action} ok={r.ok} "
                  f"obs_age={r.obs_age_ms:.1f}ms inference={r.inference_ms:.1f}ms "
                  f"total={r.total_ms:.1f}ms"
                  + (f" error={r.error}" if r.error else ""))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        agent.stop()


# ---------------------------------------------------------------------------
# core-status (P0 instrumentation dashboard)
# ---------------------------------------------------------------------------

def _summarize_metrics_file(path, max_records: int = 50000) -> dict:
    """Summarize a metrics.jsonl file: per-stage percentiles + drop window.

    Reads only the last ``max_records`` lines to bound memory on long runs.
    ``{"type": "stats"}`` samples (written by ``MetricsRecorder.observe_stats``)
    give a session drop window from the first vs last cumulative counters.
    """

    import json as _json
    from collections import defaultdict

    from anyplay.agent.metrics import _percentile

    res = {"path": str(path), "exists": False, "records": 0,
           "stages": {}, "drop_window": None}

    p = Path(path)

    if not p.exists():
        return res

    res["exists"] = True

    lines = p.read_text().splitlines()

    if max_records and len(lines) > max_records:
        lines = lines[-max_records:]

    vals = defaultdict(list)
    stats_first = None
    stats_last = None

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = _json.loads(ln)
        except Exception:
            continue
        if rec.get("type") == "stats":
            if stats_first is None:
                stats_first = rec
            stats_last = rec
            continue
        st = rec.get("stage")
        if st is not None and "duration_ms" in rec:
            res["records"] += 1
            vals[st].append(float(rec["duration_ms"]))

    for st in sorted(vals):
        arr = sorted(vals[st])
        res["stages"][st] = {
            "n": len(arr),
            "mean": round(sum(arr) / len(arr), 3),
            "p50": round(_percentile(arr, 0.50), 3),
            "p95": round(_percentile(arr, 0.95), 3),
            "p99": round(_percentile(arr, 0.99), 3),
            "max": round(arr[-1], 3),
        }

    if stats_first is not None and stats_last is not None:
        df = stats_last.get("frames", 0) - stats_first.get("frames", 0)
        dd = stats_last.get("drops", 0) - stats_first.get("drops", 0)
        res["drop_window"] = {
            "frames": df,
            "drops": dd,
            "drop_rate": round(dd / df, 6) if df > 0 else 0.0,
            "fps": stats_last.get("fps", 0.0),
            "total_frames": stats_last.get("frames", 0),
            "total_drops": stats_last.get("drops", 0),
        }

    return res


def _read_socket_stats(sock_path: str) -> dict:
    """Query a daemon control socket for live frames/fps/drops."""

    import re
    import socket as _socket

    try:
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(sock_path)
            s.sendall(b"s")
            txt = s.recv(128).decode().strip()
    except Exception as e:
        return {"error": str(e)}

    out = {"raw": txt}
    for key, pat in (("frames", r"frames=(\d+)"),
                     ("fps", r"fps=([\d.]+)"),
                     ("drops", r"drops=(\d+)")):
        m = re.search(pat, txt)
        if m:
            out[key] = int(m.group(1)) if key != "fps" else float(m.group(1))
    return out


def _print_core_dashboard(summary: dict, live, path: Path,
                          tail: list = None) -> None:
    if not summary["exists"]:
        print(f"no metrics file at {path}")
        print("run a capture/agent first, or point --metrics / $SDAI_METRICS "
              "at one.")
        if live is not None:
            print("live daemon:", live.get("raw", live))
        return

    print("AnyPlay core status")
    print(f"metrics : {summary['path']}  ({summary['records']} samples)")

    stages = summary["stages"]
    if stages:
        print()
        print(f"{'stage':<16}{'n':>7}{'mean':>9}{'p50':>9}{'p95':>9}"
              f"{'p99':>9}{'max':>9}")
        for st in sorted(stages):
            v = stages[st]
            print(f"{st:<16}{v['n']:>7}{v['mean']:>9.3f}{v['p50']:>9.3f}"
                  f"{v['p95']:>9.3f}{v['p99']:>9.3f}{v['max']:>9.3f}")
    else:
        print("  (no latency stages recorded yet)")

    dw = summary.get("drop_window")
    if dw:
        print(f"\ndrop window: {dw['frames']} frames, {dw['drops']} drops "
              f"(rate {dw['drop_rate']:.6f})  "
              f"[total {dw['total_frames']} frames, "
              f"{dw['total_drops']} drops @ {dw['fps']:.1f} fps]")

    if live is not None:
        if "error" in live:
            print(f"\nlive daemon: error: {live['error']}")
        else:
            print(f"\nlive daemon: {live.get('raw', '')}")

    if tail:
        print(f"\nlast {len(tail)} records:")
        for ln in tail:
            print("  " + ln)


def cmd_core_status(args):
    import json as _json
    import sys
    import time as _time

    from anyplay.agent.metrics import default_metrics_path

    path = Path(args.metrics) if args.metrics else default_metrics_path()

    def render() -> None:
        summary = _summarize_metrics_file(path, args.max_records)
        live = _read_socket_stats(args.sock) if args.sock else None
        tail = None
        if args.tail:
            tail = (Path(path).read_text().splitlines()[-args.tail:]
                    if path.exists() else [])
        if args.json:
            out = summary
            if live is not None:
                out["live"] = live
            print(_json.dumps(out, indent=2))
        else:
            _print_core_dashboard(summary, live, path, tail)

    if args.watch and args.watch > 0:
        try:
            while True:
                if sys.stdout.isatty():
                    sys.stdout.write("\033[2J\033[H")
                render()
                sys.stdout.flush()
                _time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        render()


def build_parser():
    p = argparse.ArgumentParser(description="AnyPlay")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="list readable input devices").set_defaults(fn=cmd_devices)

    c = sub.add_parser("capture", help="record video + inputs into a session")
    c.add_argument("--session", help="session name (default: timestamp)")
    c.add_argument("--duration", type=float, default=0.0, help="seconds (0 = until Ctrl-C)")
    c.add_argument("--region", default="",
                   help="screen region 'x,y,w,h' (default: auto-detect the game window)")
    c.add_argument("--game-name", default="shadow dungeon",
                   help="window title substring to find (default: %(default)s)")
    c.add_argument("--wait-window", type=float, default=60.0,
                   help="seconds to wait for the game window (default: %(default)s)")
    c.add_argument("--launch", action="store_true",
                   help="start the game via its .desktop launcher if no window is found")
    c.add_argument("--fps", type=int, default=60)
    c.add_argument("--backend",
                   choices=["ffmpeg", "gstreamer", "portal"], default="ffmpeg",
                   help="video backend: ffmpeg/gstreamer (X11) or portal (Wayland, "
                        "window picker -- any game) (default: %(default)s)")
    c.add_argument("--portal-types", type=int, default=2,
                   help="portal capture type (2 = interactive window picker; "
                        "default: %(default)s)")
    c.add_argument("--device", default="", help="input device name or path (default: auto)")
    c.set_defaults(fn=cmd_capture)

    b = sub.add_parser("build", help="discover the action space of a session")
    b.add_argument("--session", required=True)
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="train the policy on a session")
    t.add_argument("--session", required=True)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--window", type=int, default=4, help="frames per sample")
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="cuda", help="cuda (ROCm) or cpu")
    t.set_defaults(fn=cmd_train)

    v = sub.add_parser("vl", help="ask the vision-language model about a screenshot")
    v.add_argument("prompt", help="question, e.g. 'What is on screen and what next?'")
    v.add_argument("--image", default="",
                   help="screenshot to attach (PNG/JPEG); omit for text-only")
    v.add_argument("--system", default="",
                   help="system message (e.g. 'You are a game assistant.')")
    v.add_argument("--thinking", action="store_true",
                   help="show the model's reasoning tokens")
    v.add_argument("--max-tokens", type=int, default=1024)
    v.add_argument("--port", type=int, default=8198)
    v.add_argument("--model", default="", help="GGUF model path (default: HF cache)")
    v.add_argument("--mmproj", default="", help="mmproj GGUF path (default: HF cache)")
    v.add_argument("--stop-after", action="store_true",
                   help="shut the server down after answering")
    v.set_defaults(fn=cmd_vl)

    vs = sub.add_parser("vl-serve", help="run the VL server in the foreground")
    vs.add_argument("--port", type=int, default=8198)
    vs.add_argument("--ctx-size", type=int, default=8192)
    vs.add_argument("--model", default="", help="GGUF model path (default: HF cache)")
    vs.add_argument("--mmproj", default="", help="mmproj GGUF path (default: HF cache)")
    vs.set_defaults(fn=cmd_vl_serve)

    vst = sub.add_parser("vl-stop", help="stop the background VL server")
    vst.add_argument("--port", type=int, default=8198)
    vst.set_defaults(fn=cmd_vl_stop)

    ag = sub.add_parser(
        "agent",
        help="run the closed loop: capture -> temporal obs -> Qwen -> action -> input")
    ag.add_argument("--source", choices=["daemon", "portal"], default="daemon",
                    help="capture source: daemon (X11 region) or portal "
                         "(xdg-desktop-portal + PipeWire; real pixels on "
                         "Wayland, opens a window picker) (default: %(default)s)")
    ag.add_argument("--region", default="",
                    help="screen region 'x,y,w,h' (daemon source only; "
                         "default: 0,0,320,240)")
    ag.add_argument("--width", type=int, default=None,
                    help="output width (default: 320 daemon / auto-detect portal)")
    ag.add_argument("--height", type=int, default=None,
                    help="output height (default: 240 daemon / auto-detect portal)")
    ag.add_argument("--fps", type=int, default=60,
                    help="capture FPS (daemon; portal is compositor-driven)")
    ag.add_argument("--display", default="", help="X11 display (default: $DISPLAY)")
    ag.add_argument("--daemon", default="",
                    help="capture daemon binary (default: native/capture-daemon; "
                         "use core/build/anyplay-capture for the C++ core)")
    ag.add_argument("--portal-types", type=int, default=2,
                    help="portal SelectSources bitmask: 1=monitor 2=window "
                         "4=region (default: 2 = window picker)")
    ag.add_argument("--portal-timeout", type=float, default=180.0,
                    help="seconds to wait for the portal picker/consent")
    ag.add_argument("--decision-interval", type=float, default=0.25,
                    help="seconds between decisions (default: 0.25 = 4 FPS)")
    ag.add_argument("--model", action="store_true",
                    help="use Qwen3.5-4B (starts the llama.cpp server if needed)")
    ag.add_argument("--real-input", action="store_true",
                    help="send real uinput key events (default: dry run)")
    ag.add_argument("--steps", type=int, default=0, help="stop after N decisions")
    ag.add_argument("--dataset", default="",
                    help="append training records (JSONL) to this path")
    ag.add_argument("--record-frames", action="store_true",
                    help="write decision frames to --frames-dir (data collection)")
    ag.add_argument("--frames-dir", default="data/agent-frames")
    ag.add_argument("--metrics", default="",
                    help="P0 latency/drop metrics.jsonl path "
                         "(default: $SDAI_METRICS or ./metrics.jsonl)")
    ag.add_argument("--frame-count", type=int, default=3,
                    help="temporal frames per observation (default: %(default)s)")
    ag.add_argument("--offsets", default="400,200,0",
                    help="temporal frame ages in ms, oldest first")
    ag.set_defaults(fn=cmd_agent)

    pl = sub.add_parser("play", help="run the trained policy live")
    pl.add_argument("--checkpoint", required=True)
    pl.add_argument("--threshold", type=float, default=0.5)
    pl.add_argument("--fps", type=int, default=30)
    pl.add_argument("--source", choices=["mss", "native", "portal"],
                    default="mss",
                    help="capture backend: mss screenshots (default), the "
                         "native GStreamer daemon (zero-copy), or portal "
                         "(xdg-desktop-portal + PipeWire; real pixels on Wayland, "
                         "shows a consent dialog)")
    pl.add_argument("--window", type=int, default=None, help="override training window")
    pl.add_argument("--device", default="cuda", help="cuda (ROCm) or cpu")
    pl.add_argument("--region", default="",
                    help="screen region 'x,y,w,h' to watch (default: auto-detect the game window)")
    pl.add_argument("--game-name", default="shadow dungeon",
                    help="window title substring to find (default: %(default)s)")
    pl.add_argument("--wait-window", type=float, default=60.0,
                     help="seconds to wait for the game window (default: %(default)s)")
    pl.add_argument("--launch", action="store_true",
                    help="start the game via its .desktop launcher if no window is found")
    pl.add_argument("--device-in", dest="device_in", default="",
                    help="physical input device to clone for the virtual device")
    pl.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = forever)")
    pl.add_argument("--smoke", action="store_true",
                    help="self-test: synthetic frames, no screen, no UInput")
    pl.set_defaults(fn=cmd_play)

    st = sub.add_parser(
        "core-status",
        help="per-stage latency + drop dashboard (P0 metrics.jsonl)")
    st.add_argument("--metrics", default="",
                    help="metrics file (default: $SDAI_METRICS or metrics.jsonl)")
    st.add_argument("--sock", default="",
                    help="daemon control socket for live frames/fps/drops")
    st.add_argument("--max-records", type=int, default=50000,
                    help="max recent records to summarize (default: %(default)s)")
    st.add_argument("--watch", type=float, default=0.0,
                    help="refresh every N seconds (0 = once)")
    st.add_argument("--tail", type=int, default=0,
                    help="print the last N raw records")
    st.add_argument("--json", action="store_true",
                    help="machine-readable JSON output")
    st.set_defaults(fn=cmd_core_status)

    oc = sub.add_parser(
        "obs-check",
        help="dump timestamped temporal observations side-by-side (no Qwen, P1)")
    oc.add_argument("--source", choices=["test", "daemon", "portal"],
                    default="test",
                    help="frame source (test = headless synthetic, default)")
    oc.add_argument("--frames", type=int, default=5,
                    help="number of observations to dump (default: %(default)s)")
    oc.add_argument("--offsets", default="400,200,0",
                    help="temporal frame ages in ms, oldest first (default: %(default)s)")
    oc.add_argument("--cap-fps", type=int, default=60,
                    help="capture/sample rate (test+daemon)")
    oc.add_argument("--decision-interval", type=float, default=0.25,
                    help="seconds between dumped observations")
    oc.add_argument("--width", type=int, default=320)
    oc.add_argument("--height", type=int, default=240)
    oc.add_argument("--region", default="0,0,320,240", help="daemon region")
    oc.add_argument("--daemon", default="", help="capture daemon binary (daemon mode)")
    oc.add_argument("--portal-types", type=int, default=2,
                    help="portal capture bitmask (portal mode)")
    oc.add_argument("--out", default="",
                    help="directory to write side-by-side PNGs (default: none)")
    oc.set_defaults(fn=cmd_obs_check)

    ic = sub.add_parser(
        "input-check",
        help="verify the C++ input ring end-to-end (headless, no video; "
             "P-core step 2)")
    ic.add_argument("--keyboard", default="",
                    help="keyboard /dev/input path(s), comma-separated "
                         "(default: auto-detect)")
    ic.add_argument("--pointer", default="",
                    help="pointer /dev/input path(s), comma-separated "
                         "(default: auto-detect)")
    ic.add_argument("--daemon", default="",
                    help="capture daemon binary "
                         "(default: core/build/anyplay-capture)")
    ic.add_argument("--duration", type=float, default=10.0,
                    help="seconds to sample input (default: %(default)s)")
    ic.add_argument("--verbose", action="store_true",
                    help="print every event as it arrives")
    ic.add_argument("--self-test", action="store_true",
                    help="create virtual UInput devices and inject a scripted "
                         "sequence; verify exact capture (no hardware needed, "
                         "exit 1 on mismatch)")
    ic.set_defaults(fn=cmd_input_check)

    return p


def _write_obs_side_by_side(obs, path):
    """Write an observation's frames (oldest->newest) as one side-by-side PNG."""
    import numpy as np
    from PIL import Image

    frames = [np.asarray(f) for f in obs.frames]
    if not frames:
        return
    h = max(f.shape[0] for f in frames)
    w = max(f.shape[1] for f in frames)

    def _pad(f):
        if f.shape[0] == h and f.shape[1] == w:
            return f
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:f.shape[0], :f.shape[1]] = f
        return out

    canvas = np.concatenate([_pad(f) for f in frames], axis=1)
    Image.fromarray(canvas).save(str(path))
    print(f"  side-by-side: {path}  ({canvas.shape[1]}x{canvas.shape[0]})")


def cmd_obs_check(args):
    """P1: dump timestamped temporal observations side-by-side (no Qwen)."""
    from anyplay.capture.native_capture import NativeCapture
    from anyplay.obs import ObservationBuffer, SyntheticCapture

    offsets = tuple(int(x) for x in args.offsets.split(",") if x.strip() != "")
    obs_cfg = ObsConfig(frame_count=len(offsets), offsets_ms=offsets,
                        cap_fps=args.cap_fps,
                        policy_fps=1.0 / max(args.decision_interval, 1e-6))

    if args.source == "test":
        cap = SyntheticCapture(fps=args.cap_fps, width=args.width, height=args.height)
        print(f"source: synthetic (headless) {args.width}x{args.height} @ {args.cap_fps} fps")
    elif args.source == "portal":
        cap = NativeCapture(source="portal", portal_types=args.portal_types,
                            width=args.width, height=args.height)
        print(f"source: portal (types={args.portal_types}, "
              f"{args.width}x{args.height} or auto)")
    else:
        import os

        import anyplay.capture.native_capture as nc

        if args.daemon:
            os.environ["SDAI_DAEMON_BIN"] = args.daemon
            nc.DAEMON = Path(args.daemon)
        cap = NativeCapture(source="daemon", region=args.region, width=args.width,
                            height=args.height, fps=args.cap_fps)
        print(f"source: daemon region={args.region} "
              f"(daemon={nc.DAEMON.name})")

    buf = ObservationBuffer(cap, obs_cfg=obs_cfg)
    buf.start()
    try:
        print(f"obs_cfg: {obs_cfg.frame_count} frames, offsets_ms={offsets} "
              f"(newest-first spacings={obs_cfg.spacings}), cap_fps={args.cap_fps}")
        print(f"warming up {obs_cfg.history:.2f}s so every offset is populated ...")
        time.sleep(obs_cfg.history)

        out_dir = Path(args.out) if args.out else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        prev_wall = None
        for n in range(args.frames):
            obs = buf.observation()
            wall = time.monotonic()
            if obs is None:
                print(f"obs {n}: (no frames yet)")
            else:
                gap = (wall - prev_wall) * 1000.0 if prev_wall is not None else 0.0
                print(f"\n=== observation {n}  cadence_gap={gap:6.1f}ms  "
                      f"newest_age={obs.age_ms:6.1f}ms  frames={len(obs.frames)} ===")
                for i, fr in enumerate(obs.as_frames()):
                    rel = (obs.timestamp - fr.ts) * 1000.0
                    req = offsets[i] if i < len(offsets) else -1
                    print(f"  [{i}] req_offset={req:>4}ms  actual_age={rel:6.1f}ms  "
                          f"fid={fr.id:<6} rgb={str(fr.rgb.shape)}")
                print(f"  input_state={obs.input_state}  "
                      f"last_action={obs.last_action}  game_state={obs.game_state}")
                if out_dir is not None:
                    _write_obs_side_by_side(obs, out_dir / f"obs_{n:03d}.png")
            prev_wall = wall
            if n < args.frames - 1:
                time.sleep(obs_cfg.decision_interval)

        h = buf.health()
        print(f"\nbuffer: samples={h['samples']} misses={h['misses']} "
              f"depth={h['depth']} span_s={h['span_s']}")
    finally:
        buf.stop()
        cap.close()


def _detect_input_devices():
    """Auto-detect (keyboard_path, pointer_path) via evdev.

    Keyboard = the device with the most EV_KEY codes; pointer = the
    device with relative X+Y axes.
    """
    from anyplay.capture.input_recorder import list_devices
    from evdev.ecodes import EV_KEY, REL_X, REL_Y

    kb_path, kb_score = "", 0
    ptr_path = ""
    for dev in list_devices():
        try:
            caps = dev.capabilities()
        except Exception:
            continue
        nkeys = len(caps.get(EV_KEY, []))
        if nkeys > kb_score:
            kb_path, kb_score = dev.path, nkeys
        if not ptr_path and REL_X in caps.get(2, []) and REL_Y in caps.get(2, []):
            ptr_path = dev.path
    return kb_path, ptr_path


def _input_check_read_ready(p, timeout: float = 10.0) -> str:
    """Read daemon stdout (line by line) until a READY line arrives."""
    import select

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
    raise RuntimeError(f"no READY line within {timeout:.0f}s")


def _input_check_sock(sock_path: str, cmd: bytes) -> str:
    """One command per connection; read the response line."""
    import socket as _socket

    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
        s.sendall(cmd)
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.decode(errors="replace").strip()
    finally:
        s.close()


def _input_check_self_test():
    """Create UInput keyboard+mouse, return (kb, ms, kb_path, ms_path).

    Devices are created sequentially: the kernel allocates the event
    node only after UInput creation, so each creation waits for its
    new /dev/input/event* node.
    """
    import glob

    from evdev import UInput
    from evdev.ecodes import BTN_LEFT, EV_KEY, EV_REL, KEY_A, KEY_D, KEY_E, REL_X, REL_Y

    def _nodes():
        return set(glob.glob("/dev/input/event*"))

    before = _nodes()
    kb = UInput({EV_KEY: [KEY_A, KEY_D, KEY_E]}, name="anyplay-ic-kb")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (_nodes() - before):
        time.sleep(0.05)
    kb_path = (sorted(_nodes() - before) or [""])[0]
    if not kb_path:
        kb.close()
        raise SystemExit("UInput keyboard node did not appear (missing /dev/uinput?)")

    before = _nodes()
    ms = UInput({EV_KEY: [BTN_LEFT], EV_REL: [REL_X, REL_Y]}, name="anyplay-ic-ms")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (_nodes() - before):
        time.sleep(0.05)
    ms_path = (sorted(_nodes() - before) or [""])[0]
    if not ms_path:
        kb.close(); ms.close()
        raise SystemExit("UInput mouse node did not appear (missing /dev/uinput?)")

    return kb, ms, kb_path, ms_path


# Scripted self-test sequence: (dev, type, code, value). dev 0 = keyboard,
# dev 1 = pointer (CLI order). Mirrors core/test/regression_input.py.
_SELF_TEST_SEQ = [
    (0, 1, 30, 1), (0, 1, 30, 0),      # A down/up
    (0, 1, 32, 1), (0, 1, 32, 0),      # D down/up
    (0, 1, 18, 1), (0, 1, 18, 0),      # E down/up
    (1, 1, 272, 1),                    # left button down
    (1, 2, 0, 7), (1, 2, 0, -3), (1, 2, 1, -2),  # X+7 X-3 Y-2
    (1, 1, 272, 0),                    # left button up
]


def cmd_input_check(args):
    """P-core step 2: verify the C++ input ring end-to-end (no video).

    Launches the capture daemon in --input-only mode on the keyboard /
    pointer devices, drains the input-event ring for --duration seconds,
    and reports what was captured plus the daemon's own counters.
    With --self-test, virtual UInput devices are created and a scripted
    sequence is injected; the captured events are verified exactly.
    """
    import tempfile

    from anyplay.capture.input_ring import InputEventRing

    daemon = args.daemon or str(PROJECT_ROOT / "core" / "build" / "anyplay-capture")
    if not Path(daemon).is_file():
        raise SystemExit(
            f"capture daemon not found: {daemon}\n"
            "build it with: make -C core"
        )

    kb_uinput = ms_uinput = None
    kb = args.keyboard
    ptr = args.pointer
    if args.self_test:
        kb_uinput, ms_uinput, kb, ptr = _input_check_self_test()
        print("self-test: virtual UInput devices (no hardware needed)")
    elif not kb or not ptr:
        auto_kb, auto_ptr = _detect_input_devices()
        if not kb:
            kb = auto_kb
        if not ptr:
            ptr = auto_ptr
    if not kb or not ptr:
        raise SystemExit("could not auto-detect keyboard/pointer; pass --keyboard and --pointer")

    from evdev import InputDevice

    def _dev_desc(path):
        try:
            return f"{InputDevice(path).name!r}"
        except Exception:
            return ""

    print(f"daemon:   {daemon}")
    print(f"keyboard: {kb} {_dev_desc(kb)}")
    print(f"pointer:  {ptr} {_dev_desc(ptr)}")

    shm = tempfile.mktemp(prefix="anyplay_input_check_shm_")
    sock = tempfile.mktemp(prefix="anyplay_input_check_sock_")
    proc = None
    try:
        proc = subprocess.Popen(
            [
                daemon, "--input-only",
                "--keyboard", kb,
                "--pointer", ptr,
                "--input-shm", shm,
                "--sock", sock,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        ready = _input_check_read_ready(proc)
        print(f"ready:    {ready.strip()}")

        ring = InputEventRing(shm)
        if args.self_test:
            def _press(dev_u, t, code, value):
                dev_u.write(t, code, value)
                dev_u.syn()

            exp0 = [x[1:] for x in _SELF_TEST_SEQ if x[0] == 0]
            exp1 = [x[1:] for x in _SELF_TEST_SEQ if x[0] == 1]
            print("\nself-test: injecting scripted sequence (11 events) ...\n")
            t0 = time.monotonic()
            for dev_id, t, code, value in _SELF_TEST_SEQ:
                _press(kb_uinput if dev_id == 0 else ms_uinput, t, code, value)
                time.sleep(0.05)
            deadline = time.monotonic() + 5.0
            got = {0: [], 1: []}
            while time.monotonic() < deadline:
                for ev in ring.drain():
                    got.setdefault(ev.dev_id, []).append((ev.type, ev.code, ev.value))
                if got[0] == exp0 and got[1] == exp1:
                    break
                time.sleep(0.02)
            elapsed = time.monotonic() - t0
            ring_idx, ring_events, ring_drops, ring_alive = (
                ring.idx, ring.events, ring.drops, ring.alive)
            ring.close()

            ok = (got[0] == exp0 and got[1] == exp1
                  and ring_drops == 0 and ring_events == len(_SELF_TEST_SEQ))
            print(f"\n--- input-check self-test ({elapsed * 1000:.0f} ms) ---")
            print(f"keyboard: got {len(got[0])}/{len(exp0)}  "
                  f"{'OK' if got[0] == exp0 else 'MISMATCH'}")
            if got[0] != exp0:
                print(f"  expected: {exp0}\n  got:      {got[0]}")
            print(f"pointer:  got {len(got[1])}/{len(exp1)}  "
                  f"{'OK' if got[1] == exp1 else 'MISMATCH'}")
            if got[1] != exp1:
                print(f"  expected: {exp1}\n  got:      {got[1]}")
            print(f"ring: events={ring_events} drops={ring_drops} alive={ring_alive}")
            stats = "(no stats)"
            try:
                stats = _input_check_sock(sock, b"s")
            except OSError:
                pass
            print(f"daemon stats: {stats}")
            print("PASS" if ok else "FAIL")
            if not ok:
                args._failed = True
        else:
            print(f"\nsampling {args.duration:.0f}s -- move the mouse and press keys:\n")
            t0 = time.monotonic()
            total = 0
            per_dev: dict = {}
            last: list = []
            while time.monotonic() - t0 < args.duration:
                evs = ring.drain()
                if evs:
                    total += len(evs)
                    for ev in evs:
                        per_dev[ev.dev_id] = per_dev.get(ev.dev_id, 0) + 1
                        last.append(ev)
                        if args.verbose:
                            print(f"  dev{ev.dev_id} type={ev.type} code={ev.code} "
                                  f"value={ev.value} ts={ev.ts:.6f}")
                last = last[-20:]
                if not evs:
                    time.sleep(0.02)
            elapsed = time.monotonic() - t0
            ring_idx, ring_events, ring_drops, ring_alive = (
                ring.idx, ring.events, ring.drops, ring.alive)
            ring.close()

            stats = "(no stats)"
            try:
                stats = _input_check_sock(sock, b"s")
            except OSError:
                pass

            print(f"\n--- input-check summary ---")
            print(f"events received: {total} in {elapsed:.1f}s")
            for dev_id in sorted(per_dev):
                print(f"  dev{dev_id}: {per_dev[dev_id]} events")
            print(f"ring: idx={ring_idx} events={ring_events} "
                  f"drops={ring_drops} alive={ring_alive}")
            print(f"daemon stats: {stats}")
            if last and not args.verbose:
                print("last events:")
                for ev in last[-10:]:
                    print(f"  dev{ev.dev_id} type={ev.type} code={ev.code} "
                          f"value={ev.value} ts={ev.ts:.6f}")
            if total == 0:
                print("note: no events received (plumbing ok; nothing was pressed)")
    finally:
        if kb_uinput is not None:
            kb_uinput.close()
        if ms_uinput is not None:
            ms_uinput.close()
        if proc is not None:
            try:
                _input_check_sock(sock, b"q")
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                print("warning: daemon ignored shutdown and was killed")
            for stream in (proc.stdout, proc.stderr):
                try:
                    tail = stream.read().decode(errors="replace").strip()
                except Exception:
                    tail = ""
                if tail:
                    for line in tail.split("\n")[-10:]:
                        print(f"  daemon: {line}")
        for f in (shm, sock):
            try:
                os.unlink(f)
            except OSError:
                pass


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)
    if args.cmd == "input-check" and getattr(args, "_failed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
