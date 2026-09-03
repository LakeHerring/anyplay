"""One-command capture session: 60 FPS master video + synchronized input log.

Produces ``datasets/<session>/``:

    video.mp4     60 FPS master recording (keep forever)
    inputs.jsonl  raw evdev events, t = seconds since recorder start
    meta.json     clocks, geometry, device info, alignment offset
"""

import json
import time
from pathlib import Path

from .input_recorder import InputRecorder
from .video_capturer import VideoCapturer


def _make_capturer(video_path, capture_cfg):
    backend = getattr(capture_cfg, "backend", "ffmpeg")
    if backend == "portal":
        # Wayland / xdg-desktop-portal capture: the window picker selects the
        # game window at native resolution, so region is ignored.
        from .video_capturer_portal import PortalVideoCapturer

        return PortalVideoCapturer(
            video_path,
            fps=capture_cfg.fps,
            crf=capture_cfg.crf,
            preset=capture_cfg.preset,
            duration=capture_cfg.duration,
            portal_types=getattr(capture_cfg, "portal_types", 2),
            portal_timeout=getattr(capture_cfg, "portal_timeout", 180.0),
        )
    if backend == "gstreamer":
        from .video_capturer_gst import GstVideoCapturer

        return GstVideoCapturer(
            video_path,
            display=capture_cfg.display,
            region=capture_cfg.region,
            fps=capture_cfg.fps,
            crf=capture_cfg.crf,
            preset=capture_cfg.preset,
            duration=capture_cfg.duration,
        )
    return VideoCapturer(
        video_path,
        display=capture_cfg.display,
        region=capture_cfg.region,
        fps=capture_cfg.fps,
        crf=capture_cfg.crf,
        preset=capture_cfg.preset,
        duration=capture_cfg.duration,
    )


def run_capture(session_dir, capture_cfg):
    """Run a full capture session. Returns the meta dict."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    video_path = session_dir / "video.mp4"
    inputs_path = session_dir / "inputs.jsonl"
    meta_path = session_dir / "meta.json"

    backend = getattr(capture_cfg, "backend", "ffmpeg")
    if backend == "ffmpeg":
        quality = f"x264 crf={capture_cfg.crf}"
    elif backend == "portal":
        quality = f"portal->ffmpeg x264 crf={capture_cfg.crf}"
    else:
        quality = f"gstreamer x264enc crf≈{capture_cfg.crf}"
    recorder = InputRecorder(capture_cfg.input_device, inputs_path)
    for d in recorder.info:
        print(f"input device : {d['role']:8} {d['name']} ({d['path']})")
    print(f"video out    : {video_path}  ({capture_cfg.fps} FPS, {quality}, backend={backend})")
    print(f"input log    : {inputs_path}")

    # Start the input recorder first so no early events are missed; the
    # offset between the two clocks is stored in meta.json.
    recorder.start()
    rec_start = recorder.start_time

    capturer = _make_capturer(video_path, capture_cfg)
    capturer.start()
    video_start = capturer.start_time

    meta = {
        "session": session_dir.name,
        "created_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fps": capture_cfg.fps,
        "display": capture_cfg.display,
        "backend": getattr(capture_cfg, "backend", "ffmpeg"),
        "region": capture_cfg.region,
        "input_device": recorder.info,
        "recorder_start_monotonic": rec_start,
        "video_start_monotonic": video_start,
        # video t=0 corresponds to input-log t = video_start_offset
        "video_start_offset": video_start - rec_start,
    }

    print("\nrecording... play the game. Press Ctrl-C to stop.")
    try:
        while capturer.running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopping...")

    duration = capturer.stop()
    recorder.stop()

    meta["duration"] = duration
    meta_path.write_text(json.dumps(meta, indent=2))

    with open(inputs_path) as f:
        n_events = sum(1 for _ in f)
    print(f"\nsaved session: {session_dir}")
    print(f"  video: {duration:.1f}s @ {capture_cfg.fps} FPS | input events: {n_events}")
    print(f"  next: main.py build --session {session_dir.name}")
    return meta
