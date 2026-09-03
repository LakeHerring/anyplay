"""Project configuration for AnyPlay.

Data flow (see README):
    capture (60 FPS master) -> build (30 FPS samples) -> train (imitation) -> play

A capture session lives in ``datasets/<session>/``:

    datasets/<session>/
        video.mp4          # 60 FPS master recording (never deleted)
        inputs.jsonl       # raw evdev events, t = seconds since recorder start
        meta.json          # clocks, geometry, device info, alignment offset
        action_space.json  # discovered buttons/axes (written by build)
        checkpoints/       # policy checkpoints (written by train)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anyplay.vl.vl_model import VLConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../AnyPlay
DATA_ROOT = PROJECT_ROOT / "datasets"


@dataclass
class CaptureConfig:
    display: str = ":0.0"      # X11 display for ffmpeg x11grab
    region: str = ""           # "x,y,w,h" of the game window; "" = full screen
    fps: int = 60              # master capture rate
    crf: int = 18              # x264 quality (lower = better, bigger)
    preset: str = "veryfast"   # x264 speed preset
    duration: float = 0.0      # seconds; 0 = run until Ctrl-C
    input_device: str = ""     # device name substring or /dev/input/eventN path
    # Video source backend:
    #   "ffmpeg"    - ffmpeg x11grab (X11 only)
    #   "gstreamer" - gstreamer ximagesrc (X11 only)
    #   "portal"    - xdg-desktop-portal / PipeWire (Wayland, any window;
    #                 the window picker selects the game, so region is ignored)
    backend: str = "ffmpeg"
    # Portal window-picker type (2 = interactive picker, any game).
    portal_types: int = 2
    # Seconds to wait for the portal consent dialog.
    portal_timeout: float = 180.0


@dataclass
class DatasetConfig:
    train_fps: int = 30        # downsampled training rate
    width: int = 128           # training frame size (4:3, Shadow Dungeon is 640x480)
    height: int = 96
    window: int = 4            # frames per training sample (4 @ 30 FPS = 133 ms)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    button_weight: float = 1.0
    axis_weight: float = 1.0
    motion_weight: float = 1.0  # weight of the relative-mouse-motion (EV_REL) loss
    device: str = "cuda"       # "cuda" (ROCm) or "cpu"


@dataclass
class PlayConfig:
    fps: int = 30
    threshold: float = 0.5     # sigmoid threshold for button presses
    input_device: str = ""     # physical device cloned into a UInput virtual device
    display: str = ":0.0"
    region: str = ""
    source: str = "mss"        # capture backend: "mss" | "native" (GStreamer daemon) | "portal" (PipeWire, Wayland)
    portal_types: int = 2      # portal source type: 1=desktop, 2=window picker (default), 3=both


@dataclass
class ObsConfig:
    """Temporal observation + policy cadence (plan §P1).

    All rates the closed loop depends on live here so nothing downstream
    is hardcoded. ``offsets_ms`` are ages of the temporal frames, **oldest
    first** (matching the plan); the observation buffer consumes them as
    newest-first seconds via :meth:`spacings`.
    """

    frame_count: int = 3             # temporal frames per observation
    offsets_ms: tuple = (400, 200, 0)  # ages, oldest first (0 = now)
    cap_fps: int = 60                # capture-ring sampling rate (background thread)
    policy_fps: float = 4.0          # decision cadence (steps / sec)
    vl_hz: float = 0.0               # 0 => VL at policy cadence; >0 => independent

    @property
    def spacings(self) -> tuple:
        """Ages in seconds, newest first, for ``ObservationBuffer.observation``."""
        return tuple(o / 1000.0 for o in reversed(self.offsets_ms))

    @property
    def sample_interval(self) -> float:
        """Seconds between background-thread samples (1 / cap_fps)."""
        return 1.0 / max(self.cap_fps, 1)

    @property
    def history(self) -> float:
        """Seconds of frame history to keep (largest offset + margin)."""
        return max(self.offsets_ms) / 1000.0 + 0.3

    @property
    def decision_interval(self) -> float:
        """Seconds between decisions (1 / policy_fps)."""
        return 1.0 / self.policy_fps if self.policy_fps > 0 else 0.0


@dataclass
class ProjectConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vl: VLConfig = field(default_factory=VLConfig)
    play: PlayConfig = field(default_factory=PlayConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    data_root: Path = field(default_factory=lambda: DATA_ROOT)

    def session_dir(self, name: str) -> Path:
        return self.data_root / name

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# Backwards-compatible alias for the old scaffold class name.
Config = ProjectConfig
