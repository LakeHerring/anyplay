"""Closed-loop agent: capture -> observation -> Qwen -> action -> input.

Wires the components into the loop from the implementation plan:

    capture (60 FPS, C++ daemon + SHM)
        -> ObservationBuffer (temporal frames t, t-200ms, t-400ms)
        -> VLModel.decide (Qwen3.5-4B, 2-5 FPS)
        -> parse_decision (constrained action space; Rule 3)
        -> InputController (uinput virtual keyboard; Rule 12)
        -> game

Design rules in force:

* Rule 1  - capture runs at 60 FPS regardless of how slow Qwen is; the
  decision loop simply samples the buffer.
* Rule 2  - no disk I/O on the real-time frame path. Dataset records
  (JSONL) contain metadata + frame IDs only; frame images are written
  only when ``record_frames=True`` (explicit data-collection mode).
* Rule 3  - only validated actions reach the input controller.
* Rule 4  - every step is timed: capture wait, observation age,
  inference, parsing, input dispatch. These numbers are the basis for
  the later latency budget (plan §6).
* Rule 5  - every step appends a training record (observation metadata,
  action, reason, latencies) for the later RL student (plan §8).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..actions import ActionSpace, InputController, UInputBackend
from ..capture.native_capture import NativeCapture
from ..obs import ObservationBuffer
from ..vl import VLModel
from ..utils.config.config import ObsConfig
from .metrics import MetricsRecorder


@dataclass
class AgentConfig:
    source: str = "daemon"          # "daemon" (X11) or "portal" (Wayland/PipeWire)
    region: str = "0,0,320,240"     # daemon mode only
    width: int = 320                # 0 = auto-detect (portal) / source size
    height: int = 240
    fps: int = 60                   # daemon mode; portal is compositor-driven
    display: str = ""
    daemon_bin: str = ""            # "" = NativeCapture default
    portal_types: int = 2           # 1=monitor 2=window 4=region (bitmask)
    portal_timeout: float = 180.0   # wait for picker + consent
    decision_interval: float = 0.25  # 4 FPS target
    spacings: tuple = (0.0, 0.2, 0.4)   # legacy; buffer now uses obs.spacings
    obs: ObsConfig = field(default_factory=ObsConfig)  # plan §P1 temporal obs rates
    dry_run: bool = True            # no uinput device, no model
    use_model: bool = False
    dataset_path: str = ""          # "" = no dataset logging
    record_frames: bool = False
    frames_dir: str = ""
    metrics_path: str = ""          # "" = default (SDAI_METRICS or metrics.jsonl)
    max_steps: int = 0              # 0 = run until stopped


@dataclass
class StepResult:
    step: int
    action: str
    ok: bool
    error: str = ""
    confidence: Optional[float] = None
    reason: str = ""
    # latencies (ms)
    obs_age_ms: float = 0.0
    capture_wait_ms: float = 0.0
    inference_ms: float = 0.0
    input_ms: float = 0.0
    total_ms: float = 0.0


class Agent:
    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.space = ActionSpace()

        if self.cfg.daemon_bin:
            # NativeCapture resolves the daemon from $SDAI_DAEMON_BIN at
            # import time; override both to be safe.
            os.environ["SDAI_DAEMON_BIN"] = self.cfg.daemon_bin
            import anyplay.capture.native_capture as nc

            nc.DAEMON = Path(self.cfg.daemon_bin)

        self.capture = NativeCapture(
            source=self.cfg.source,
            region="" if self.cfg.source == "portal" else self.cfg.region,
            width=self.cfg.width,
            height=self.cfg.height,
            fps=self.cfg.fps,
            display=self.cfg.display or os.environ.get("DISPLAY", ":0"),
            portal_timeout=self.cfg.portal_timeout,
            portal_types=self.cfg.portal_types,
        )
        self.buffer = ObservationBuffer(self.capture, obs_cfg=self.cfg.obs)
        if self.cfg.dry_run:
            self.controller = InputController(backend=None)
        else:
            self.controller = InputController(backend=UInputBackend())
        self.vl: Optional[VLModel] = None
        if self.cfg.use_model:
            self.vl = VLModel()

        self._stop = threading.Event()
        self.steps = 0
        self.rejections = 0
        # P0 instrumentation: shared per-stage latency sink. The capture
        # client records the shm stages; the decision loop records the rest.
        self.metrics = MetricsRecorder(path=self.cfg.metrics_path or None)
        self.capture.metrics = self.metrics
        self._last_stats_poll = 0.0
        self._dataset_f = None
        if self.cfg.dataset_path:
            self._dataset_f = open(self.cfg.dataset_path, "a", buffering=1)
        self._frames_dir = (Path(self.cfg.frames_dir).mkdir(parents=True,
                                                             exist_ok=True)
                            if self.cfg.record_frames else None)
        self.results: list = []

    # ------------------------------------------------------------------

    def start(self) -> None:
        self.buffer.start()

    def stop(self) -> None:
        self._stop.set()
        self.buffer.stop()
        self.controller.stop()
        self.capture.close()
        self.metrics.close()
        if self._dataset_f is not None:
            self._dataset_f.close()
            self._dataset_f = None

    # ------------------------------------------------------------------
    # One decision cycle
    # ------------------------------------------------------------------

    def step(self) -> StepResult:
        t0 = time.monotonic()
        self.steps += 1

        obs = self.buffer.observation()
        if obs is None:
            return StepResult(step=self.steps, action="NONE", ok=False,
                              error="no observation (capture not producing "
                                    "frames?)", total_ms=(time.monotonic() - t0) * 1000)

        t_obs = time.monotonic()
        res = StepResult(step=self.steps, action="NONE", ok=False,
                         obs_age_ms=obs.age_ms,
                         capture_wait_ms=(t_obs - t0) * 1000.0)

        # fill observation metadata
        obs.input_state = self.controller.state()
        if self.results:
            obs.previous_action = self.results[-1].action

        if self.vl is None:
            res.action = "WAIT"
            res.ok = True
            self._record(obs, res)
        else:
            try:
                out = self.vl.decide(obs, self.space)
                res.inference_ms = out["ms"]
                d = out["decision"]
                res.ok = d.ok
                res.error = d.error
                res.confidence = d.confidence
                res.reason = d.reason
                if d.ok and d.action is not None:
                    res.action = d.action.name
                    t_in = time.monotonic()
                    self.controller.execute(d.action)
                    res.input_ms = (time.monotonic() - t_in) * 1000.0
                    self._record(obs, res)
                else:
                    self.rejections += 1
                    self._log({"type": "reject", "step": self.steps,
                               "raw": out["content"][:500],
                               "error": d.error})
            except Exception as e:  # never crash the loop on a model error
                res.error = f"{type(e).__name__}: {e}"

        res.total_ms = (time.monotonic() - t0) * 1000.0
        self._record_metrics(res)
        self.results.append(res)
        if len(self.results) > 200:
            del self.results[:100]
        return res

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> list:
        interval = self.cfg.decision_interval
        while not self._stop.is_set():
            if (self.cfg.max_steps and self.steps >= self.cfg.max_steps):
                break
            self._poll_stats()
            t0 = time.monotonic()
            self.step()
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                self._stop.wait(interval - elapsed)
        return self.results

    # ------------------------------------------------------------------
    # P0 instrumentation helpers
    # ------------------------------------------------------------------

    def _record_metrics(self, res: StepResult) -> None:
        """Record the agent-side decision stages to the shared metrics sink."""
        self.metrics.record(
            "obs_fetch", res.capture_wait_ms,
            obs_age_ms=round(res.obs_age_ms, 1),
        )
        if res.inference_ms:
            self.metrics.record("inference", res.inference_ms)
        if res.input_ms:
            self.metrics.record("input", res.input_ms)
        self.metrics.record(
            "step_total", res.total_ms,
            action=res.action, ok=int(res.ok),
        )

    def _poll_stats(self) -> None:
        """Feed daemon frame/drop counters into metrics at most once per second."""
        now = time.monotonic()
        if now - self._last_stats_poll < 1.0:
            return
        self._last_stats_poll = now
        try:
            d = self.capture.stats_dict()
        except Exception:
            return
        if d["frames"]:
            self.metrics.observe_stats(d["frames"], d["fps"], d["drops"])

    # ------------------------------------------------------------------
    # Data collection (Rule 5)
    # ------------------------------------------------------------------

    def _record(self, obs, res: StepResult) -> None:
        rec = {
            "type": "step",
            "t_wall": time.time(),
            "t_mono": obs.timestamp,
            "step": res.step,
            "frame_ages_ms": [round((obs.timestamp - ts) * 1000.0, 1)
                              for ts in obs.timestamps],
            "action": res.action,
            "confidence": res.confidence,
            "reason": res.reason,
            "obs_age_ms": round(res.obs_age_ms, 1),
            "inference_ms": round(res.inference_ms, 1),
            "total_ms": round(res.total_ms, 1),
            "input_state": obs.input_state,
        }
        self._log(rec)
        if self._frames_dir is not None:
            for i, frame in enumerate(obs.frames):
                from ..vl.vl_model import _write_png
                _write_png(self._frames_dir / f"step{res.step:06d}_f{i}.png",
                           frame)

    def _log(self, rec: dict) -> None:
        if self._dataset_f is not None:
            self._dataset_f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health(self) -> dict:
        latest = self.results[-1] if self.results else None
        return {
            "steps": self.steps,
            "rejections": self.rejections,
            "buffer": self.buffer.health(),
            "input": self.controller.state(),
            "metrics": self.metrics.summary(),
            "last": latest,
        }
