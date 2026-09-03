"""Live play loop: screenshots -> policy inference -> virtual controller.

Captures the game screen at the play FPS (mss, or the native GStreamer
daemon via ``source="native"`` for zero-copy frames), keeps a sliding
window of frames (same window the policy was trained on), runs the
policy, and writes the predicted buttons/axes to a UInput virtual device.

    main.py play --checkpoint datasets/<session>/checkpoints/policy_best.pt
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from mss import MSS
from PIL import Image


class GameIntegration:
    def __init__(self, checkpoint_path, play_cfg, dataset_cfg, device="cuda", window=None):
        from ..training.training import load_checkpoint

        self.model, self.action_space, self.meta = load_checkpoint(
            Path(checkpoint_path), device
        )
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.cfg = play_cfg
        self.fps = play_cfg.fps
        self.source = getattr(play_cfg, "source", "mss")
        self.portal_types = getattr(play_cfg, "portal_types", 2)
        self.threshold = play_cfg.threshold
        self.display = play_cfg.display
        self.region = play_cfg.region
        self.input_device = play_cfg.input_device

        # Match the geometry the network was trained on.
        # Match the geometry the network was trained on (checkpoint config
        # wins over the dataset config when present).
        model_cfg = self.meta.get("config", {})
        self.width = int(model_cfg.get("width", dataset_cfg.width))
        self.height = int(model_cfg.get("height", dataset_cfg.height))
        self.window = int(window or model_cfg.get("window") or dataset_cfg.window)
        self.frame_window = deque(maxlen=self.window)

        self.button_codes = list(self.action_space.get("buttons", []))
        self.axis_codes = list(self.action_space.get("axes", []))
        self.motion_codes = list(self.action_space.get("motion", []))
        self.axis_bounds = self.action_space.get("axis_bounds", {})

    # ------------------------------------------------------------------
    def _monitor(self):
        """mss monitor dict for the full screen or the configured region."""
        if not self.region:
            return None  # mss default = primary monitor
        x, y, w, h = (int(v) for v in self.region.split(","))
        return {"left": x, "top": y, "width": w, "height": h}

    def grab_frame(self, screen):
        """One mss screen grab -> (3, H, W) float tensor in [0, 1]."""
        shot = screen.grab(self._monitor())
        bgra = np.asarray(shot, dtype=np.uint8)
        rgb = bgra[..., [2, 1, 0]][..., :3]  # BGRA -> RGB
        return self.frame_from_rgb(rgb)

    def frame_from_rgb(self, arr):
        """(H, W, 3) uint8 RGB array -> (3, H, W) float tensor in [0, 1].

        Skips the resize when the frame is already at training size (the
        native capture daemon scales in the pipeline).
        """
        if arr.shape[0] != self.height or arr.shape[1] != self.width:
            img = Image.fromarray(arr).resize((self.width, self.height), Image.BILINEAR)
            arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        return (
            torch.from_numpy(arr.astype(np.float32, copy=False))
            .permute(2, 0, 1)
            .div_(255.0)
        )

    @torch.no_grad()
    def predict(self, frame):
        """Feed one (3,H,W) frame into the sliding window; return action if ready."""
        self.frame_window.append(frame.squeeze(0).to(self.device))
        if len(self.frame_window) < self.window:
            return None
        frames = torch.stack(list(self.frame_window), dim=0).unsqueeze(0)  # (1,T,3,H,W)
        out = self.model(frames)
        buttons = torch.sigmoid(out["buttons"]).squeeze(0).cpu()
        axes = out["axes"].squeeze(0).cpu()
        result = {
            "buttons": {c: int(b) for c, b in
                        zip(self.button_codes, (buttons > self.threshold).int())},
            "axes": {c: float(a) for c, a in zip(self.axis_codes, axes)},
        }
        if "motion" in out:
            motion = out["motion"].squeeze(0).cpu()
            result["motion"] = {c: float(m) for c, m in
                                zip(self.motion_codes, motion)}
        return result

    # ------------------------------------------------------------------
    def run(self, controller=None, max_steps=0):
        """Run the live loop until Ctrl-C.

        ``controller=None`` builds a real GameController (needs input group).
        Pass a fake with the same write_action() interface to test headless.
        """
        if controller is None:
            from .controllers.game_controller import GameController
            controller = GameController(self.input_device, self.action_space)

        print(f"policy loaded ({self.meta.get('loss') and round(self.meta['loss'], 4)})")
        print(f"playing on {self.display} at {self.fps} FPS, "
              f"threshold={self.threshold}, window={self.window}, "
              f"source={self.source}")
        print("Ctrl-C to stop (all keys will be released).")

        period = 1.0 / self.fps
        try:
            if self.source in ("native", "portal"):
                steps = self._run_native(controller, period, max_steps)
            else:
                steps = self._run_mss(controller, period, max_steps)
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            controller.reset()
            if controller is not None and hasattr(controller, "close"):
                controller.close()
            print(f"ran {steps} steps. keys released.")

    def _run_mss(self, controller, period, max_steps):
        """mss screenshot loop (legacy path)."""
        with MSS() as screen:
            steps = 0
            while True:
                t0 = time.monotonic()
                frame = self.grab_frame(screen)
                action = self.predict(frame)
                if action is not None:
                    controller.write_action(action["buttons"], action["axes"],
                                            action.get("motion"))
                steps += 1
                if max_steps and steps >= max_steps:
                    break
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        return steps

    def _run_native(self, controller, period, max_steps):
        """Zero-copy loop on a C capture daemon (GStreamer or portal/PipeWire)."""
        from ..capture.native_capture import NativeCapture

        cap = NativeCapture(
            region=self.region,
            width=self.width, height=self.height,
            fps=self.fps, display=self.display,
            source="portal" if self.source == "portal" else "daemon",
            portal_types=self.portal_types,
        )
        try:
            steps = 0
            while True:
                t0 = time.monotonic()
                arr = cap.get_frame()
                if arr is not None:
                    # copy into a tensor now; the daemon may overwrite the
                    # shared slot while inference runs on the GPU
                    frame = self.frame_from_rgb(arr)
                    action = self.predict(frame)
                    if action is not None:
                        controller.write_action(action["buttons"], action["axes"],
                                                action.get("motion"))
                steps += 1
                if max_steps and steps >= max_steps:
                    break
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        finally:
            cap.close()
        return steps
