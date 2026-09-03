"""Policy network: temporal frame features -> controller action.

    frames (B, T, 3, H, W)
      -> FrameEncoder per frame -> (B, T, feat)
      -> GRU -> last hidden state (B, hidden)
      -> button_head: (B, n_buttons) logits        (sigmoid for play)
      -> axis_head:   (B, n_axes)   tanh, [-1, 1]  (denormalize for play)
"""

import torch
import torch.nn as nn

from .encoder import FrameEncoder


class Policy(nn.Module):
    def __init__(self, n_buttons: int, n_axes: int, n_motion: int = 0,
                 feat: int = 256, hidden: int = 256):
        super().__init__()
        if n_buttons <= 0 and n_axes <= 0 and n_motion <= 0:
            raise ValueError("policy needs at least one button, axis, or motion")
        self.n_buttons = int(n_buttons)
        self.n_axes = int(n_axes)
        self.n_motion = int(n_motion)
        self.encoder = FrameEncoder(feat)
        self.temporal = nn.GRU(feat, hidden, batch_first=True)
        self.button_head = nn.Linear(hidden, self.n_buttons)
        self.axis_head = nn.Linear(hidden, self.n_axes)
        self.motion_head = (nn.Linear(hidden, self.n_motion)
                            if self.n_motion else None)

    def frame_features(self, frames):
        """(B, T, 3, H, W) -> (B, T, feat)"""
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * t, c, h, w))
        return feats.reshape(b, t, -1)

    def forward(self, frames):
        last = self.temporal(self.frame_features(frames))[0][:, -1, :]
        out = {
            "buttons": self.button_head(last),
            "axes": torch.tanh(self.axis_head(last)),
        }
        if self.n_motion:
            out["motion"] = torch.tanh(self.motion_head(last))
        return out
