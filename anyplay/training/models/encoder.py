"""CNN frame encoder: raw RGB frame -> fixed-size game-state feature.

Input:  (B, 3, H, W) float in [0, 1]
Output: (B, feat)  in [-1, 1] (tanh)

Small and fast by design; it runs at 30 FPS during play, so a few MParams
is plenty for a 128x96 pixel-art game.
"""

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, feat: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, feat)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return torch.tanh(self.head(h))
