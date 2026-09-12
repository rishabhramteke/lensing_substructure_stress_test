"""CNN embedding network: sbi expects flat feature vectors for x, so this
reshapes back to an image internally before convolving.

v1 (2026-09-11): deepened relative to v0 after v0's posterior collapsed near
the training-set marginal (first_results.md's "Family D" section) -- v0's
4-block/32-dim embedding was noticeably shallower than the U-Net's own
encoder at an equivalent stage, and evidently under-powered for extracting
enough signal to beat a single scalar's own prior. Now 5 blocks, 256 peak
channels, 64-dim output -- closer to (though still smaller than) the U-Net's
effective capacity, on the theory that a global-summary task needs at least
as rich a feature extractor as a per-pixel one, not less.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ImageEmbedding(nn.Module):
    def __init__(self, num_pix: int = 64, out_dim: int = 64):
        super().__init__()
        self.num_pix = num_pix
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),    # 32x32
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 16x16
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 8x8
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),# 4x4
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),           # 4x4, extra block
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dim))

    def forward(self, x):
        x = x.view(-1, 1, self.num_pix, self.num_pix)
        h = self.net(x).flatten(1)
        return self.fc(h)
