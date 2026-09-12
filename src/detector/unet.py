"""A small U-Net for 64x64 single-channel lens images.

No architecture details are specified in Ostdiek+2020/22 or Tsang+2024 beyond
"U-Net image segmentation architecture" — this is a standard, unremarkable
encoder-decoder with skip connections, sized for 64x64 inputs on a laptop.
The point of this reimplementation is not architectural novelty; it is
finding out whether *a* reasonable U-Net, trained on our simulator's Tier-0
images, reproduces the completeness numbers the literature reports.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.bottleneck = conv_block(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c2, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # logits, (N,1,H,W)
