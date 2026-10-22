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


class ConvNeXtUNet(nn.Module):
    """The same segmentation task with a modern backbone: a ConvNeXt encoder (Liu et al. 2022; timm `convnext_atto`,
    randomly initialized, features at strides 4-32) under a U-Net decoder, with two shallow convolutional stages at
    strides 1 and 2 as the high-resolution skips so the output keeps pixel-level localization."""
    def __init__(self):
        super().__init__()
        import timm
        self.enc = timm.create_model("convnext_atto", pretrained=False, in_chans=1, features_only=True)
        ch = self.enc.feature_info.channels()          # [40, 80, 160, 320]
        self.s1 = conv_block(1, 16); self.s2 = conv_block(16, 32)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2), nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2),
                                 nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2), nn.ConvTranspose2d(ch[0], 32, 2, stride=2),
                                 nn.ConvTranspose2d(32, 16, 2, stride=2)])
        self.dec = nn.ModuleList([conv_block(2 * ch[2], ch[2]), conv_block(2 * ch[1], ch[1]), conv_block(2 * ch[0], ch[0]),
                                  conv_block(64, 32), conv_block(32, 16)])
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        f1 = self.s1(x); f2 = self.s2(nn.functional.max_pool2d(f1, 2))
        e4, e8, e16, e32 = (t.contiguous() for t in self.enc(x))   # timm hands back permuted views; MPS backward needs contiguous
        d = e32
        for up, dec, skip in zip(self.up, self.dec, (e16, e8, e4, f2, f1)):
            d = dec(torch.cat([up(d), skip], 1))
        return self.out(d)


def build_model(ckpt: dict | None = None, arch: str = "unet"):
    arch = (ckpt or {}).get("arch", arch)
    return ConvNeXtUNet() if arch == "convnext_unet" else UNet(in_ch=1, base=(ckpt or {}).get("base", 16))
