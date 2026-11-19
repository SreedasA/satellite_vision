"""U-Net height regressor with a timm encoder: the CNN baseline for ROADMAP P2.2.

Inputs are padded to a multiple of 32 internally (encoder stride) and the output is cropped
back, so it accepts the same multiple-of-14 sizes as DAv2. Output: (B, H, W) heights >= 0.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class _Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class UNetHeight(nn.Module):
    encoder_prefix = "encoder."

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True, decoder=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        chs = self.encoder.feature_info.channels()  # e.g. resnet34: [64, 64, 128, 256, 512]
        c = chs[-1]
        ups = []
        for skip, out in zip(reversed(chs[:-1]), decoder):
            ups.append(_Up(c, skip, out))
            c = out
        self.ups = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        x = F.pad(pixel_values, (0, pw, 0, ph), mode="reflect") if (ph or pw) else pixel_values
        feats = self.encoder(x)
        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = up(y, skip)
        y = F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.relu(y[:, 0, :h, :w])
