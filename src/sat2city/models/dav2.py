"""Depth Anything V2 wrapper (Hugging Face weights, Apache-2.0 for the Small model)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForDepthEstimation

DAV2_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"

# DINOv2 / Depth Anything preprocessing (ImageNet statistics; inputs must be multiples of 14).
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dav2(name: str = DAV2_SMALL, cache_dir: str | None = None) -> nn.Module:
    return AutoModelForDepthEstimation.from_pretrained(name, cache_dir=cache_dir)


def normalize(rgb_uint8: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) uint8/float in [0,255] -> normalised float tensor."""
    x = rgb_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class DAv2Height(nn.Module):
    """DAv2 returning a (B,H,W) map. Pretrained = relative inverse depth; after
    fine-tuning on AGL targets the same output is read as height in metres."""

    encoder_prefix = "net.backbone."  # parameters that get the (lower) encoder learning rate

    def __init__(self, name: str = DAV2_SMALL, cache_dir: str | None = None):
        super().__init__()
        self.net = load_dav2(name, cache_dir)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=pixel_values).predicted_depth
