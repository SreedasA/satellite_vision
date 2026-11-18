"""PyTorch datasets: RGB tile -> height map (m) with a validity mask.

Training samples are random `crop` x `crop` windows at native resolution (tiles smaller than
`crop` are upsampled first), with flips/rot90 and photometric jitter. Evaluation samples are
whole tiles resized to the nearest multiple of 14 (DAv2 patch size); targets stay at native
size so metrics are computed on the original grid.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sat2city.data.io import read_height, read_rgb
from sat2city.models.dav2 import normalize


def dataset_items(dataset: str, root: Path, ids: list[str]) -> list[tuple[str, str]]:
    """(rgb_path, height_path) pairs for the given sample ids / tile names."""
    if dataset == "dfc19":
        return [(str(root / f"{i}_RGB.tif"), str(root / f"{i}_AGL.tif")) for i in ids]
    if dataset == "dfc23":
        return [(str(root / "rgb" / n), str(root / "dsm" / n)) for n in ids]
    raise ValueError(dataset)


def multiple_of_14(n: int) -> int:
    return max(14, int(round(n / 14)) * 14)


class HeightDataset(Dataset):
    def __init__(self, items, train: bool, crop: int = 518, min_height: float = 0.0):
        self.items, self.train, self.crop, self.min_height = items, train, crop, min_height
        self.photo = v2.Compose([
            v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.03)], p=0.8),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        ])

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        rgb_path, h_path = self.items[i]
        rgb = torch.from_numpy(np.ascontiguousarray(read_rgb(rgb_path))).permute(2, 0, 1)  # uint8 3HW
        h = torch.from_numpy(read_height(h_path))
        valid = torch.isfinite(h)
        h = torch.nan_to_num(h, nan=0.0).clamp_min(self.min_height)
        return rgb, h, valid

    def __getitem__(self, i):
        rgb, h, valid = self._load(i)
        if not self.train:
            H, W = h.shape
            size = (multiple_of_14(H), multiple_of_14(W))
            x = F.interpolate(normalize(rgb[None]), size=size, mode="bilinear", align_corners=False)[0]
            return {"image": x, "height": h, "valid": valid, "id": Path(self.items[i][0]).name}

        c = self.crop
        H, W = h.shape
        if min(H, W) < c:  # e.g. DFC23 512 tiles -> upsample so a full crop fits
            s = c / min(H, W)
            size = (int(np.ceil(H * s)), int(np.ceil(W * s)))
            rgb = F.interpolate(rgb[None].float(), size=size, mode="bilinear", align_corners=False)[0]
            h = F.interpolate(h[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
            valid = F.interpolate(valid[None, None].float(), size=size, mode="nearest")[0, 0].bool()
            H, W = size
        y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
        rgb, h, valid = rgb[:, y0:y0 + c, x0:x0 + c], h[y0:y0 + c, x0:x0 + c], valid[y0:y0 + c, x0:x0 + c]

        k = random.randint(0, 3)
        rgb, h, valid = torch.rot90(rgb, k, (1, 2)), torch.rot90(h, k), torch.rot90(valid, k)
        if random.random() < 0.5:
            rgb, h, valid = rgb.flip(2), h.flip(1), valid.flip(1)
        rgb = self.photo(rgb.to(torch.uint8))
        return {"image": normalize(rgb[None])[0], "height": h.contiguous(), "valid": valid.contiguous()}
