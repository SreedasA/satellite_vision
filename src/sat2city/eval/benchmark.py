"""Benchmark adapters + model evaluation shared by training, baselines and reports.

Per-building instances: DFC19 = connected components of CLS==6 (>= 20 px);
DFC23 = annotated footprints with reference median >= 2 m (docs/DATA.md, DECISIONS D10).
Per-building errors are pooled over all evaluated tiles and also reported per height bin,
because a single number is dominated by the many low houses (docs/LOG.md, Phase 0).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sat2city.config import REPO_ROOT
from sat2city.data.datasets import multiple_of_14
from sat2city.data.io import read_height, read_label, read_rgb
from sat2city.eval.metrics import HEIGHT_BINS, PixelAccumulator, label_buildings, per_building_heights
from sat2city.models.dav2 import normalize


class DFC19Bench:
    name = "dfc19"

    def __init__(self, root: Path):
        self.root = root

    def load(self, sid: str):
        g = read_height(self.root / f"{sid}_AGL.tif")
        bmask = read_label(self.root / f"{sid}_CLS.tif") == 6
        return read_rgb(self.root / f"{sid}_RGB.tif"), g, bmask, label_buildings(bmask, min_pixels=20)


class DFC23Bench:
    name = "dfc23"

    def __init__(self, root: Path):
        from sat2city.data.dfc23 import load_index

        self.root = root
        tiles, self.shapes = load_index(root)
        self.ids = {t.name: t.image_id for t in tiles}

    def load(self, name: str):
        from sat2city.data.dfc23 import instance_labels

        g = read_height(self.root / "dsm" / name)
        lab = instance_labels(self.shapes.get(self.ids[name], []), g.shape)
        unreliable = [i for i, h in per_building_heights(g, lab).items() if h < 2.0]
        lab[np.isin(lab, unreliable)] = 0
        return read_rgb(self.root / "rgb" / name), g, lab > 0, lab


def make_bench(dataset: str, processed: Path):
    return DFC19Bench(processed / "dfc19") if dataset == "dfc19" else DFC23Bench(processed / "dfc23" / "track2")


def load_split(dataset: str, split: str) -> dict[str, list[str]]:
    return json.loads((REPO_ROOT / "configs" / "splits" / f"{dataset}_{split}.json").read_text())


def sample_ids(ids: list[str], n: int | None, seed: int = 0) -> list[str]:
    """Deterministic subset (same rule as scripts/baselines.py, so numbers are comparable)."""
    if not n or n >= len(ids):
        return list(ids)
    return random.Random(seed).sample(ids, n)


@torch.no_grad()
def predict_tile(model, rgb: np.ndarray, device="cuda") -> np.ndarray:
    """Full-tile prediction: resize to a multiple of 14, predict, resize back to native size."""
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(device)
    x = F.interpolate(normalize(x), size=(multiple_of_14(h), multiple_of_14(w)), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        y = model(x)
    return F.interpolate(y[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


class BuildingPool:
    """Collects per-building (pred, ref) heights across tiles."""

    def __init__(self, floor_height: float = 3.0):
        self.p, self.g, self.floor_height = [], [], floor_height

    def update(self, pred, gt, labels):
        valid = np.isfinite(gt) & np.isfinite(pred)
        hp, hg = per_building_heights(pred, labels, valid), per_building_heights(gt, labels, valid)
        ids = sorted(set(hp) & set(hg))
        self.p += [hp[i] for i in ids]
        self.g += [hg[i] for i in ids]

    def _stats(self, p, g):
        if not len(p):
            return {"n": 0}
        e = p - g
        fp = np.maximum(np.round(p / self.floor_height), 1)
        fg = np.maximum(np.round(g / self.floor_height), 1)
        return {"n": int(len(p)), "mae": round(float(np.abs(e).mean()), 3),
                "rmse": round(float(np.sqrt((e**2).mean())), 3), "bias": round(float(e.mean()), 3),
                "floor_exact": round(float((fp == fg).mean()), 3), "floor_pm1": round(float((np.abs(fp - fg) <= 1).mean()), 3)}

    def compute(self):
        p, g = np.array(self.p), np.array(self.g)
        out = {"all": self._stats(p, g)}
        for lo, hi in HEIGHT_BINS:
            name = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
            sel = (g >= lo) & (g < hi)
            out[name] = self._stats(p[sel], g[sel])
        return out


def evaluate_model(model, bench, ids: list[str], device="cuda") -> dict:
    acc_all, acc_bld, pool = PixelAccumulator(), PixelAccumulator(), BuildingPool()
    model.eval()
    for sid in ids:
        rgb, g, bmask, labels = bench.load(sid)
        p = predict_tile(model, rgb, device)
        acc_all.update(p, g)
        acc_bld.update(p, g, bmask)
        pool.update(p, g, labels)
    return {"n_images": len(ids), "all": acc_all.compute(), "building": acc_bld.compute(), "per_building": pool.compute()}
