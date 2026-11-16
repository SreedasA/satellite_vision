"""Height-estimation metrics for nDSM / AGL maps (all heights in metres).

Conventions (documented in docs/DATA.md):
- A pixel is valid when the ground truth is finite and not equal to `nodata`.
- MAE, RMSE, SI-RMSE follow Depth2Elevation (TGRS 2025): SI-RMSE = RMSE / std(gt).
- delta_k = share of pixels with max(pred/gt, gt/pred) < 1.25**k, computed only where
  gt >= `delta_min_gt` (ratios are meaningless near 0 m); predictions are clamped to
  `eps` so negative/zero predictions count as errors rather than crashing.
- acc_1m = share of pixels with |pred - gt| < 1 m (DFC19 Track 1 "mIoU-3" height test).
- Per-building height = a robust statistic (median by default) of the pixels inside each
  building instance; floors assume a fixed storey height (default 3 m).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

HEIGHT_BINS = ((0.0, 10.0), (10.0, 30.0), (30.0, np.inf))


def valid_mask(gt: np.ndarray, nodata: float | None = None) -> np.ndarray:
    mask = np.isfinite(gt)
    if nodata is not None:
        mask &= gt != nodata
    return mask


def pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    nodata: float | None = None,
    delta_min_gt: float = 1.0,
    eps: float = 1e-3,
) -> dict[str, float]:
    """Pixel-wise metrics over valid pixels, optionally restricted by `mask`."""
    sel = valid_mask(gt, nodata) & np.isfinite(pred)
    if mask is not None:
        sel &= mask.astype(bool)
    p, g = pred[sel].astype(np.float64), gt[sel].astype(np.float64)
    n = p.size
    if n == 0:
        return {"n": 0}

    err = p - g
    rmse = float(np.sqrt(np.mean(err**2)))
    std = float(np.std(g))
    out = {
        "n": n,
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "si_rmse": rmse / std if std > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "acc_1m": float(np.mean(np.abs(err) < 1.0)),
    }

    dsel = g >= delta_min_gt
    if dsel.any():
        ratio = np.maximum(np.maximum(p[dsel], eps) / g[dsel], g[dsel] / np.maximum(p[dsel], eps))
        for k in (1, 2, 3):
            out[f"delta{k}"] = float(np.mean(ratio < 1.25**k))
    return out


class PixelAccumulator:
    """Pools pixels across many images so dataset metrics equal pixel_metrics on the union."""

    def __init__(self, delta_min_gt: float = 1.0, eps: float = 1e-3):
        self.delta_min_gt, self.eps = delta_min_gt, eps
        self.n = self.abs = self.sq = self.err = self.g = self.g2 = self.acc1 = 0.0
        self.nd = 0
        self.d = np.zeros(3)

    def update(self, pred, gt, mask=None, nodata=None):
        sel = valid_mask(gt, nodata) & np.isfinite(pred)
        if mask is not None:
            sel &= mask.astype(bool)
        p, g = pred[sel].astype(np.float64), gt[sel].astype(np.float64)
        e = p - g
        self.n += p.size
        self.abs += np.abs(e).sum()
        self.sq += (e**2).sum()
        self.err += e.sum()
        self.g += g.sum()
        self.g2 += (g**2).sum()
        self.acc1 += (np.abs(e) < 1.0).sum()
        dsel = g >= self.delta_min_gt
        if dsel.any():
            pe = np.maximum(p[dsel], self.eps)
            ratio = np.maximum(pe / g[dsel], g[dsel] / pe)
            self.nd += int(dsel.sum())
            self.d += [(ratio < 1.25**k).sum() for k in (1, 2, 3)]

    def compute(self) -> dict[str, float]:
        if self.n == 0:
            return {"n": 0}
        rmse = np.sqrt(self.sq / self.n)
        std = np.sqrt(max(self.g2 / self.n - (self.g / self.n) ** 2, 0.0))
        out = {"n": int(self.n), "mae": self.abs / self.n, "rmse": float(rmse),
               "si_rmse": float(rmse / std) if std > 0 else float("nan"),
               "bias": self.err / self.n, "acc_1m": self.acc1 / self.n}
        if self.nd:
            out |= {f"delta{k + 1}": float(self.d[k] / self.nd) for k in range(3)}
        return {k: (round(float(v), 4) if k != "n" else v) for k, v in out.items()}


def binned_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    nodata: float | None = None,
    bins=HEIGHT_BINS,
) -> dict[str, dict[str, float]]:
    """Pixel metrics split by ground-truth height bin, e.g. '0-10', '10-30', '30+'."""
    sel = valid_mask(gt, nodata)
    if mask is not None:
        sel &= mask.astype(bool)
    out = {}
    for lo, hi in bins:
        name = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
        in_bin = sel & (gt >= lo) & (gt < hi)
        out[name] = pixel_metrics(pred, gt, in_bin)
    return out


def label_buildings(building_mask: np.ndarray, min_pixels: int = 10) -> np.ndarray:
    """Connected components of a building mask (8-connectivity); drops tiny blobs."""
    labels, _ = ndimage.label(building_mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    if min_pixels > 1:
        sizes = np.bincount(labels.ravel())
        small = sizes < min_pixels
        small[0] = False
        labels[small[labels]] = 0
        labels, _ = ndimage.label(labels > 0, structure=np.ones((3, 3), dtype=int))
    return labels


def per_building_heights(
    height: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray | None = None,
    stat: str = "median",
) -> dict[int, float]:
    """Robust height per building instance (label 0 = background)."""
    fn = {"median": np.median, "mean": np.mean, "p75": lambda x: np.percentile(x, 75),
          "p90": lambda x: np.percentile(x, 90)}[stat]
    sel = labels > 0
    if valid is not None:
        sel &= valid
    ids = labels[sel]
    vals = height[sel]
    order = np.argsort(ids, kind="stable")
    ids, vals = ids[order], vals[order]
    uniq, starts = np.unique(ids, return_index=True)
    groups = np.split(vals, starts[1:])
    return {int(i): float(fn(v)) for i, v in zip(uniq, groups) if v.size}


def building_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    labels: np.ndarray,
    nodata: float | None = None,
    stat: str = "median",
    floor_height: float = 3.0,
) -> dict[str, float]:
    """Per-building errors: MAE/RMSE of instance heights plus floor-count accuracy."""
    valid = valid_mask(gt, nodata) & np.isfinite(pred)
    hp = per_building_heights(pred, labels, valid, stat)
    hg = per_building_heights(gt, labels, valid, stat)
    ids = sorted(set(hp) & set(hg))
    if not ids:
        return {"n_buildings": 0}
    p = np.array([hp[i] for i in ids])
    g = np.array([hg[i] for i in ids])
    err = p - g
    fp = np.maximum(np.round(p / floor_height), 1)
    fg = np.maximum(np.round(g / floor_height), 1)
    return {
        "n_buildings": len(ids),
        "bld_mae": float(np.mean(np.abs(err))),
        "bld_rmse": float(np.sqrt(np.mean(err**2))),
        "bld_bias": float(np.mean(err)),
        "floor_exact": float(np.mean(fp == fg)),
        "floor_pm1": float(np.mean(np.abs(fp - fg) <= 1)),
    }
