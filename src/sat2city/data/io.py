"""Raster readers shared by DFC19 / DFC23 code."""

from __future__ import annotations

import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def _open(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return rasterio.open(path)


def read_rgb(path) -> np.ndarray:
    """(H, W, 3) uint8."""
    with _open(path) as ds:
        arr = ds.read([1, 2, 3])
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, (1, 99))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def read_height(path, nodata_values=(-999.0, -9999.0, -10000.0)) -> np.ndarray:
    """(H, W) float32 metres; nodata -> NaN."""
    with _open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
    for v in nodata_values:
        arr[arr == v] = np.nan
    return arr


def read_label(path) -> np.ndarray:
    with _open(path) as ds:
        return ds.read(1)


def describe(path) -> dict:
    """Georeferencing facts for P0.5 (globe vs local frame)."""
    with _open(path) as ds:
        return {
            "shape": (ds.height, ds.width, ds.count),
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs) if ds.crs else None,
            "transform_is_identity": ds.transform.is_identity,
            "transform": tuple(round(v, 6) for v in ds.transform[:6]),
            "has_rpcs": ds.rpcs is not None,
            "nodata": ds.nodata,
            "tags": dict(list(ds.tags().items())[:10]),
        }
