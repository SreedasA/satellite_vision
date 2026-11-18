"""Image -> height map (GeoTIFF) -> LoD1 buildings (GeoJSON).

Footprint sources (`--footprints`):
  reference  benchmark building instances (DFC19 CLS==6 components / DFC23 annotations); for demos
             on benchmark tiles, clearly labelled "reference footprints".
  height     connected blobs where predicted height >= --min-height. Works on any image but also
             picks up trees and other tall non-building objects (Phase 1 placeholder until a
             building-mask head exists).

Georeferenced inputs keep their CRS/transform; non-georeferenced inputs (DFC19) are placed in a
local frame around --origin lon,lat with --gsd metres/pixel (approximate placement).

Usage:
  python -m sat2city.infer --checkpoint RUN/best.pt --dataset dfc23 --ids SV_NewDelhi_...tif [...] --out web/data/x.geojson
  python -m sat2city.infer --checkpoint RUN/best.pt --image any.tif --footprints height --origin -81.66,30.33 --gsd 1.3 --out x.geojson
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning

from sat2city.config import load_paths
from sat2city.data.io import read_rgb
from sat2city.eval.benchmark import make_bench, predict_tile
from sat2city.eval.metrics import label_buildings
from sat2city.geo.lod1 import instances_to_features, instances_to_features_geo, write_geojson
from sat2city.models.dav2 import DAv2Height

TILE, OVERLAP = 1022, 128


def load_model(checkpoint: str | Path, device: str = "cuda") -> torch.nn.Module:
    paths = load_paths()
    model = DAv2Height(cache_dir=str(paths["cache"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def predict_large(model, rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Whole-image prediction; images larger than one tile are processed in overlapping
    windows blended with a linear ramp so seams don't show."""
    h, w = rgb.shape[:2]
    if h <= TILE + 14 and w <= TILE + 14:
        return predict_tile(model, rgb, device)
    step = TILE - OVERLAP
    out = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    ramp = np.minimum(np.arange(TILE) + 1, OVERLAP).astype(np.float32) / OVERLAP
    win = np.minimum.outer(np.minimum(ramp, ramp[::-1]), np.minimum(ramp, ramp[::-1]))
    ys = list(range(0, max(h - TILE, 0) + 1, step)) + ([h - TILE] if (h - TILE) % step else [])
    xs = list(range(0, max(w - TILE, 0) + 1, step)) + ([w - TILE] if (w - TILE) % step else [])
    for y in ys:
        for x in xs:
            crop = rgb[y:y + TILE, x:x + TILE]
            p = predict_tile(model, crop, device)
            wy, wx = p.shape
            out[y:y + wy, x:x + wx] += p * win[:wy, :wx]
            weight[y:y + wy, x:x + wx] += win[:wy, :wx]
    return out / np.maximum(weight, 1e-6)


def write_height_tif(path: Path, height: np.ndarray, profile_src: Path | None) -> None:
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": height.shape[1],
               "height": height.shape[0], "compress": "deflate"}
    if profile_src is not None:
        with rasterio.open(profile_src) as src:
            if src.crs:
                profile |= {"crs": src.crs, "transform": src.transform}
    path.parent.mkdir(parents=True, exist_ok=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(height.astype(np.float32), 1)


def buildings_for(height, rgb_path: Path, labels, origin, gsd, min_height):
    """Returns (features, placement) where placement is 'georeferenced' or 'local'."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(rgb_path) as src:
            crs, transform = src.crs, src.transform
    if crs:
        return instances_to_features_geo(labels, height, transform, crs, min_height=min_height), "georeferenced"
    return instances_to_features(labels, height, gsd, origin, min_height=min_height), "local"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=["dfc19", "dfc23"])
    ap.add_argument("--ids", nargs="*", default=[], help="benchmark sample ids / tile names")
    ap.add_argument("--image", nargs="*", default=[], help="arbitrary RGB rasters")
    ap.add_argument("--footprints", choices=["reference", "height"], default="reference")
    ap.add_argument("--min-height", type=float, default=2.5)
    ap.add_argument("--origin", default="-81.6581,30.3270", help="lon,lat for non-georeferenced images")
    ap.add_argument("--gsd", type=float, default=1.3)
    ap.add_argument("--out", required=True, help="output GeoJSON (height GeoTIFFs go next to it)")
    args = ap.parse_args()

    paths = load_paths()
    model = load_model(args.checkpoint)
    origin = tuple(float(v) for v in args.origin.split(","))
    out = Path(args.out)
    feats, stats = [], []

    jobs = []
    if args.ids:
        bench = make_bench(args.dataset, paths["processed"])
        for sid in args.ids:
            rgb_path = (bench.root / f"{sid}_RGB.tif") if args.dataset == "dfc19" else (bench.root / "rgb" / sid)
            jobs.append((sid, rgb_path, bench))
    jobs += [(Path(p).stem, Path(p), None) for p in args.image]

    for i, (name, rgb_path, bench) in enumerate(jobs):
        t0 = time.perf_counter()
        if bench is not None:
            rgb, _, _, ref_labels = bench.load(name)
        else:
            rgb, ref_labels = read_rgb(rgb_path), None
        height = predict_large(model, rgb)
        t1 = time.perf_counter()
        if args.footprints == "reference" and ref_labels is not None:
            labels = ref_labels
        else:
            labels = label_buildings(height >= args.min_height, min_pixels=20)
        tile_origin = (origin[0] + i * 0.015, origin[1])  # spread local-frame tiles apart
        f, placement = buildings_for(height, rgb_path, labels, tile_origin, args.gsd, args.min_height)
        for x in f:
            x["properties"]["tile"] = name
        feats += f
        write_height_tif(out.parent / f"{Path(name).stem}_height.tif", height, rgb_path)
        stats.append({"tile": name, "buildings": len(f), "placement": placement,
                      "predict_s": round(t1 - t0, 2), "lod1_s": round(time.perf_counter() - t1, 2)})

    write_geojson(feats, out, meta={"checkpoint": str(args.checkpoint), "footprints": args.footprints,
                                    "placement": sorted({s["placement"] for s in stats})})
    print(json.dumps({"out": str(out), "buildings": len(feats), "tiles": stats}, indent=1))


if __name__ == "__main__":
    main()
