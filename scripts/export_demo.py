"""Export DFC2019 test tiles as a static viewer demo (no API needed, e.g. for GitHub Pages).

Writes web/demo/<tile>.geojson (LoD1 buildings, model heights on reference footprints),
web/demo/<tile>.png (downscaled image with a transparent 1-px border) and web/demo/index.json.
Only DFC2019 (CC-BY, see docs/DATA.md): DFC2023 imagery stays private until its licence is checked.

Usage: python scripts/export_demo.py --checkpoint S:/sat2city-data/runs/p2d_dfc19_tile/best.pt
       [--tiles JAX_164_008 JAX_122_013 OMA_331_001] [--size 768]
"""

import argparse
import json

import numpy as np
from PIL import Image

from sat2city.config import REPO_ROOT, load_paths
from sat2city.eval.benchmark import load_split, make_bench
from sat2city.geo.lod1 import instances_to_features, pixel_to_lonlat
from sat2city.infer import load_model, predict_large

GSD = 1.3  # DFC19, m per pixel (docs/DATA.md)
ORIGIN = {"JAX": (-81.6581, 30.3270), "OMA": (-95.9345, 41.2565)}  # same approximate origins as the API
ATTRIBUTION = "Imagery and reference: DFC2019 / US3D (JHU/APL, IARPA), CC-BY. Le Saux et al. 2019."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tiles", nargs="*", default=["JAX_164_008", "JAX_122_013", "OMA_331_001"])
    ap.add_argument("--size", type=int, default=768, help="exported image size (px)")
    args = ap.parse_args()

    test = set(load_split("dfc19", "tile")["test"])
    bench = make_bench("dfc19", load_paths()["processed"])
    model = load_model(args.checkpoint, "cuda")
    out_dir = REPO_ROOT / "web" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for tile in args.tiles:
        if tile not in test:
            raise SystemExit(f"{tile} is not a DFC19 tile-split test view (the demo must show unseen tiles)")
        rgb, _, _, labels = bench.load(tile)
        height = predict_large(model, rgb, "cuda")
        origin = ORIGIN[tile[:3]]
        feats = instances_to_features(labels, height, GSD, origin, min_height=2.0)
        w, n = pixel_to_lonlat(0, 0, *origin, GSD)
        e, s = pixel_to_lonlat(rgb.shape[1], rgb.shape[0], *origin, GSD)
        hs = [f["properties"]["height_m"] for f in feats]
        meta = {"tile": tile, "dataset": "dfc19", "footprints": "reference", "placement": "local",
                "bounds": [round(float(v), 7) for v in (w, s, e, n)], "image": f"demo/{tile}.png",
                "buildings": len(feats), "height_median_m": round(float(np.median(hs)), 1) if hs else None,
                "height_max_m": round(float(max(hs)), 1) if hs else None,
                "checkpoint": args.checkpoint.replace("\\", "/").split("/")[-2], "attribution": ATTRIBUTION}
        (out_dir / f"{tile}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats, "metadata": meta}))
        img = Image.fromarray(rgb).resize((args.size, args.size), Image.LANCZOS).convert("RGBA")
        a = np.asarray(img).copy()
        a[[0, -1], :, 3] = 0
        a[:, [0, -1], 3] = 0
        Image.fromarray(a, "RGBA").save(out_dir / f"{tile}.png", optimize=True)
        index.append({"tile": tile, "file": f"demo/{tile}.geojson", "buildings": len(feats),
                      "label": f"{'Jacksonville' if tile.startswith('JAX') else 'Omaha'} · {tile}"})
        print(tile, len(feats), "buildings")
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
