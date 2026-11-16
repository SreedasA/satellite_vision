"""P0.8 spike: DFC23 footprints + reference nDSM -> LoD1 GeoJSON for the Cesium viewer.

Writes web/data/<city>_gt.geojson for a few neighbouring tiles of one city and times the steps.
Usage: python scripts/spike_lod1_dfc23.py [--city NewDelhi] [--n-tiles 4]
"""

import argparse
import json
import time

import numpy as np
import rasterio

from sat2city.config import REPO_ROOT, load_paths
from sat2city.data.dfc23 import instance_labels, load_index
from sat2city.geo.lod1 import instances_to_features_geo, write_geojson


def main():
    paths = load_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="NewDelhi")
    ap.add_argument("--n-tiles", type=int, default=4)
    args = ap.parse_args()
    root = paths["processed"] / "dfc23" / "track2"

    tiles, shapes = load_index(root)
    city = sorted((t for t in tiles if t.city == args.city), key=lambda t: (t.lat, t.lon))
    # Pick the densest cluster: the tile with most buildings plus its nearest neighbours.
    seed = max(city, key=lambda t: len(shapes.get(t.image_id, [])))
    chosen = sorted(city, key=lambda t: (t.lat - seed.lat) ** 2 + (t.lon - seed.lon) ** 2)[: args.n_tiles]

    feats, timings = [], {"rasterise": 0.0, "lod1": 0.0}
    for t in chosen:
        with rasterio.open(root / "dsm" / t.name) as ds:
            h, transform, crs = ds.read(1), ds.transform, ds.crs
        t0 = time.perf_counter()
        lab = instance_labels(shapes.get(t.image_id, []), h.shape)
        t1 = time.perf_counter()
        f = instances_to_features_geo(lab, h, transform, crs, min_height=2.0)
        t2 = time.perf_counter()
        timings["rasterise"] += t1 - t0
        timings["lod1"] += t2 - t1
        for x in f:
            x["properties"]["tile"] = t.name
        feats += f

    out = REPO_ROOT / "web" / "data" / f"{args.city.lower()}_gt.geojson"
    write_geojson(feats, out, meta={"source": "DFC2023 Track 2 reference nDSM + footprints",
                                    "city": args.city, "tiles": [t.name for t in chosen]})
    hs = np.array([x["properties"]["height_m"] for x in feats])
    print(json.dumps({"city": args.city, "tiles": len(chosen), "buildings": len(feats),
                      "height_m_median": round(float(np.median(hs)), 1), "height_m_max": round(float(hs.max()), 1),
                      "sec_per_tile": {k: round(v / len(chosen), 3) for k, v in timings.items()},
                      "km2": round(len(chosen) * (512 * 0.5) ** 2 / 1e6, 3), "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
