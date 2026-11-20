"""Spatial overlap between DFC2023 train and val/test tiles (project audit, 20 Nov; DECISIONS D18).

DFC23 tiles are 512 px x 0.5 m = 256 m wide but their centres are ~120 m apart, so neighbouring
tiles overlap; blocks are assigned by tile centre, so tiles across a block edge can share pixels.
Prints overlap statistics per split and writes configs/splits/dfc23_block_clean.json
(block split with only the test tiles that share no area with any training tile).

Usage: python scripts/check_split_overlap.py
"""

import json
import warnings

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.strtree import STRtree

from sat2city.config import REPO_ROOT, load_paths

warnings.simplefilter("ignore")
ROOT = load_paths()["processed"] / "dfc23" / "track2" / "rgb"
_cache = {}


def bounds(name):
    if name not in _cache:
        with rasterio.open(ROOT / name) as s:
            _cache[name] = box(*transform_bounds(s.crs, "EPSG:4326", *s.bounds))
    return _cache[name]


def overlap_fractions(train, others):
    tr = [bounds(n) for n in train]
    tree = STRtree(tr)
    return np.array([max((bounds(n).intersection(tr[i]).area / bounds(n).area for i in tree.query(bounds(n))), default=0.0)
                     for n in others])


def main():
    split_dir = REPO_ROOT / "configs" / "splits"
    for sp in ("block", "cross_city", "random_tile"):
        s = json.loads((split_dir / f"dfc23_{sp}.json").read_text())
        for part in ("val", "test"):
            f = overlap_fractions(s["train"], s[part])
            print(f"{sp:12s} {part:4s} n={len(f):3d}  any overlap {int((f > 0).sum()):3d}  >25% {int((f > .25).sum()):3d}  "
                  f"mean share {f.mean():.3f}")
    s = json.loads((split_dir / "dfc23_block.json").read_text())
    f = overlap_fractions(s["train"], s["test"])
    clean = [n for n, x in zip(s["test"], f) if x == 0]
    (split_dir / "dfc23_block_clean.json").write_text(json.dumps({"train": s["train"], "val": s["val"], "test": clean}, indent=1))
    print("dfc23_block_clean test tiles:", len(clean))


if __name__ == "__main__":
    main()
