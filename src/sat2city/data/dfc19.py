"""DFC2019 Track 1 (US3D) indexing, manifest and leakage-safe splits.

Track 1 ships one unrectified RGB view per file plus per-view reference rasters:
  <CITY>_<TILE>_<VIEW>_RGB.tif, ..._AGL.tif (height above ground, m, NaN = nodata),
  ..._CLS.tif (LAS codes: 2 ground, 5 trees, 6 building, 9 water, 17 bridge, 65 unlabeled).
Several views share one geographic tile, so splits must be made by (city, tile), never by image.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

NAME_RE = re.compile(r"^(?P<city>[A-Z]{3})_(?P<tile>\d+)_(?P<view>\d+)_(?P<kind>RGB|AGL|CLS)\.tif$")

# Pixel size of the pan-sharpened WorldView-3 RGB tiles. The files carry no georeferencing; measured on
# the imagery (cars ~15 px long, parking bays ~8 px wide), 20 Nov. The earlier 1.3 m was the MSI band
# resolution, not these tiles (docs/LOG.md, project audit).
GSD = 0.3

LAS = {"ground": 2, "trees": 5, "building": 6, "water": 9, "bridge": 17, "unlabeled": 65}


@dataclass
class Sample:
    city: str
    tile: str
    view: str
    rgb: str
    agl: str
    cls: str

    @property
    def tile_id(self) -> str:
        return f"{self.city}_{self.tile}"

    @property
    def sample_id(self) -> str:
        return f"{self.city}_{self.tile}_{self.view}"


def index_samples(root: Path) -> tuple[list[Sample], list[str]]:
    """Find RGB/AGL/CLS triplets under `root`. Returns (samples, unmatched_filenames)."""
    groups: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    unmatched = []
    for path in sorted(root.rglob("*.tif")):
        m = NAME_RE.match(path.name)
        if not m:
            unmatched.append(path.name)
            continue
        groups[(m["city"], m["tile"], m["view"])][m["kind"]] = str(path)
    samples = [
        Sample(city, tile, view, files["RGB"], files["AGL"], files["CLS"])
        for (city, tile, view), files in sorted(groups.items())
        if {"RGB", "AGL", "CLS"} <= files.keys()
    ]
    return samples, unmatched


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(samples: list[Sample], out_csv: Path, with_hash: bool = True) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "tile_id", "city", "kind", "path", "sha256"])
        for s in samples:
            for kind in ("rgb", "agl", "cls"):
                p = getattr(s, kind)
                w.writerow([s.sample_id, s.tile_id, s.city, kind, p, sha256(p) if with_hash else ""])


def make_splits(samples: list[Sample], seed: int = 0, frac=(0.70, 0.15, 0.15)) -> dict[str, dict[str, list[str]]]:
    """Three split protocols, each mapping train/val/test -> sample_ids.

    random_image: views shuffled independently (reproduces prior work; expected to leak).
    tile:         whole tiles assigned to one split, stratified per city.
    cross_city:   train+val on JAX tiles, test on every OMA view (and the reverse via swap).
    """
    rng = random.Random(seed)
    ids = [s.sample_id for s in samples]

    def cut(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        a, b = round(n * frac[0]), round(n * (frac[0] + frac[1]))
        return items[:a], items[a:b], items[b:]

    shuffled = ids[:]
    rng.shuffle(shuffled)
    tr, va, te = cut(shuffled)
    splits = {"random_image": {"train": tr, "val": va, "test": te}}

    by_city_tile: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_city_tile[s.city][s.tile_id].append(s.sample_id)
    tile_split = {"train": [], "val": [], "test": []}
    for city in sorted(by_city_tile):
        tiles = sorted(by_city_tile[city])
        rng.shuffle(tiles)
        for name, part in zip(("train", "val", "test"), cut(tiles)):
            for t in part:
                tile_split[name].extend(by_city_tile[city][t])
    splits["tile"] = tile_split

    if {"JAX", "OMA"} <= by_city_tile.keys():
        jax_tiles = sorted(by_city_tile["JAX"])
        rng.shuffle(jax_tiles)
        n_val = max(1, round(len(jax_tiles) * 0.15))
        splits["cross_city_jax2oma"] = {
            "train": [i for t in jax_tiles[n_val:] for i in by_city_tile["JAX"][t]],
            "val": [i for t in jax_tiles[:n_val] for i in by_city_tile["JAX"][t]],
            "test": [i for t in by_city_tile["OMA"] for i in by_city_tile["OMA"][t]],
        }
    return splits


def leakage_report(samples: list[Sample], split: dict[str, list[str]]) -> dict[str, int]:
    """Number of tiles that appear in more than one partition of a split."""
    tile_of = {s.sample_id: s.tile_id for s in samples}
    parts = {k: {tile_of[i] for i in v} for k, v in split.items()}
    return {
        "train&val": len(parts["train"] & parts["val"]),
        "train&test": len(parts["train"] & parts["test"]),
        "val&test": len(parts["val"] & parts["test"]),
    }


def sample_records(samples: list[Sample]) -> list[dict]:
    return [asdict(s) | {"tile_id": s.tile_id, "sample_id": s.sample_id} for s in samples]
