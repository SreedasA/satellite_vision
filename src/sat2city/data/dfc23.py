"""DFC2023 Track 2 (train): 512x512 georeferenced tiles from 11 cities incl. New Delhi.

Layout: track2/{rgb,dsm,sar}/<SENSOR>_<City>_<lat>_<lon>.tif ("dsm" holds nDSM heights, m)
        track2/buildings_only_train.json (COCO; values are stored as strings)
Tiles of one city are spatial neighbours, so leakage-safe splits group tiles into ~1 km
spatial blocks (`block_deg`) or hold out whole cities.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio import features
from shapely.geometry import Polygon

NAME_RE = re.compile(r"^(?P<sensor>[A-Z0-9]+)_(?P<city>[A-Za-z]+)_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)\.tif$")


@dataclass
class Tile:
    name: str
    sensor: str
    city: str
    lat: float
    lon: float
    image_id: int

    def block(self, block_deg: float) -> str:
        return f"{self.city}_{int(np.floor(self.lat / block_deg))}_{int(np.floor(self.lon / block_deg))}"


def rle_string_to_counts(s: str) -> list[int]:
    """Decode a compressed COCO RLE string (same scheme as pycocotools rleFrString)."""
    counts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_to_mask(counts: list[int], size: tuple[int, int]) -> np.ndarray:
    """COCO RLE (column-major, starting with a run of zeros) -> bool mask of `size` (h, w)."""
    h, w = size
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((w, h)).T


def load_index(root: Path) -> tuple[list[Tile], dict[int, list]]:
    """Tiles plus building shapes per image_id. A shape is either a flat polygon ring
    [x0,y0,x1,y1,...] in pixel coords or a bool mask (from compressed RLE annotations)."""
    coco = json.loads((root / "buildings_only_train.json").read_text())
    tiles = []
    for im in coco["images"]:
        m = NAME_RE.match(im["file_name"])
        if m:
            tiles.append(Tile(im["file_name"], m["sensor"], m["city"], float(m["lat"]), float(m["lon"]), int(im["id"])))
    shapes: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        seg = ann["segmentation"]
        seg = ast.literal_eval(seg) if isinstance(seg, str) else seg
        if isinstance(seg, dict):
            counts = seg["counts"]
            counts = rle_string_to_counts(counts) if isinstance(counts, str) else counts
            shapes[int(ann["image_id"])].append(rle_to_mask(counts, tuple(seg["size"])))
            continue
        for ring in seg:
            if len(ring) >= 6:
                shapes[int(ann["image_id"])].append([float(v) for v in ring])
    return tiles, shapes


def instance_labels(shapes: list, shape=(512, 512)) -> np.ndarray:
    """Rasterise building polygons/masks into an instance map (0 = background, 1..N = buildings)."""
    polys, masks = [], []
    for i, s in enumerate(shapes, start=1):
        if isinstance(s, np.ndarray):
            masks.append((s, i))
            continue
        poly = Polygon(list(zip(s[0::2], s[1::2])))
        if poly.is_valid and poly.area > 0:
            polys.append((poly, i))
    lab = (features.rasterize(polys, out_shape=shape, fill=0, dtype="int32") if polys
           else np.zeros(shape, np.int32))
    for m, i in masks:
        lab[m & (lab == 0)] = i
    return lab


def make_splits(tiles: list[Tile], seed: int = 0, block_deg: float = 0.01, frac=(0.70, 0.15, 0.15),
                holdout_cities=("NewDelhi",)) -> dict[str, dict[str, list[str]]]:
    """random_tile (leaky reference), block (spatial blocks per city), cross_city (hold out cities)."""
    rng = random.Random(seed)

    def cut(items):
        a, b = round(len(items) * frac[0]), round(len(items) * (frac[0] + frac[1]))
        return items[:a], items[a:b], items[b:]

    names = [t.name for t in tiles]
    rng.shuffle(names)
    splits = {"random_tile": dict(zip(("train", "val", "test"), cut(names)))}

    blocks: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in tiles:
        blocks[t.city][t.block(block_deg)].append(t.name)
    block_split = {"train": [], "val": [], "test": []}
    for city in sorted(blocks):
        keys = sorted(blocks[city])
        rng.shuffle(keys)
        for part, ks in zip(("train", "val", "test"), cut(keys)):
            for k in ks:
                block_split[part].extend(blocks[city][k])
    splits["block"] = block_split

    rest = [t for t in tiles if t.city not in holdout_cities]
    rest_blocks = sorted({t.block(block_deg) for t in rest})
    rng.shuffle(rest_blocks)
    n_val = max(1, round(len(rest_blocks) * 0.15))
    val_blocks = set(rest_blocks[:n_val])
    splits["cross_city"] = {
        "train": [t.name for t in rest if t.block(block_deg) not in val_blocks],
        "val": [t.name for t in rest if t.block(block_deg) in val_blocks],
        "test": [t.name for t in tiles if t.city in holdout_cities],
    }
    return splits
