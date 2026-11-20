"""Controlled leakage study splits for DFC2019 (DECISIONS D13).

Fixed test views on the tile split's test tiles; two training sets of the SAME size:
  leak_a: the tile split's training set (no view of any test location) -> model A = p2d_dfc19_tile
  leak_b: the same set with N random views swapped for N *other* views of the test tiles
Both share val (tile split val) and test (every other view of each test tile with >= 8 views).
Evaluating A and B on identical test views isolates leakage from data amount and test content.

Usage: python scripts/make_leakage_splits.py   (writes configs/splits/dfc19_leak_{a,b}.json)
"""

import json
import random
from collections import defaultdict

from sat2city.config import REPO_ROOT

MIN_VIEWS = 8


def make(tile_split: dict, seed: int = 0) -> tuple[dict, dict, dict]:
    by_tile = defaultdict(list)
    for sid in tile_split["test"]:
        by_tile[sid.rsplit("_", 1)[0]].append(sid)
    test, leak = [], []
    for t in sorted(by_tile):
        views = sorted(by_tile[t])
        if len(views) < MIN_VIEWS:
            continue
        test += views[0::2]  # alternate views, so both halves span the acquisition dates
        leak += views[1::2]
    rng = random.Random(seed)
    train_b = sorted(set(tile_split["train"]) - set(rng.sample(sorted(tile_split["train"]), len(leak)))) + leak
    a = {"train": tile_split["train"], "val": tile_split["val"], "test": test}
    b = {"train": train_b, "val": tile_split["val"], "test": test}
    info = {"test_tiles": sorted({s.rsplit("_", 1)[0] for s in test}), "n_test": len(test), "n_leak_views": len(leak),
            "n_train": len(train_b)}
    return a, b, info


def main():
    split_dir = REPO_ROOT / "configs" / "splits"
    a, b, info = make(json.loads((split_dir / "dfc19_tile.json").read_text()))
    (split_dir / "dfc19_leak_a.json").write_text(json.dumps(a, indent=1))
    (split_dir / "dfc19_leak_b.json").write_text(json.dumps(b, indent=1))
    print(info)


if __name__ == "__main__":
    main()
