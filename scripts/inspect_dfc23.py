"""P0.4/P0.8 for DFC23 Track 2: facts, splits, and footprint/height alignment check.

Usage: python scripts/inspect_dfc23.py [--root S:/sat2city-data/processed/dfc23/track2]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from sat2city.config import REPO_ROOT, load_paths
from sat2city.data.dfc23 import instance_labels, load_index, make_splits
from sat2city.data.io import describe, read_height
from sat2city.eval.metrics import per_building_heights


def main():
    paths = load_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths["processed"] / "dfc23" / "track2"))
    args = ap.parse_args()
    root = Path(args.root)

    tiles, polys = load_index(root)
    facts = {
        "n_tiles": len(tiles),
        "cities": dict(Counter(t.city for t in tiles).most_common()),
        "sensors": dict(Counter(t.sensor for t in tiles)),
        "n_buildings": sum(len(v) for v in polys.values()),
        "rgb_raster": describe(root / "rgb" / tiles[0].name),
        "ndsm_raster": describe(root / "dsm" / tiles[0].name),
    }

    # Alignment check: building pixels should be much taller than background pixels,
    # and the median height per footprint should be well above 0.
    rng = random.Random(0)
    bld, bg, medians, empty = [], [], [], 0
    for t in rng.sample(tiles, 200):
        h = read_height(root / "dsm" / t.name)
        lab = instance_labels(polys.get(t.image_id, []), h.shape)
        if lab.max() == 0:
            empty += 1
            continue
        bld.append(h[lab > 0])
        bg.append(h[lab == 0])
        medians += list(per_building_heights(h, lab).values())
    bld, bg, medians = np.concatenate(bld), np.concatenate(bg), np.array(medians)
    facts["alignment_check_200_tiles"] = {
        "tiles_without_buildings": empty,
        "building_px_median_m": round(float(np.median(bld)), 2),
        "background_px_median_m": round(float(np.median(bg)), 2),
        "building_px_share_below_1m": round(float((bld < 1).mean()), 3),
        "per_building_median_height_pct": {p: round(float(np.percentile(medians, p)), 1) for p in (5, 25, 50, 75, 95, 99)},
        "footprints_with_median_below_2m": round(float((medians < 2).mean()), 3),
    }

    splits = make_splits(tiles, seed=0)
    for name, sp in splits.items():
        (REPO_ROOT / "configs" / "splits" / f"dfc23_{name}.json").write_text(json.dumps(sp, indent=0))
    facts["splits"] = {n: {k: len(v) for k, v in sp.items()} for n, sp in splits.items()}

    out = paths["processed"] / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dfc23_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(json.dumps(facts, indent=1, default=str))


if __name__ == "__main__":
    main()
