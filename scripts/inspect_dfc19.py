"""P0.4/P0.5: index DFC19 Track 1, write manifest + splits, report data facts.

Usage: python scripts/inspect_dfc19.py --root S:/sat2city-data/processed/dfc19 [--no-hash]
"""

import argparse
import json
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sat2city.config import REPO_ROOT, load_paths
from sat2city.data.dfc19 import LAS, index_samples, leakage_report, make_splits, write_manifest
from sat2city.data.io import describe, read_height, read_label, read_rgb


def main():
    paths = load_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(paths["processed"] / "dfc19"))
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--n-stats", type=int, default=150, help="images sampled for height statistics")
    args = ap.parse_args()

    from pathlib import Path

    root = Path(args.root)
    samples, unmatched = index_samples(root)
    facts = {"n_samples": len(samples), "n_unmatched_tifs": len(unmatched), "unmatched_examples": unmatched[:5]}
    if not samples:
        print(json.dumps(facts, indent=1))
        raise SystemExit("No RGB/AGL/CLS triplets found - check the naming pattern in dfc19.NAME_RE")

    tiles = Counter(s.tile_id for s in samples)
    facts |= {
        "cities": dict(Counter(s.city for s in samples)),
        "n_tiles": len(tiles),
        "views_per_tile": {"min": min(tiles.values()), "median": int(np.median(list(tiles.values()))),
                           "max": max(tiles.values())},
        "example_names": [s.sample_id for s in samples[:3]],
        "rgb_raster": describe(samples[0].rgb),
        "agl_raster": describe(samples[0].agl),
    }

    rng = random.Random(0)
    pick = rng.sample(samples, min(args.n_stats, len(samples)))
    all_h, bld_h, nan_frac, cls_counts, same_shape = [], [], [], Counter(), 0
    for s in pick:
        agl, cls, rgb = read_height(s.agl), read_label(s.cls), read_rgb(s.rgb)
        same_shape += agl.shape == cls.shape == rgb.shape[:2]
        nan_frac.append(float(np.isnan(agl).mean()))
        cls_counts.update(dict(zip(*np.unique(cls, return_counts=True))))
        v = agl[np.isfinite(agl)]
        all_h.append(rng.choices(v.tolist(), k=min(4000, v.size)) if v.size else [])
        b = agl[(cls == LAS["building"]) & np.isfinite(agl)]
        if b.size:
            bld_h.append(rng.choices(b.tolist(), k=min(4000, b.size)))
    all_h = np.concatenate([np.asarray(a) for a in all_h if len(a)])
    bld_h = np.concatenate([np.asarray(a) for a in bld_h]) if bld_h else np.array([])
    total = sum(cls_counts.values())
    facts |= {
        "rgb_agl_cls_same_shape": f"{same_shape}/{len(pick)}",
        "agl_nan_fraction_mean": round(float(np.mean(nan_frac)), 4),
        "agl_percentiles_all": {p: round(float(np.percentile(all_h, p)), 2) for p in (1, 50, 90, 99, 99.9)},
        "agl_percentiles_building": {p: round(float(np.percentile(bld_h, p)), 2) for p in (5, 50, 90, 99)} if bld_h.size else {},
        "agl_min_max": [round(float(all_h.min()), 2), round(float(all_h.max()), 2)],
        "cls_pixel_share": {int(k): round(v / total, 4) for k, v in sorted(cls_counts.items())},
    }

    out = paths["processed"] / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(all_h, bins=100, log=True, color="#4a7")
    ax[0].set(title="AGL, all valid pixels (sample)", xlabel="m")
    if bld_h.size:
        ax[1].hist(bld_h, bins=100, log=True, color="#a64")
    ax[1].set(title="AGL, building pixels (sample)", xlabel="m")
    fig.tight_layout()
    fig.savefig(out / "dfc19_agl_hist.png", dpi=120)

    write_manifest(samples, paths["processed"] / "manifests" / "dfc19_manifest.csv", with_hash=not args.no_hash)
    splits = make_splits(samples, seed=0)
    split_dir = REPO_ROOT / "configs" / "splits"
    for name, split in splits.items():
        (split_dir / f"dfc19_{name}.json").write_text(json.dumps(split, indent=0))
    facts["splits"] = {name: {k: len(v) for k, v in sp.items()} | {"tile_leakage": leakage_report(samples, sp)}
                       for name, sp in splits.items()}

    (out / "dfc19_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(json.dumps(facts, indent=1, default=str))


if __name__ == "__main__":
    main()
