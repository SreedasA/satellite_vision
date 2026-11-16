"""P0.7: trivial and zero-shot baselines on a dataset split's test partition.

Baselines:
  zero            predict 0 m everywhere
  train_mean      predict the mean height of the training sample
  dav2_global     zero-shot DAv2-S, one affine map (a*d+b) fitted on training images -> deployable
  dav2_oracle     zero-shot DAv2-S, affine fitted per test image with its ground truth -> upper bound

Per-building metrics use building instances: DFC19 = connected components of CLS==6,
DFC23 = annotated footprints whose reference median height is >= 2 m (see docs/DATA.md).

Usage: python scripts/baselines.py --dataset dfc23 --split block [--n-test 150] [--n-fit 60] [--size 518]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sat2city.config import REPO_ROOT, load_paths
from sat2city.data.io import read_height, read_label, read_rgb
from sat2city.eval.metrics import PixelAccumulator, building_metrics, label_buildings, per_building_heights
from sat2city.models.dav2 import DAv2Height, normalize


class DFC19:
    def __init__(self, root: Path):
        from sat2city.data.dfc19 import LAS, index_samples

        self.las_building = LAS["building"]
        self.samples = {s.sample_id: s for s in index_samples(root)[0]}

    def load(self, sid):
        s = self.samples[sid]
        g, cls = read_height(s.agl), read_label(s.cls)
        bmask = cls == self.las_building
        return read_rgb(s.rgb), g, bmask, label_buildings(bmask, min_pixels=20)


class DFC23:
    def __init__(self, root: Path):
        from sat2city.data.dfc23 import load_index

        self.root = root
        tiles, self.shapes = load_index(root)
        self.ids = {t.name: t.image_id for t in tiles}

    def load(self, name):
        from sat2city.data.dfc23 import instance_labels

        g = read_height(self.root / "dsm" / name)
        lab = instance_labels(self.shapes.get(self.ids[name], []), g.shape)
        ref = per_building_heights(g, lab)
        unreliable = [i for i, h in ref.items() if h < 2.0]
        lab[np.isin(lab, unreliable)] = 0
        return read_rgb(self.root / "rgb" / name), g, lab > 0, lab


@torch.no_grad()
def predict_relative(model, rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].cuda()
    x = F.interpolate(normalize(x), size=(size, size), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model(x)
    return F.interpolate(d[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def fit_affine(d: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    sel = np.isfinite(g)
    A = np.stack([d[sel], np.ones(sel.sum())], 1)
    (a, b), *_ = np.linalg.lstsq(A, g[sel], rcond=None)
    return float(a), float(b)


def pool_buildings(rows):
    rows = [r for r in rows if r.get("n_buildings")]
    n = sum(r["n_buildings"] for r in rows)
    if not n:
        return {"n_buildings": 0}
    w = lambda k: sum(r[k] * r["n_buildings"] for r in rows) / n  # noqa: E731
    rmse = np.sqrt(sum(r["bld_rmse"] ** 2 * r["n_buildings"] for r in rows) / n)
    return {"n_buildings": n, "bld_mae": round(w("bld_mae"), 3), "bld_rmse": round(float(rmse), 3),
            "floor_exact": round(w("floor_exact"), 3), "floor_pm1": round(w("floor_pm1"), 3)}


def main():
    paths = load_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dfc19", "dfc23"], required=True)
    ap.add_argument("--split", default="block")
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--size", type=int, default=518)
    args = ap.parse_args()

    root = paths["processed"] / ("dfc19" if args.dataset == "dfc19" else "dfc23/track2")
    ds = DFC19(root) if args.dataset == "dfc19" else DFC23(root)
    split = json.loads((REPO_ROOT / "configs" / "splits" / f"{args.dataset}_{args.split}.json").read_text())
    rng = random.Random(0)
    test_ids = rng.sample(split["test"], min(args.n_test, len(split["test"])))
    fit_ids = rng.sample(split["train"], min(args.n_fit, len(split["train"])))

    model = DAv2Height(cache_dir=str(paths["cache"])).cuda().eval()

    ds_, gs_ = [], []
    nrng = np.random.default_rng(0)
    for sid in fit_ids:
        rgb, g, _, _ = ds.load(sid)
        d = predict_relative(model, rgb, args.size)
        sel = np.flatnonzero(np.isfinite(g))
        idx = nrng.choice(sel, size=min(20000, sel.size), replace=False)
        ds_.append(d.ravel()[idx])
        gs_.append(g.ravel()[idx])
    ds_, gs_ = np.concatenate(ds_), np.concatenate(gs_)
    a_glob, b_glob = fit_affine(ds_, gs_)
    train_mean = float(gs_.mean())

    names = ["zero", "train_mean", "dav2_global", "dav2_oracle"]
    acc_all = {n: PixelAccumulator() for n in names}
    acc_bld = {n: PixelAccumulator() for n in names}
    per_bld = {n: [] for n in names}
    for sid in test_ids:
        rgb, g, bmask, labels = ds.load(sid)
        d = predict_relative(model, rgb, args.size)
        a, b = fit_affine(d, g)
        preds = {"zero": np.zeros_like(g), "train_mean": np.full_like(g, train_mean),
                 "dav2_global": a_glob * d + b_glob, "dav2_oracle": a * d + b}
        for n, p in preds.items():
            acc_all[n].update(p, g)
            acc_bld[n].update(p, g, bmask)
            per_bld[n].append(building_metrics(p, g, labels))

    report = {
        "dataset": args.dataset, "split": args.split, "n_test_images": len(test_ids), "n_fit_images": len(fit_ids),
        "size": args.size, "global_affine": [round(a_glob, 4), round(b_glob, 4)], "train_mean": round(train_mean, 3),
        "results": {n: {"all": acc_all[n].compute(), "building": acc_bld[n].compute(),
                        "per_building": pool_buildings(per_bld[n])} for n in names},
    }
    out = paths["processed"] / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dataset}_{args.split}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
