"""Phase 3 failure gallery: where and how a run's height predictions fail.

Numbers (pooled over the run's test sample):
  * pixel error by surface class (DFC19 CLS: ground, trees, building, water, bridge; DFC23: building / other)
  * building edges (3 px rim) vs interiors
  * per-building error by height and footprint size; the largest tall-building misses
  * DFC23: by sensor (SV = SuperView-1, GF2 = Gaofen-2)
Pictures (RGB | reference | prediction | error) for the worst tiles of each failure type.
DFC23 imagery licence is unverified (docs/DATA.md), so pictures go to S:/sat2city-data/gallery, never the repo.

Usage: python scripts/failure_gallery.py RUN_NAME [--checkpoint best] [--rows 6]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from sat2city.config import load_paths
from sat2city.data.io import read_label
from sat2city.eval.benchmark import load_split, make_bench, predict_tile, sample_ids
from sat2city.eval.metrics import per_building_heights
from sat2city.models import build_model

DFC19_CLASSES = {2: "ground", 5: "trees", 6: "building", 9: "water", 17: "bridge"}


class Acc:
    def __init__(self):
        self.n = self.s = self.a = self.q = 0.0

    def add(self, e):
        self.n += e.size; self.s += e.sum(); self.a += np.abs(e).sum(); self.q += (e**2).sum()

    def out(self):
        if not self.n:
            return {"n": 0}
        return {"n": int(self.n), "mae": round(float(self.a / self.n), 3), "rmse": round(float(np.sqrt(self.q / self.n)), 3),
                "bias": round(float(self.s / self.n), 3)}


def per_building_table(b, px_m2):
    """Footprint size in m² (pixel area x GSD²: DFC19 ~1.3 m per docs/DATA.md, DFC23 0.5 m)."""
    out = {}
    for lo, hi in ((2, 10), (10, 30), (30, 999)):
        for alo, ahi, aname in ((0, 100, "small <100 m²"), (100, 500, "mid 100-500 m²"), (500, 1e12, "large >500 m²")):
            m = (b["ref"] >= lo) & (b["ref"] < hi) & (b["area"] * px_m2 >= alo) & (b["area"] * px_m2 < ahi)
            if m.sum() >= 5:
                e = b["pred"][m] - b["ref"][m]
                out[f"{lo}-{hi} m, {aname}"] = {"n": int(m.sum()), "mae": round(float(np.abs(e).mean()), 2),
                                                 "bias": round(float(e.mean()), 2)}
    return out


def panel(rows, title, path, vmax_err=15):
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.2 * len(rows)), squeeze=False)
    for r, (sid, rgb, g, p, note) in enumerate(rows):
        hmax = max(float(np.nanpercentile(g, 99.5)), 10)
        ims = [(rgb, None, None, sid), (g, "magma", (0, hmax), "reference (m)"), (p, "magma", (0, hmax), "prediction (m)"),
               (p - g, "RdBu_r", (-vmax_err, vmax_err), f"error (m): {note}")]
        for c, (im, cmap, lim, lab) in enumerate(ims):
            ax = axes[r, c]
            h = ax.imshow(im, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            ax.set_title(lab, fontsize=9)
            ax.axis("off")
            if cmap:
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--rows", type=int, default=6)
    args = ap.parse_args()

    paths = load_paths()
    run_dir = paths["data_root"] / "runs" / args.run
    out_dir = paths["data_root"] / "gallery" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cache_dir=str(paths["cache"]), pretrained=False)
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.cuda().eval()

    ds = cfg["dataset"]
    bench = make_bench(ds, paths["processed"])
    ids = sample_ids(load_split(ds, cfg["split"])["test"], cfg["n_test"], seed=0)

    cls_acc, edge_acc, inner_acc, sensor_acc = {}, Acc(), Acc(), {}
    tiles, b = [], {"tile": [], "pred": [], "ref": [], "area": []}
    for k, sid in enumerate(ids):
        rgb, g, bmask, labels = bench.load(sid)
        p = predict_tile(model, rgb)
        valid = np.isfinite(g)
        e = np.where(valid, p - g, 0)
        if ds == "dfc19":
            cls = read_label(bench.root / f"{sid}_CLS.tif")
            for code, name in DFC19_CLASSES.items():
                cls_acc.setdefault(name, Acc()).add(e[valid & (cls == code)])
            tree_bias = float(e[valid & (cls == 5)].mean()) if (valid & (cls == 5)).sum() > 500 else 0.0
        else:
            cls_acc.setdefault("building", Acc()).add(e[valid & bmask])
            cls_acc.setdefault("other", Acc()).add(e[valid & ~bmask])
            sensor_acc.setdefault(sid.split("_")[0], Acc()).add(e[valid & bmask])
            tree_bias = 0.0
        inner = ndimage.binary_erosion(bmask, iterations=3)
        edge_acc.add(e[valid & bmask & ~inner])
        inner_acc.add(e[valid & inner])
        hp, hg = per_building_heights(p, labels, valid), per_building_heights(g, labels, valid)
        area = np.bincount(labels.ravel())
        common = sorted(set(hp) & set(hg))
        for i in common:
            b["tile"].append(k); b["pred"].append(hp[i]); b["ref"].append(hg[i]); b["area"].append(area[i])
        be = np.array([hp[i] - hg[i] for i in common]) if common else np.zeros(0)
        tall = [hp[i] - hg[i] for i in common if hg[i] >= 30]
        tiles.append({"id": sid, "b_rmse": float(np.sqrt((e[valid & bmask] ** 2).mean())) if (valid & bmask).any() else 0.0,
                      "bld_bias": float(be.mean()) if be.size else 0.0, "worst_tall": float(min(tall)) if tall else 0.0,
                      "tree_bias": tree_bias, "max_over": float(np.percentile(e[valid], 99.9)),
                      "ref_max": float(np.nanmax(g)), "pred_max": float(p.max())})
    b = {k: np.array(v, dtype=float) for k, v in b.items()}

    res = {"run": args.run, "dataset": ds, "n_tiles": len(ids), "by_class": {k: v.out() for k, v in cls_acc.items()},
           "building_edge": edge_acc.out(), "building_interior": inner_acc.out(),
           "by_sensor_building_pixels": {k: v.out() for k, v in sensor_acc.items()},
           "per_building_by_height_and_size": per_building_table(b, 1.69 if ds == "dfc19" else 0.25)}
    tall = np.where(b["ref"] >= 30)[0]
    worst = tall[np.argsort(b["pred"][tall] - b["ref"][tall])][:10]
    res["largest_tall_misses"] = [{"tile": ids[int(b["tile"][i])], "ref": round(float(b["ref"][i]), 1),
                                   "pred": round(float(b["pred"][i]), 1), "area_px": int(b["area"][i])} for i in worst]
    # Ratio pred/ref for tall buildings: compression (constant fraction) or a ceiling (pred saturates)?
    if tall.size:
        res["tall_pred_over_ref"] = {f"{lo}-{hi}": round(float(np.median(b["pred"][m] / b["ref"][m])), 3)
                                     for lo, hi in ((30, 40), (40, 60), (60, 999))
                                     if (m := (b["ref"] >= lo) & (b["ref"] < hi)).sum() >= 3}
    (out_dir / "failures.json").write_text(json.dumps(res, indent=1))
    (out_dir / "tiles.json").write_text(json.dumps(tiles, indent=1))

    cats = {"worst_building_rmse": sorted(tiles, key=lambda t: -t["b_rmse"]),
            "tall_building_misses": sorted([t for t in tiles if t["worst_tall"] < 0], key=lambda t: t["worst_tall"]),
            "hallucinated_heights": sorted(tiles, key=lambda t: -t["max_over"])}
    if ds == "dfc19":
        cats["tree_overprediction"] = sorted([t for t in tiles if t["tree_bias"]], key=lambda t: -t["tree_bias"])
    else:
        cats["gaofen2_tiles"] = sorted([t for t in tiles if t["id"].startswith("GF2")], key=lambda t: -t["b_rmse"])
    for name, lst in cats.items():
        rows = []
        for t in lst[: args.rows]:
            rgb, g, _, _ = bench.load(t["id"])
            p = predict_tile(model, rgb)
            note = (f"B-RMSE {t['b_rmse']:.1f}, worst tall {t['worst_tall']:.0f}, trees {t['tree_bias']:+.1f}, "
                    f"max {t['pred_max']:.0f}/{t['ref_max']:.0f}")
            rows.append((t["id"], rgb, g, p, note))
        if rows:
            panel(rows, f"{args.run}: {name.replace('_', ' ')}", out_dir / f"{name}.png")
    print(json.dumps({k: res[k] for k in ("by_class", "building_edge", "building_interior", "by_sensor_building_pixels",
                                           "tall_pred_over_ref") if k in res}, indent=0))
    print(json.dumps(res["per_building_by_height_and_size"]))
    print(json.dumps(res["largest_tall_misses"][:5]))
    print("pictures:", out_dir)


if __name__ == "__main__":
    main()
