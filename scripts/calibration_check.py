"""Phase 3: is a run's per-building height error a single scale factor that a few labelled tiles can fix?

For each test tile: predict, take per-building medians (same instances as benchmark.py).
Then:
  1. ratio ref/pred per height bin (constant ratio = pure scale; rising ratio = compression of tall buildings);
  2. spread of the best scale between tiles (a city-level factor only works if tiles agree);
  3. few-shot calibration: fit on K random tiles (scale through the origin, L1; or affine, least squares),
     score the remaining tiles, repeat R times; plus an oracle fit on all tiles (upper bound).

Run it on the New Delhi held-out run and on the block run (seen cities, control).
Usage: python scripts/calibration_check.py RUN_NAME [--checkpoint best] [--repeats 50]
Writes <run>/calibration_<checkpoint>.json and <run>/buildings_<checkpoint>.npz.
"""

import argparse
import json
import random

import numpy as np
import torch

from sat2city.config import load_paths
from sat2city.eval.benchmark import BuildingPool, load_split, make_bench, predict_tile, sample_ids
from sat2city.eval.metrics import per_building_heights
from sat2city.models import build_model

SCALES = np.arange(0.5, 3.0001, 0.0025)


def collect(model, bench, ids):
    rows = []  # tile index, pred median, ref median, area px
    for k, sid in enumerate(ids):
        rgb, g, _, labels = bench.load(sid)
        p = predict_tile(model, rgb)
        valid = np.isfinite(g) & np.isfinite(p)
        hp, hg = per_building_heights(p, labels, valid), per_building_heights(g, labels, valid)
        area = np.bincount(labels.ravel())
        for i in sorted(set(hp) & set(hg)):
            rows.append((k, hp[i], hg[i], area[i]))
    a = np.array(rows, dtype=float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3]


def fit_scale(p, g):
    """Scale through the origin minimising per-building L1 error (grid search; robust to outliers)."""
    err = np.abs(SCALES[:, None] * p[None] - g[None]).sum(1)
    return float(SCALES[err.argmin()])


def fit_affine(p, g):
    a, b = np.linalg.lstsq(np.stack([p, np.ones_like(p)], 1), g, rcond=None)[0]
    return float(a), float(b)


def stats(p, g):
    pool = BuildingPool()
    pool.p, pool.g = list(p), list(g)
    return pool.compute()


def few_shot(tile, p, g, n_tiles, k, repeats, rng):
    res = {"scale": [], "affine": [], "s": []}
    for _ in range(repeats):
        fit = set(rng.sample(range(n_tiles), k))
        m = np.isin(tile, list(fit))
        if m.sum() < 3:
            continue
        s = fit_scale(p[m], g[m])
        a, b = fit_affine(p[m], g[m])
        res["s"].append(s)
        res["scale"].append(stats(s * p[~m], g[~m]))
        res["affine"].append(stats(np.maximum(a * p[~m] + b, 0), g[~m]))
        res.setdefault("none", []).append(stats(p[~m], g[~m]))

    def mean(lst, key):
        return {b: {f: round(float(np.mean([r[b][f] for r in lst if r[b]["n"]])), 3) for f in ("mae", "bias", "floor_pm1")}
                for b in ("all", "0-10", "10-30", "30+") if key in res}

    return {"k_tiles": k, "repeats": len(res["s"]), "scale_mean": round(float(np.mean(res["s"])), 3),
            "scale_sd": round(float(np.std(res["s"])), 3),
            **{name: mean(res[name], name) for name in ("none", "scale", "affine")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    paths = load_paths()
    run_dir = paths["data_root"] / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cache_dir=str(paths["cache"]), pretrained=False)
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.cuda().eval()

    ids = sample_ids(load_split(cfg["dataset"], cfg["split"])["test"], cfg["n_test"], seed=0)
    tile, p, g, area = collect(model, make_bench(cfg["dataset"], paths["processed"]), ids)
    np.savez(run_dir / f"buildings_{args.checkpoint}.npz", tile=tile, pred=p, ref=g, area=area, ids=np.array(ids))

    out = {"run": args.run, "checkpoint": args.checkpoint, "n_tiles": len(ids), "n_buildings": int(len(p))}
    # 1. ratio by reference-height bin and by predicted-height bin (only the latter is usable at inference)
    for key, h in (("ratio_by_ref_bin", g), ("ratio_by_pred_bin", p)):
        out[key] = {f"{lo}-{hi}": {"n": int(m.sum()), "median_ref_over_pred": round(float(np.median(g[m] / np.maximum(p[m], 0.5))), 3)}
                    for lo, hi in ((2, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 999))
                    if (m := (h >= lo) & (h < hi)).sum() >= 5}
    # 2. per-tile best scale (tiles with >= 5 buildings of reference >= 5 m)
    per_tile = []
    for k in range(len(ids)):
        m = (tile == k) & (g >= 5)
        if m.sum() >= 5:
            per_tile.append(fit_scale(p[m], g[m]))
    q = np.percentile(per_tile, [10, 25, 50, 75, 90])
    out["per_tile_scale"] = {"n_tiles": len(per_tile), "p10_p25_p50_p75_p90": [round(float(x), 3) for x in q]}
    # sensors (DFC23 names start with SV_ / GF2_)
    sensors = np.array([ids[k].split("_")[0] for k in tile])
    out["by_sensor"] = {s: {"n": int((sensors == s).sum()), "scale": fit_scale(p[sensors == s], g[sensors == s]),
                            **stats(p[sensors == s], g[sensors == s])["all"]} for s in np.unique(sensors)}
    # 3. oracle and few-shot calibration
    s_all = fit_scale(p, g)
    a_all, b_all = fit_affine(p, g)
    out["oracle"] = {"scale": s_all, "affine": [round(a_all, 3), round(b_all, 3)], "none": stats(p, g),
                     "scaled": stats(s_all * p, g), "affine_fit": stats(np.maximum(a_all * p + b_all, 0), g)}
    rng = random.Random(0)
    out["few_shot"] = [few_shot(tile, p, g, len(ids), k, args.repeats, rng) for k in (1, 3, 5, 10)]

    (run_dir / f"calibration_{args.checkpoint}.json").write_text(json.dumps(out, indent=1))
    o = out["oracle"]
    print(f"{args.run}: {len(p)} buildings on {len(ids)} tiles | oracle scale {s_all:.3f}: "
          f"bld MAE {o['none']['all']['mae']} -> {o['scaled']['all']['mae']} (affine {o['affine_fit']['all']['mae']})")
    print("ratio by ref bin:", {k: v["median_ref_over_pred"] for k, v in out["ratio_by_ref_bin"].items()})
    print("ratio by pred bin:", {k: v["median_ref_over_pred"] for k, v in out["ratio_by_pred_bin"].items()})
    print("per-tile scale p10..p90:", out["per_tile_scale"]["p10_p25_p50_p75_p90"])
    print("by sensor:", {s: (v["n"], v["scale"], v["mae"], v["bias"]) for s, v in out["by_sensor"].items()})
    for f in out["few_shot"]:
        print(f"K={f['k_tiles']:>2}: s={f['scale_mean']}±{f['scale_sd']}  none {f['none']['all']['mae']}  "
              f"scale {f['scale']['all']['mae']} (10-30 {f['scale']['10-30']['mae']}, 30+ {f['scale']['30+']['mae']}, "
              f"floor±1 {f['scale']['all']['floor_pm1']})  affine {f['affine']['all']['mae']}")


if __name__ == "__main__":
    main()
