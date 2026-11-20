"""Phase 3: how much of the building error is misregistration rather than height?

Off-nadir images show tall buildings displaced (leaning) relative to the nadir reference nDSM, so a
correct height in the wrong place counts twice (a red/blue pair in the failure gallery). For each test
tile we search the integer shift of the prediction (on a 4x downsampled grid, up to +-12 cells =
+-48 px) that minimises building-pixel RMSE and report how much error the best shift removes.
A large best shift with a big RMSE drop = geometry, not height estimation.

Usage: python scripts/shift_check.py RUN_NAME [--checkpoint best]   (writes <run>/shift_check_<ckpt>.json)
"""

import argparse
import json

import numpy as np
import torch

from sat2city.config import load_paths
from sat2city.eval.benchmark import load_split, make_bench, predict_tile, sample_ids
from sat2city.models import build_model

F, R = 4, 12  # downsample factor, max shift in downsampled cells
GSD = {"dfc19": 0.3, "dfc23": 0.5}  # m per pixel (DFC19 measured, see sat2city.data.dfc19.GSD)


def pool(a):
    h, w = (a.shape[0] // F) * F, (a.shape[1] // F) * F
    return a[:h, :w].reshape(h // F, F, w // F, F).mean((1, 3))


def best_shift(p, g, m):
    best = (np.sqrt(((p - g)[m] ** 2).mean()), 0, 0)
    pp = np.pad(p, R, mode="edge")
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            q = pp[R + dy:R + dy + p.shape[0], R + dx:R + dx + p.shape[1]]
            r = np.sqrt(((q - g)[m] ** 2).mean())
            if r < best[0]:
                best = (r, dy, dx)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = ap.parse_args()
    paths = load_paths()
    run_dir = paths["data_root"] / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cache_dir=str(paths["cache"]), pretrained=False)
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.cuda().eval()
    bench = make_bench(cfg["dataset"], paths["processed"])
    ids = sample_ids(load_split(cfg["dataset"], cfg["split"])["test"], cfg["n_test"], seed=0)

    rows = []
    for sid in ids:
        rgb, g, bmask, _ = bench.load(sid)
        p = predict_tile(model, rgb)
        g = np.where(np.isfinite(g), g, 0)
        pg, gg, mg = pool(p), pool(g), pool(bmask.astype(float)) > 0.5
        if mg.sum() < 20:
            continue
        r0 = float(np.sqrt(((pg - gg)[mg] ** 2).mean()))
        r1, dy, dx = best_shift(pg, gg, mg)
        rows.append({"id": sid, "rmse": round(r0, 2), "rmse_shifted": round(float(r1), 2),
                     "shift_m": round(float(np.hypot(dy, dx) * F * GSD[cfg["dataset"]]), 1), "n": int(mg.sum())})

    n = np.array([r["n"] for r in rows], float)
    r0 = np.sqrt((np.array([r["rmse"] for r in rows]) ** 2 * n).sum() / n.sum())
    r1 = np.sqrt((np.array([r["rmse_shifted"] for r in rows]) ** 2 * n).sum() / n.sum())
    sh = np.array([r["shift_m"] for r in rows])
    big = sh >= 3
    out = {"run": args.run, "n_tiles": len(rows), "building_rmse_coarse": round(float(r0), 3),
           "building_rmse_best_shift": round(float(r1), 3), "share_removed": round(float(1 - r1**2 / r0**2), 3),
           "tiles_shift_ge_3m": int(big.sum()), "median_shift_m": float(np.median(sh)),
           "rmse_on_shifted_tiles": [round(float(np.sqrt((np.array([r['rmse'] for r in rows])[big] ** 2 * n[big]).sum() / max(n[big].sum(), 1))), 2),
                                     round(float(np.sqrt((np.array([r['rmse_shifted'] for r in rows])[big] ** 2 * n[big]).sum() / max(n[big].sum(), 1))), 2)],
           "worst": sorted(rows, key=lambda r: r["rmse_shifted"] - r["rmse"])[:10]}
    (run_dir / f"shift_check_{args.checkpoint}.json").write_text(json.dumps(out | {"tiles": rows}, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "worst"}))
    print([(w["id"], w["rmse"], w["rmse_shifted"], w["shift_m"]) for w in out["worst"][:6]])


if __name__ == "__main__":
    main()
