"""Re-evaluate a finished run's checkpoint on its test sample with the current code.

Use when an evaluation bug is fixed after training (e.g. the U-Net padding fix, 19 Nov).
Writes <run>/test_metrics_<checkpoint>_reeval.json; the original test_metrics.json is kept.

Usage: python scripts/evaluate_run.py RUN_NAME [--checkpoint best|last] [--split-part test|val]
"""

import argparse
import json

import torch

from sat2city.config import load_paths
from sat2city.eval.benchmark import evaluate_model, load_split, make_bench, sample_ids
from sat2city.models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best")
    ap.add_argument("--split-part", choices=["test", "val"], default="test")
    args = ap.parse_args()

    paths = load_paths()
    run_dir = paths["data_root"] / "runs" / args.run
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cache_dir=str(paths["cache"]), pretrained=False)
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.cuda().eval()

    split = load_split(cfg["dataset"], cfg["split"])
    n, seed = (cfg["n_test"], 0) if args.split_part == "test" else (cfg["n_val"], 1)
    ids = sample_ids(split[args.split_part], n, seed=seed)
    res = evaluate_model(model, make_bench(cfg["dataset"], paths["processed"]), ids)
    res |= {"run": args.run, "checkpoint": args.checkpoint, "part": args.split_part,
            "epoch": state.get("epoch") if isinstance(state, dict) and "model" in state else None}
    out = run_dir / f"{args.split_part}_metrics_{args.checkpoint}_reeval.json"
    out.write_text(json.dumps(res, indent=1))
    a, b, p = res["all"], res["building"], res["per_building"]["all"]
    print(f"{args.run} [{args.checkpoint}] {args.split_part}: RMSE {a['rmse']} MAE {a['mae']} | B-RMSE {b['rmse']} | "
          f"bld MAE {p['mae']} floor±1 {p['floor_pm1']} -> {out.name}")


if __name__ == "__main__":
    main()
