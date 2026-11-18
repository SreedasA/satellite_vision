"""Fine-tune DAv2 for height (nDSM) estimation.

Usage:
  python -m sat2city.train --config configs/train/dav2s_base.yaml [--split tile] [--epochs 5] [--run-name NAME]

Every run: logs params/metrics to MLflow (sqlite store under data_root/mlruns), keeps the best
checkpoint by validation building-RMSE, then evaluates it on the deterministic test sample and
writes data_root/runs/<run>/test_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sat2city.config import REPO_ROOT, load_paths
from sat2city.data.datasets import HeightDataset, dataset_items
from sat2city.eval.benchmark import evaluate_model, load_split, make_bench, sample_ids
from sat2city.models.dav2 import DAv2Height
from sat2city.models.losses import height_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train" / "dav2s_base.yaml"))
    ap.add_argument("--dataset")
    ap.add_argument("--split")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="debug: cap steps per epoch")
    ap.add_argument("--run-name")
    return ap.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def flat(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}".replace("+", "plus")  # MLflow metric names reject '+'
        if isinstance(v, dict):
            out |= flat(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for k in ("dataset", "split", "epochs", "batch_size", "n_val", "n_test"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    paths = load_paths()
    run_name = args.run_name or f"dav2s_{cfg['dataset']}_{cfg['split']}_{time.strftime('%m%d-%H%M')}"
    run_dir = paths["data_root"] / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    processed = paths["processed"]
    root = processed / "dfc19" if cfg["dataset"] == "dfc19" else processed / "dfc23" / "track2"
    split = load_split(cfg["dataset"], cfg["split"])
    bench = make_bench(cfg["dataset"], processed)
    val_ids = sample_ids(split["val"], cfg["n_val"], seed=1)
    test_ids = sample_ids(split["test"], cfg["n_test"], seed=0)

    train_ds = HeightDataset(dataset_items(cfg["dataset"], root, split["train"]), train=True, crop=cfg["crop"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=True, drop_last=True, persistent_workers=cfg["num_workers"] > 0)

    model = DAv2Height(cfg["model"], cache_dir=str(paths["cache"])).cuda()
    enc = [p for n, p in model.named_parameters() if n.startswith("net.backbone.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("net.backbone.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr_encoder"]}, {"params": rest, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16

    db = paths["mlruns"] / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    mlflow.set_experiment("sat2city")
    (run_dir / "config.json").write_text(json.dumps(cfg | {"git": git_commit(), "run_name": run_name}, indent=1))

    best, step = float("inf"), 0
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flat(cfg) | {"git": git_commit(), "n_train": len(train_ds), "n_val_eval": len(val_ids)})
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            t0, losses = time.perf_counter(), []
            for i, batch in enumerate(loader):
                if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                    break
                x = batch["image"].cuda(non_blocking=True)
                y = batch["height"].cuda(non_blocking=True)
                v = batch["valid"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=dtype):
                    pred = model(x)
                loss, parts = height_loss(pred.float(), y, v, **cfg["loss"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} step {i}: {parts}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                losses.append(float(loss.detach()))
                step += 1
                if step % 50 == 0:
                    mlflow.log_metrics({f"train.{k}": v for k, v in parts.items()} | {"train.loss": losses[-1]}, step=step)

            val = evaluate_model(model, bench, val_ids)
            score = val["building"]["rmse"]
            dt = time.perf_counter() - t0
            mlflow.log_metrics(flat(val, "val.") | {"epoch_sec": dt, "train.loss_epoch": float(np.mean(losses))}, step=step)
            line = (f"epoch {epoch}/{cfg['epochs']}  loss {np.mean(losses):.3f}  val RMSE {val['all']['rmse']:.3f}  "
                    f"val B-RMSE {score:.3f}  bld MAE {val['per_building']['all']['mae']:.3f}  ({dt:.0f}s)")
            print(line, flush=True)
            with open(run_dir / "log.txt", "a") as f:
                f.write(line + "\n")
            if score < best:
                best = score
                torch.save(model.state_dict(), run_dir / "best.pt")

        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cuda"))
        test = evaluate_model(model, bench, test_ids)
        test |= {"dataset": cfg["dataset"], "split": cfg["split"], "run": run_name, "best_val_building_rmse": best}
        (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=1))
        mlflow.log_metrics(flat(test, "test."))
        print(json.dumps({k: test[k] for k in ("all", "building")} | {"per_building": test["per_building"]["all"]}, indent=1))


if __name__ == "__main__":
    main()
