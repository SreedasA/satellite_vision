"""Fine-tune a height (nDSM) model: DAv2 (default) or the U-Net baseline.

Usage:
  python -m sat2city.train --config configs/train/dav2s_base.yaml [--split tile] [--epochs 5] [--run-name NAME]
  python -m sat2city.train --run-name NAME --resume [--epochs 40]    # continue an interrupted run

Every run logs to MLflow (sqlite under data_root/mlruns) and keeps, in data_root/runs/<run>/:
  best.pt          weights with the best validation score (`select:` rule in the config)
  last.pt          full training state after the latest epoch (model, optimizer, scheduler, RNG,
                   MLflow run id) - what --resume continues from
  test_metrics.json, log.txt, progress.json, config.json
An interrupted run loses at most the epoch in progress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from sat2city.config import REPO_ROOT, load_paths, load_train_config
from sat2city.data.datasets import HeightDataset, dataset_items
from sat2city.eval.benchmark import evaluate_model, load_split, make_bench, sample_ids
from sat2city.models import build_model
from sat2city.models.losses import height_loss
from sat2city.progress import ProgressWriter

DEFAULTS = {"schedule": "constant", "warmup_epochs": 0.0, "min_lr_ratio": 0.05, "select": "building_rmse",
            "select_tall_weight": 1.0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "train" / "dav2s_base.yaml"))
    ap.add_argument("--dataset")
    ap.add_argument("--split")
    ap.add_argument("--epochs", type=int, help="total epochs (with --resume: raise it to extend a finished run)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--seed", type=int, help="random seed for data order and augmentation (val/test samples stay fixed)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="debug: cap steps per epoch")
    ap.add_argument("--run-name")
    ap.add_argument("--resume", action="store_true", help="continue <run-name> from its last.pt")
    return ap.parse_args(argv)


def keep_awake() -> None:
    """Stop Windows from sleeping while this process runs (no admin needed; released on exit).
    Does not override closing the laptop lid if Windows is set to sleep on lid close."""
    if os.name == "nt":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


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


# ---------------------------------------------------------------- helpers (unit-tested)

def lr_lambda(cfg: dict, steps_per_epoch: int):
    """Multiplier on the base LRs per optimizer step: optional linear warm-up, then constant or cosine
    decay to `min_lr_ratio`."""
    total = max(cfg["epochs"] * steps_per_epoch, 1)
    warm = int(cfg["warmup_epochs"] * steps_per_epoch)

    def fn(step: int) -> float:
        if warm and step < warm:
            return (step + 1) / warm
        if cfg["schedule"] == "cosine":
            t = min((step - warm) / max(total - warm, 1), 1.0)
            return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * t))
        return 1.0

    return fn


def selection_score(val: dict, cfg: dict) -> float:
    """Lower is better.
    building_rmse: val building-pixel RMSE (Phase 1 rule).
    tall_aware:    building RMSE + w * per-building MAE of buildings >= 10 m (pooled over the 10-30
                   and 30+ bins), so a checkpoint can't win by only fitting the many low houses."""
    score = val["building"]["rmse"]
    if cfg["select"] == "tall_aware":
        pb = val["per_building"]
        bins = [pb[k] for k in ("10-30", "30+") if pb.get(k, {}).get("n")]
        n = sum(b["n"] for b in bins)
        if n:
            score += cfg["select_tall_weight"] * sum(b["mae"] * b["n"] for b in bins) / n
    return float(score)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, **state) -> None:
    """Atomic save: an interruption during the write never leaves a corrupt last.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------- main

def resolve_config(args) -> tuple[dict, str, Path, dict | None]:
    paths = load_paths()
    if args.resume:
        if not args.run_name:
            raise SystemExit("--resume needs --run-name of the run to continue")
        run_dir = paths["data_root"] / "runs" / args.run_name
        if not (run_dir / "last.pt").exists():
            raise SystemExit(f"nothing to resume: {run_dir / 'last.pt'} not found")
        cfg = json.loads((run_dir / "config.json").read_text())  # same recipe as the original run
        if args.epochs:
            cfg["epochs"] = args.epochs
        return cfg, args.run_name, run_dir, load_checkpoint(run_dir / "last.pt")

    cfg = DEFAULTS | load_train_config(Path(args.config))
    for k in ("dataset", "split", "epochs", "batch_size", "n_val", "n_test", "seed"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    tag = "unet" if str(cfg["model"]).startswith("unet") else "dav2s"
    run_name = args.run_name or f"{tag}_{cfg['dataset']}_{cfg['split']}_{time.strftime('%m%d-%H%M')}"
    run_dir = paths["data_root"] / "runs" / run_name
    if (run_dir / "last.pt").exists():
        raise SystemExit(f"run '{run_name}' already exists - use --resume to continue it or pick a new --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Stored so a resumed run keeps the same steps/epoch (the LR schedule depends on it).
    cfg |= {"git": git_commit(), "run_name": run_name, "max_steps_per_epoch": args.max_steps_per_epoch}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    return cfg, run_name, run_dir, None


def main(argv=None):
    args = parse_args(argv)
    keep_awake()
    cfg, run_name, run_dir, ckpt = resolve_config(args)
    cfg = DEFAULTS | cfg
    if args.max_steps_per_epoch is None:
        args.max_steps_per_epoch = cfg.get("max_steps_per_epoch")  # resumed runs keep the original cap
    paths = load_paths()

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
    steps_per_epoch = min(len(loader), args.max_steps_per_epoch or len(loader))

    model = build_model(cfg["model"], cache_dir=str(paths["cache"])).cuda()
    prefix = getattr(model, "encoder_prefix", "")
    enc = [p for n, p in model.named_parameters() if prefix and n.startswith(prefix)]
    rest = [p for n, p in model.named_parameters() if not (prefix and n.startswith(prefix))]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr_encoder"]}, {"params": rest, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps_per_epoch))

    start_epoch, step, best, mlflow_id, history = 1, 0, float("inf"), None, []
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])  # schedule already uses the (possibly extended) epoch count
        set_rng_state(ckpt["rng"])
        start_epoch, step, best = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"]
        mlflow_id, history = ckpt.get("mlflow_run_id"), ckpt.get("history", [])
        print(f"Resuming {run_name} from epoch {start_epoch}/{cfg['epochs']} (best so far {best:.3f})", flush=True)
        with open(run_dir / "log.txt", "a") as f:
            f.write(f"--- resumed at epoch {start_epoch} ({time.strftime('%Y-%m-%d %H:%M')}) ---\n")

    db = paths["mlruns"] / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    mlflow.set_experiment("sat2city")

    progress = ProgressWriter(run_dir, run_name, cfg["epochs"], steps_per_epoch)
    progress.resume_from(start_epoch - 1, step, best, history)
    state = {"cfg": cfg, "args": args, "run_dir": run_dir, "run_name": run_name, "model": model, "opt": opt,
             "sched": sched, "loader": loader, "bench": bench, "val_ids": val_ids, "test_ids": test_ids,
             "n_train": len(train_ds), "progress": progress, "start_epoch": start_epoch, "step": step,
             "best": best, "mlflow_id": mlflow_id, "history": history}
    try:
        train_loop(state)
    except BaseException as e:
        progress.failed(f"{type(e).__name__}: {e}")
        raise


def train_loop(s: dict):
    cfg, args, run_dir, model, opt, sched, progress = (s[k] for k in ("cfg", "args", "run_dir", "model", "opt",
                                                                     "sched", "progress"))
    dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    best, step, history = s["best"], s["step"], s["history"]

    with mlflow.start_run(run_id=s["mlflow_id"], run_name=None if s["mlflow_id"] else s["run_name"]) as run:
        if not s["mlflow_id"]:
            mlflow.log_params(flat(cfg) | {"n_train": s["n_train"], "n_val_eval": len(s["val_ids"])})
        for epoch in range(s["start_epoch"], cfg["epochs"] + 1):
            model.train()
            progress.epoch_start(epoch)
            t0, losses = time.perf_counter(), []
            for i, batch in enumerate(s["loader"]):
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
                sched.step()
                losses.append(float(loss.detach()))
                step += 1
                progress.step(i + 1, losses[-1])
                if step % 50 == 0:
                    mlflow.log_metrics({f"train.{k}": v for k, v in parts.items()}
                                       | {"train.loss": losses[-1], "train.lr_head": sched.get_last_lr()[-1]}, step=step)

            progress.phase("evaluating")
            t_eval = time.perf_counter()
            val = evaluate_model(model, s["bench"], s["val_ids"])
            eval_sec = time.perf_counter() - t_eval
            score = selection_score(val, cfg)
            dt = time.perf_counter() - t0
            mlflow.log_metrics(flat(val, "val.") | {"val.select_score": score, "epoch_sec": dt,
                                                   "train.loss_epoch": float(np.mean(losses))}, step=step)
            line = (f"epoch {epoch}/{cfg['epochs']}  loss {np.mean(losses):.3f}  val RMSE {val['all']['rmse']:.3f}  "
                    f"val B-RMSE {val['building']['rmse']:.3f}  bld MAE {val['per_building']['all']['mae']:.3f}  "
                    f"select {score:.3f}  ({dt:.0f}s)")
            print(line, flush=True)
            with open(run_dir / "log.txt", "a") as f:
                f.write(line + "\n")
            history.append(line)
            if score < best:
                best = score
                torch.save(model.state_dict(), run_dir / "best.pt")
            save_checkpoint(run_dir / "last.pt", model=model.state_dict(), optimizer=opt.state_dict(),
                            scheduler=sched.state_dict(), rng=rng_state(), epoch=epoch, step=step, best=best,
                            mlflow_run_id=run.info.run_id, history=history[-200:])
            progress.epoch_end(line, eval_sec, best)

        progress.phase("testing")
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cuda"))
        test = evaluate_model(model, s["bench"], s["test_ids"])
        test |= {"dataset": cfg["dataset"], "split": cfg["split"], "run": s["run_name"], "best_val_score": best,
                 "select": cfg["select"]}
        (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=1))
        mlflow.log_metrics(flat(test, "test."))
        progress.done({"rmse": test["all"]["rmse"], "building_rmse": test["building"]["rmse"],
                       "bld_mae": test["per_building"]["all"]["mae"]})
        print(json.dumps({k: test[k] for k in ("all", "building")} | {"per_building": test["per_building"]["all"]}, indent=1))


if __name__ == "__main__":
    main()
