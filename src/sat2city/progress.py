"""Live training progress: written by sat2city.train, displayed by `python -m sat2city.progress`.

The trainer writes <data_root>/runs/<run>/progress.json (atomically, every few steps) and
<data_root>/runs/LATEST (name of the most recent run). The CLI prints a progress bar with ETA:

  python -m sat2city.progress            # follows the latest run live, including runs started later
  python -m sat2city.progress --once     # print once and exit
  python -m sat2city.progress --run p1_dav2s_dfc19_tile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from sat2city.config import load_paths


class ProgressWriter:
    """Tracks step timing (EMA) and epoch durations to estimate the remaining time."""

    def __init__(self, run_dir: Path, run_name: str, epochs: int, steps_per_epoch: int, every: int = 10):
        self.path = run_dir / "progress.json"
        self.every = every
        self.state = {
            "run": run_name, "status": "starting", "epoch": 0, "epochs": epochs, "step_in_epoch": 0,
            "steps_per_epoch": steps_per_epoch, "global_step": 0, "total_steps": epochs * steps_per_epoch,
            "started": time.time(), "updated": time.time(), "sec_per_step": None, "eval_sec": None,
            "eta_sec": None, "last_loss": None, "best_val_building_rmse": None, "history": [], "error": None,
        }
        self._t_warm = None
        latest = run_dir.parent / "LATEST"
        latest.write_text(run_name)
        self._write()

    def _eta(self) -> float | None:
        s = self.state
        if not s["sec_per_step"]:
            return None
        remaining_steps = s["total_steps"] - s["global_step"]
        remaining_epochs = s["epochs"] - s["epoch"] + (1 if s["status"] == "training" else 0)
        eval_sec = s["eval_sec"] or 0.0
        return remaining_steps * s["sec_per_step"] + max(remaining_epochs, 0) * eval_sec + eval_sec  # + final test

    def _write(self):
        """Best-effort atomic write. On Windows os.replace fails while a reader (the watcher)
        has the file open, so retry briefly and otherwise skip: progress must never stop training."""
        self.state["updated"] = time.time()
        self.state["eta_sec"] = self._eta()
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.state))
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
        except OSError:
            pass  # skipped this update; the next one will catch up

    def resume_from(self, epochs_done: int, global_step: int, best: float, history: list[str]):
        """Continue the bar of a resumed run instead of starting again at 0%."""
        self.state |= {"epoch": epochs_done, "global_step": global_step, "history": list(history)[-20:],
                       "best_val_building_rmse": round(best, 3) if best != float("inf") else None}
        self._write()

    WARMUP_STEPS = 3  # DataLoader workers start slowly on Windows; don't let that skew the ETA

    def epoch_start(self, epoch: int):
        self.state |= {"epoch": epoch, "step_in_epoch": 0, "status": "training"}
        self._t_warm = None
        self._write()

    def step(self, step_in_epoch: int, loss: float):
        now = time.perf_counter()
        if step_in_epoch == self.WARMUP_STEPS:
            self._t_warm = now
        elif self._t_warm is not None:
            # steady-state rate of this epoch (excludes worker start-up)
            self.state["sec_per_step"] = (now - self._t_warm) / (step_in_epoch - self.WARMUP_STEPS)
        self.state["global_step"] += 1
        self.state["step_in_epoch"] = step_in_epoch
        self.state["last_loss"] = round(loss, 4)
        if step_in_epoch % self.every == 0 or step_in_epoch <= self.WARMUP_STEPS + 2:
            self._write()

    def phase(self, status: str):
        self.state["status"] = status
        self._write()

    def epoch_end(self, line: str, eval_sec: float, best: float):
        self.state["history"].append(line)
        self.state["eval_sec"] = eval_sec
        self.state["best_val_building_rmse"] = round(best, 3)
        self._write()

    def done(self, summary: dict):
        self.state |= {"status": "done", "summary": summary, "eta_sec": 0}
        self._write()

    def failed(self, error: str):
        self.state |= {"status": "failed", "error": error[:500]}
        self._write()


def fmt_time(sec: float | None) -> str:
    if sec is None:
        return "estimating…"
    sec = int(max(sec, 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def render(p: dict, width: int = 34) -> str:
    total = max(p["total_steps"], 1)
    frac = min(p["global_step"] / total, 1.0) if p["status"] != "done" else 1.0
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    age = time.time() - p["updated"]
    stale = "  ⚠ no update for %s" % fmt_time(age) if p["status"] in ("training", "evaluating") and age > 300 else ""
    lines = [
        f"Run      {p['run']}",
        f"Status   {p['status'].upper()}{stale}",
        f"Progress [{bar}] {frac * 100:5.1f}%",
        f"Epoch    {p['epoch']}/{p['epochs']}   step {p['step_in_epoch']}/{p['steps_per_epoch']}"
        + (f"   loss {p['last_loss']}" if p["last_loss"] is not None else ""),
        f"Elapsed  {fmt_time(time.time() - p['started'] if p['status'] != 'done' else p['updated'] - p['started'])}"
        f"   Remaining ≈ {fmt_time(p['eta_sec']) if p['status'] != 'done' else '0'}",
    ]
    if p.get("best_val_building_rmse") is not None:
        lines.append(f"Best val score so far: {p['best_val_building_rmse']} (lower is better)")
    if p["history"]:
        lines.append("Last epoch: " + p["history"][-1])
    if p.get("error"):
        lines.append("Error: " + p["error"])
    if p["status"] == "done" and p.get("summary"):
        s = p["summary"]
        lines.append(f"Test: RMSE {s.get('rmse')} m · building RMSE {s.get('building_rmse')} m · per-building MAE {s.get('bld_mae')} m")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=5.0, help="refresh seconds")
    args = ap.parse_args()
    runs = load_paths()["data_root"] / "runs"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    while True:
        name = args.run or ((runs / "LATEST").read_text().strip() if (runs / "LATEST").exists() else None)
        path = runs / name / "progress.json" if name else None
        if not path or not path.exists():
            text = "No training run found yet (waiting)…"
            p = None
        else:
            try:
                p = json.loads(path.read_text())
                text = render(p)
            except (json.JSONDecodeError, OSError):
                p, text = None, "reading progress…"
        if args.once:
            print(text)
            return
        os.system("cls" if os.name == "nt" else "clear")
        follow = "" if args.run else " · follows the latest run"
        print(text + f"\n\n(refreshing every {args.every:g}s{follow} · Ctrl+C to stop)", flush=True)
        if args.run and p and p["status"] in ("done", "failed"):
            return  # a named run has ended; without --run keep following new runs
        time.sleep(args.every)


if __name__ == "__main__":
    main()
