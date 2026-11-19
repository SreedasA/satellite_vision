# Phase 2 readiness checklist (18 Nov 2026)

Phase 2 (core model, target v0.5) starts only when every **Must** item is ticked.
Owner: **C** = Claude, **Y** = you.

## A. Close Phase 1

| ✓ | Item | Owner | Priority | Notes |
|---|---|---|---|---|
| [x] | Phase 1 code committed & pushed | C | Must | `a57ec87` |
| [x] | README rewritten for v0.2 | C | Must | Results, quick start, limitations, acknowledgements |
| [x] | Live training progress bar | C | Must | `scripts\watch_training.cmd` → bar, %, epoch/step, ETA, best val; launched in the Terminal panel for every run |
| [x] | Verify the progress bar's ETA on a short demo run | C | Must | Done: ETA within ~10%; fixed a Windows file-lock crash (progress can no longer stop training); watcher follows new runs automatically |
| [x] | Commit today's changes (progress bar, checklist) | C | Must | `b1b765d` |
| [ ] | Tag `v0.1` and `v0.2` | C | Nice | Roadmap version ladder |
| [ ] | LinkedIn post about v0.2 | Y | Should | I can draft it |
| [ ] | Check DFC2023 licence/terms on DataPort / contest T&C | Y | Should | Needed before any public DFC2023 image or GIF |
| [ ] | README visual (daylight render of a DFC2019 tile, CC-BY) | C | Nice | Fine to show publicly |

## B. Training infrastructure (needed for 30–50-epoch runs)

| ✓ | Item | Owner | Priority | Why |
|---|---|---|---|---|
| [x] | **Resume from checkpoint** (`last.pt` with model + optimizer + epoch; `--resume`) | C | Must | Done: `last.pt` saved atomically every epoch (model, optimizer, LR scheduler, RNG, MLflow run id). `scripts\train.cmd --run-name X --resume`. Tested by killing a run after epoch 1 and resuming: same MLflow run, exact step count. At most the current epoch is lost |
| [~] | **Keep the laptop awake during runs** | C+Y | Must | Done in code: training blocks Windows sleep while it runs (no admin needed). You: keep it plugged in and the lid open (or set "lid close → do nothing" when plugged in) |
| [x] | **Model selection that counts tall buildings** (e.g. val building-RMSE + per-building MAE for ≥ 10 m) | C | Must | Done: `select: tall_aware` = building RMSE + per-building MAE of ≥ 10 m buildings; logged as `val.select_score` |
| [x] | **Height-bin-weighted loss option** | C | Must | Done: `loss.w_tall` (weight 1 + w·min(h/10 m, 5); a 30 m pixel counts 4×), normalised so the loss scale stays comparable |
| [x] | **LR schedule option** (warm-up + cosine) | C | Should | Done: `schedule: cosine`, `warmup_epochs`, `min_lr_ratio`; resumes correctly |
| [x] | **U-Net baseline model** (`--model unet`, timm encoder) | C | Must | Done: `model: unet:resnet34` (ImageNet init; weights cached for offline use); GPU smoke run OK; inference loads either model type from the run's config |
| [x] | Phase 2 experiment configs (one change per config) | C | Must | Done: `configs/train/p2/` A→B→C→D chain via `base:` inheritance + U-Net baseline, with a README |
| [x] | Unit tests for resume, weighted loss, U-Net | C | Must | 43 tests pass (config inheritance, weighted loss, LR schedule, selection, checkpoint/RNG round trip, resume guards, U-Net shapes + loading) |

## C. Machine & environment

| ✓ | Item | Owner | Priority | Status today |
|---|---|---|---|---|
| [x] | GPU environment works | C | Must | torch 2.14.1+cu126, RTX 3050, 43 tests pass |
| [x] | Disk space | C | Must | S: 224 GB free; runs use 0.5 GB |
| [x] | Data + leak-free splits in place | C | Must | DFC19 (2,783) + DFC23 (1,773); 6 split files |
| [ ] | Laptop plugged in, Windows power mode "Best performance", sleep off during runs | Y | Must | — |
| [ ] | Close GPU-heavy apps (games, video editing) during training | Y | Should | 4 GB VRAM is shared |

## D. Decisions to confirm

| ✓ | Decision | Owner | Recommendation |
|---|---|---|---|
| [ ] | Training budget per run | Y | 30 epochs (~1.5 h DFC23, ~2 h DFC19), local, can run overnight |
| [ ] | Primary benchmark for model selection | Y | **DFC23 block** (georeferenced, 11 cities, New Delhi tests), plus DFC19 tile for comparison with published work |
| [ ] | Order of fixes for tall buildings | Y | 1) longer training → 2) height-bin-weighted loss → 3) higher LR → 4) MSE term; one change at a time |

## Definition of ready (go/no-go for Phase 2)
- All **Must** items in A–C ticked
- Decisions in D confirmed
- Test suite green, everything committed
