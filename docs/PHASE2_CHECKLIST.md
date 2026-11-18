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
| [ ] | Commit today's changes (progress bar, checklist) | C | Must | via `save.ps1` |
| [ ] | Tag `v0.1` and `v0.2` | C | Nice | Roadmap version ladder |
| [ ] | LinkedIn post about v0.2 | Y | Should | I can draft it |
| [ ] | Check DFC2023 licence/terms on DataPort / contest T&C | Y | Should | Needed before any public DFC2023 image or GIF |
| [ ] | README visual (daylight render of a DFC2019 tile, CC-BY) | C | Nice | Fine to show publicly |

## B. Training infrastructure (needed for 30–50-epoch runs)

| ✓ | Item | Owner | Priority | Why |
|---|---|---|---|---|
| [ ] | **Resume from checkpoint** (`last.pt` with model + optimizer + epoch; `--resume`) | C | Must | Phase 2 runs take ~1.5–2.5 h; sleep, crashes or a closed session currently lose everything |
| [~] | **Keep the laptop awake during runs** | C+Y | Must | Done in code: training blocks Windows sleep while it runs (no admin needed). You: keep it plugged in and the lid open (or set "lid close → do nothing" when plugged in) |
| [ ] | **Model selection that counts tall buildings** (e.g. val building-RMSE + per-building MAE for ≥ 10 m) | C | Must | Current best-checkpoint rule can favour the many low houses (D14) |
| [ ] | **Height-bin-weighted loss option** | C | Must | First fix to try for tall-building underestimation (D14) |
| [ ] | **LR schedule option** (warm-up + cosine) | C | Should | Needed for longer runs; Depth2Elevation used a constant LR |
| [ ] | **U-Net baseline model** (`--model unet`, timm encoder) | C | Must | ROADMAP P2.2: checks our pipeline against published DFC19 numbers (U-Net ≈ 2.98 m RMSE) |
| [ ] | Phase 2 experiment configs (one change per config) | C | Must | Keeps the ablations clean |
| [ ] | Unit tests for resume, weighted loss, U-Net | C | Must | Keep the suite green |

## C. Machine & environment

| ✓ | Item | Owner | Priority | Status today |
|---|---|---|---|---|
| [x] | GPU environment works | C | Must | torch 2.14.1+cu126, RTX 3050, 34 tests pass |
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
