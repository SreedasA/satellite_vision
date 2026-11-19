# Lab notebook

## 2026-11-17 · Phase 0 (Groundwork) → v0.1

**Environment**
- venv on `S:\sat2city-data\venv`; Python 3.11.9; torch 2.14.1+cu126 (RTX 3050 Laptop, 4 GiB, bf16 ok); `requirements.lock` frozen.
- Problems fixed:
  - pip timed out on the 2.6 GB wheel → downloaded with curl (resumable).
  - A parallel install pulled in CPU torch 2.15 → clean reinstall from the local wheel.
  - numpy/scipy/rasterio pins needed Python 3.12 → re-pinned (D7).
  - onnxruntime-gpu needs CUDA 13 → CPU onnxruntime (D6).
- 19/19 unit tests pass (metrics, splits, LoD1, DFC23 RLE/polygons).

**Data** (all in `S:\sat2city-data\raw`, extracted to `processed/`)
- DFC19 Track 1:
  - 2,783 RGB/AGL/CLS triplets, 1024², JAX 1,015 / OMA 1,768.
  - **108 tiles, views per tile min 1 / median 24 / max 43.**
  - Not georeferenced (identity transform, no CRS/RPC).
  - AGL in m, no NaNs, range −2.9…137 m; building px p50 5.9 m / p99 46 m.
  - CLS share: ground 65%, trees 15%, building 15%, water 3%, bridge 1%, unlabeled 1%.
- DFC23 Track 2 train:
  - 1,773 tiles, 512², 0.5 m, UTM-georeferenced, 11 cities incl. **New Delhi (131)**.
  - nDSM 0…83 m.
  - 125k footprints; 17% have a reference median < 2 m (D10).

**Splits** (`configs/splits/`)
- DFC19:
  - `random_image` 1948/418/417, with **96 tiles shared between train and test** (leaky by design).
  - `tile` 1997/376/410 (0 shared).
  - `cross_city_jax2oma` 851/164/1768 (0 shared).
- DFC23:
  - `random_tile` 1241/266/266
  - `block` 1214/277/282
  - `cross_city` (New Delhi test) 1375/267/131

**Profiling (DAv2-S fine-tune, bf16):** 518², batch 4: 2.05 GiB, ~12 img/s. With gradient checkpointing: 1.1 GiB, ~10 img/s (D8).

**Quick tests**
- ONNX: FP32/FP16/INT8 export ok. CPU: FP32 1.2 s/518², FP16 rel. err 0.04%, INT8 rel. err 5.1% (D6).
- Cesium 1.146: extruded entities cast moving sun shadows; flood colouring works; 0 ion requests. The water surface doesn't render (Phase 1).
- LoD1 on real data: New Delhi, 4 tiles → 991 buildings, 0.35 s/tile for non-ML steps, rendered on the globe with shadows + flood stats.

**Baselines** (150 test images each; "oracle" uses GT per image, not deployable)

| Dataset/split | Baseline | MAE | RMSE | B-RMSE | bld MAE | floor ±1 |
|---|---|---|---|---|---|---|
| DFC19 tile | zero | 2.12 | 5.42 | 12.93 | 4.82 | 0.90 |
| | train mean | 3.76 | 5.15 | 10.97 | 1.97 | 0.90 |
| | DAv2 global | 3.63 | 5.23 | 9.99 | 2.58 | 0.89 |
| | DAv2 oracle | 2.10 | 3.91 | 8.27 | 2.58 | 0.92 |
| DFC23 block | zero | 4.89 | 10.79 | 19.07 | 12.68 | 0.30 |
| | train mean | 7.19 | 9.64 | 15.04 | 7.74 | 0.54 |
| | DAv2 global | 6.78 | 9.42 | 13.82 | 7.09 | 0.49 |
| | DAv2 oracle | 5.32 | 8.18 | 12.09 | 6.74 | 0.42 |

**Observations**
1. Zero-shot DAv2 is barely better than predicting the mean, even with oracle scaling → fine-tuning is essential (D11).
2. DFC19 per-building metrics are dominated by many low houses (train-mean gets 1.97 m MAE) → **report per-building errors by height bin and building size**. Never quote a single per-building number for DFC19.
3. DFC19 views are not georeferenced → globe placement uses DFC23 (UTM). DFC19 is shown in a local frame (ROADMAP P0.5 Plan B).

**Gate G0: PASS.** Data loaded and licences recorded; splits by tile/block are leak-free; evaluation tests pass; every quick test has a decided path.

## 2026-11-17 · Phase 1 (rough end-to-end version) → v0.2 in progress

**Built:**
- `sat2city.train`: crops at native resolution + rot/flip/photometric augmentation; L1 + SiLog(log1p) + gradient loss; AdamW (encoder 5e-6 / head 5e-5); bf16; MLflow (sqlite); best checkpoint by val building RMSE; test on the baseline sample.
- `sat2city.infer`: tiled prediction with blending → height GeoTIFF → LoD1 GeoJSON.
- `api/main.py`: `/health`, `/predict`, `/benchmark/{ds}/{tile}`, `/…/image.png`.
- `web/index.html`: image drape, rise / sun / flood sliders, stats.
- Tests: 31 pass.

**Results (DAv2-S, 5 epochs; test = same deterministic sample as the baselines)**

| Dataset / split | MAE | RMSE | SI | B-RMSE | bld MAE | floor ±1 | bias 10–30 m | bias 30+ m |
|---|---|---|---|---|---|---|---|---|
| DFC19 tile (leak-free) | 1.26 | 3.38 | 0.68 | 7.69 | 2.06 | 0.93 | −7.70 | −25.6 (n=17) |
| DFC19 random_image (leaky) | 1.77 | 4.23 | 0.62 | 8.52 | 1.94 | – | −5.61 | −31.7 (n=42) |
| DFC23 block | 2.30 | 5.74 | 0.60 | 7.89 | 3.28 | 0.79 | −1.93 | −9.95 |
| DFC23 cross-city (New Delhi unseen) | 1.93 | 5.38 | 0.67 | 10.02 | 6.15 | 0.48 | −6.96 | −19.4 |

Baselines for comparison: DFC19 train-mean RMSE 5.15 / B-RMSE 10.97; DFC23 train-mean 9.64 / 15.04.

**Observations**
1. Fine-tuning beats every baseline on every dataset after only 5 epochs (~9–16 min per run).
2. DFC23 block per-building MAE of 3.28 m is already in the range of published results on other datasets (3.23–3.3 m).
3. **New Delhi held out:** per-building MAE nearly doubles (3.28 → 6.15 m) and floor ±1 falls from 0.79 → 0.48. This is the honest generalisation gap (S3).
4. Tall buildings are under-predicted everywhere → the Phase 2 target (D14).
5. Random vs tile comparison is confounded by test content → controlled S4 design (D13).

**Viewer fixes:**
- Sun time is set from the tile longitude (D15).
- Image layer clipped to the tile, and a 1-px transparent PNG border stops Cesium smearing edge texels across the globe.
- The water surface now renders (extruded polygon with a CallbackProperty height).

## 2026-11-18 · Phase 2 run A: `p2a_dfc23_block` (run by you, offline via `scripts\train.cmd`)

Config `p2/a_long.yaml`: 30 epochs, cosine LR (1-epoch warm-up), tall-aware selection; otherwise identical to Phase 1. Took 53 min (~106 s/epoch). Best checkpoint at epoch 24 (select 8.73).

| DFC23 block test (150 tiles, 9,284 buildings) | MAE | RMSE | SI | δ1 | B-RMSE | bld MAE | floor ±1 | 10–30 m MAE / bias | 30+ m MAE / bias (n=369) |
|---|---|---|---|---|---|---|---|---|---|
| Train-mean baseline | 7.19 | 9.64 | 1.00 | 0.11 | 15.04 | 7.74 | 0.54 | – | – |
| Phase 1 (5 ep, constant LR) | 2.30 | 5.74 | 0.60 | 0.40 | 7.89 | 3.28 | 0.79 | 3.50 / −1.93 | 10.04 / −9.95 |
| **Phase 2-A (30 ep, cosine)** | **1.90** | **5.14** | **0.53** | **0.50** | **7.18** | **2.79** | **0.84** | 3.24 / −1.77 | **7.23 / −6.92** |

**Findings**
1. Longer training improves every metric: RMSE −10%, building RMSE −9%, per-building MAE −15% (3.28 → 2.79 m), floor ±1 79 → 84%.
2. Tall buildings improve most: 30 m+ MAE 10.0 → 7.2 m and bias −9.9 → −6.9 m. **They are still clearly under-predicted.** The 10–30 m bias barely moved (−1.9 → −1.8 m).
3. Validation plateaus after ~epoch 20 (select score 8.7–9.0 while training loss keeps falling). More epochs at this recipe won't fix the remaining bias → run **B (height-weighted loss)** next, as planned.
4. 2.79 m per-building MAE is below the ~3.2–3.3 m reported on *other* datasets (not directly comparable).
