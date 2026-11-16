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
