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

## 2026-11-19 · Phase 2 run B: `p2b_dfc23_block` (run by you, offline)

Config `p2/b_tall_weight.yaml` = A + height-weighted L1 (`w_tall: 1.0`, ref 10 m, cap 5). This is the **only** config difference from A. Took 54 min; best checkpoint at epoch 24.

| DFC23 block test | MAE | RMSE | δ1 | <1 m | B-RMSE | B-bias | bld MAE | floor ±1 | <10 m MAE / bias | 10–30 m MAE / bias | 30+ m MAE / bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (30 ep, cosine) | **1.90** | **5.14** | 0.50 | **0.70** | 7.18 | −1.06 | 2.79 | 0.84 | **2.06** / −0.81 | 3.24 / −1.77 | 7.23 / −6.92 |
| **B (A + w_tall)** | 2.06 | 5.32 | **0.53** | 0.68 | **7.10** | **−0.28** | **2.65** | **0.85** | 2.11 / **−0.41** | **2.94 / −1.27** | **6.32 / −5.63** |

**Findings**
1. **Building metrics improve**, and the weighting did what it was meant to: per-building MAE −5% (2.79 → 2.65 m), 10–30 m MAE −9%, 30 m+ MAE −13% (7.23 → 6.32 m). Building-pixel bias is almost gone (−1.06 → −0.28 m).
2. **Trade-off:** all-pixel MAE +8% (1.90 → 2.06), RMSE +3%, and overall bias turned slightly positive (+0.17 m). Likely cause: more weight on tall targets pushes up other tall-ish pixels too (trees, edges).
3. **Tall buildings are still under-predicted** (−5.6 m above 30 m), but the gap is shrinking: −9.9 (P1) → −6.9 (A) → −5.6 (B).
4. Caveat: one seed per config, one 150-tile test sample. Differences of 3–13% are suggestive, not conclusive. Repeat the best config with a second seed before Phase 2 is closed.

**Decision:** B is the current best for the project goal (building heights / LoD1). C (higher LR) builds on B as planned.

## 2026-11-19 · Phase 2 run C: `p2c_dfc23_block` (run by you, offline)

Config `p2/c_higher_lr.yaml` = B + 2× learning rates (encoder 5e-6 → 1e-5, head 5e-5 → 1e-4). These are the **only** config differences from B. Took 54 min; best checkpoint at epoch 21.

| DFC23 block test | MAE | RMSE | δ1 | bias | B-RMSE | B-bias | bld MAE | floor ±1 | <10 m MAE / bias | 10–30 m MAE / bias | 30+ m MAE / bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (30 ep) | **1.90** | **5.14** | 0.50 | −0.40 | 7.18 | −1.06 | 2.79 | 0.84 | 2.06 / −0.81 | 3.24 / −1.77 | 7.23 / −6.92 |
| B (A + w_tall) | 2.06 | 5.32 | 0.53 | +0.17 | 7.10 | −0.28 | 2.65 | 0.85 | 2.11 / −0.41 | 2.94 / −1.27 | 6.32 / −5.63 |
| **C (B + 2× LR)** | 2.03 | 5.25 | **0.56** | +0.30 | **6.92** | **−0.01** | **2.48** | **0.87** | **2.03 / +0.05** | **2.69 / −0.90** | **5.97 / −5.11** |

Validation select score (lower = better): best A 8.73 / B 8.65 / **C 8.32**; mean of last 5 epochs 8.95 / 8.71 / **8.57**. Epoch-to-epoch noise over the last 10 epochs is σ ≈ 0.13, so C's gain is about 1–3 σ: probably real, not conclusive.

**Findings**
1. **C is best on every building metric:** per-building MAE −6% vs B (2.65 → 2.48 m), 10–30 m MAE −9%, 30 m+ MAE −6%. Building-pixel bias is now ~0. Floors ±1: 87%.
2. **All-pixel metrics sit between A and B** (MAE 2.03, RMSE 5.25), but overall bias keeps creeping up (+0.30 m): the model slightly over-predicts non-building areas.
3. **Cumulative vs Phase 1:** per-building MAE 3.28 → 2.48 m (−24%); 30 m+ MAE 10.0 → 6.0 m (−41%); 30 m+ bias −9.9 → −5.1 m. The tallest buildings are still the main weakness.
4. With a higher LR, validation is noisier early (epochs 1–3) but converges well. The model was likely under-trained at the Phase 1 LR.

**Decision:** C is the new best. Next: **D** (C + small MSE term) per plan. Then the **U-Net baseline** (gate G1) and a second seed of the winner.
