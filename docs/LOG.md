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

## 2026-11-19 · Phase 2 run D: `p2d_dfc23_block` (run by you, offline)

Config `p2/d_mse.yaml` = C + small MSE term (`w_mse: 0.05`). This is the **only** config difference from C. Took 56 min (some epochs ~115 s, probably other load on the PC); best checkpoint at epoch 27.

| DFC23 block test | MAE | RMSE | δ1 | bias | B-RMSE | B-bias | bld MAE | bld RMSE | floor ±1 | <10 m MAE / bias | 10–30 m MAE / bias | 30+ m MAE / bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | **1.90** | 5.14 | 0.50 | −0.40 | 7.18 | −1.06 | 2.79 | 4.50 | 0.84 | 2.06 / −0.81 | 3.24 / −1.77 | 7.23 / −6.92 |
| B | 2.06 | 5.32 | 0.53 | +0.17 | 7.10 | −0.28 | 2.65 | 4.22 | 0.85 | 2.11 / −0.41 | 2.94 / −1.27 | 6.32 / −5.63 |
| C | 2.03 | 5.25 | **0.56** | +0.30 | 6.92 | −0.01 | **2.48** | 4.04 | **0.87** | 2.03 / +0.05 | **2.69** / −0.90 | 5.97 / −5.11 |
| **D (C + MSE)** | 1.99 | **5.03** | 0.55 | +0.38 | **6.69** | +0.20 | **2.48** | **3.95** | **0.87** | 2.03 / −0.01 | 2.72 / **−0.77** | **5.52 / −4.64** |

Validation select score: best A 8.73 / B 8.65 / C 8.32 / **D 8.01**; last-5-epoch mean 8.95 / 8.71 / 8.57 / **8.13**. D's margin over C (~0.4) is ~3× the epoch-to-epoch noise, so the clearest step after A→B.

**Findings**
1. **D is the best overall model.**
   - Lowest RMSE (5.03), building RMSE (6.69) and per-building RMSE (3.95).
   - Best on the tallest buildings: 30 m+ MAE 5.97 → 5.52 m (−8%), bias −5.1 → −4.6 m.
   - Per-building MAE (2.48 m) and floor ±1 (87%) tie C.
   - As expected from an MSE term, it mainly reduces *large* errors (RMSE-type metrics), while MAE-type metrics stay level.
2. All-pixel MAE (1.99) is better than B/C, only A is lower (1.90).
3. **Watch item:** overall bias keeps rising (−0.40 → +0.17 → +0.30 → +0.38 m across A→D); non-building pixels are increasingly over-predicted. Check in the failure gallery (trees, edges) before tuning further.

**Phase 2 config sweep summary (DFC23 block, vs Phase 1):**
- Per-building MAE 3.28 → 2.48 m (−24%)
- Building RMSE 7.89 → 6.69 m (−15%)
- 30 m+ MAE 10.0 → 5.5 m (−45%), bias −9.9 → −4.6 m
- Floor ±1: 79 → 87%

**Decision:** D is the Phase 2 recipe. Remaining before gate G1:
1. U-Net baseline (same schedule)
2. Second seed of D (confirm)
3. D on DFC19 tile (comparison with published work) and DFC23 cross-city (New Delhi)

## 2026-11-19 · Phase 2 U-Net baseline: `p2u_dfc23_block` (run by you, offline)

Config `p2/unet_baseline.yaml`: ResNet-34 U-Net with ImageNet init, on **A's recipe** (30 epochs, cosine, tall-aware selection, *no* height weighting or MSE). LRs are set for a CNN (encoder 1e-4, head 1e-3). Took 32 min (62 s/epoch, ~40% faster than DAv2-S); best checkpoint at epoch 28. **Model sizes are almost identical: U-Net 24.5 M vs DAv2-S 24.8 M parameters.**

| DFC23 block test | MAE | RMSE | δ1 | B-RMSE | B-bias | bld MAE | bld RMSE | floor ±1 | <10 m MAE | 10–30 m MAE / bias | 30+ m MAE / bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Train-mean baseline | 7.19 | 9.64 | 0.11 | 15.04 | – | 7.74 | – | 0.54 | – | – | – |
| **U-Net (A recipe)** | 2.37 | 5.88 | 0.40 | 8.38 | −1.75 | 3.48 | 5.44 | 0.78 | 2.45 | 4.11 / −2.35 | 9.43 / −9.11 |
| DAv2-S A (same recipe) | **1.90** | 5.14 | 0.50 | 7.18 | −1.06 | 2.79 | 4.50 | 0.84 | 2.06 | 3.24 / −1.77 | 7.23 / −6.92 |
| DAv2-S D (best) | 1.99 | **5.03** | **0.55** | **6.69** | +0.20 | **2.48** | **3.95** | **0.87** | **2.03** | **2.72 / −0.77** | **5.52 / −4.64** |

Validation select score: U-Net best 10.53 vs DAv2-S A 8.73 vs D 8.01.

**Findings**
1. **At the same model size and recipe, DAv2-S clearly beats the CNN:** per-building MAE −20% (3.48 → 2.79 m), building RMSE −14%, overall RMSE −13%, 30 m+ MAE −23%. The depth-foundation pretraining, not model size, accounts for the gain.
2. **The 30-epoch U-Net (3.48 m) is worse than a 5-epoch DAv2-S (3.28 m, Phase 1).**
3. The U-Net shares the same tall-building weakness, more strongly (30 m+ bias −9.1 m vs −6.9 m for DAv2-S A). The bias comes from the data and loss as well as the model, which is why the B–D loss changes matter.
4. The U-Net trains ~40% faster per epoch; its validation was still improving slightly at the end. A longer U-Net run would narrow the gap, but not close it.

**Gate G1 (DAv2 vs CNN): PASS on DFC23.** The other half of G1, checking the pipeline against the published DFC2019 U-Net (~2.98 m RMSE), still needs a U-Net run on the DFC2019 tile split.

## 2026-11-19 · U-Net on DFC2019 tile split: `p2u_dfc19_tile` (run by you, offline): first diagnosis, SUPERSEDED

> **Superseded by the next entry.** The instability was not caused by the learning rate: it was a U-Net padding bug at evaluation time (fixed). The lower-LR rerun proposed below is not needed.

Same config as the DFC2023 U-Net (`p2/unet_baseline.yaml`: A recipe, encoder LR 1e-4, head LR 1e-3). Took 59 min (~112 s/epoch on 1024² DFC2019 tiles). **No second-seed run of D exists yet:** every run so far uses seed 0.

**Training was unstable.** Validation RMSE rose from 4.17 m (epoch 2) to 12.8 m (epoch 30) while training loss kept falling and validation per-building MAE stayed ~3 m. A diagnosis on 20 validation tiles found the epoch-30 model **hallucinates 140–155 m structures on ground and water** (true maxima 24–62 m). RMSE on ground pixels 3.1 → 13.7 m, on water 2.1 → 15.2 m; buildings 7.3 → 10.0 m. The tall-aware selection kept the **epoch-3 checkpoint** (select 13.21), so the test numbers come from the healthy early model.

| DFC19 tile test (150 images, 7,459 buildings) | MAE | RMSE | SI | <1 m | B-RMSE | bld MAE | floor ±1 | 10–30 m MAE / bias | 30+ m MAE / bias (n=17) |
|---|---|---|---|---|---|---|---|---|---|
| Train-mean baseline | 3.76 | 5.15 | 1.03 | 0.06 | 10.97 | 1.97 | 0.90 | – | – |
| U-Net (best = epoch 3) | 1.92 | 4.38 | 0.88 | 0.68 | 8.43 | 2.90 | 0.84 | **5.47** / −1.81 | 23.9 / −23.9 |
| DAv2-S Phase 1 (5 epochs, old recipe) | **1.26** | **3.38** | **0.68** | **0.74** | **7.69** | **2.06** | **0.93** | 7.97 / −7.70 | 25.6 / −25.6 |

**Findings**
1. **DAv2-S beats the U-Net on DFC2019 too**, even its 5-epoch Phase 1 version against the U-Net's best checkpoint: RMSE 3.38 vs 4.38 m, per-building MAE 2.06 vs 2.90 m. The U-Net is better only on 10–30 m buildings (5.5 vs 8.0 m MAE), where the old DAv2 recipe had a large bias that B–D since reduced.
2. **The pipeline check against the published U-Net (~2.98 m RMSE) is inconclusive.**
   - Our U-Net stopped improving after epoch 3, so it isn't a converged baseline.
   - The published figure probably comes from a random image split, which leaks (D13).
   - **Next step:** rerun the U-Net on DFC19 with ~3× lower LRs (stability), and optionally on the random split, to compare like with like.
3. The tall-aware checkpoint rule did its job: it protected the test result from a model that degraded late in training.
4. On DFC2019 with only 17 buildings above 30 m, the 30+ m bin is too small to compare models reliably.

**Gate G1:**
- DAv2 vs CNN: **pass on both datasets**.
- Pipeline vs published numbers: **pending** (stable U-Net rerun).

## 2026-11-19 · U-Net on DFC2019: `p2u_dfc19_tile` (run by you) + U-Net padding bug

Config `p2/unet_baseline.yaml` on the DFC2019 tile split (30 epochs, 112 s/epoch, 58 min).

**What looked wrong:** validation RMSE rose from 4.2 m (epoch 2) to 12.8 m (epoch 30) while training loss kept falling, so best-checkpoint selection kept epoch 3. About 2% of pixels had errors > 20 m, with predictions up to 165–172 m where the true maximum was 23–60 m; a third of them were in the bottom rows of each tile.

**Cause: an inference bug in the U-Net, not the model.** `UNetHeight` reflect-padded only bottom/right up to a multiple of 32. In training (518 px crops) that is 26 px; at evaluation (full 1022 px DFC2019 tiles) it is 2 px. The model's border behaviour broke on the thin one-sided pad and got worse as training went on. DAv2 doesn't pad and is unaffected. DFC23 is evaluated at 518 px (same padding as in training), so it was barely affected.

**Fix:** symmetric reflect padding with a 32 px margin on every side (`pad_amounts()`, tested). Same checkpoint, validation: RMSE 12.84 → **3.55 m**, max prediction 172 → 66 m. New `scripts/evaluate_run.py` re-scores saved checkpoints after a fix (writes `*_reeval.json`, keeps the originals).

**Re-evaluated test results (150 tiles, fixed code)**

| Model | Dataset | RMSE | MAE | B-RMSE | bld MAE | floor ±1 |
|---|---|---|---|---|---|---|
| U-Net, last epoch (30) | DFC19 tile | 3.52 | 1.30 | 8.03 | 1.89 | 0.94 |
| U-Net, "best" (epoch 3, chosen under the bug) | DFC19 tile | 4.13 | 1.73 | 8.39 | 2.69 | 0.88 |
| DAv2-S Phase 1 (5 ep, old recipe) | DFC19 tile | **3.38** | **1.26** | **7.69** | 2.06 | 0.93 |
| U-Net best / last | DFC23 block | 5.94 / 5.96 | 2.38 / 2.39 | 8.50 / 8.66 | 3.48 / 3.48 | 0.78 |

**Findings**
1. **Pipeline check (gate G1, part 2): PASS.** Our U-Net reaches 3.52 m RMSE on the leak-free DFC2019 tile split. That's in the expected range of the published ~2.98 m, which was probably measured on an easier, leaky random split.
2. **DAv2-S still wins on DFC2019:** even the 5-epoch Phase 1 recipe beats a 30-epoch U-Net on RMSE (3.38 vs 3.52) and building RMSE (7.69 vs 8.03). The U-Net's slightly better per-building MAE (1.89 vs 2.06) comes from DFC2019's many low houses (see Phase 0 note: never quote a single per-building number for DFC2019).
3. **DFC23 U-Net results are unchanged by the fix** (3.48 m per-building), so the 19 Nov DAv2-vs-U-Net conclusion stands.
4. The DFC2019 U-Net's checkpoint selection was corrupted by the bug; its number comes from the last epoch. A re-run with the fixed code (~1 h) would give properly selected numbers. Optional, since the conclusion is clear.

**Gate G1: PASS** (DAv2 beats the CNN on both datasets; pipeline matches the published range). Remaining before closing Phase 2: config D second seed, D with New Delhi held out, D on DFC2019 tile.

## 2026-11-20 · Config D, second seed: `p2d_dfc23_block_s1` (run by you, offline)

Same config as `p2d_dfc23_block` with `--seed 1` (training seed only; validation and test samples are fixed). 30 epochs, ~106 s/epoch, 54 min.

| DFC23 block test (150 tiles, 9,284 buildings) | seed 0 | seed 1 | spread |
|---|---|---|---|
| per-building MAE | 2.48 | 2.51 | 0.03 |
| building-pixel RMSE | 6.69 | 6.75 | 0.06 |
| all-pixel RMSE | 5.03 | 4.98 | 0.05 |
| floor ±1 | 0.866 | 0.863 | 0.003 |
| MAE <10 / 10–30 / 30+ m | 2.03 / 2.72 / 5.52 | 1.97 / 2.85 / 5.78 | – |
| bias 10–30 / 30+ m | −0.77 / −4.64 | −1.24 / −5.14 | – |
| overall pixel bias | +0.38 | +0.17 | – |
| best epoch (val select) | 27 (8.01) | 27 (8.19) | – |

**Findings**
1. **Seed noise ≈ 0.03 m per-building MAE** (two seeds, rough). The A→B (0.14) and B→C (0.17) gains are real; C vs D (2.48 vs 2.48) is a tie overall, and D is kept for its tall-building and building-RMSE gains.
2. **Tall-building under-prediction is systematic** (30 m+ bias −4.6 / −5.1 m with both seeds) → Phase 3 failure gallery; candidate v1.1 fixes: higher `w_tall`, height-range calibration.
3. **The "rising overall bias" was mostly seed noise** (+0.38 vs +0.17 m): dropped as a concern.
4. Both seeds plateau from epoch ~21 and select epoch 27: 30 epochs is enough.
5. Report D on DFC23 block as **per-building MAE 2.50 ± 0.02 m, building RMSE 6.72 m, floor ±1 86.5%** (mean of 2 seeds).

## 2026-11-20 · Config D on New Delhi held out and on DFC2019: `p2d_dfc23_cross_city`, `p2d_dfc19_tile` (run by you, offline)

Both seed 0, 30 epochs, code at 62d1d02. Cross-city: ~120 s/epoch, 61 min; validation comes from the training cities. DFC19 tile: ~190 s/epoch, 95 min.

**DFC23 cross-city (New Delhi unseen; 131 tiles, 5,010 buildings)**

| Model | RMSE | B-RMSE | B bias | bld MAE | floor ±1 | <10 m MAE / bias | 10–30 m MAE / bias | 30+ m MAE / bias (n=198) |
|---|---|---|---|---|---|---|---|---|
| Phase 1 (5 ep) | 5.38 | 10.02 | – | 6.15 | 0.48 | – | – / −6.96 | – / −19.4 |
| **Config D (30 ep)** | 5.09 | 9.47 | −6.38 | **5.76** | 0.50 | 2.42 / −0.11 | 6.60 / −6.54 | 17.3 / −17.3 |

**DFC2019 tile split (150 tiles, 7,459 buildings)**

| Model | MAE | RMSE | B-RMSE | bld MAE | floor ±1 | 10–30 m MAE / bias (n=318) | 30+ m MAE / bias (n=17) |
|---|---|---|---|---|---|---|---|
| U-Net, last epoch (fixed padding) | 1.30 | 3.52 | 8.03 | 1.89 | 0.94 | – | – |
| DAv2-S Phase 1 (5 ep) | 1.26 | 3.38 | 7.69 | 2.06 | 0.93 | 7.97 / −7.70 | 25.6 / −25.6 |
| **DAv2-S config D (30 ep)** | **1.21** | **3.18** | **7.12** | **1.62** | **0.95** | 5.79 / −5.31 | 20.0 / −20.0 |

**Findings**
1. **DFC2019: config D is the best model on every metric**, now including per-building MAE (1.62 vs U-Net 1.89). RMSE 3.18 m on the leak-free tile split is close to the published ~2.98 m (probably a leaky random split). The DFC23-tuned recipe transfers to a different sensor and region without changes.
2. **New Delhi: the recipe helps only a little out of domain.** Per-building MAE 6.15 → 5.76 m (−6%), against −24% in-domain (3.28 → 2.48 m). The generalisation gap is now the biggest weakness: 5.76 vs 2.50 m in-domain.
3. **The New Delhi error is almost all under-prediction of mid- and high-rise buildings.** Low buildings are fine (2.4 m MAE, bias −0.1 m). In the 10–30 m bin, MAE 6.60 against bias −6.54 means nearly every building is predicted about 2 floors too low; every 30 m+ building is under-predicted (bias = −MAE = −17.3 m). New Delhi is also 66% 10–30 m buildings, against 45% in the block test set.
4. Validation (training cities) does not see this gap (val select 10.2, smooth curve). City-level held-out testing is needed to report generalisation honestly (S3).
5. DFC2019 validation was noisier (select 9.8–12.8 through training); the tall-aware rule picked epoch 16. With only 17 buildings above 30 m, the DFC2019 tall-building bin stays too small to rank models.

**Phase 2: complete.** Recipe = config D. Headline numbers: DFC23 block 2.50 ± 0.02 m per-building MAE (2 seeds); DFC19 tile RMSE 3.18 m; New Delhi unseen 5.76 m.

Carried into Phase 3:
- New Delhi failure analysis: is the under-prediction a global scale offset (fixable by per-city linear calibration) or appearance-dependent? Check by fitting a single scale factor on a few New Delhi tiles and re-scoring the rest.
- Tall-building under-prediction on all datasets (failure gallery; v1.1 fixes).
- Controlled leakage study (D13), JAX→OMA, viewer polish, gate G2.
