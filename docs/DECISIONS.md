# Decision log

Short records: date · decision · why · alternatives considered.

### D1 · 2026-11-17 · Data and environment live on `S:\sat2city-data`
**Why:** C: has 43 GB free, and OneDrive would sync gigabytes. S: is an internal SATA SSD with 271 GB free.
**Alternative:** C:\sat2city-data (too little space).

### D2 · 2026-11-17 · Stack v1 pinned (see ROADMAP §1; `requirements.txt`)
**Choices:** PyTorch 2.14.1 with **cu126** wheels (portable to Kaggle/Colab drivers); CesiumJS 1.146.0.

### D3 · 2026-11-17 · Download order: PyTorch → DFC23 → DFC19
**Why:** measured ~1.1 MB/s in total. A partial zip can't be opened, so finishing complete files first lets work start sooner. DFC23 (3.9 GB) is a complete dataset on its own. DFC19 downloads resume with fresh signed links.

### D4 · 2026-11-17 · Viewer: CesiumJS extruded polygon entities, no Cesium ion (P0.8 quick test passed)
**Evidence:** `web/spike_cesium.html`: 5 extruded buildings cast sun shadows that move with the time slider; 0 requests to cesium.com; flood colouring works.
**Consequence:** the glTF and three.js fallbacks are not needed. Open issue: the translucent water surface doesn't render (Phase 1 fix).

### D5 · 2026-11-17 · No Docker / Node on this machine
**Consequence:** P1.5 follows Plan B (run uvicorn natively; build Docker on Hugging Face Spaces). The viewer uses CesiumJS from a CDN, so no Node build is needed.

### D6 · 2026-11-17 · ONNX Runtime: CPU build (`onnxruntime==1.31.1`) instead of `onnxruntime-gpu`
**Why:** the PyPI GPU build needs CUDA 13 + cuDNN 9 (`cublasLt64_13.dll` missing), while our PyTorch is cu126. The demo host (HF Spaces) is CPU-only, and GPU inference already runs in PyTorch.
**Quick-test results (pretrained DAv2-S, 518², CPU):** FP32 1.2 s; FP16 relative error 0.04% but slower on CPU (1.6–1.7 s); INT8 dynamic (MatMul only) relative error 5.1%, 1.1 s.
**Alternatives rejected for now:** installing CUDA 13 runtime wheels (~1.5 GB), or switching PyTorch to cu130 (another 2.6 GB download).

### D7 · 2026-11-17 · numpy 2.4.6 / scipy 1.17.1 / rasterio 1.4.4
**Why:** the newer pins need Python ≥ 3.12. We stay on Python 3.11 rather than re-download the cp312 PyTorch wheel.

### D8 · 2026-11-17 · Train DAv2-S locally; cloud only for Base/LoRA and sweeps
**Evidence:** profiling at 518², batch 4, bf16, no gradient checkpointing: 2.05 GiB peak, ~12 img/s. Gradient checkpointing cuts memory to 1.1 GiB at ~10 img/s.
**Consequence:** a 50-epoch DFC19 run is about 2.5 h on the laptop. ROADMAP P0.7 Plan C (Kaggle) isn't needed for DAv2-S.

### D9 · 2026-11-17 · DFC23 is a full second benchmark, and gives the Indian-city demo
**Facts:** 1,773 georeferenced (UTM) 512² tiles at 0.5 m GSD from 11 cities, **including New Delhi (131 tiles)**; nDSM ground truth; 125k footprints (COCO polygons + 0.4% compressed RLE).
**Splits:** `random_tile` (leaky reference), `block` (~1 km spatial blocks per city), `cross_city` (train on the other 10 cities, test on New Delhi).
**Consequence:** the Indian demo moves from "qualitative stretch" to a measurable held-out-city test.

### D10 · 2026-11-17 · DFC23 per-building evaluation uses only footprints with reference median ≥ 2 m
**Why:** the alignment check (200 tiles) found 17.4% of footprints have a reference median height < 2 m, and 13% of building pixels are < 1 m. This is likely a timing or resolution mismatch between footprints, 0.5 m imagery and ~2 m stereo nDSM. Pixel metrics still use all valid pixels.

### D11 · 2026-11-17 · Zero-shot DAv2 is not a usable height estimator here, so fine-tuning is required
**Evidence (DFC23 block split, 150 test tiles):** a per-image *oracle* affine fit of zero-shot DAv2-S reaches RMSE 8.18 m, against 9.64 m for predicting the training mean. The deployable global fit gives 9.42 m.
**Consequence:** Phase 1/2 fine-tuning is the core of the project. The zero-shot floor-count approach (IGARSS 2026, aerial imagery) does not transfer to 0.5 m satellite tiles.

### D12 · 2026-11-17 · Phase 1 runs offline from the HF cache (`HF_HUB_OFFLINE=1`)
**Why:** the first training launch hung with no workers and an idle GPU after startup. The cause is most likely the Hugging Face library checking online for model updates over the slow link. Runs now set `HF_HUB_OFFLINE=1` and write periodic faulthandler stack dumps (`runs_*.stack`).

### D13 · 2026-11-17 · S4 (split leakage) needs a controlled design: same test views, different training sets
**Evidence:** after 5 epochs, the random-image model scored *worse* in metres than the tile model (RMSE 4.23 vs 3.38). Its own test sample has 2.5× more buildings above 30 m (42 vs 17); on SI-RMSE it looks better (0.62 vs 0.68). Comparing on different test sets mixes up leakage with differences in test content.
**Design for Phase 3:** fix one set of test views on held-out tiles. Train model A on tiles that exclude those locations, and model B on the same data plus *other dates of the test locations*. Evaluate both on the identical test views; the gap measures leakage.

### D14 · 2026-11-17 · Phase 2's main target: tall buildings are badly under-predicted
**Evidence (DFC19 tile test, DAv2-S 5 epochs):** per-building bias −1.3 m (< 10 m, n=7,124), −7.7 m (10–30 m, n=318), −25.6 m (> 30 m, n=17). The model regresses towards the many low houses.
**Candidate fixes (one at a time):** longer training; height-bin-weighted loss; higher head LR / encoder LR; MSE term weight (penalises large errors); a log-space target.

### D15 · 2026-11-17 · Viewer sets the sun to ~10:30 local solar time per tile
**Why:** a fixed 05:00 UTC default lit New Delhi but drew Jacksonville at night. UTC hour = 10.5 − lon/15.

### D16 · 2026-11-20 · Unseen cities: report zero-shot, offer 5-tile scale calibration as a documented option
**Evidence:** New Delhi held out, config D: one scale factor (×1.48) fitted on 5 labelled tiles cuts per-building MAE 5.77 → 4.24 m; on seen cities the fitted factor is 1.00 and calibration doesn't help (docs/LOG.md, 20 Nov).
**Decision:** the headline unseen-city number stays uncalibrated. Calibration is a separate, clearly labelled row and a README "deploying to a new city" step. Affine calibration is not used (no better, unstable with 1 tile).

### D17 · 2026-11-20 · Leakage study: same test views, same training-set size
Refines D13. Model A = the existing `p2d_dfc19_tile` run; model B swaps 200 random training views for the 200 other views of the test tiles, so both train on 1,997 views and only B has seen the test locations. Both are scored on the same 205 test views. One new run (B) instead of two.

### D18 · 2026-11-20 · Report DFC23 on `dfc23_block_clean`; DFC19 GSD is 0.3 m
The block split's test tiles overlap training tiles (40% of tiles, 15% of pixels; tiles are 256 m wide, ~120 m apart). Re-scoring on the 170 test tiles with no overlap changes no conclusion (D: 2.45 / 2.49 m), so models are not retrained; all DFC23 headline numbers come from the clean test set from now on. DFC19 pixel size measured at ~0.3 m (was wrongly 1.3 m); heights and metrics unaffected, 3D placement and areas fixed.
