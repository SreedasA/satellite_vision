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
