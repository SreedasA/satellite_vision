# Sat2City3D — Execution Roadmap (versions, steps, best/worst cases, fallbacks)

> Companion to [`PLAN.md`](PLAN.md) (strategy) and [`RESEARCH.md`](RESEARCH.md) (evidence). Draft 16 Nov 2026.
> **Rule:** every step has a time box. When a step hits its worst case or runs past its time box, switch to its Plan B immediately and log it in `docs/DECISIONS.md`. Don't keep pushing on the original approach.

---

## 1. Version strategy: which version are we building?

### Release ladder

| Version | Name | Date | Contents (definition of done) |
|---|---|---|---|
| v0.1 | Groundwork | Fri 20 Nov | Environment pinned; data downloaded and inspected; tile-ID splits; eval code + tests; trivial and zero-shot baselines; quick-test results; G0 passed |
| v0.2 | Rough end-to-end | Fri 27 Nov | Image → (short-trained) DAv2-S → nDSM → per-building heights → LoD1 → CesiumJS with sun slider, served by FastAPI. First tile-split vs random-split numbers |
| v0.5 | Core model | Fri 4 Dec | Full-resolution DAv2-S fine-tune; U-Net baseline; S1 + S2 metrics on the tile split; G1 passed |
| **v0.8** | **Minimum release (floor)** | **must ship by Fri 18 Dec** | v0.5 + FP16 ONNX + Docker API + viewer (image↔3D, sun/shadow, basic flood colouring) + README with honest results + 90-s video |
| **v1.0** | **Target release** | **Fri 18 Dec** | v0.8 + **S4 split-leakage study** + **S3 cross-city (JAX↔OMA) and DFC23 test** + failure gallery (S7) + INT8 experiment (S6) + public demo (HF Spaces + static web) |
| v1.1 | Synthetic study | Jan 2027 | **S5 SynRS3D label-efficiency curve** |
| v2.0 | Research+ | Jan–Feb 2027 | DAv2-Base + LoRA, height + mask multi-task head, offset-aware footprints, Gaussian-splat comparison tab, Indian-city qualitative demo |

### Which version to commit to: **v1.0 as the target, v0.8 as the minimum**

| Option | Value for jobs/articles | Cost | Risk | Verdict |
|---|---|---|---|---|
| Aim for v0.8 only | Working demo, but "yet another fine-tune" | Low | Low | Too weak: no original finding |
| **Aim for v1.0** | Demo **plus** an original, cheap finding (S4 reuses the same model with only the split changed) **plus** a generalisation story (S3) | Medium | Medium | **Chosen.** Best value per hour |
| Aim for v1.0 + S5 by December | Adds a second research finding | High: a SynRS3D subset is a 4–16 GB download, plus pre-training compute. The full set is 101.8 GB | High | **Deferred to v1.1.** If S5 squeezes the schedule it puts v1.0 at risk |

**How this affects every step:** each step below is tagged with the version it belongs to.
- Steps tagged **v1.1 / v2.0** are not started before v1.0 ships, even if there is spare time; spare time goes to v1.0 quality.
- If the schedule slips, drop v1.0 items in this order: INT8 → DFC23 test → failure gallery. **Never drop S4 or the viewer.**

### Pinned stack ("Stack v1", fixed for v0.1–v1.0)

Chosen 16 Nov 2026. Each version is at least about 2 weeks old, and the CUDA wheel works with both the local driver and cloud GPUs.

| Component | Pin | Why |
|---|---|---|
| Python | 3.11.9 (installed) | Wheels available for every library below |
| PyTorch / torchvision | **2.14.1 / 0.29.1**, `cu126` wheels | Local driver 621.24 supports CUDA 13.5, but **cu126 is the most portable to Kaggle/Colab** drivers. Not 2.15.0 (only 3 weeks old). |
| transformers | 5.22.x | Loads DAv2 weights (`depth-anything/Depth-Anything-V2-Small-hf`) |
| timm | 1.0.30 | U-Net encoders / utilities |
| ONNX / ONNX Runtime (GPU) | 1.23.2 / 1.31.1 | Not 1.32.0 (released 13 Nov) |
| rasterio / geopandas / shapely | 1.5.2 / 1.1.4 / 2.1.2 | Stable lines; avoid fresh minors (geopandas 1.2.0, shapely 2.2.0) |
| MLflow | 3.17.2 | Local file store |
| FastAPI / uvicorn | 0.162.x / 0.53.0 | Avoid FastAPI releases from the last few days |
| pytest | 9.1.1 | — |
| CesiumJS | **1.146.0** | 1.147.0 is too new; **avoid Cesium ion dependencies** (see P0.8) |

**Upgrade policy:** no upgrades until v1.0 ships, except for security fixes or a blocking bug. Each upgrade is logged in DECISIONS.md. Exact versions are frozen in `requirements.lock` on Day 1.

---

## 2. Step template

Each step lists:
- **Version** — which release it belongs to
- **Questions** — what this step must answer
- **Best case** — the best realistic outcome
- **Worst case** — the realistic worst outcome
- **Trigger** — the signal that we're heading for the worst case
- **Plan B / Plan C** — researched fallbacks
- **Time box** — maximum time before switching to the fallback

---

## 3. Phase 0 — Groundwork (17–20 Nov) → v0.1

### P0.1 Environment & pinned stack (v0.1)
- **Questions:** Does PyTorch see the GPU with the cu126 wheel? Do geospatial wheels install cleanly on Windows?
- **Best case:** everything installs from pip in under 30 min; `torch.cuda.is_available() == True`.
- **Worst case:** CUDA not detected, or rasterio/GDAL wheel errors.
- **Trigger:** `cuda.is_available()` is False, or the import fails.
- **Plan B:** use the `cu130` wheel (the local driver supports CUDA 13.5). For geospatial wheels, install via `conda-forge` in a separate env.
- **Plan C:** develop in WSL2 Ubuntu, or do GPU work on Kaggle and keep the laptop for CPU work and the web front end.
- **Time box:** 3 h.

### P0.2 Dataset downloads (v0.1) — *you do the downloads from DataPort*
- **Questions:** Can we get DFC19 Track 1 RGB + Reference (14.8 GB) and DFC23 track2 (3.9 GB) in time?
- **Best case:** both done on Day 1.
- **Worst case:** DataPort downloads stall, fail repeatedly, or need extra approval.
- **Trigger:** no progress after about 4 h, or repeated failures.
- **Plan B:** start with **DFC23 Track 2** (smaller; has footprints + nDSM) and make it the primary dev set until DFC19 arrives. Use a download manager that can resume.
- **Plan C:** develop on **SynRS3D** (CC BY 4.0 on Zenodo, with MD5 checksums; take one 4–6 GB grid zip). Switch to real data as soon as it lands.
- **Time box:** Day 1–2 (in parallel with other steps).

### P0.3 Licence & terms check (v0.1)
- **Questions:**
  - What exactly may we publish: metrics, predictions, screenshots, a demo?
  - Is redistribution of raw data allowed?
  - Is approval needed before publication?
- **Best case:** CC-BY (DFC19 shows "Creative Commons Attribution"; SynRS3D is CC BY 4.0). Attribution is enough.
- **Worst case:** DFC23 is NC/ND, or DFC19 requires approval before publication.
- **Plan B:** the public demo shows only **our predictions** and a few attributed sample tiles. No raw data is redistributed. Results are framed as a portfolio project, not a paper.
- **Plan C:** drop the restricted dataset from the public demo and keep it only for internal evaluation numbers.
- **Time box:** 1 h.

### P0.4 Data inspection & tile IDs (v0.1) — *critical for S4*
- **Questions:**
  - Do filenames encode city/tile/view?
  - How many views per tile?
  - AGL units and nodata value?
  - Class codes?
  - How large is the off-nadir lean?
  - What does the height histogram look like?
- **Best case:** names like `JAX_004_009_RGB.tif` give the tile ID directly. There are about 25 views per tile. AGL is in metres with a clear nodata value.
- **Worst case:** no tile ID in the filenames, so a leakage-safe split isn't possible yet.
- **Plan B:** group images by identical or near-identical AGL/CLS reference rasters (the same tile → the same ground truth), using a hash of the downsampled reference.
- **Plan C:** cluster images by content using DINOv2 embeddings, check the clusters by eye on a sample, and treat each cluster as one location.
- **Time box:** 1 day.

### P0.5 Geo-metadata → globe or local 3D frame (v0.1)
- **Questions:** Do Track 1 images carry a CRS/transform or RPCs we can use to place buildings on the Cesium globe?
- **Best case:** a usable transform → real lon/lat on the globe.
- **Worst case:** unrectified images with no usable geo-transform.
- **Plan B:** a **local ENU frame**. Convert pixel coordinates to metres (GSD ≈ 1.3 m) around an approximate city origin (Jacksonville / Omaha), clearly labelled as approximate placement.
- **Plan C:** render the 3D scene in **three.js** (no globe needed). Keep Cesium for DFC23, if its tiles are georeferenced.
- **Time box:** 2 h.

### P0.6 Evaluation code + tests (v0.1)
- **Questions:** Are MAE/RMSE/SI-RMSE/δ/RMSE-B/per-building/floor metrics correct?
- **Best case:** all unit tests pass first time; numbers match hand calculations.
- **Worst case:** ambiguity about nodata handling or about how building instances are defined.
- **Plan B:** follow the DFC19 baseline evaluation conventions (pubgeo code) and document them in DATA.md.
- **Plan C:** report the same metrics under two documented conventions.
- **Time box:** 1 day.

### P0.7 Baselines + zero-shot + profiling (v0.1)
- **Questions:**
  - What do the trivial baselines score?
  - How good is zero-shot DAv2-S after oracle alignment?
  - Local memory and seconds per iteration at 256/384/518 px?
- **Best case:** zero-shot+oracle is clearly better than the trivial baselines; 518 px training fits locally at batch ≥ 2 in FP16.
- **Worst case:** 518 px won't fit even at batch 1.
- **Plan B:** 384 px + gradient checkpointing + gradient accumulation; or freeze the encoder for warm-up epochs.
- **Plan C:** run all training on Kaggle. Reported free tier: about 30 GPU-h/week, 12-h sessions, T4/P100; check your actual quota in account settings. Colab is the second option. The laptop keeps inference + development.
- **Time box:** 1 day.

### P0.8 Quick risk tests (v0.1)

| Quick test | Best case | Worst case | Plan B | Plan C | Box |
|---|---|---|---|---|---|
| DAv2-S → ONNX (FP32/FP16), outputs compared with PyTorch | Exports; max diff tiny | Export fails or FP16 drifts | Change opset / use the dynamo exporter; replace LayerNorm/interpolation ops (Jetson-paper tricks) | Serve PyTorch directly in FastAPI; drop ONNX from v0.8 | 2 h |
| Building instances from CLS → per-building heights | Clean separate buildings | Adjacent buildings merge into blocks | Split merged blobs with distance-transform watershed | Report per-"block" heights, labelled as such | 2 h |
| CesiumJS 1.146: extruded GeoJSON + sun shadows, **no ion token** | Extruded polygons cast shadows; no ion calls | Shadows don't work for polygon entities, or ion is required | Export buildings as **glTF** (trimesh) and load them as models (models support shadows) | three.js scene with a directional "sun" light | 3 h |
| Timing of non-ML steps for 1 tile | < a few seconds | Very slow polygonisation | Vectorise with rasterio.features + shapely 2 bulk operations | Pre-compute offline for the demo tiles | 1 h |

### P0.9 Gate G0 → tag v0.1
- **Pass if:** data loaded, splits by tile, eval tests pass, every quick test has a decided path.
- **Fail path:** if real data is still missing, continue Phase 1 on DFC23 or SynRS3D and slot DFC19 in later. The phase dates stay fixed.

---

## 4. Phase 1 — Rough end-to-end version (21–27 Nov) → v0.2

### P1.1 First DAv2-S fine-tune on the tile split (v0.2)
- **Questions:** Does the Depth2Elevation-style recipe (AdamW, lr 5e-6, MSE + SiLog + grad) converge for DAv2-S?
- **Best case:** loss falls steadily; it beats trivial and zero-shot baselines within a few epochs.
- **Worst case:** NaNs, divergence, or no gain over zero-shot.
- **Trigger:** NaN loss, or validation stays flat for 2 epochs.
- **Plan B:** switch FP16 → **bf16** (Ampere GPUs support it); gradient clipping; lr 1e-5 for the head and 5e-6 for the encoder; L1 loss only at first.
- **Plan C:** freeze the encoder (train the head only), then unfreeze the last N blocks. Check the data pipeline: visualise inputs and targets.
- **Time box:** 2 days.

### P1.2 Same quick run on the random-image split (v0.2, previews S4)
- **Questions:** Is there a visible gap between random-split and tile-split error?
- **Best case:** a clear gap (random split looks much better), confirming the leakage hypothesis.
- **Worst case:** no meaningful gap.
- **Plan B:** a negative result is still a result. S4 becomes "splits matter less than expected for DFC19", and the article angle shifts to cross-city generalisation (S3).
- **Time box:** runs alongside P1.1.

### P1.3 Full-tile inference (v0.2)
- **Best case:** a sliding window with overlap blending gives seamless nDSMs.
- **Worst case:** visible seams or out-of-memory on 1024 px tiles.
- **Plan B:** larger overlap + Hann-window blending; run inference at 518 and upsample.
- **Plan C:** CPU inference for the demo tiles (offline, cached).
- **Time box:** 1 day.

### P1.4 Per-building heights → LoD1 (v0.2)
- **Questions:** Do image-space building masks give sensible per-building heights despite off-nadir lean?
- **Best case:** median AGL per building gives plausible LoD1 blocks.
- **Worst case:** leaning façades smear heights onto neighbours, so blocks look wrong.
- **Plan B:** use a robust statistic (e.g. 70–90th percentile) inside an eroded mask; drop pixels near edges.
- **Plan C (G2 fallback):** render the nDSM directly as a **height-field mesh** (no per-building blocks) for the viewer. Keep per-building metrics on DFC23 footprints.
- **Time box:** 1 day.

### P1.5 ONNX FP16 + FastAPI + Docker (v0.2)
- **Best case:** a container returns GeoJSON + stats for an uploaded tile.
- **Worst case:** Docker Desktop/WSL problems on Windows.
- **Plan B:** run uvicorn natively for v0.2; build Docker later on Hugging Face Spaces (it builds remotely).
- **Plan C:** pre-computed demo outputs served as static files (no live inference) for v0.2.
- **Time box:** 1 day.

### P1.6 CesiumJS viewer, rough version (v0.2)
- **Best case:** extruded buildings, sun slider with shadows, basic flood colouring, stats panel.
- **Worst case:** the P0.8 Cesium quick test failed.
- **Plan B / C:** the glTF-model path, or three.js, as decided in P0.8.
- **Time box:** 2 days.

### P1.7 Publish the rough version (v0.2)
- **Best case:** README with GIF, first numbers, caveats; LinkedIn post; start applying.
- **Worst case:** parts are unfinished on 27 Nov.
- **Plan B:** publish what works plus a screen recording. **Never move the publish date**; scope shrinks instead.

---

## 5. Phase 2 — Core model (28 Nov – 4 Dec) → v0.5

| Step | Questions | Best case | Worst case | Plan B | Plan C | Box |
|---|---|---|---|---|---|---|
| P2.1 Full-res fine-tune (cloud) | Does training at 518 px and testing at 1022 px beat the short local run? | Clear gain; stable | Kaggle quota/session limits interrupt runs | Checkpoint every N steps + resume; split epochs across sessions | Colab, or a short paid GPU rental | 3 days |
| P2.2 U-Net baseline | Is our pipeline in the expected range? | U-Net ≈ published-range RMSE | U-Net far worse than published numbers | Check normalisation, nodata, resolution, augmentations | Run the official pubgeo Keras baseline for a sanity check | 2 days |
| P2.3 S1 + S2 metrics, tile split | How do we compare with published numbers? | DAv2-S beats U-Net; per-building MAE ≤ ~3.5 m | DAv2-S ≤ U-Net (**G1 fails**) | Tune resolution/LR/loss weights; height-bin re-weighting | Head-only → partial unfreeze; or adopt U-Net as the v0.8 model and report DAv2 honestly | 2 days |
| P2.4 Gate G1 → tag v0.5 | — | Pass | Fail | Pause S3/S4, fix the recipe | Ship v0.8 with the best available model | — |

---

## 6. Phase 3 — Research + 3D (5–11 Dec) → v1.0 features

| Step | Version | Questions | Best case | Worst case | Plan B | Plan C | Box |
|---|---|---|---|---|---|---|---|
| P3.1 S4 split-leakage study (3 protocols × same model) | v1.0 | How much does the split inflate the numbers? | Large, clear gap → headline result + article 1 | Small gap | Report it as a robustness result | Fold it into S3 as one table | 2 days |
| P3.2 S3 cross-city JAX↔OMA + DFC23 | v1.0 | How much worse on an unseen city or dataset? | Moderate drop, explained by the failure gallery | DFC23 format/GSD mismatch eats the time | JAX↔OMA only | DFC23 qualitative only | 2 days |
| P3.3 Viewer polish (image↔3D, sun, flood, stats) | v0.8 | Is it understandable in 10 seconds? | Smooth, labelled, works on a laptop browser | Performance issues with many buildings | Simplify geometry; limit to demo tiles | Pre-rendered video for the README | 2 days |
| P3.4 Failure gallery (S7) | v1.0 | Where and why does it fail? | 5–6 categories with examples | — | — | — | 0.5 day |
| P3.5 Gate G2 | — | Are per-building heights trustworthy? | Yes | No | Height-field mesh in the viewer | Per-building metrics on DFC23 only | — |

---

## 7. Phase 4 — Ship (12–18 Dec) → v0.8 guaranteed, v1.0 target

| Step | Version | Best case | Worst case | Plan B | Plan C | Box |
|---|---|---|---|---|---|---|
| P4.1 FP16 ONNX accuracy + throughput (S6) | v0.8 | FP16 within ~2% of FP32 | FP16 drifts | Keep sensitive ops in FP32 (mixed) | Ship FP32 ONNX | 1 day |
| P4.2 INT8 experiment (G3) | v1.0 | INT8 within +10% RMSE-B | Collapse (likely for DINOv2-based ViTs) | MatMul-only or SmoothQuant-style | Document the failure → article 3 | 1 day |
| P4.3 Public demo | v1.0 | HF Spaces API + static web viewer | Spaces build or size limits | CPU Spaces with small demo tiles | Static pre-computed demo + video | 1.5 days |
| P4.4 README, results, video, article pitch | v0.8 | Polished, honest, linked | Time short | Results table + GIF + short write-up | — | 1.5 days |
| P4.5 Tag release | — | **v1.0 on 18 Dec** | Not everything is done | **Tag v0.8** with a roadmap to v1.0 | — | — |

---

## 8. Whole-plan contingencies (if a core assumption fails)

| Assumption that fails | Plan B | Plan C |
|---|---|---|
| DFC19 unusable (access, terms or format) | DFC23 Track 2 becomes the primary benchmark (footprints + nDSM, 17 cities) | SynRS3D (CC BY 4.0) + small real set for evaluation |
| DAv2-S can't beat a U-Net | Ship the best model; the story becomes "foundation model vs CNN under honest splits" | Try DAv2-Base + LoRA on cloud (pulls v2.0 work forward) |
| Leakage hypothesis false | Headline becomes cross-city generalisation (S3) | Lead with the product/demo; research is secondary |
| No cloud GPU available | Local training at 384 px with fewer epochs, results stated as such | Head-only fine-tuning |
| Per-building heights unreliable (off-nadir) | Height-field mesh viewer | Per-building results on DFC23 footprints only |
| Cesium blocked (shadows/ion) | glTF models in Cesium | three.js viewer |
| Schedule slips more than 3 days | Drop in order: INT8 → DFC23 → failure gallery | Ship **v0.8** on 18 Dec; v1.0 in early January |

---

## 9. Questions per phase (answer them in `docs/LOG.md` at each phase end)

**Phase 0:**
1. Is the data downloaded, licensed for our use, and split by tile?
2. Do tile IDs exist, and how many views per tile?
3. Globe or local frame?
4. Does 518 px training fit locally, or does it need the cloud?
5. Which path did each quick test pick (ONNX, instances, Cesium)?

**Phase 1:**
1. Does the recipe converge?
2. Is there a random-vs-tile gap?
3. Can a stranger understand the demo in 10 seconds?

**Phase 2:**
1. Does DAv2-S beat U-Net on the tile split?
2. What is the per-building MAE?
3. G1 pass or fail?

**Phase 3:**
1. How big is the leakage gap?
2. How much worse on an unseen city?
3. Are per-building heights trustworthy (G2)?

**Phase 4:**
1. FP16/INT8 outcome?
2. Is the demo public and stable?
3. v1.0 or v0.8?

**Weekly review (every Friday):**
- Are we on the version ladder?
- What's the riskiest open item?
- What do we drop if we're behind?

---

## Sources (fallback research)
- Stack versions: PyPI JSON API and download.pytorch.org wheel index (checked 16 Nov 2026); npm registry for CesiumJS
- SynRS3D licence and files (CC BY 4.0; 17 zips, 101.8 GB): https://zenodo.org/records/13905264
- DFC2019 baseline/code: https://github.com/pubgeo/dfc2019 · DataPort: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
- Kaggle free GPU (≈30 h/week reported; 12 h sessions per docs, per a 2026 review): https://gpuperhour.com/blog/free-cloud-gpus-and-credits · https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/
- CesiumJS without ion (disable geocoder, avoid world terrain; shadows on polygon entities unconfirmed): https://community.cesium.com/t/is-it-possible-to-use-cesiumjs-without-cesiumion/13587 · https://github.com/jguerrer/cesium-experimental/blob/main/Documentation/OfflineGuide/README.md · https://community.cesium.com/t/cesium-1-22-shadows-feature/4184
- DAv2-S ONNX/TensorRT deployment tricks and FP16 results: IEEE Xplore 11212965
