# Sat2City3D — Project Plan & Strategy

> Status: **Draft v2 (16 Nov 2026)**, revised after the literature and dataset review in [`RESEARCH.md`](RESEARCH.md).
> Update this file when decisions change, and log why in `docs/DECISIONS.md`.
> **Execution details** (release versions, pinned stack, best/worst case and fallbacks for every step): [`ROADMAP.md`](ROADMAP.md).

---

## 1. Summary

**Sat2City3D** turns a single satellite image into an explorable 3D city in the browser. You can then run simple "what if" simulations on it: sun/shadow and flood water level.

- **Core ML:** fine-tune **Depth Anything V2-Small** (Apache-2.0) to predict per-pixel above-ground height (nDSM) from one RGB satellite image.
- **3D:** building masks/footprints + predicted heights → per-building heights → LoD1 blocks → GeoJSON / 3D Tiles.
- **Product:** a CesiumJS viewer (image↔3D slider, sun/shadow time slider, flood-level slider, stats), backed by a FastAPI inference service.

### Positioning after the literature review
Fine-tuning Depth Anything for satellite height **already exists** (Depth2Elevation, TGRS 2025; DAv2 transfer, 2025; zero-shot DAv2 floor counts, IGARSS 2026). So Sat2City3D does **not** claim novelty for the model. Its contributions are:

1. **Leakage-aware evaluation:** how much do random-image splits inflate DFC19 results compared with tile-level and cross-city (Jacksonville ↔ Omaha) splits?
2. **Per-building and floor-level accuracy** that feeds a working **LoD1 3D product**, not only pixel RMSE.
3. **Synthetic label-efficiency:** how much real labelled data does SynRS3D pre-training save for DAv2-S?
4. **Deployment:** FP16/INT8 ONNX accuracy and throughput, measured end to end.
5. **Interactive 3D viewer** with sun/shadow and flood simulations.

### Why it matters
3D city data normally needs LiDAR or stereo imagery. Both are expensive, rarely updated and missing for most Indian cities. Single-image height estimation makes 3D data much cheaper. It helps planning, flood risk, rooftop solar and insurance exposure modelling.

### What it is for (personal)
A flagship portfolio piece for EO/geospatial CV roles, and material for paid technical articles. **Not** a commercial product.

---

## 2. Scope

| In scope (MVP) | Stretch | Out of scope |
|---|---|---|
| DAv2-S nDSM fine-tune on DFC19 (+ DFC23 for cross-domain) | DAv2-Base + LoRA; Depth2Elevation-style scale modulator | Engineering-grade flood hydraulics |
| Split study: random vs tile vs cross-city | Multi-task head (height + building mask) | New ground truth / LiDAR |
| SynRS3D label-efficiency curve | Offset-aware footprints (SingleRecon/BDH-style) | Buying imagery |
| Per-building heights → LoD1 → CesiumJS with sun + flood sliders | Multi-view Gaussian-splat comparison tab | Mobile app |
| ONNX FP16 (+INT8 experiment), FastAPI, Docker, public demo | Indian-city qualitative demo | Selling / marketing |

---

## 3. Success criteria

| # | Metric | Target | Reference point |
|---|---|---|---|
| S1 | DFC19 pixel MAE / RMSE / SI-RMSE on a **tile-level** held-out split | Beat our own reproduced baselines (U-Net, direct DAv2-S fine-tune). Report alongside published numbers | Depth2Elevation reports 2.45 m RMSE (direct DAM 2.60 m), probably on a random split |
| S2 | Building-pixel RMSE-B, δ1, and **per-building MAE** + **±1-floor accuracy** | Per-building MAE ≤ ~3.5 m | SAM2+LoRA 3.23 m; SingleRecon 3.3 m (other datasets); floor count 63% (zero-shot) |
| S3 | Cross-city and cross-dataset drop: JAX→OMA, DFC19→DFC23 | Report honestly | Depth2Elevation GAMUS→DFC19 RMSE 2.62 m |
| S4 | **Split-leakage gap**: random-image vs tile vs cross-city split, same model | Quantify (headline finding) | Not reported in prior DFC19 work we found |
| S5 | **Synthetic label-efficiency**: 10/25/50/100% real, with vs without SynRS3D (**v1.1, Jan 2027**; see ROADMAP §1) | Quantify | SynRS3D shows gains mainly in low-data settings (ViT-L, figure only) |
| S6 | Deployment: FP16 ONNX accuracy change + end-to-end km²/min | FP16 within ~2% of FP32. INT8 ships only if RMSE-B rises ≤ 10% | FP16 ≈ 66% faster on Jetson with minimal loss (DAv2-S) |
| S7 | Failure analysis | Categorised gallery: off-nadir lean, shadows, tall towers, dense housing, trees on roofs | — |

**Metric conventions:** report MAE, RMSE and SI-RMSE (comparable with Depth2Elevation), δ1 (comparable with DFC23), and mIoU-3-style "height error < 1 m" (comparable with DFC19). All on **valid pixels**, and separately on **building pixels**.

**Decision gates:**
- **G0 (end of Phase 0):** data loaded, tiles identifiable, eval tests pass → go. If DFC19 can't be georeferenced for Cesium → use a local-coordinate 3D scene (see risk R7).
- **G1 (end of Phase 2):** if tile-split RMSE-B is worse than the U-Net baseline → fix the recipe (resolution, loss, LR) before any research experiments.
- **G2 (Phase 3):** if per-building heights are poor because masks and roofs are misaligned → switch to predicted roof masks (multi-task head) or a height-field mesh.
- **G3 (Phase 4):** if INT8 fails S6 → ship FP16 and write up the result.

---

## 4. Data strategy

| Dataset | Role | Access / licence | Size |
|---|---|---|---|
| **DFC2019 Track 1 (US3D)** | Primary benchmark | IEEE DataPort, CC-BY (verify terms; acknowledge JHU/APL, IARPA; cite Le Saux et al. 2019) | RGB 6.6 GB + Reference 8.2 GB |
| **DFC2023 Track 2** | Cross-domain test; per-building eval (has footprints) | IEEE DataPort, licence to verify | 3.9 GB (+0.9 GB test) |
| **SynRS3D** | Synthetic pre-training (S5) | Zenodo; licence not stated, so check before using | Subset first (~10–20k) |
| GAMUS, GBH | Optional | Licence/availability unclear | — |

**Rules**
1. **Inspect before training** (QGIS + notebook): AGL units/nodata, height histogram (long tail), image↔label alignment, views per tile, off-nadir lean, class map codes.
2. **Splits are defined by tile ID, never by image.** Three protocols, all fixed in `configs/splits/`:
   - `random_image`: reproduces the prior-work setting (expected to leak; used only for S4).
   - `tile`: JAX + OMA tiles split ~70/15/15 by tile ID.
   - `cross_city`: train JAX → test OMA (and the reverse).
3. **Frozen test sets:** touched only for final reporting.
4. **Storage outside OneDrive:** `C:\sat2city-data\` (raw, tiles, checkpoints, mlruns). Set in `configs/paths.yaml`; only code, configs and manifests go in git.
5. **Manifest:** CSV of every file + SHA-256 + tile ID + city + split.
6. **Licence log:** `docs/DATA.md` with each dataset's licence, citation and acknowledgement text.

---

## 5. Modelling strategy

1. **Trivial baselines:** zero everywhere; mean height; per-class mean height.
2. **Zero-shot DAv2-S + oracle affine alignment** (per-image scale/shift fitted with GT). An upper bound for relative depth, not deployable.
3. **U-Net baseline** (and optionally the DFC19 Keras baseline from `pubgeo/dfc2019`) to validate the pipeline.
4. **Direct DAv2-S metric fine-tune**: the main model. Starting recipe from Depth2Elevation:
   - AdamW, lr 5e-6 (constant to start)
   - loss = MSE + SiLog + gradient
   - train 518 px / test full tile (1022 px, sliding window if needed)
   - augmentation: flips, colour jitter, Gaussian blur
   - mixed precision, gradient checkpointing if needed
5. **Experiments (one change at a time):**
   - split study (S4)
   - SynRS3D pre-train → fine-tune at 10/25/50/100% (S5)
   - cross-city / cross-dataset (S3)
6. **Stretch:** DAv2-Base + LoRA (weights CC-BY-NC, fine for a portfolio, not for commercial use); scale-modulator variant; height + building-mask multi-task head.

**Compute:** local GPU for development, debugging, short runs and inference. Cloud GPU (Kaggle/Colab first, rented if needed) for full-resolution training and the sweeps in S4/S5. The same `train.py` + YAML runs in both places, with checkpoint/resume always on.

---

## 6. Evaluation strategy

- `src/sat2city/eval/` implements the metrics: MAE, RMSE, SI-RMSE, δ1/δ2/δ3, RMSE-B, < 1 m accuracy, per-building height (median AGL inside each building instance), floor accuracy (3 m/floor assumption, stated clearly), and errors by height bin (< 10, 10–30, > 30 m).
- **Unit tests** on hand-made arrays for every metric.
- Building instances for per-building evaluation:
  - DFC19: connected components of the building class.
  - DFC23: the provided footprints.
- Every run logs: config, git commit, manifest hash, split protocol, metrics, sample predictions (MLflow).

---

## 7. 3D & product strategy

```
RGB tile ─► FastAPI ─► DAv2-S (ONNX FP16) ─► nDSM
                         │
building masks / footprints ──► per-building height ─► LoD1 extrusion ─► GeoJSON / 3D Tiles
                                                                           │
                        CesiumJS viewer ◄──────────────────────────────────┘
      (image↔3D slider · sun/shadow time slider · flood level slider · stats panel)
```

- **Footprints must match the image geometry.** DFC19 views are unrectified (off-nadir), so OSM footprints will not line up. Use building masks from labels or predictions in image space.
- **Placing tiles on the globe:** check in Phase 0 whether tiles carry usable geo-metadata. If not, render in a local ENU frame (Cesium with an arbitrary origin, or plain three.js).
- **Sun/shadow:** use Cesium's built-in sun lighting and shadows, driven by date/time.
- **Flood:** a water slider (0–5 m) colours buildings by inundation depth and counts affected buildings. Labelled as illustrative.
- **Textures (stretch):** drape the source image on roofs (SingleRecon-style).

---

## 8. Engineering practices

```
configs/            experiment YAMLs, splits/, paths.yaml
src/sat2city/
  data/             readers, tile-ID parsing, splits, manifest
  models/           DAv2 wrapper, U-Net, losses
  eval/             metrics (+ tests)
  geo/              instances, per-building heights, LoD1, export
  train.py  infer.py  export_onnx.py
api/                FastAPI + Dockerfile
web/                CesiumJS viewer
notebooks/          exploration only
tests/
docs/               PLAN, RESEARCH, DATA, DECISIONS, LOG
```

- Python 3.11 venv; **CUDA PyTorch** (the installed build is currently CPU-only).
- Config-driven runs, fixed seeds, git commit recorded per run; MLflow (local file store under `C:\sat2city-data\mlruns`).
- `docs/LOG.md`: dated lab notebook. `docs/DECISIONS.md`: short decision records.
- Commit often with `save.ps1`.

---

## 9. Detailed plan — Phase 0 & Phase 1

### Phase 0 — Groundwork (Tue 17 – Fri 20 Nov)

| Day | Tasks | Done when |
|---|---|---|
| **Day 1 — Environment & downloads** | (1) Create the venv; install CUDA PyTorch + torchvision; check `torch.cuda.is_available()`. (2) Create the repo skeleton (§8), `configs/paths.yaml`, `C:\sat2city-data\`. (3) **You** download from IEEE DataPort: DFC19 *Track 1 RGB* (6.6 GB) + *Track 1 Reference* (8.2 GB), DFC23 *track2.zip* (3.9 GB). (4) Read and record licences/terms in `docs/DATA.md`. (5) Clone `pubgeo/dfc2019` and the DAv2 repo for reference | GPU visible in PyTorch; downloads started; licences recorded |
| **Day 2 — Data inspection** | (1) Parse filenames → city/tile/view; count views per tile; build `manifest.csv` with SHA-256. (2) Open samples in QGIS: AGL units/nodata, image↔AGL alignment, CLS codes, off-nadir lean. (3) Height histograms (all pixels, building pixels). (4) Check for geo-metadata (CRS/transform/RPC) → decide globe vs local 3D frame. (5) Same pass for DFC23 Track 2 (footprint format, nDSM GSD vs image GSD). (6) Write the three split files by **tile ID** | `docs/DATA.md` facts table; histograms saved; splits committed; geo decision logged |
| **Day 3 — Evaluation code + baselines** | (1) `eval/metrics.py` (§6) + unit tests. (2) Dataloader with tiling, nodata masking, augmentations. (3) Trivial baselines on the tile split. (4) **Zero-shot DAv2-S** + oracle affine alignment. (5) Profile DAv2-S forward/backward at 256/384/518 px: memory and s/iter; decide which runs go local vs cloud | `pytest` green; baseline table in `docs/LOG.md`; compute budget written down |
| **Day 4 — Quick risk tests + tracking** | (1) Export pretrained DAv2-S → ONNX FP32/FP16 (+ try INT8) and compare outputs. (2) Building instances from CLS → per-building median AGL → GeoJSON. (3) Render a hand-made GeoJSON with heights in CesiumJS (sun + shadows on). (4) Time the non-ML steps on one tile. (5) Set up MLflow and the run-config template. (6) **Gate G0** | Each quick test has a one-line result in `docs/LOG.md`; G0 decision recorded in `docs/DECISIONS.md` |

### Phase 1 — Rough end-to-end version (Sat 21 – Fri 27 Nov)

Goal: **one working pipeline from image to 3D city in the browser, however rough**, plus the first honest numbers.

| Day | Tasks | Done when |
|---|---|---|
| **21–22 Nov** | `train.py`: direct DAv2-S fine-tune (recipe §5.4), short run on the **tile split** (local, 518 px, a few epochs). Log to MLflow. Same quick run on the **random_image** split → first look at the leakage gap (S4 preview) | Two runs logged; tile-split vs random-split numbers in `LOG.md` |
| **23 Nov** | `infer.py`: full-tile prediction (sliding window + blending) → nDSM GeoTIFF/NPY. `geo/`: instances → per-building heights → LoD1 GeoJSON | One tile → GeoJSON with building heights |
| **24 Nov** | `export_onnx.py` (FP16) + check ONNX output against PyTorch. `api/`: FastAPI `POST /predict` (image → GeoJSON + stats). Dockerfile | `curl` returns GeoJSON from the container |
| **25–26 Nov** | `web/`: CesiumJS page loads the GeoJSON, extrudes buildings, adds a sun time slider (shadows), a basic flood slider (colour by height < water level) and a stats panel | Screen recording of image → 3D → shadows → flood |
| **27 Nov** | README update (pipeline GIF, first numbers, honest caveats); LinkedIn post #1; **start applying with the link**; plan Phase 2 runs (cloud vs local) | Public repo shows a working rough version |

### Later phases (summary)

| Phase | Dates | Goal |
|---|---|---|
| 2 — Core model | 28 Nov – 4 Dec | Full-res training on cloud; U-Net baseline; S1/S2 numbers; **G1** |
| 3 — Research + 3D | 5 – 11 Dec | S4 split study, S3 cross-city/DFC23; viewer polish; **G2** (S5 SynRS3D curve moved to v1.1) |
| 4 — Ship | 12 – 18 Dec | FP16/INT8 study (S6, **G3**), public demo (HF Spaces + static web), README with results, failure gallery (S7), 90-s video, article pitch |
| Stretch | Jan 2027 | LoRA-Base, multi-task mask head, offset-aware footprints, splat comparison, Indian-city demo |

---

## 10. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Dataset terms block a public demo or redistribution | Medium | High | Record terms Day 1; demo shows our predictions, never redistributes raw data |
| R2 | Evaluation bug / leakage gives misleading numbers | Medium | High | Tile-ID splits, unit tests, U-Net sanity check, S4 study |
| R3 | Off-nadir lean misaligns roofs and footprints | High | High | Image-space masks; G2 fallback; offset-aware methods as stretch |
| R4 | US-trained model fails on other regions | High | Medium | Report S3 honestly; DFC23 cross-domain test |
| R5 | INT8 collapses on the DINOv2 encoder | Medium | Low | FP16 is the default; INT8 is an experiment (G3) |
| R6 | SynRS3D licence or size problems | Medium | Medium | Check on Zenodo; use a subset; S5 can be dropped without hurting the MVP |
| R7 | DFC19 tiles lack usable geo-referencing for the globe | Medium | Medium | Local ENU 3D scene; still interactive |
| R8 | Local training too slow / thermal throttling | Medium | Low | Cloud GPU for long runs; checkpoint/resume |
| R9 | OneDrive syncing gigabytes | High if ignored | Medium | Data outside OneDrive |
| R10 | Scope creep | High | Medium | Stretch work only after Phase 4 |

---

## 11. Career & output strategy

- From Phase 1, link the rough version when applying; update the CV line at each phase.
- Milestone posts: rough end-to-end version → first honest numbers → split-leakage finding → public demo.
- Article pitches:
  1. "Your satellite height benchmark might be leaking: random vs tile vs cross-city splits"
  2. "Do synthetic cities save real labels? DAv2-S on SynRS3D"
  3. "From one satellite image to an interactive 3D city"
- Interview stories: the leakage finding, the off-nadir footprint problem, the FP16/INT8 trade-off.

---

## 12. Open questions (resolve in Phase 0)

1. DFC19 / DFC23 exact licence and terms text (Day 1).
2. Do DFC19 Track 1 tiles carry geo-metadata (CRS, transform, RPC)? → globe vs local frame (Day 2).
3. Confirm that DFC19 filenames encode tile IDs and count views per tile (Day 2). This is the basis of the leakage study.
4. SynRS3D licence on Zenodo (before Phase 3).
5. Kaggle vs Colab for the first cloud runs (Day 3, after profiling).
6. Final name: keep **Sat2City3D**?
