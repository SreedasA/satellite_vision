# Data: sources, licences, conventions

Local storage: `S:\sat2city-data\` (outside OneDrive; set in `configs/paths.yaml`, override with `SAT2CITY_DATA`).

| Folder | Contents |
|---|---|
| `raw/` | Original zips, never modified |
| `processed/` | Extracted rasters, manifests, inspection outputs, baselines |
| `checkpoints/`, `mlruns/` | Model weights, MLflow store |
| `cache/` | Hugging Face model cache |

---

## DFC2019 Track 1 (US3D): primary benchmark
- **Source:** IEEE DataPort, DOI 10.21227/c6tm-vw12. Files: `Train-Track1-RGB.zip` (6.59 GB), `Train-Track1-Truth.zip` (8.17 GB), `Validate-Track1.zip` (0.68 GB, images only).
- **Licence (DataPort field):** Creative Commons Attribution.
- **Required acknowledgement:** "The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest."
- **Citation:** Le Saux et al., 2019 IEEE GRSS Data Fusion Contest, IEEE GRSS Magazine 2019.
- **Older contest terms:** these mention case-by-case approval for *scientific publications*. Sat2City3D is a portfolio project. Before any formal paper submission, re-check the terms.
- **Our public use:** metrics, our own predictions, and a few attributed sample tiles. **No redistribution** of the raw dataset.
- **Files (from the official baseline code):**
  - `<CITY>_<TILE>_<VIEW>_RGB.tif`
  - `..._AGL.tif`: height above ground in metres; NaN = nodata (the official metric maps NaN → -999)
  - `..._CLS.tif`: LAS codes 2 ground, 5 trees, 6 building, 9 water, 17 bridge/elevated road, 65 unlabeled
- **Geometry:** images are **unrectified single views** (off-nadir). AGL/CLS are given per view, in that view's image geometry. WorldView-3, ~1.3 m GSD (pan-sharpened).
- **Official metric:** mIoU-3 (a true positive needs the correct class *and* |AGL error| < 1 m).
- **Verified 2026-11-17:**
  - 2,783 triplets of 1024² (JAX 1,015, OMA 1,768).
  - **108 tiles; views per tile min 1 / median 24 / max 43.**
  - RGB uint8; AGL float32 with no NaNs in the sample, range −2.9…137 m.
  - **No georeferencing** (identity transform, no CRS/RPC).

## DFC2023 Track 2: cross-domain + per-building
- **Source:** IEEE DataPort, DOI 10.21227/mrnt-8w27. File: `track2.zip` (3.88 GB).
- **Licence:** **not shown on the DataPort page**; access is under the contest terms and conditions. Treat as research use, publish only metrics and our predictions, and verify before any redistribution.
- **Content (outcome paper):** SuperView-1 / Gaofen-2 / Gaofen-3 imagery at 0.5 / 0.8 / 1 m; nDSM from Gaofen-7 & WorldView stereo at ~2 m GSD; building annotations; 17 cities on 6 continents. Contest height metric: threshold accuracy (δ).
- **Verified 2026-11-17 (train set):**
  - Layout: `track2/{rgb,dsm,sar}/<SENSOR>_<City>_<lat>_<lon>.tif` plus `buildings_only_train.json` (COCO with string-encoded fields; 124,670 polygons + 483 compressed-RLE masks).
  - 1,773 tiles of 512², 0.5 m, **UTM-georeferenced**.
  - 11 cities: Berlin 566, Brasília 246, Copenhagen 216, Portsmouth 151, **New Delhi 131**, New York 108, San Diego 108, Rio 93, Barcelona 90, Sydney 34, São Luís 30.
  - The `dsm` folder holds **nDSM** heights (0…83 m).
  - `val/` has RGB+SAR only (no labels).
- **Footprint/height agreement:** 17% of footprints have a reference median < 2 m. Per-building evaluation keeps only footprints with median ≥ 2 m (DECISIONS D10).

## SynRS3D (v1.1, not used before v1.0)
- Zenodo 13905264, **CC BY 4.0**. 17 zips, 101.8 GB in total, with MD5 checksums. Per sample: `opt`, `gt_nDSM`, `gt_ss_mask`, …

---

## Metric conventions (see `src/sat2city/eval/metrics.py`)
- **Valid pixel:** finite ground truth (NaN/nodata excluded).
- **MAE, RMSE, SI-RMSE** (= RMSE / std(gt)) follow Depth2Elevation. Dataset numbers **pool pixels** across images (`PixelAccumulator`); they are not averages of per-image scores.
- **δk:** share of pixels with max(p/g, g/p) < 1.25^k, only where gt ≥ 1 m; predictions clamped at 1e-3.
- **acc_1m:** share of pixels with |error| < 1 m (the height test used by DFC19's mIoU-3).
- **Building pixels:** CLS == 6 (DFC19), or footprints (DFC23).
- **Per building:** median height inside each 8-connected building component (≥ 20 px). Floors = round(h / 3 m), minimum 1. Adjacent roofs can merge into one component (logged risk).
