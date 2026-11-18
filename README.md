# Sat2City3D

**One satellite image → building heights → an explorable 3D city in the browser**, with sun/shadow and flood-level views.

A fine-tuned depth foundation model (Depth Anything V2-Small) predicts height above ground for every pixel of a single RGB satellite image. The heights are combined with building footprints into LoD1 blocks (extruded footprints) and shown in a CesiumJS viewer.

> **Status: v0.2 (work in progress, 18 Nov 2026).** The full pipeline runs end to end: training → evaluation → inference → API → 3D viewer. Accuracy work (Phase 2) is next. Target release: **v1.0 on 18 Dec 2026**. See the [roadmap](docs/ROADMAP.md).

---

## What works today

| Component | What it does |
|---|---|
| `sat2city.train` | Fine-tunes DAv2 on height maps (bf16, MLflow tracking, best checkpoint, test on a fixed sample) |
| `sat2city.infer` | Image → height GeoTIFF → per-building LoD1 GeoJSON (tiled prediction with seam blending) |
| `api/` (FastAPI) | `POST /predict` (upload an image), `GET /benchmark/{dataset}/{tile}` (predict a benchmark tile), tile image + bounds |
| `web/index.html` (CesiumJS) | Satellite image drape, *rise* slider (buildings grow out of the image), sun/shadow time slider, flood slider with water surface and affected-building counts |
| `src/sat2city/eval` | Pixel, building-pixel, per-building and floor-level metrics, broken down by height bin |
| `tests/` | 31 unit tests |

---

## First results (v0.2: DAv2-Small, 5 epochs, about 10–16 min per run on a laptop RTX 3050)

All test sets are **geographically separate** from training (no shared tiles or blocks).

| Dataset · split | RMSE (m) | Building RMSE (m) | Per-building MAE (m) | Floors within ±1 |
|---|---|---|---|---|
| DFC2019 (Jacksonville + Omaha) · held-out tiles | **3.38** | 7.69 | 2.06 | 93% |
|   *baseline: predict training mean* | 5.15 | 10.97 | 1.97 | 90% |
| DFC2023 (11 cities) · held-out blocks | **5.74** | 7.89 | **3.28** | 79% |
|   *baseline: predict training mean* | 9.64 | 15.04 | 7.74 | 54% |
| DFC2023 · **New Delhi never seen in training** | 5.38 | 10.02 | 6.15 | 48% |

**What these numbers say, honestly:**
- After only 5 epochs, fine-tuning beats every baseline. Zero-shot Depth Anything barely beats predicting the mean, even with an oracle scale fit, so fine-tuning is essential.
- 3.28 m per-building MAE (DFC2023 blocks) is in the range of recent published results on other datasets (about 3.2–3.3 m).
- **On a city the model has never seen (New Delhi), the per-building error almost doubles.** Generalising to new cities is the real challenge.
- **Tall buildings are badly under-predicted:** about −8 m for 10–30 m buildings and −10 to −26 m above 30 m. Fixing this is the next milestone.
- DFC2019 has about 24 photos of each location (108 locations). Random image splits put the same places in train and test. This project only reports location-level splits. A controlled leakage study is planned for v1.0.

Full numbers, baselines and decisions: [docs/LOG.md](docs/LOG.md) · [docs/DECISIONS.md](docs/DECISIONS.md).

---

## Quick start

> Windows + Python 3.11. Large data lives **outside** the repo (default `S:/sat2city-data`; override with `SAT2CITY_DATA`).

```bash
python -m venv venv
venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
venv/Scripts/python -m pip install -r requirements.txt -e .
venv/Scripts/python -m pytest -q
```

1. **Data:** download DFC2019 Track 1 and DFC2023 Track 2 from IEEE DataPort (see [docs/DATA.md](docs/DATA.md)), extract them under `processed/`, then run `scripts/inspect_dfc19.py` and `scripts/inspect_dfc23.py`. These write the manifests and the leak-free split files in `configs/splits/`.

2. **Train:**
   ```bash
   HF_HUB_OFFLINE=1 python -m sat2city.train --dataset dfc23 --split block --epochs 5
   ```

3. **Serve:**
   ```bash
   SAT2CITY_CKPT=<data>/runs/<run>/best.pt uvicorn api.main:app --port 8000
   ```

4. **View:** serve `web/` with any static server, then open
   `index.html?api=http://127.0.0.1:8000&dataset=dfc23&tile=<tile name>`.

---

## Repository layout

```
configs/      paths, training recipes, split files (by tile / spatial block / city)
src/sat2city/ data loaders · models (DAv2 wrapper, losses) · eval (metrics, benchmarks) · geo (LoD1) · train · infer
api/          FastAPI service
web/          CesiumJS viewer (+ Phase 0 quick-test pages)
scripts/      inspection, baselines, profiling, ONNX test
docs/         PLAN · ROADMAP · RESEARCH · DATA · DECISIONS · LOG
```

## Limitations
- Heights are **model estimates**. They are least reliable for tall buildings and unseen cities (see the results above).
- The flood view is **illustrative** (water level vs building height), **not a hydraulic simulation**.
- Uploaded images use *height-based blobs* as footprints, which can include trees. Benchmark tiles use reference footprints.
- DFC2019 images aren't georeferenced, so they are placed approximately in a local frame. DFC2023 tiles are georeferenced (UTM).

## Data and acknowledgements
- **DFC2019 / US3D** (IEEE DataPort, CC-BY): *The authors would like to thank the Johns Hopkins University Applied Physics Laboratory and IARPA for providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee for organizing the Data Fusion Contest.* Le Saux et al., 2019.
- **DFC2023 Track 2** (IEEE DataPort): used under the contest terms. Redistribution terms are still being verified, so no DFC2023 imagery or labels are published in this repo.
- **Depth Anything V2-Small** (Apache-2.0).
- No raw dataset files are redistributed here.

## Docs
[Plan](docs/PLAN.md) · [Roadmap](docs/ROADMAP.md) · [Research review](docs/RESEARCH.md) · [Data](docs/DATA.md) · [Decisions](docs/DECISIONS.md) · [Lab log](docs/LOG.md)
