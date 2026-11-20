# Literature & Dataset Review (Phase 0 input)

> Compiled 16 Nov 2026 from IEEE Xplore (signed in), IEEE DataPort, arXiv and author repositories.
> "Abstract only" means only the abstract was read; numbers should be checked against the full paper before citing them in public.

---

## 1. Closest prior work: what already exists

| Paper | Venue | What it does | Key numbers | Read |
|---|---|---|---|---|
| **Depth2Elevation** (Hong et al.) | IEEE TGRS 2025 | First use of Depth Anything (ViT-B) for remote-sensing height. Adds a scale modulator + resolution-agnostic decoder. Loss = MSE + SiLog + Gradient | **DFC2019:** direct DAM fine-tune MAE 1.537 / RMSE 2.603 m; Depth2Elevation 1.138 / 2.450 m; IMG2HEIGHT 2.728; HTC-DC 3.341 (their rerun). **GAMUS→DFC19 cross-dataset:** RMSE 2.623 m. Training: 4× RTX 4090, batch 16, AdamW, lr 5e-6 constant, 50 epochs, train 518 px, test 1022 px | Full text (author manuscript, AUT repo) |
| Height Estimation via Transfer Learning from MDE models (Tsukanov & Chekanin) | RusAutoCon 2025 | Full transfer of DAv2 to aerial height (DFC18, Potsdam, Vaihingen) | Keeping the depth decoder gives a 7.2% lower MAE than the backbone-only variant | Abstract only |
| Probabilistic Floor Count Inference using Depth Foundation Models (Sankhe et al.) | IGARSS 2026 | **Zero-shot DAv2** → depth aggregated over footprints → probabilistic floor model calibrated from partial labels + graph Laplacian smoothing. Tiled geospatial pipeline | 63% floor-count accuracy (Florida) | Abstract only |
| Instance-Level Building Height by Adapting a VFM (Zhao et al.) | IGARSS 2026 | SAM2 encoder + LoRA + Mask2Former + AdaBins; joint instances + height | mAP 36.2, **MAE 3.23 m per building** (Google Earth imagery) | Abstract only |
| SingleRecon (LoD1 from one off-nadir image) | IEEE JSTARS 2024 | Roof polygons + roof-to-footprint offsets → LoD1 models with textures taken from the image | **Average height error 3.3 m** | Abstract only (open access) |
| BDH-Net (instance-level multitask, off-nadir) | IEEE TGRS 2025 | Roof-to-footprint offset vectors; BDH dataset from Google Earth. Code: github.com/wishx98/BDHNet | — | Abstract only |
| Weak supervision from imperfect labels (Chen, Shi, Zhu) | IEEE JSTARS 2026 (open access) | Uses noisy out-of-domain labels; balanced soft losses + ordinal constraints for the long tail | RMSE −22.9% (DFC23), −18.6% (GBH) vs baselines | Abstract only |
| Prior2DSM (test-time adaptation) | IEEE TGRS 2026 | DINOv3 features + MDE + LoRA test-time optimisation; spatially varying scale/shift | RMSE −41% vs global affine rescaling | Abstract only |
| SynRS3D (Song et al.) | NeurIPS 2024 | 69,667 synthetic 512² images, GSD 0.09–1 m; DINOv2 ViT-L + DPT; RS3DAda unsupervised domain adaptation. Targets include **JAX & OMA (DFC19)** | T.D.1 avg: real-trained MAE 1.272 / RMSE 3.363; synthetic-only 2.557 / 5.617; RS3DAda 2.148 / 4.921. FT/JT low-data gains shown only in a figure. **Dataset licence not stated** | Full text (arXiv HTML) |
| DAv2-Small on Jetson Nano (UAV MDE) | ETAAV 2025 | PyTorch → ONNX → simplify → replace LayerNorm, fix interpolation → TensorRT | **FP16: ~66% faster with minimal accuracy loss** | Abstract only |
| DFC23 outcome paper | IEEE JSTARS 2024 (open access) | Contest summary; data openly available | 17 cities, 6 continents; 0.5–1 m optical (+SAR); nDSM ~2 m GSD | Partial full text |

Other relevant: MPG-Net (JSTARS 2025; RMSE 1.613 m Vaihingen, 1.947 m Potsdam), BuildMamba (IGARSS 2025; DFC23 δ1 0.872), MAMHESet multi-angle dataset (JSTARS 2025), SSRecon single-image 3DGS (JSTARS 2026), diffusion+LoRA semantic 3D (TGRS 2026), HTC-DC Net (TGRS 2023), GlobalBuildingAtlas (5.5 m global LoD1 height RMSE).

---

## 2. What this means for Sat2City3D

1. **Fine-tuning Depth Anything for remote-sensing height is not new** (Depth2Elevation 2025, RusAutoCon 2025, IGARSS 2026 floor counts). Our project must **not** claim "first DAv2 for satellite height".
2. **A likely evaluation weakness in prior DFC19 results.** DFC19 Track 1 has *dozens of images (different dates/angles) per 500 m tile*, 111 training tiles, and about 2,783 images. Depth2Elevation split those 2,783 images **2,226 / 557 without saying the split is by tile**. If the split was random, the same locations appear in both train and test, so the reported numbers are probably too optimistic. *(Inference from the paper text; verify by checking image→tile IDs.)*
3. **Our positioning (honest contributions):**
   - **Leakage-aware benchmark:** compare random-image vs tile-level vs cross-city (JAX↔OMA) splits for DAv2-S. *How much does the split inflate the numbers?*
   - **Per-building and floor-level metrics** feeding an actual **LoD1 3D product**, not only pixel RMSE.
   - **Synthetic label-efficiency curve** for DAv2-S (10/25/50/100% real data, with vs without SynRS3D pre-training). SynRS3D shows this only as a figure, for ViT-L, and not per building.
   - **Deployment study:** FP16/INT8 ONNX (TensorRT later), with accuracy measured after export.
   - **Interactive 3D viewer** with sun/shadow and flood simulation.
4. **Realistic accuracy bars:**
   - Pixel level (DFC19): about 2.45–2.60 m RMSE is reported **under a probably leaky split**. With an honest tile split, expect worse.
   - Per building: about **3.2–3.3 m MAE** is the current level in other datasets (SAM2+LoRA; SingleRecon).
   - Floors: 63% exact floor-count accuracy (zero-shot DAv2, partial labels).
5. **Proven training recipe to start from:** AdamW, lr ≈ 5e-6 (constant), loss = MSE + SiLog + gradient, train ≈ 518 px / test ≈ 1022 px, flips + colour jitter + blur.
6. **Deployment order:** FP16 first (shown to be close to lossless for DAv2-S); INT8 is an experiment.
7. **Off-nadir geometry matters.** DFC19 Track 1 images are *unrectified*, and roofs shift relative to footprints. OSM footprints will **not** line up with the image. Get footprints from the image/labels (building class masks) or use offset-aware methods (SingleRecon/BDH-Net). Getting this wrong ruins the per-building metrics.

---

## 3. Datasets: verified facts

| Dataset | Access / licence | Size | Content | Use |
|---|---|---|---|---|
| **DFC2019 / US3D Track 1** — IEEE DataPort DOI 10.21227/c6tm-vw12 | Open access; DataPort licence field: **Creative Commons Attribution**. Acknowledge JHU/APL & IARPA and cite Le Saux et al. 2019. Older terms mention approval for publications, so re-read on download | RGB 6.59 GB + Reference 8.17 GB (MSI 30 GB, not needed) | WorldView-3 pan-sharpened RGB ~0.3 m GSD (corrected 20 Nov; 1.3 m is the MSI band), **unrectified single views**; **AGL (nDSM) labels**; semantic classes (building, elevated road, high veg., ground, water); Jacksonville + Omaha, ~100 km² total; 111 train tiles (val/test ground truth not released) | **Primary benchmark**; cross-city JAX↔OMA test |
| **DFC2023 Track 2** — IEEE DataPort DOI 10.21227/mrnt-8w27 | DataPort listing says CC Attribution (search result); outcome paper possibly CC BY-NC-ND, so **verify on the page** | track2.zip 3.88 GB (+ test 0.88 GB) | Optical 0.5–1 m (+SAR), **building footprints + nDSM (~2 m)**, 17 cities / 6 continents; metric = threshold accuracy (δ) | **Second benchmark**: cross-domain test, per-building eval (has footprints) |
| SynRS3D — Zenodo 13905264; code github.com/JTRNEO/SynRS3D | **Licence not stated in the paper**; check Zenodo | 69,667 × 512² | Synthetic RGB + nDSM + land cover, GSD 0.09–1 m | Synthetic pre-training study |
| GAMUS — via EarthNets | Licence unclear | 8,724 × 1024², 0.33 m, 5 cities | RGB + nDSM + semantics | Optional only |
| GBH (HTC-DC Net) | Public availability unconfirmed | 20,532 × 256², 3 m | Planet RGB + nDSM | Optional only |

---

## Sources
- IEEE Xplore: Depth2Elevation (10978076), RusAutoCon transfer (11177401), floor count IGARSS 2026 (11730282), SAM2 instance height IGARSS 2026 (11730425), SingleRecon (10723241), BDH-Net (11087657), weak supervision (11498687), Prior2DSM (11755060), DAv2 Jetson (11212965), DFC23 outcome (10535214), MPG-Net (11048889), BuildMamba (11242380), MAMHESet (11196932), SSRecon (11493928), diffusion semantic 3D (11346819)
- Depth2Elevation author manuscript: https://openrepository.aut.ac.nz/items/4d5713c5-f7ea-4a08-9a73-4c2555e29743/full
- SynRS3D: https://arxiv.org/html/2406.18151v2 · https://zenodo.org/records/13905264
- DFC2019 DataPort: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019 · baseline code: https://github.com/pubgeo/dfc2019
- DFC2023 DataPort: https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest-large-scale-fine-grained-building-classification
- GAMUS: https://arxiv.org/abs/2305.14914 · Depth Any Canopy: https://github.com/DarthReca/depth-any-canopy (Apache-2.0; Small/Base checkpoints on Hugging Face)
