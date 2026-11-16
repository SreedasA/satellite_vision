# Sat2City3D

Turn a single satellite image into an explorable 3D city in the browser, with sun/shadow and flood water-level simulations.

- **Model:** Depth Anything V2 fine-tuned to predict building height (nDSM) from one RGB satellite image
- **3D:** footprints + predicted heights → LoD1 city model → CesiumJS viewer
- **Serving:** ONNX + FastAPI + Docker

Status: **planning**. Target release **v1.0 on 18 Dec 2026** (minimum: v0.8).

- [docs/PLAN.md](docs/PLAN.md): goals, success criteria, data and modelling strategy, risks
- [docs/ROADMAP.md](docs/ROADMAP.md): release versions, pinned stack, every step with best/worst case and fallbacks
- [docs/RESEARCH.md](docs/RESEARCH.md): literature and dataset review
