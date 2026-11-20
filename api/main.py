"""Sat2City3D inference API (Phase 1: PyTorch backend; ONNX backend comes in Phase 4).

Run:  SAT2CITY_CKPT=S:/sat2city-data/runs/<run>/best.pt uvicorn api.main:app --port 8000
      Optional per-dataset checkpoints for benchmark tiles: SAT2CITY_CKPT_DFC19, SAT2CITY_CKPT_DFC23.
Endpoints:
  GET  /health                      model + device info
  POST /predict  (multipart file)   RGB image -> LoD1 GeoJSON + stats
       query: footprints=height (only option for uploads), min_height, gsd, origin_lon, origin_lat
  GET  /benchmark/{dataset}/tiles    test tiles for the viewer's picker (grouped by city / split)
  GET  /benchmark/{dataset}/{tile}  benchmark tile -> GeoJSON with reference footprints
CORS is open so the static Cesium viewer (served from another origin) can call it.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from sat2city.config import load_paths
from sat2city.eval.benchmark import make_bench
from sat2city.eval.metrics import label_buildings
from sat2city.infer import buildings_for, load_model, predict_large

MAX_PIXELS = 4096 * 4096

app = FastAPI(title="Sat2City3D", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_state: dict = {}


def model(dataset: str | None = None):
    """Default model (SAT2CITY_CKPT), or a dataset-specific one (SAT2CITY_CKPT_DFC19 / _DFC23) if set."""
    ckpt = (dataset and os.environ.get(f"SAT2CITY_CKPT_{dataset.upper()}")) or os.environ.get("SAT2CITY_CKPT")
    models = _state.setdefault("models", {})
    if ckpt not in models:
        if not ckpt or not Path(ckpt).exists():
            raise HTTPException(503, "SAT2CITY_CKPT is not set to an existing checkpoint")
        _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        models[ckpt] = load_model(ckpt, _state["device"])
    _state["ckpt"] = ckpt
    return models[ckpt]


def summary(feats: list[dict]) -> dict:
    hs = np.array([f["properties"]["height_m"] for f in feats]) if feats else np.zeros(0)
    return {"buildings": len(feats), "height_median_m": round(float(np.median(hs)), 1) if hs.size else None,
            "height_max_m": round(float(hs.max()), 1) if hs.size else None,
            "floors_total": int(sum(f["properties"]["floors"] for f in feats))}


@app.get("/health")
def health():
    loaded = bool(_state.get("models"))
    return {"status": "ok", "model_loaded": loaded, "device": _state.get("device"),
            "checkpoint": _state.get("ckpt") or os.environ.get("SAT2CITY_CKPT")}


@app.post("/predict")
async def predict(file: UploadFile = File(...), min_height: float = Query(2.5, ge=0),
                  gsd: float = Query(1.3, gt=0), origin_lon: float = -81.6581, origin_lat: float = 30.3270):
    data = await file.read()
    suffix = Path(file.filename or "upload").suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix or '.png'}"
        path.write_bytes(data)
        try:
            if suffix in (".tif", ".tiff"):
                from sat2city.data.io import read_rgb

                rgb = read_rgb(path)
            else:
                rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:  # noqa: BLE001 - surface any decode failure to the client
            raise HTTPException(400, f"could not read image: {e}") from e
        if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
            raise HTTPException(413, "image too large (max 4096x4096)")
        t0 = time.perf_counter()
        height = predict_large(model(), rgb, _state["device"])
        labels = label_buildings(height >= min_height, min_pixels=20)
        if suffix not in (".tif", ".tiff"):
            from sat2city.geo.lod1 import instances_to_features

            feats, placement = instances_to_features(labels, height, gsd, (origin_lon, origin_lat), min_height=min_height), "local"
        else:
            feats, placement = buildings_for(height, path, labels, (origin_lon, origin_lat), gsd, min_height)
    return {"type": "FeatureCollection", "features": feats,
            "metadata": summary(feats) | {"footprints": "height-blobs", "placement": placement,
                                          "seconds": round(time.perf_counter() - t0, 2)}}


LOCAL_ORIGIN = {"JAX": (-81.6581, 30.3270), "OMA": (-95.9345, 41.2565)}  # approximate, DFC19 is not georeferenced


def _bench_tile(dataset: str, tile: str):
    if dataset not in ("dfc19", "dfc23"):
        raise HTTPException(404, "unknown dataset")
    key = f"bench_{dataset}"
    if key not in _state:
        _state[key] = make_bench(dataset, load_paths()["processed"])
    bench = _state[key]
    rgb_path = bench.root / f"{tile}_RGB.tif" if dataset == "dfc19" else bench.root / "rgb" / tile
    if not rgb_path.exists():
        raise HTTPException(404, "unknown tile")
    return bench, rgb_path


def _bounds(rgb_path: Path, shape, origin, gsd) -> list[float]:
    """[west, south, east, north] in degrees for draping the tile image under the buildings."""
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.warp import transform_bounds

    from sat2city.geo.lod1 import pixel_to_lonlat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(rgb_path) as src:
            if src.crs:
                return [round(v, 7) for v in transform_bounds(src.crs, "EPSG:4326", *src.bounds)]
    w_lon, n_lat = pixel_to_lonlat(0, 0, *origin, gsd)
    e_lon, s_lat = pixel_to_lonlat(shape[1], shape[0], *origin, gsd)
    return [round(float(v), 7) for v in (w_lon, s_lat, e_lon, n_lat)]


# Test tiles offered in the viewer. Whether a tile is unseen depends on the checkpoint: the New Delhi
# tiles are unseen for the cross_city model, the held-out blocks for the block model.
PICKER = {"dfc23": (("cross_city", "New Delhi (unseen city)"), ("block", "held-out blocks")),
          "dfc19": (("tile", "held-out tiles"),)}


@app.get("/benchmark/{dataset}/tiles")
def benchmark_tiles(dataset: str):
    from sat2city.eval.benchmark import load_split

    if dataset not in PICKER:
        raise HTTPException(404, "unknown dataset")
    out, seen = [], set()
    for split, label in PICKER[dataset]:
        for t in sorted(load_split(dataset, split)["test"]):
            if t in seen:
                continue
            seen.add(t)
            city = t.split("_")[1] if dataset == "dfc23" else t[:3]
            out.append({"tile": t, "group": f"{city} · {label}"})
    return out


@app.get("/benchmark/{dataset}/{tile}")
def benchmark_tile(dataset: str, tile: str, min_height: float = 2.0):
    bench, rgb_path = _bench_tile(dataset, tile)
    t0 = time.perf_counter()
    rgb, _, _, labels = bench.load(tile)
    height = predict_large(model(dataset), rgb, _state["device"])
    origin = LOCAL_ORIGIN.get(tile[:3], LOCAL_ORIGIN["JAX"])
    feats, placement = buildings_for(height, rgb_path, labels, origin, 1.3, min_height)
    return {"type": "FeatureCollection", "features": feats,
            "metadata": summary(feats) | {"footprints": "reference", "placement": placement, "tile": tile,
                                          "dataset": dataset, "checkpoint": Path(_state["ckpt"]).parent.name,
                                          "bounds": _bounds(rgb_path, height.shape, origin, 1.3),
                                          "seconds": round(time.perf_counter() - t0, 2)}}


@app.get("/benchmark/{dataset}/{tile}/image.png")
def benchmark_image(dataset: str, tile: str):
    from fastapi.responses import Response

    from sat2city.data.io import read_rgb

    _, rgb_path = _bench_tile(dataset, tile)
    rgb = read_rgb(rgb_path)
    # 1-px transparent border: Cesium clamps edge texels on coarse globe tiles, which smears
    # the outermost pixels across the globe; transparent edges make that smear invisible.
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    rgba[[0, -1], :, 3] = 0
    rgba[:, [0, -1], 3] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-cache"})
