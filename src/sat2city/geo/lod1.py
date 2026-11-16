"""Building instances + height map -> LoD1 footprints (GeoJSON) ready for extrusion.

Coordinates: if the tile has no usable geo-transform (DFC19 Track 1 views are unrectified),
pixels are mapped to a local east/north frame in metres around an approximate origin and
then to lon/lat with a flat-earth approximation. This is labelled "approximate placement".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import transform

from sat2city.eval.metrics import per_building_heights

EARTH_R = 6_378_137.0


def pixel_to_lonlat(col, row, origin_lon: float, origin_lat: float, gsd: float):
    """Pixel (col, row) -> (lon, lat); works on scalars or arrays."""
    east, north = np.asarray(col) * gsd, -np.asarray(row) * gsd
    lat = origin_lat + np.degrees(north / EARTH_R)
    lon = origin_lon + np.degrees(east / (EARTH_R * math.cos(math.radians(origin_lat))))
    return lon, lat


def instances_to_features(
    labels: np.ndarray,
    height: np.ndarray,
    gsd: float,
    origin_lonlat: tuple[float, float],
    valid: np.ndarray | None = None,
    stat: str = "median",
    min_height: float = 2.0,
    simplify_px: float = 1.0,
) -> list[dict]:
    """One GeoJSON Feature per building with properties {id, height_m, floors, area_m2}."""
    heights = per_building_heights(height, labels, valid, stat)
    lon0, lat0 = origin_lonlat
    out = []
    for geom, value in features.shapes(labels.astype(np.int32), mask=labels > 0):
        bid = int(value)
        h = heights.get(bid)
        if h is None or h < min_height:
            continue
        poly = shape(geom).simplify(simplify_px)
        if poly.is_empty:
            continue
        poly_ll = transform(lambda c, r, z=None: pixel_to_lonlat(c, r, lon0, lat0, gsd), poly)
        out.append({
            "type": "Feature",
            "geometry": mapping(poly_ll),
            "properties": {
                "id": bid,
                "height_m": round(h, 2),
                "floors": max(1, round(h / 3.0)),
                "area_m2": round(poly.area * gsd * gsd, 1),
            },
        })
    return out


def instances_to_features_geo(
    labels: np.ndarray,
    height: np.ndarray,
    transform,
    crs,
    valid: np.ndarray | None = None,
    stat: str = "median",
    min_height: float = 2.0,
    simplify_m: float = 0.5,
) -> list[dict]:
    """Same as instances_to_features, for georeferenced rasters (e.g. DFC23 UTM tiles)."""
    from rasterio.warp import transform_geom

    heights = per_building_heights(height, labels, valid, stat)
    px_area = abs(transform.a * transform.e)
    out = []
    for geom, value in features.shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        bid = int(value)
        h = heights.get(bid)
        if h is None or h < min_height:
            continue
        poly = shape(geom).simplify(simplify_m)
        if poly.is_empty:
            continue
        out.append({
            "type": "Feature",
            "geometry": transform_geom(crs, "EPSG:4326", mapping(poly)),
            "properties": {
                "id": bid,
                "height_m": round(h, 2),
                "floors": max(1, round(h / 3.0)),
                "area_m2": round(float((labels == bid).sum()) * px_area, 1),
            },
        })
    return out


def write_geojson(feats: list[dict], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats, "metadata": meta or {}}))
