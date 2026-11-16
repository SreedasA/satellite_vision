import numpy as np
import pytest

from sat2city.geo.lod1 import instances_to_features, pixel_to_lonlat


def test_pixel_to_lonlat_scale():
    lon, lat = pixel_to_lonlat(0, 0, -81.66, 30.33, gsd=1.3)
    assert (lon, lat) == pytest.approx((-81.66, 30.33))
    _, lat_s = pixel_to_lonlat(0, 1000, -81.66, 30.33, gsd=1.0)  # 1 km south
    assert (30.33 - lat_s) * 111_320 == pytest.approx(1000, rel=0.01)


def test_instances_to_features():
    labels = np.zeros((20, 20), int)
    labels[2:8, 2:8] = 1  # tall building
    labels[12:18, 12:18] = 2  # too low -> dropped
    height = np.where(labels == 1, 12.0, 1.0)
    feats = instances_to_features(labels, height, gsd=1.3, origin_lonlat=(-81.66, 30.33), min_height=2.0)
    assert len(feats) == 1
    props = feats[0]["properties"]
    assert props["height_m"] == 12.0 and props["floors"] == 4
    assert props["area_m2"] == pytest.approx(36 * 1.3**2, rel=0.05)
    assert feats[0]["geometry"]["type"] == "Polygon"
