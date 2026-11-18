import numpy as np
import pytest
import rasterio
import torch

from sat2city.data.datasets import HeightDataset, multiple_of_14
from sat2city.eval.benchmark import BuildingPool, sample_ids
from sat2city.models.losses import gradient_loss, height_loss, silog_loss


def test_losses_zero_for_perfect_prediction():
    t = torch.rand(2, 16, 16) * 20
    m = torch.ones_like(t)
    total, parts = height_loss(t.clone(), t, m.bool(), w_mse=1.0)
    assert float(total) == pytest.approx(0.0, abs=1e-3)
    assert float(gradient_loss(t, t, m)) == 0.0


def test_losses_respect_mask_and_constant_offset():
    t = torch.zeros(1, 8, 8)
    p = torch.full_like(t, 2.0)
    m = torch.zeros_like(t)
    m[..., :4, :] = 1
    _, parts = height_loss(p, t, m.bool())
    assert parts["l1"] == pytest.approx(2.0)
    assert parts["grad"] == pytest.approx(0.0)  # constant offset has no gradient error
    assert float(silog_loss(p, t, m)) > 0


def test_multiple_of_14():
    assert multiple_of_14(1024) == 1022 and multiple_of_14(512) == 518


def _write(path, arr):
    arr = np.asarray(arr)
    bands = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1],
                       count=bands.shape[0], dtype=bands.dtype) as ds:
        ds.write(bands)


def test_height_dataset_train_and_eval(tmp_path):
    rng = np.random.default_rng(0)
    _write(tmp_path / "a_RGB.tif", rng.integers(0, 255, (3, 64, 80), dtype=np.uint8))
    h = rng.uniform(0, 30, (64, 80)).astype(np.float32)
    h[0, 0] = np.nan
    _write(tmp_path / "a_AGL.tif", h)
    items = [(str(tmp_path / "a_RGB.tif"), str(tmp_path / "a_AGL.tif"))]

    s = HeightDataset(items, train=True, crop=56)[0]
    assert s["image"].shape == (3, 56, 56) and s["height"].shape == (56, 56) and s["valid"].dtype == torch.bool

    small = HeightDataset(items, train=True, crop=98)[0]  # tile smaller than crop -> upsampled
    assert small["image"].shape == (3, 98, 98)

    e = HeightDataset(items, train=False)[0]
    assert e["image"].shape == (3, 70, 84) and e["height"].shape == (64, 80)
    assert not bool(e["valid"][0, 0]) and float(e["height"][0, 0]) == 0.0


def test_building_pool_bins():
    pool = BuildingPool()
    lab = np.zeros((4, 12), int)
    lab[:, :4], lab[:, 4:8], lab[:, 8:] = 1, 2, 3
    gt = np.choose(lab - 1, [5.0, 20.0, 40.0]).astype(float)
    pool.update(gt + 1, gt, lab)
    out = pool.compute()
    assert out["all"]["n"] == 3 and out["all"]["mae"] == pytest.approx(1.0)
    assert out["0-10"]["n"] == out["10-30"]["n"] == out["30+"]["n"] == 1


def test_sample_ids_matches_baseline_rule():
    import random

    ids = [f"t{i}" for i in range(500)]
    assert sample_ids(ids, 150, seed=0) == random.Random(0).sample(ids, 150)
