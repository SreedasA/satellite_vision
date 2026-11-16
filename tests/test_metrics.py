import numpy as np
import pytest

from sat2city.eval.metrics import (
    PixelAccumulator,
    binned_metrics,
    building_metrics,
    label_buildings,
    per_building_heights,
    pixel_metrics,
)


def test_perfect_prediction():
    gt = np.array([[0.0, 5.0], [10.0, 40.0]])
    m = pixel_metrics(gt.copy(), gt)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["acc_1m"] == 1
    assert m["delta1"] == m["delta2"] == m["delta3"] == 1


def test_hand_computed_errors():
    gt = np.array([0.0, 2.0, 4.0, 6.0])
    pred = np.array([1.0, 2.0, 2.0, 6.0])  # errors +1, 0, -2, 0
    m = pixel_metrics(pred, gt)
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 4))
    assert m["bias"] == pytest.approx(-0.25)
    assert m["si_rmse"] == pytest.approx(np.sqrt(5 / 4) / np.std(gt))
    assert m["acc_1m"] == pytest.approx(0.5)  # |err| < 1 only for the two zeros


def test_delta_ignores_low_gt_and_handles_zero_pred():
    gt = np.array([0.2, 10.0, 10.0])
    pred = np.array([5.0, 12.0, 0.0])  # 0.2 m pixel excluded; 12/10=1.2 ok; 0 -> huge ratio
    m = pixel_metrics(pred, gt, delta_min_gt=1.0)
    assert m["delta1"] == pytest.approx(0.5)


def test_nodata_and_mask():
    gt = np.array([-9999.0, 3.0, np.nan, 5.0])
    pred = np.array([100.0, 3.0, 7.0, 6.0])
    m = pixel_metrics(pred, gt, nodata=-9999.0)
    assert m["n"] == 2 and m["mae"] == pytest.approx(0.5)
    mb = pixel_metrics(pred, gt, mask=np.array([0, 0, 0, 1]), nodata=-9999.0)
    assert mb["n"] == 1 and mb["mae"] == pytest.approx(1.0)


def test_empty_selection():
    assert pixel_metrics(np.zeros(3), np.zeros(3), mask=np.zeros(3))["n"] == 0


def test_accumulator_matches_pooled_pixel_metrics():
    rng = np.random.default_rng(0)
    gts = [rng.uniform(0, 40, (8, 8)) for _ in range(3)]
    preds = [g + rng.normal(0, 2, g.shape) for g in gts]
    acc = PixelAccumulator()
    for p, g in zip(preds, gts):
        acc.update(p, g)
    pooled = pixel_metrics(np.concatenate(preds), np.concatenate(gts))
    got = acc.compute()
    for k in ("mae", "rmse", "si_rmse", "bias", "acc_1m", "delta1"):
        assert got[k] == pytest.approx(pooled[k], abs=1e-4)


def test_bins():
    gt = np.array([1.0, 15.0, 50.0])
    out = binned_metrics(gt + 1, gt)
    assert set(out) == {"0-10", "10-30", "30+"}
    assert all(v["n"] == 1 and v["mae"] == pytest.approx(1.0) for v in out.values())


def test_label_buildings_drops_small_blobs():
    mask = np.zeros((10, 10), bool)
    mask[1:5, 1:5] = True  # 16 px building
    mask[8, 8] = True  # 1 px speck
    labels = label_buildings(mask, min_pixels=4)
    assert labels.max() == 1 and (labels > 0).sum() == 16


def test_building_metrics_and_floors():
    labels = np.zeros((4, 8), int)
    labels[:, :4] = 1
    labels[:, 4:] = 2
    gt = np.where(labels == 1, 9.0, 30.0)  # 3 floors, 10 floors
    pred = np.where(labels == 1, 10.0, 24.0)  # 3 floors (exact), 8 floors (off by 2)
    assert per_building_heights(gt, labels) == {1: 9.0, 2: 30.0}
    m = building_metrics(pred, gt, labels)
    assert m["n_buildings"] == 2
    assert m["bld_mae"] == pytest.approx(3.5)
    assert m["floor_exact"] == pytest.approx(0.5)
    assert m["floor_pm1"] == pytest.approx(0.5)
