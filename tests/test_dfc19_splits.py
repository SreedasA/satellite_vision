from pathlib import Path

from sat2city.data.dfc19 import index_samples, leakage_report, make_splits


def _fake_dataset(root: Path, cities=("JAX", "OMA"), tiles=10, views=4) -> None:
    for city in cities:
        for t in range(tiles):
            for v in range(views):
                for kind in ("RGB", "AGL", "CLS"):
                    (root / f"{city}_{t:03d}_{v:03d}_{kind}.tif").touch()
    (root / "README.tif").touch()  # does not match the naming pattern


def test_index_and_unmatched(tmp_path):
    _fake_dataset(tmp_path)
    samples, unmatched = index_samples(tmp_path)
    assert len(samples) == 2 * 10 * 4
    assert unmatched == ["README.tif"]


def test_tile_and_cross_city_splits_do_not_leak(tmp_path):
    _fake_dataset(tmp_path)
    samples, _ = index_samples(tmp_path)
    splits = make_splits(samples, seed=0)
    for name in ("tile", "cross_city_jax2oma"):
        assert leakage_report(samples, splits[name]) == {"train&val": 0, "train&test": 0, "val&test": 0}
        assert sum(len(v) for v in splits[name].values()) == len(samples)
    assert all(i.startswith("OMA") for i in splits["cross_city_jax2oma"]["test"])


def test_random_image_split_leaks(tmp_path):
    _fake_dataset(tmp_path, views=6)
    samples, _ = index_samples(tmp_path)
    report = leakage_report(samples, make_splits(samples, seed=0)["random_image"])
    assert report["train&test"] > 0
