# Phase 3 experiment configs

## Controlled leakage study (DECISIONS D13, D17)

`scripts/make_leakage_splits.py` writes `configs/splits/dfc19_leak_{a,b}.json`:

- **Test (shared):** every other view of each DFC2019 tile-split test tile with at least 8 views (14 tiles, 205 views).
- **Model A (no leak):** the tile split's training set (1,997 views). This is the existing `p2d_dfc19_tile` run, re-scored on the shared test views:
  `python scripts/evaluate_run.py p2d_dfc19_tile --split leak_a --n-test 205`
- **Model B (leak):** the same training set with 200 random views swapped for the 200 *other* views of the test tiles. Same size, same recipe, same seed: the only difference is whether the model has seen other dates of the test locations.

Run model B (~1.5 h):

    scripts\train.cmd --config configs/train/p3/leak.yaml --split leak_b --run-name p3_leak_b

## Cross-city transfer: Jacksonville → Omaha (~1 h)

Config D trained on Jacksonville only, tested on Omaha:

    scripts\train.cmd --config configs/train/p2/d_mse.yaml --dataset dfc19 --split cross_city_jax2oma --run-name p3_dfc19_jax2oma

## Analysis scripts (no training)

| Script | Question |
|---|---|
| `scripts/calibration_check.py RUN` | Is the error one scale factor that a few labelled tiles fix? |
| `scripts/failure_gallery.py RUN` | Where does it fail (surface class, edges, size, tall buildings, sensor)? Pictures in `S:/sat2city-data/gallery/RUN` |
| `scripts/shift_check.py RUN` | Is part of the building error image/reference misregistration? |
| `scripts/export_demo.py --checkpoint …` | Static DFC2019 viewer demo in `web/demo/` (no API needed) |
