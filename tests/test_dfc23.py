import numpy as np

from sat2city.data.dfc23 import NAME_RE, Tile, instance_labels, make_splits, rle_string_to_counts, rle_to_mask


def test_name_parsing():
    m = NAME_RE.match("GF2_Brasilia_-15.8652_-47.9337.tif")
    assert m and m["city"] == "Brasilia" and float(m["lon"]) == -47.9337
    assert NAME_RE.match("SV_NewDelhi_28.6100_77.2000.tif")["city"] == "NewDelhi"


def test_instance_labels():
    rings = [[1, 1, 5, 1, 5, 5, 1, 5], [10, 10, 14, 10, 14, 14, 10, 14]]
    lab = instance_labels(rings, shape=(20, 20))
    assert set(np.unique(lab)) == {0, 1, 2}
    assert (lab == 1).sum() == 16


def _encode_rle_string(counts):
    """Reference encoder (pycocotools rleToString) used to round-trip the decoder."""
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def test_rle_round_trip():
    mask = np.zeros((6, 5), bool)
    mask[1:4, 2:4] = True
    mask[5, 0] = True
    flat = mask.T.ravel()  # column-major
    counts, val, run = [], False, 0
    for v in flat:
        if v != val:
            counts.append(run)
            val, run = v, 0
        run += 1
    counts.append(run)
    assert rle_string_to_counts(_encode_rle_string(counts)) == counts
    assert (rle_to_mask(counts, (6, 5)) == mask).all()
    lab = instance_labels([rle_to_mask(counts, (6, 5))], shape=(6, 5))
    assert (lab > 0).sum() == mask.sum()


def _tiles():
    out, i = [], 0
    for city, (lat0, lon0) in {"Berlin": (52.5, 13.4), "NewDelhi": (28.6, 77.2)}.items():
        for a in range(6):
            for b in range(6):
                out.append(Tile(f"SV_{city}_{lat0 + a * 0.004:.4f}_{lon0 + b * 0.004:.4f}.tif", "SV", city,
                                lat0 + a * 0.004, lon0 + b * 0.004, i))
                i += 1
    return out


def test_block_split_has_no_shared_blocks():
    tiles = _tiles()
    sp = make_splits(tiles, block_deg=0.01)["block"]
    block_of = {t.name: t.block(0.01) for t in tiles}
    parts = {k: {block_of[n] for n in v} for k, v in sp.items()}
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["val"])
    assert sum(map(len, sp.values())) == len(tiles)


def test_cross_city_holds_out_new_delhi():
    sp = make_splits(_tiles())["cross_city"]
    assert sp["test"] and all("NewDelhi" in n for n in sp["test"])
    assert not any("NewDelhi" in n for n in sp["train"] + sp["val"])
