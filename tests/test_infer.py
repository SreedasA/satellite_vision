import numpy as np
import torch

from sat2city.infer import TILE, predict_large


class GreenModel(torch.nn.Module):
    """Predicts the normalised green channel: a known, position-independent answer."""

    def forward(self, x):
        return x[:, 1]


def test_predict_large_blends_without_seams():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (TILE + 400, TILE + 300, 3), dtype=np.uint8)
    rgb[..., 1] = 120  # constant green -> constant prediction everywhere
    out = predict_large(GreenModel(), rgb, device="cpu")
    assert out.shape == rgb.shape[:2]
    assert np.allclose(out, out[0, 0], atol=1e-3)  # no seams from tiling/blending


def test_predict_large_single_tile_path():
    rgb = np.full((300, 200, 3), 50, np.uint8)
    out = predict_large(GreenModel(), rgb, device="cpu")
    assert out.shape == (300, 200)
