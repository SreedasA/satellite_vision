import io

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api import main  # noqa: E402


class TallSquareModel(torch.nn.Module):
    """Predicts 12 m inside a bright square, 0 m elsewhere."""

    def forward(self, x):
        return (x[:, 0] > 1.0).float() * 12.0


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SAT2CITY_CKPT", "fake")
    monkeypatch.delenv("SAT2CITY_CKPT_DFC19", raising=False)
    monkeypatch.delenv("SAT2CITY_CKPT_DFC23", raising=False)
    monkeypatch.setattr(main, "_state", {"models": {"fake": TallSquareModel()}, "device": "cpu"})
    return TestClient(main.app)


def test_per_dataset_checkpoint(client, monkeypatch):
    other = TallSquareModel()
    main._state["models"]["dfc19.pt"] = other
    monkeypatch.setenv("SAT2CITY_CKPT_DFC19", "dfc19.pt")
    assert main.model("dfc19") is other and main.model("dfc23") is main._state["models"]["fake"]
    assert main.model() is main._state["models"]["fake"]


def test_tile_picker_lists_test_tiles(client):
    r = client.get("/benchmark/dfc23/tiles")
    assert r.status_code == 200
    tiles = r.json()
    assert tiles[0]["group"].startswith("NewDelhi") and len({t["tile"] for t in tiles}) == len(tiles)
    assert client.get("/benchmark/nope/tiles").status_code == 404


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_png_returns_buildings(client):
    img = np.zeros((140, 140, 3), np.uint8)
    img[30:80, 40:100] = 255
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    r = client.post("/predict?min_height=2.5", files={"file": ("tile.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["buildings"] == 1 and body["metadata"]["placement"] == "local"
    assert body["features"][0]["properties"]["height_m"] == pytest.approx(12.0, abs=0.5)


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_missing_checkpoint(monkeypatch):
    monkeypatch.setattr(main, "_state", {})
    monkeypatch.delenv("SAT2CITY_CKPT", raising=False)
    r = TestClient(main.app).post("/predict", files={"file": ("t.png", b"x", "image/png")})
    assert r.status_code in (400, 503)
