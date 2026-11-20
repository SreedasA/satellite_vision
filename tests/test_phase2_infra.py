import json
import random

import numpy as np
import pytest
import torch

from sat2city.config import REPO_ROOT, load_train_config
from sat2city.models.losses import height_loss, tall_weights
from sat2city.train import (
    DEFAULTS,
    load_checkpoint,
    lr_lambda,
    parse_args,
    resolve_config,
    rng_state,
    save_checkpoint,
    selection_score,
    set_rng_state,
)

P2 = REPO_ROOT / "configs" / "train" / "p2"


def test_config_inheritance_chain():
    d = load_train_config(P2 / "d_mse.yaml")
    assert d["epochs"] == 30 and d["schedule"] == "cosine" and d["select"] == "tall_aware"  # from A
    assert d["loss"]["w_tall"] == 1.0  # from B
    assert d["lr_head"] == 1.0e-4  # from C
    assert d["loss"]["w_mse"] == 0.05 and d["loss"]["w_l1"] == 1.0 and d["loss"]["w_grad"] == 0.5  # D + base
    assert d["model"].startswith("depth-anything")
    assert load_train_config(P2 / "unet_baseline.yaml")["model"] == "unet:resnet34"


def test_tall_weighted_loss_emphasises_tall_errors():
    target = torch.tensor([[[0.0, 30.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    err_tall = target + torch.tensor([[[0.0, 2.0]]])  # 2 m error on the 30 m pixel
    err_low = target + torch.tensor([[[2.0, 0.0]]])  # 2 m error on the ground pixel
    kw = dict(w_silog=0, w_grad=0, w_tall=1.0)
    _, tall = height_loss(err_tall, target, valid, **kw)
    _, low = height_loss(err_low, target, valid, **kw)
    assert tall["l1"] > low["l1"] * 3  # weights 1 vs 4
    assert float(tall_weights(torch.tensor(30.0), 1.0)) == pytest.approx(4.0)
    _, off = height_loss(err_tall, target, valid, w_silog=0, w_grad=0)
    assert off["l1"] == pytest.approx(1.0)  # unweighted mean when w_tall = 0


def test_lr_schedule():
    cfg = DEFAULTS | {"epochs": 10, "schedule": "cosine", "warmup_epochs": 1, "min_lr_ratio": 0.1}
    f = lr_lambda(cfg, steps_per_epoch=100)
    assert f(0) == pytest.approx(0.01) and f(99) == pytest.approx(1.0)  # linear warm-up
    assert f(100) == pytest.approx(1.0, abs=1e-3)
    assert f(999) == pytest.approx(0.1, abs=1e-3)  # decays to min_lr_ratio
    assert lr_lambda(DEFAULTS | {"epochs": 5}, 10)(37) == 1.0  # constant


def test_selection_score():
    val = {"building": {"rmse": 7.0},
           "per_building": {"10-30": {"n": 30, "mae": 8.0}, "30+": {"n": 10, "mae": 24.0}}}
    assert selection_score(val, DEFAULTS) == 7.0
    tall = selection_score(val, DEFAULTS | {"select": "tall_aware"})
    assert tall == pytest.approx(7.0 + (8 * 30 + 24 * 10) / 40)


def test_checkpoint_round_trip_restores_rng(tmp_path):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    save_checkpoint(tmp_path / "last.pt", rng=state, epoch=3, best=1.5, model={"w": torch.ones(2)})
    assert not (tmp_path / "last.tmp").exists()
    ck = load_checkpoint(tmp_path / "last.pt")
    set_rng_state(ck["rng"])
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == pytest.approx(expected)
    assert ck["epoch"] == 3 and torch.equal(ck["model"]["w"], torch.ones(2))


def test_resume_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("SAT2CITY_DATA", str(tmp_path))
    with pytest.raises(SystemExit, match="needs --run-name"):
        resolve_config(parse_args(["--resume"]))
    with pytest.raises(SystemExit, match="nothing to resume"):
        resolve_config(parse_args(["--resume", "--run-name", "missing"]))
    run = tmp_path / "runs" / "taken"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="already exists"):
        resolve_config(parse_args(["--run-name", "taken"]))


def test_resume_reads_saved_config_and_extends_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("SAT2CITY_DATA", str(tmp_path))
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"epochs": 30, "dataset": "dfc23", "model": "unet:resnet34",
                                                 "max_steps_per_epoch": 30}))
    save_checkpoint(run / "last.pt", epoch=12)
    cfg, name, run_dir, ck = resolve_config(parse_args(["--resume", "--run-name", "r", "--epochs", "40"]))
    assert cfg["epochs"] == 40 and cfg["model"] == "unet:resnet34" and ck["epoch"] == 12 and name == "r"
    assert cfg["max_steps_per_epoch"] == 30  # main() applies it when --max-steps-per-epoch isn't given


def test_seed_override_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("SAT2CITY_DATA", str(tmp_path))
    cfg, name, run_dir, ck = resolve_config(parse_args(["--config", str(P2 / "d_mse.yaml"), "--seed", "1",
                                                         "--run-name", "seed1"]))
    assert cfg["seed"] == 1 and ck is None
    assert json.loads((run_dir / "config.json").read_text())["seed"] == 1  # a resume reuses it
    default, *_ = resolve_config(parse_args(["--config", str(P2 / "d_mse.yaml"), "--run-name", "seed0"]))
    assert default["seed"] == 0


def test_eval_samples_do_not_depend_on_training_seed():
    from sat2city.eval.benchmark import sample_ids

    ids = [f"t{i}" for i in range(300)]
    assert sample_ids(ids, 150, seed=0) == sample_ids(ids, 150, seed=0)  # test sample seed is fixed in train.py


def test_unet_shapes_and_loading(tmp_path):
    from sat2city.infer import load_model
    from sat2city.models.unet import UNetHeight

    net = UNetHeight(pretrained=False).eval()
    with torch.no_grad():
        for hw in ((518, 518), (126, 98)):
            y = net(torch.randn(1, 3, *hw))
            assert y.shape == (1, *hw) and float(y.min()) >= 0
    assert any(n.startswith(net.encoder_prefix) for n, _ in net.named_parameters())

    # Padding is symmetric with a margin on every side (bottom/right-only padding caused border
    # spikes on DFC19 1022 px tiles).
    from sat2city.models.unet import pad_amounts

    for h, w in ((1022, 1022), (518, 518), (70, 98)):
        l, r, t, b = pad_amounts(h, w, margin=32)
        assert min(l, r, t, b) >= 32 and abs(l - r) <= 1 and abs(t - b) <= 1
        assert (h + t + b) % 32 == 0 and (w + l + r) % 32 == 0

    torch.save(net.state_dict(), tmp_path / "best.pt")
    (tmp_path / "config.json").write_text(json.dumps({"model": "unet:resnet34"}))
    loaded = load_model(tmp_path / "best.pt", device="cpu")
    x = torch.randn(1, 3, 98, 98)
    with torch.no_grad():
        assert torch.allclose(loaded(x), net(x), atol=1e-5)
