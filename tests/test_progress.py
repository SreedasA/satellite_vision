import json

from sat2city.progress import ProgressWriter, fmt_time, render


def test_progress_lifecycle(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    pw = ProgressWriter(run_dir, "r1", epochs=2, steps_per_epoch=4, every=1)
    assert (tmp_path / "runs" / "LATEST").read_text() == "r1"

    pw.epoch_start(1)
    for i in range(1, 5):
        pw.step(i, loss=1.0 / i)
    p = json.loads((run_dir / "progress.json").read_text())
    assert p["global_step"] == 4 and p["status"] == "training" and p["eta_sec"] is not None
    assert "50.0%" in render(p)

    pw.phase("evaluating")
    pw.epoch_end("epoch 1/2 ...", eval_sec=3.0, best=5.5)
    pw.done({"rmse": 3.4, "building_rmse": 7.7, "bld_mae": 2.1})
    p = json.loads((run_dir / "progress.json").read_text())
    text = render(p)
    assert p["status"] == "done" and "100.0%" in text and "building RMSE 7.7" in text


def test_failed_status(tmp_path):
    pw = ProgressWriter(tmp_path, "r2", epochs=1, steps_per_epoch=1)
    pw.failed("FloatingPointError: nan")
    p = json.loads((tmp_path / "progress.json").read_text())
    assert p["status"] == "failed" and "nan" in render(p)


def test_write_never_raises_when_target_is_locked(tmp_path, monkeypatch):
    import os

    pw = ProgressWriter(tmp_path, "r3", epochs=1, steps_per_epoch=5, every=1)

    def locked(*_a, **_k):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(os, "replace", locked)
    pw.epoch_start(1)
    for i in range(1, 6):
        pw.step(i, loss=1.0)  # must not raise even though every replace fails
    assert pw.state["global_step"] == 5


def test_fmt_time():
    assert fmt_time(None) == "estimating…"
    assert fmt_time(75) == "1m 15s" and fmt_time(3725) == "1h 02m"
