"""P0.8 spike: export pretrained DAv2-S to ONNX and compare FP32 / FP16 / INT8 with PyTorch.

Usage: python scripts/spike_onnx.py [--size 518]
"""

import argparse
import json
import time

import numpy as np
import onnxruntime as ort
import torch

from sat2city.config import load_paths
from sat2city.models.dav2 import DAv2Height


def ort_run(path, x, providers):
    sess = ort.InferenceSession(str(path), providers=providers)
    name = sess.get_inputs()[0].name
    feed = {name: x.astype(np.float16 if "fp16" in path.name else np.float32)}
    sess.run(None, feed)  # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        out = sess.run(None, feed)[0]
    return out.astype(np.float32), (time.perf_counter() - t0) / 5, sess.get_providers()[0]


def rel_err(a, b):
    return float(np.abs(a - b).mean() / (np.abs(b).mean() + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=518)
    args = ap.parse_args()
    paths = load_paths()
    out_dir = paths["processed"] / "onnx_spike"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DAv2Height(cache_dir=str(paths["cache"])).eval()
    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        ref = model(x).numpy()

    report = {"size": args.size}
    fp32 = out_dir / "dav2s_fp32.onnx"
    torch.onnx.export(model, (x,), str(fp32), input_names=["pixel_values"], output_names=["height"],
                      opset_version=18, dynamo=True)
    report["export_fp32"] = "ok"

    fp16 = out_dir / "dav2s_fp16.onnx"
    try:
        torch.onnx.export(model.half().cuda(), (x.half().cuda(),), str(fp16), input_names=["pixel_values"],
                          output_names=["height"], opset_version=18, dynamo=True)
        model.float().cpu()
        report["export_fp16"] = "ok"
    except Exception as e:  # noqa: BLE001 - record any exporter failure in the spike report
        report["export_fp16"] = f"failed: {type(e).__name__}: {e}"[:300]

    int8 = out_dir / "dav2s_int8_dynamic.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
        report["export_int8"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["export_int8"] = f"failed: {type(e).__name__}: {e}"[:300]

    xn = x.numpy()
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    for path, providers in ((fp32, gpu), (fp32, ["CPUExecutionProvider"]), (fp16, gpu), (int8, ["CPUExecutionProvider"])):
        if not path.exists():
            continue
        out, sec, used = ort_run(path, xn, providers)
        report[f"{path.stem}@{used}"] = {"rel_err_vs_torch": round(rel_err(out, ref), 5), "ms": round(sec * 1000, 1)}
    print(json.dumps(report, indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
