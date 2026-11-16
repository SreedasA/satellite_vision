"""P0.7: peak GPU memory and step time for DAv2-S fine-tuning at several sizes.

Usage: python scripts/profile_dav2.py [--sizes 252 378 518] [--batches 1 2 4]
"""

import argparse
import json
import time

import torch

from sat2city.config import load_paths
from sat2city.models.dav2 import DAv2Height


def step_stats(model, opt, size, batch, ckpt, dtype, steps=6):
    size = size // 14 * 14  # DAv2 patch size is 14
    x = torch.randn(batch, 3, size, size, device="cuda")
    y = torch.rand(batch, size, size, device="cuda") * 30
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            loss = torch.nn.functional.l1_loss(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:  # skip warm-up
            times.append(time.perf_counter() - t0)
    return torch.cuda.max_memory_allocated() / 2**30, sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[252, 378, 518])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    paths = load_paths()
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB, bf16={torch.cuda.is_bf16_supported()}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    results = []
    for ckpt in (False, True):
        model = DAv2Height(cache_dir=str(paths["cache"])).cuda().train()
        if ckpt:
            model.net.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-6)
        for size in args.sizes:
            for batch in args.batches:
                try:
                    mem, sec = step_stats(model, opt, size, batch, ckpt, dtype)
                    row = {"ckpt": ckpt, "size": size, "batch": batch, "peak_GiB": round(mem, 2),
                           "s_per_step": round(sec, 3), "img_per_s": round(batch / sec, 2)}
                except torch.OutOfMemoryError:
                    row = {"ckpt": ckpt, "size": size, "batch": batch, "peak_GiB": "OOM"}
                torch.cuda.empty_cache()
                print(json.dumps(row))
                results.append(row)
        del model, opt
        torch.cuda.empty_cache()
    out = paths["processed"] / "profile_dav2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
