"""Sweep tile sizes and launch settings for the Triton matmul and compare with torch.matmul.

Run on a GPU runtime (for example Colab):  python scripts/tune_matmul.py --size 2048
Writes tuning_results.csv. Configurations that fail to compile or run are recorded as failed,
not hidden, because some settings do not fit on every GPU.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys

from gpu_robot_lab.bench import gflops, matmul_flops, speedup, time_fn
from gpu_robot_lab.kernels import gpu_ready

TILES = [(32, 32), (64, 64), (128, 64), (64, 128), (128, 128)]
BLOCK_KS = [16, 32, 64]
WARPS = [2, 4, 8]
STAGES = [2, 3, 4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch
    import triton

    from gpu_robot_lab.kernels.matmul import matmul

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}, triton {triton.__version__}, CUDA {torch.version.cuda}")
    n = args.size
    torch.manual_seed(0)
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    flops = matmul_flops(n, n, n)
    reference = torch.matmul(a, b)
    torch_ms = time_fn(lambda: torch.matmul(a, b), iters=args.iters, sync=torch.cuda.synchronize).median_ms
    print(f"torch.matmul at {n}: {torch_ms:.3f} ms, {gflops(flops, torch_ms):.0f} GFLOPS\n")

    rows = []
    for (bm, bn), bk, warps, stages in itertools.product(TILES, BLOCK_KS, WARPS, STAGES):
        row = {"block_m": bm, "block_n": bn, "block_k": bk, "num_warps": warps, "num_stages": stages}
        try:
            out = matmul(a, b, bm, bn, bk, warps, stages)
            correct = bool(torch.allclose(out, reference, rtol=1e-2, atol=1e-1))
            ms = time_fn(
                lambda bm=bm, bn=bn, bk=bk, w=warps, s=stages: matmul(a, b, bm, bn, bk, w, s),
                iters=args.iters,
                sync=torch.cuda.synchronize,
            ).median_ms
            row.update(status="ok" if correct else "WRONG", ms=round(ms, 4), gflops=round(gflops(flops, ms), 1),
                       vs_torch=round(speedup(torch_ms, ms), 3))
        except Exception as error:  # noqa: BLE001 - record any failure and keep sweeping
            row.update(status=f"failed: {type(error).__name__}", ms="", gflops="", vs_torch="")
        rows.append(row)
        print(row)

    with open("tuning_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    good = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: -r["gflops"])
    print("\nBest 5 configurations:")
    for r in good[:5]:
        print(r)
    print(f"\nSaved tuning_results.csv ({len(good)} ok, {len(rows) - len(good)} failed or wrong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
