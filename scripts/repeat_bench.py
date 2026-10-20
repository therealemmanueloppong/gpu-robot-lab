"""Measure how much the benchmark results vary from run to run.

Times PyTorch, v1, v1 with the best swept settings, and v2 several times each, taking turns so that
slow changes in the shared GPU affect every version equally. Reports the median and range of each.

Run on a GPU runtime:  python scripts/repeat_bench.py          Writes repeat_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys

from gpu_robot_lab.bench import gflops, matmul_flops, summarize, time_fn
from gpu_robot_lab.kernels import gpu_ready

BEST_V1 = (64, 64, 16, 4, 3)  # block_m, block_n, block_k, num_warps, num_stages from the sweep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", default="repeat_results.csv")
    args = parser.parse_args()
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch

    from gpu_robot_lab.kernels.matmul import matmul
    from gpu_robot_lab.kernels.matmul_v2 import matmul_v2

    print(f"GPU: {torch.cuda.get_device_name(0)}, {args.repeats} repeats")
    rows = []
    for n in args.sizes:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        flops = matmul_flops(n, n, n)
        versions = {
            "torch.matmul": lambda a=a, b=b: torch.matmul(a, b),
            "v1 default": lambda a=a, b=b: matmul(a, b),
            "v1 best swept settings": lambda a=a, b=b: matmul(a, b, *BEST_V1),
            "v2 autotuned": lambda a=a, b=b: matmul_v2(a, b),
        }
        for fn in versions.values():
            fn()  # compile / autotune once, outside the timing
        samples: dict[str, list[float]] = {name: [] for name in versions}
        for _ in range(args.repeats):
            for name, fn in versions.items():
                ms = time_fn(fn, warmup=3, iters=20, sync=torch.cuda.synchronize).median_ms
                samples[name].append(gflops(flops, ms))
        torch_median = summarize(samples["torch.matmul"]).median
        for name, vals in samples.items():
            s = summarize(vals)
            row = {"size": n, "version": name, "median_gflops": round(s.median, 1), "min_gflops": round(s.low, 1),
                   "max_gflops": round(s.high, 1), "spread_percent": round(s.spread_percent, 1),
                   "vs_torch_median": round(s.median / torch_median, 3)}
            rows.append(row)
            print(row)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
