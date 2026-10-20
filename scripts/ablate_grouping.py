"""Isolate the effect of the grouped launch order: same tile settings, only GROUP_M changes.

GROUP_M = 1 is the plain row-by-row order. Larger values launch tiles in groups of that many rows,
so tiles that run together share more cached data. Versions take turns across repeats so slow drift
in the shared GPU does not favour one of them.

Run on a GPU runtime:  python scripts/ablate_grouping.py        Writes ablation_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys

from gpu_robot_lab.bench import gflops, matmul_flops, summarize, time_fn
from gpu_robot_lab.kernels import gpu_ready

CONFIGS = [(64, 64, 16, 4, 3), (64, 128, 16, 8, 3)]  # block_m, block_n, block_k, num_warps, num_stages
GROUPS = [1, 2, 4, 8, 16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch

    from gpu_robot_lab.kernels.matmul_v2 import matmul_grouped

    print(f"GPU: {torch.cuda.get_device_name(0)}, {args.repeats} repeats")
    rows = []
    for n in args.sizes:
        a = torch.randn((n, n), device="cuda", dtype=torch.float16)
        b = torch.randn((n, n), device="cuda", dtype=torch.float16)
        flops = matmul_flops(n, n, n)
        reference = torch.matmul(a, b)
        for bm, bn, bk, warps, stages in CONFIGS:
            fns = {
                g: (lambda g=g, a=a, b=b, bm=bm, bn=bn, bk=bk, warps=warps, stages=stages:
                    matmul_grouped(a, b, bm, bn, bk, g, warps, stages))
                for g in GROUPS
            }
            for g, fn in fns.items():
                assert torch.allclose(fn(), reference, rtol=1e-2, atol=1e-1), f"wrong result for GROUP_M={g}"
            samples: dict[int, list[float]] = {g: [] for g in GROUPS}
            for _ in range(args.repeats):
                for g, fn in fns.items():
                    ms = time_fn(fn, warmup=3, iters=20, sync=torch.cuda.synchronize).median_ms
                    samples[g].append(gflops(flops, ms))
            base = summarize(samples[1]).median
            for g, vals in samples.items():
                s = summarize(vals)
                row = {"size": n, "tile": f"{bm}x{bn}x{bk}", "warps": warps, "group_m": g,
                       "median_gflops": round(s.median, 1), "min_gflops": round(s.low, 1),
                       "max_gflops": round(s.high, 1), "vs_group1": round(s.median / base, 3)}
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
