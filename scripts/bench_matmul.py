"""Benchmark the Triton matmul against torch.matmul. Run on a GPU runtime (for example Colab)."""
from __future__ import annotations

import csv
import sys

from gpu_robot_lab.bench import gflops, matmul_flops, speedup, time_fn
from gpu_robot_lab.kernels import gpu_ready


def main() -> int:
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch

    from gpu_robot_lab.kernels.matmul import matmul

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    rows = []
    for size in [256, 512, 1024, 2048, 4096]:
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        flops = matmul_flops(size, size, size)
        torch_t = time_fn(lambda a=a, b=b: torch.matmul(a, b), sync=torch.cuda.synchronize)
        mine_t = time_fn(lambda a=a, b=b: matmul(a, b), sync=torch.cuda.synchronize)
        row = {
            "size": size,
            "torch_ms": round(torch_t.median_ms, 4),
            "triton_ms": round(mine_t.median_ms, 4),
            "torch_gflops": round(gflops(flops, torch_t.median_ms), 1),
            "triton_gflops": round(gflops(flops, mine_t.median_ms), 1),
            "triton_vs_torch": round(speedup(torch_t.median_ms, mine_t.median_ms), 3),
        }
        rows.append(row)
        print(row)
    with open("matmul_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved matmul_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
