"""Benchmark the Triton matmul against torch.matmul. Run on a GPU runtime (for example Colab)."""
from __future__ import annotations

import csv
import sys

from gpu_robot_lab.bench import gflops, matmul_flops, speedup, time_fn
from gpu_robot_lab.kernels import gpu_ready

# Best v1 settings from the 2026-10-20 sweep: block_m, block_n, block_k, num_warps, num_stages
BEST_V1 = (64, 64, 16, 4, 3)


def main() -> int:
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch
    import triton

    from gpu_robot_lab.kernels.matmul import matmul
    from gpu_robot_lab.kernels.matmul_v2 import matmul_v2

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}, triton {triton.__version__}, CUDA {torch.version.cuda}")
    rows = []
    for size in [256, 512, 1024, 2048, 4096]:
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        flops = matmul_flops(size, size, size)
        torch_t = time_fn(lambda a=a, b=b: torch.matmul(a, b), sync=torch.cuda.synchronize)
        mine_t = time_fn(lambda a=a, b=b: matmul(a, b), sync=torch.cuda.synchronize)
        best_t = time_fn(lambda a=a, b=b: matmul(a, b, *BEST_V1), sync=torch.cuda.synchronize)
        matmul_v2(a, b)  # first call runs the autotuner, so keep it out of the timing
        v2_t = time_fn(lambda a=a, b=b: matmul_v2(a, b), sync=torch.cuda.synchronize)
        row = {
            "size": size,
            "torch_ms": round(torch_t.median_ms, 4),
            "triton_ms": round(mine_t.median_ms, 4),
            "torch_gflops": round(gflops(flops, torch_t.median_ms), 1),
            "triton_gflops": round(gflops(flops, mine_t.median_ms), 1),
            "triton_vs_torch": round(speedup(torch_t.median_ms, mine_t.median_ms), 3),
            "v1_tuned_ms": round(best_t.median_ms, 4),
            "v1_tuned_gflops": round(gflops(flops, best_t.median_ms), 1),
            "v1_tuned_vs_torch": round(speedup(torch_t.median_ms, best_t.median_ms), 3),
            "v2_ms": round(v2_t.median_ms, 4),
            "v2_gflops": round(gflops(flops, v2_t.median_ms), 1),
            "v2_vs_torch": round(speedup(torch_t.median_ms, v2_t.median_ms), 3),
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
