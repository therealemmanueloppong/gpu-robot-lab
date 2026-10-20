"""List the GPU kernels PyTorch runs for a float16 matrix multiply, and how long each takes.

Run on a GPU runtime:  python scripts/profile_torch.py
The kernel names show what library code PyTorch uses (usually cuBLAS, with names that mention
tensor-core shapes such as s1688 or s884). Compare with scripts/inspect_kernel.py.
"""
from __future__ import annotations

import sys

from gpu_robot_lab.kernels import gpu_ready


def main() -> int:
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch
    from torch.profiler import ProfilerActivity, profile

    n = 2048
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            torch.matmul(a, b)
        torch.cuda.synchronize()
    rows = [e for e in prof.key_averages() if e.device_time_total > 0]
    rows.sort(key=lambda e: -e.device_time_total)
    print(f"GPU kernels run by torch.matmul at {n} x {n}, float16 (10 calls):")
    for e in rows[:5]:
        print(f"  {e.device_time_total / e.count / 1000:.3f} ms per call  x{e.count}  {e.key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
