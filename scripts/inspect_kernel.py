"""Print register use, spills and shared memory for some v1 settings, to see why some are slow.

Run on a GPU runtime:  python scripts/inspect_kernel.py
A "spill" means the compiler ran out of registers and used slow memory instead. Values that
this version of Triton does not report are printed as n/a.
"""
from __future__ import annotations

import sys

from gpu_robot_lab.kernels import gpu_ready

SETTINGS = [  # block_m, block_n, block_k, num_warps
    (64, 64, 32, 4),
    (64, 64, 32, 8),
    (64, 64, 16, 4),
    (64, 64, 16, 8),
    (64, 64, 32, 2),
    (32, 32, 32, 4),
    (128, 64, 16, 8),
]


def main() -> int:
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found.")
        return 2
    import torch
    import triton

    from gpu_robot_lab.kernels.matmul import _matmul_kernel

    n = 1024
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    c = torch.empty((n, n), device="cuda", dtype=torch.float16)
    print(f"{'bm':>4} {'bn':>4} {'bk':>4} {'warps':>5} {'regs':>6} {'spills':>7} {'shared B':>9}")
    for bm, bn, bk, warps in SETTINGS:
        grid = (triton.cdiv(n, bm), triton.cdiv(n, bn))
        compiled = _matmul_kernel[grid](
            a, b, c, n, n, n,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
        )
        regs = getattr(compiled, "n_regs", "n/a")
        spills = getattr(compiled, "n_spills", "n/a")
        shared = getattr(getattr(compiled, "metadata", None), "shared", "n/a")
        print(f"{bm:>4} {bn:>4} {bk:>4} {warps:>5} {regs!s:>6} {spills!s:>7} {shared!s:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
