"""Check the GPU kernels against PyTorch. Run this on a GPU runtime (for example Colab)."""
from __future__ import annotations

import sys

from gpu_robot_lab.kernels import gpu_ready


def main() -> int:
    if not gpu_ready():
        print("No CUDA GPU with torch and triton found. On Colab choose Runtime, then Change runtime type, then a GPU.")
        return 2
    import torch

    from gpu_robot_lab.kernels.matmul import matmul
    from gpu_robot_lab.kernels.matmul_v2 import matmul_v2
    from gpu_robot_lab.kernels.softmax import softmax

    torch.manual_seed(0)
    failures = 0

    for m, n, k in [(64, 64, 64), (100, 77, 130), (512, 512, 512), (1000, 1000, 1000)]:
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        err = (matmul(a, b).float() - (a @ b).float()).abs().max().item()
        ok = torch.allclose(matmul(a, b), a @ b, rtol=1e-2, atol=1e-1)
        failures += not ok
        print(f"matmul  {m}x{k} @ {k}x{n}: max abs error {err:.4f}  {'PASS' if ok else 'FAIL'}")
        ok2 = torch.allclose(matmul_v2(a, b), a @ b, rtol=1e-2, atol=1e-1)
        failures += not ok2
        print(f"matmul2 {m}x{k} @ {k}x{n}: {'PASS' if ok2 else 'FAIL'}")

    for rows, cols in [(1, 10), (128, 257), (1024, 1000)]:
        x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
        ok = torch.allclose(softmax(x), torch.softmax(x, dim=1), rtol=1e-4, atol=1e-5)
        failures += not ok
        print(f"softmax {rows}x{cols}: {'PASS' if ok else 'FAIL'}")

    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
