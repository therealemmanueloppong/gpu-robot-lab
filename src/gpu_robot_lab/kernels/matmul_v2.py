"""Matmul v2 in Triton: automatic tuning plus a cache-friendly launch order.

Not yet run on a GPU. Run ``scripts/check_kernels.py`` and ``scripts/bench_matmul.py`` on a Colab GPU.

Two changes from v1:
1. ``triton.autotune`` tries several tile sizes, warp counts and pipeline depths the first time it
   sees a new matrix shape and keeps the fastest. Settings that do not fit on the GPU are skipped.
2. Programs are launched in groups of GROUP_M rows of tiles (see ``tiling.grouped_order``), so tiles
   that run at the same time reuse the same data from the GPU's L2 cache.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=2, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=2),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel_v2(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Same arithmetic as tiling.grouped_order, which is unit tested on the CPU.
    per_group = GROUP_M * num_pid_n
    first_m = (pid // per_group) * GROUP_M
    size_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % per_group) % size_m)
    pid_n = (pid % per_group) // size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B for float16 CUDA tensors, with autotuned tile settings."""
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul_v2 expects float16 inputs")
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("matmul_v2 expects CUDA tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {tuple(a.shape)} and {tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _matmul_kernel_v2[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c
