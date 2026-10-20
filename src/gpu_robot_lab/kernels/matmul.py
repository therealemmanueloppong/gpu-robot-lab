"""Tiled matrix multiply in Triton, stage 1 of the plan.

Not yet run on a GPU. Run ``scripts/check_kernels.py`` on a Colab GPU to verify it.
Each program computes one BLOCK_M x BLOCK_N tile of C by sweeping along K,
loading a tile of A and a tile of B at each step and accumulating in float32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
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


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """C = A @ B for float16 CUDA tensors, accumulating in float32.

    ``num_warps`` and ``num_stages`` are Triton launch settings. Leave them as None for Triton's defaults.
    """
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("matmul expects float16 inputs")
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("matmul expects CUDA tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {tuple(a.shape)} and {tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    launch = {}
    if num_warps is not None:
        launch["num_warps"] = num_warps
    if num_stages is not None:
        launch["num_stages"] = num_stages
    _matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        **launch,
    )
    return c
