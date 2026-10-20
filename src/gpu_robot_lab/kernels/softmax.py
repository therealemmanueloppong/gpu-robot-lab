"""Row-wise softmax in Triton, stage 1 of the plan.

Not yet run on a GPU. Each program handles one row: it loads the row, subtracts the
row maximum so exp cannot overflow, then divides by the sum.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension of a 2-D float32 CUDA tensor."""
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("softmax expects a 2-D float32 CUDA tensor")
    rows, cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=triton.next_power_of_2(cols))
    return out
