from __future__ import annotations

import numpy as np


def matmul_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain CPU matrix multiply in float32. This is the ground truth the GPU kernels are checked against."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {a.shape} and {b.shape}")
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


def softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtracting the max stops exp from overflowing."""
    shifted = x.astype(np.float32) - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
