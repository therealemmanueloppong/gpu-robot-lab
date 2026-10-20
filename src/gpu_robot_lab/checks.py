from __future__ import annotations

import numpy as np


def max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def assert_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-2, atol: float = 1e-2) -> None:
    """Fail with the size of the worst error, so a wrong kernel says how wrong it is."""
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise AssertionError("result contains NaN or infinity")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"results differ, max abs error {max_abs_error(actual, expected):.6f}")
