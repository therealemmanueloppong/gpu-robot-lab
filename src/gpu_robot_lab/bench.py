from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    runs: int


def time_fn(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    sync: Callable[[], None] | None = None,
) -> Timing:
    """Time ``fn`` after warm-up runs.

    GPU work is asynchronous, so pass ``sync`` (for example ``torch.cuda.synchronize``).
    It runs before and after every call, otherwise you would time only the launch.
    """
    if iters < 1 or warmup < 0:
        raise ValueError("iters must be at least 1 and warmup cannot be negative")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return Timing(median_ms=statistics.median(samples), min_ms=min(samples), runs=iters)


def matmul_flops(m: int, n: int, k: int) -> int:
    """A matrix multiply does one multiply and one add for each of m*n*k terms."""
    return 2 * m * n * k


def gflops(flops: float, milliseconds: float) -> float:
    if milliseconds <= 0:
        raise ValueError("milliseconds must be positive")
    return flops / (milliseconds / 1000.0) / 1e9


def speedup(baseline_ms: float, candidate_ms: float) -> float:
    """How many times faster the candidate is than the baseline (above 1 means faster)."""
    if candidate_ms <= 0:
        raise ValueError("candidate time must be positive")
    return baseline_ms / candidate_ms


@dataclass(frozen=True)
class Summary:
    median: float
    low: float
    high: float
    spread_percent: float  # (high - low) / median, in percent


def summarize(samples: list[float]) -> Summary:
    """Median, range and relative spread of repeated measurements."""
    if not samples:
        raise ValueError("need at least one sample")
    median = statistics.median(samples)
    low, high = min(samples), max(samples)
    return Summary(median=median, low=low, high=high, spread_percent=100.0 * (high - low) / median if median else 0.0)
