"""A CPU model of the tiled matmul kernel.

It follows the Triton kernel step by step: one "program" per output tile, a loop over K,
masked loads at the edges, float32 accumulation. Because it runs anywhere, the tiling and
edge handling are tested in CI, and the loads it counts back up the claim made on the demo page.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileStats:
    programs: int  # output tiles, each handled by one GPU program
    k_steps: int  # iterations of the K loop per program
    loads: int  # elements actually read from A and B (masked lanes are not counted)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def tiled_matmul(
    a: np.ndarray, b: np.ndarray, block_m: int = 64, block_n: int = 64, block_k: int = 32
) -> tuple[np.ndarray, TileStats]:
    """Compute a @ b the way the Triton kernel does, and count how much it reads."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {a.shape} and {b.shape}")
    if min(block_m, block_n, block_k) < 1:
        raise ValueError("block sizes must be positive")
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float32)
    loads = 0
    for pid_m in range(ceil_div(m, block_m)):
        for pid_n in range(ceil_div(n, block_n)):
            rows = slice(pid_m * block_m, min((pid_m + 1) * block_m, m))
            cols = slice(pid_n * block_n, min((pid_n + 1) * block_n, n))
            acc = np.zeros((block_m, block_n), dtype=np.float32)
            for step in range(ceil_div(k, block_k)):
                ks = slice(step * block_k, min((step + 1) * block_k, k))
                a_tile = np.zeros((block_m, block_k), dtype=np.float32)  # masked lanes read as 0
                b_tile = np.zeros((block_k, block_n), dtype=np.float32)
                a_part = a[rows, ks].astype(np.float32)
                b_part = b[ks, cols].astype(np.float32)
                a_tile[: a_part.shape[0], : a_part.shape[1]] = a_part
                b_tile[: b_part.shape[0], : b_part.shape[1]] = b_part
                loads += a_part.size + b_part.size
                acc += a_tile @ b_tile
            c[rows, cols] = acc[: rows.stop - rows.start, : cols.stop - cols.start]
    stats = TileStats(
        programs=ceil_div(m, block_m) * ceil_div(n, block_n), k_steps=ceil_div(k, block_k), loads=loads
    )
    return c, stats


def naive_loads(m: int, n: int, k: int) -> int:
    """With no tiling every output element reads a full row of A and a full column of B."""
    return 2 * m * n * k


def traffic_reduction(block_m: int, block_n: int) -> float:
    """How many times fewer elements tiling reads, for matrices that divide evenly.

    A tile of BM x BN outputs reads (BM + BN) * K elements instead of 2 * BM * BN * K,
    so the saving is 2 * BM * BN / (BM + BN).
    """
    return 2 * block_m * block_n / (block_m + block_n)


def row_major_order(num_pid_m: int, num_pid_n: int) -> list[tuple[int, int]]:
    """The simple launch order: finish a whole row of output tiles, then the next row."""
    return [(i, j) for i in range(num_pid_m) for j in range(num_pid_n)]


def grouped_order(num_pid_m: int, num_pid_n: int, group_m: int) -> list[tuple[int, int]]:
    """The launch order used by matmul_v2: walk down groups of ``group_m`` rows of tiles.

    This is the same arithmetic the kernel does with its program id. Programs that run at the
    same time then work on a compact block of output tiles and share loaded data in cache.
    """
    if group_m < 1:
        raise ValueError("group_m must be positive")
    order = []
    per_group = group_m * num_pid_n
    for pid in range(num_pid_m * num_pid_n):
        first_m = (pid // per_group) * group_m
        size_m = min(num_pid_m - first_m, group_m)
        order.append((first_m + (pid % per_group) % size_m, (pid % per_group) // size_m))
    return order


def tiles_needed(order: list[tuple[int, int]]) -> int:
    """How many different row-bands of A plus column-bands of B a set of programs must load."""
    return len({m for m, _ in order}) + len({n for _, n in order})
