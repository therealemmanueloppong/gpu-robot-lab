"""Turn a benchmark CSV into a chart. Usage: python scripts/plot_results.py results/matmul_t4_2026-10-20.csv"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(path: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    sizes = [int(r["size"]) for r in rows]
    torch_g = [float(r["torch_gflops"]) for r in rows]
    triton_g = [float(r["triton_gflops"]) for r in rows]
    tuned_g = [float(r["v1_tuned_gflops"]) for r in rows] if "v1_tuned_gflops" in rows[0] else None
    v2_g = [float(r["v2_gflops"]) for r in rows] if "v2_gflops" in rows[0] else None

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(sizes, torch_g, marker="o", linewidth=2, color="#2a7ae2", label="torch.matmul")
    ax.plot(sizes, triton_g, marker="o", linewidth=2, color="#d9534f", label="Triton v1 (untuned)")
    if tuned_g:
        ax.plot(sizes, tuned_g, marker="o", linewidth=2, color="#e08a1e", label="Triton v1, best swept settings")
    if v2_g:
        ax.plot(sizes, v2_g, marker="o", linewidth=2, color="#2e9e6b", label="Triton v2 (autotuned)")
    newest, newest_color = (v2_g, "#2e9e6b") if v2_g else (triton_g, "#d9534f")
    for x, t, y in zip(sizes, torch_g, newest):
        ax.annotate(f"{y / t:.0%}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=newest_color)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size (N x N, float16)")
    ax.set_ylabel("Throughput (GFLOPS, higher is better)")
    ax.set_title("Matrix multiply on a GPU: Triton vs PyTorch")
    ax.set_ylim(bottom=-1000)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(0.01, 0.01, "Labels show the newest Triton version as a percentage of torch.matmul in the same run. Data: " + Path(path).name, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(path).with_suffix(".png")
    fig.savefig(out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "matmul_results.csv"))
