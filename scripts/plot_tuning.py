"""Chart the best tuning configurations. Usage: python scripts/plot_tuning.py tuning_results.csv"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

BASELINE = ("64", "64", "32", "4", "3")  # block_m, block_n, block_k, num_warps, num_stages of Triton v1


def label(row: dict[str, str]) -> str:
    return f"{row['block_m']}x{row['block_n']}x{row['block_k']}  w{row['num_warps']} s{row['num_stages']}"


def main(path: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    if not rows:
        print("No successful configurations in the file.")
        return 1
    rows.sort(key=lambda r: -float(r["gflops"]))
    torch_gflops = float(rows[0]["gflops"]) / float(rows[0]["vs_torch"])
    baseline = next(
        (r for r in rows if (r["block_m"], r["block_n"], r["block_k"], r["num_warps"], r["num_stages"]) == BASELINE),
        None,
    )

    top = rows[:10]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    ax.barh([label(r) for r in top][::-1], [float(r["gflops"]) for r in top][::-1], color="#d9534f")
    ax.axvline(torch_gflops, color="#2a7ae2", linewidth=2, label=f"torch.matmul ({torch_gflops:,.0f})")
    if baseline:
        ax.axvline(float(baseline["gflops"]), color="gray", linestyle="--", label=f"Triton v1 ({float(baseline['gflops']):,.0f})")
    best = float(top[0]["gflops"])
    ax.set_title(f"Tuning the Triton matmul: best is {best / torch_gflops:.0%} of torch.matmul")
    ax.set_xlabel("Throughput (GFLOPS, higher is better)")
    ax.set_ylabel("tile MxNxK, warps, stages")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, 0.01, "Top 10 configurations. Data: " + Path(path).name, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(path).with_suffix(".png")
    fig.savefig(out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "tuning_results.csv"))
