"""Chart simulation speed from bench_sim's CSV. Usage: python scripts/plot_sim.py sim_results.csv"""
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
    batch = [int(r["batch"]) for r in rows]
    gpu = [float(r["mjx_steps_per_s"]) for r in rows]
    cpu = float(rows[0]["cpu_steps_per_s"])

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(batch, gpu, marker="o", linewidth=2, color="#2e9e6b", label="MJX on the GPU")
    ax.axhline(cpu, color="#d9534f", linestyle="--", linewidth=2, label=f"MuJoCo, 1 CPU thread ({cpu:,.0f})")
    for x, y in zip(batch, gpu):
        ax.annotate(f"{y / cpu:.2f}x" if y / cpu < 1 else f"{y / cpu:.0f}x", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#2e9e6b")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(top=max(gpu) * 5)
    ax.set_xticks(batch)
    ax.set_xticklabels([f"{b:,}" for b in batch], fontsize=8)
    ax.set_xlabel("Robots simulated at the same time")
    ax.set_ylabel("Simulation steps per second (all robots)")
    ax.set_title("Simulation speed: one CPU thread vs many robots on a GPU")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, "Labels show speed compared with one CPU thread. Data: " + Path(path).name, fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(path).with_suffix(".png")
    fig.savefig(out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sim_results.csv"))
