"""Drawing helpers for the simulated pendulum. Needs the ``plot`` extra (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .sim import POLE_LENGTH


def animate_pendulum(
    angles: np.ndarray, path: str | Path, dt: float, every: int = 4, title: str = "Swing-up found by planning"
) -> Path:
    """Save a GIF of the pole. ``every`` draws one frame per that many simulation steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = list(range(0, len(angles), every)) + [len(angles) - 1]
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=80)
    lim = POLE_LENGTH * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    ax.plot([-lim * 0.4, lim * 0.4], [POLE_LENGTH, POLE_LENGTH], alpha=0)  # keeps the frame steady
    ax.axhline(POLE_LENGTH, xmin=0.42, xmax=0.58, color="#2e9e6b", linewidth=2, linestyle=":")
    (rod,) = ax.plot([], [], color="#2a7ae2", linewidth=5, solid_capstyle="round")
    (tip,) = ax.plot([], [], "o", color="#d9534f", markersize=14)
    ax.plot([0], [0], "o", color="black", markersize=5)
    clock = ax.text(0, -lim * 0.95, "", ha="center", fontsize=9)

    def draw(i: int):
        a = angles[i]
        x, y = POLE_LENGTH * np.sin(a), -POLE_LENGTH * np.cos(a)
        rod.set_data([0, x], [0, y])
        tip.set_data([x], [y])
        clock.set_text(f"t = {(i + 1) * dt:.2f} s")
        return rod, tip, clock

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=round(1 / (dt * every))))
    plt.close(fig)
    return out


def plot_scores(best_scores: list[float], final_heights: list[float], path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=130)
    x = np.arange(1, len(best_scores) + 1)
    ax.plot(x, final_heights, marker="o", color="#2e9e6b", label="tip height at the end of the best plan")
    ax.axhline(POLE_LENGTH, color="gray", linestyle="--", label="top (0.5 m)")
    ax.set_xlabel("Planner iteration")
    ax.set_ylabel("Height of pole tip (m)")
    ax.set_title("Planner finds the swing-up")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
