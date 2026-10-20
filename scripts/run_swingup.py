"""Plan a swing-up with thousands of parallel simulations, replay it, and save a GIF and a chart.

Run on a GPU runtime for the big batch:  python scripts/run_swingup.py --batch 4096
Writes results/swingup.gif, results/swingup_curve.png and results/swingup.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    import jax

    from gpu_robot_lab import sim, viz

    device = jax.devices()[0]
    print(f"JAX device: {device.platform} ({device.device_kind})")
    model = sim.make_model()

    start = time.perf_counter()
    plan = sim.plan_swing_up(model, batch=args.batch, iterations=args.iterations, seed=args.seed)
    seconds = time.perf_counter() - start
    print(f"Planning took {seconds:.1f} s with {args.batch} plans per iteration")
    for i, (s, h) in enumerate(zip(plan.best_scores, plan.final_heights), start=1):
        print(f"  iteration {i:>2}: best score {s:+.3f}, final tip height {h:+.3f} m")

    # Replay the plan in the CPU simulator, an independent check that the GPU result is real.
    angles = sim.rollout_cpu(model, plan.controls.astype("float64"))
    replay_height = float(sim.tip_height(angles[-1]))
    print(f"CPU replay: final tip height {replay_height:+.3f} m (top is {sim.POLE_LENGTH} m)")
    print("SUCCESS" if plan.success and replay_height > 0.9 * sim.POLE_LENGTH else "DID NOT REACH THE TOP")

    args.out.mkdir(parents=True, exist_ok=True)
    title = f"Swing-up from {args.batch} parallel simulations ({device.platform})"
    viz.animate_pendulum(angles, args.out / "swingup.gif", dt=sim.DT, title=title)
    viz.plot_scores(plan.best_scores, plan.final_heights, args.out / "swingup_curve.png")
    (args.out / "swingup.json").write_text(json.dumps({
        "device": f"{device.platform} ({device.device_kind})",
        "batch": args.batch,
        "iterations": args.iterations,
        "seed": args.seed,
        "planning_seconds": round(seconds, 2),
        "best_scores": [round(x, 4) for x in plan.best_scores],
        "final_heights_planner": [round(x, 4) for x in plan.final_heights],
        "final_height_cpu_replay": round(replay_height, 4),
        "success": bool(plan.success and replay_height > 0.9 * sim.POLE_LENGTH),
    }, indent=2))
    print(f"Saved {args.out}/swingup.gif, swingup_curve.png, swingup.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
