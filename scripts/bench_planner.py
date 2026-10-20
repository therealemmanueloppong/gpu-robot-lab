"""Time the swing-up planner itself, without the one-time compile, to compare CPU and GPU fairly.

The earlier timing (9.1 s on a GPU against 2.8 s on a laptop CPU) was not a fair comparison: the
batch sizes differed and it included compiling. Here the simulator is compiled once, then each
planner iteration is timed with the same batch size on whichever device JAX uses.

GPU:  python scripts/bench_planner.py
CPU:  JAX_PLATFORMS=cpu python scripts/bench_planner.py     Writes planner_<device>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    import jax

    from gpu_robot_lab import sim

    device = jax.devices()[0]
    platform = device.platform
    print(f"JAX device: {platform} ({device.device_kind})")
    model = sim.make_model()
    rollout = sim.make_batched_rollout(model)

    rows = []
    for batch in args.batches:
        sim.plan_swing_up(model, batch=batch, iterations=1, rollout=rollout)  # compiles for this batch size
        start = time.perf_counter()
        plan = sim.plan_swing_up(model, batch=batch, iterations=args.iterations, rollout=rollout)
        seconds = time.perf_counter() - start
        row = {"device": platform, "batch": batch, "iterations": args.iterations,
               "seconds_per_iteration": round(seconds / args.iterations, 4),
               "final_height": round(plan.final_heights[-1], 3), "success": plan.success}
        rows.append(row)
        print(row)
    out = f"planner_{platform}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
