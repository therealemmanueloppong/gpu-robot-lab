"""Compare simulation speed: one CPU thread of MuJoCo versus many robots at once in MJX.

Run on a GPU runtime to see the difference:  python scripts/bench_sim.py
Writes sim_results.csv. Units are simulation steps per second summed over all robots.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 256, 1024, 4096, 16384])
    parser.add_argument("--out", default="sim_results.csv")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco

    from gpu_robot_lab import sim

    device = jax.devices()[0]
    print(f"JAX device: {device.platform} ({device.device_kind}); jax {jax.__version__}; mujoco {mujoco.__version__}")
    model = sim.make_model()

    # CPU baseline: one MuJoCo environment on one thread. More robots just take proportionally longer.
    data = mujoco.MjData(model)
    n_cpu = 20000
    start = time.perf_counter()
    for _ in range(n_cpu):
        mujoco.mj_step(model, data)
    cpu_rate = n_cpu / (time.perf_counter() - start)
    print(f"CPU, 1 thread: {cpu_rate:,.0f} steps per second\n")

    rollout = sim.make_batched_rollout(model)
    rng = np.random.default_rng(0)
    rows = []
    for batch in args.batches:
        controls = jnp.asarray(rng.uniform(-1, 1, size=(batch, args.steps)).astype(np.float32))
        np.asarray(rollout(controls))  # first call compiles, keep it out of the timing
        times = []
        for _ in range(3):
            start = time.perf_counter()
            np.asarray(rollout(controls))  # converting to numpy waits for the device to finish
            times.append(time.perf_counter() - start)
        rate = batch * args.steps / min(times)
        row = {"batch": batch, "device": device.platform, "mjx_steps_per_s": round(rate),
               "cpu_steps_per_s": round(cpu_rate), "speedup_vs_cpu_1_thread": round(rate / cpu_rate, 2)}
        rows.append(row)
        print(row)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
