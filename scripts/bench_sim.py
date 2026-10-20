"""Compare simulation speed: one CPU thread of MuJoCo versus many robots at once in MJX.

Run on a GPU runtime to see the difference:
  python scripts/bench_sim.py                    the pendulum (cheap physics)   -> sim_results.csv
  python scripts/bench_sim.py --model box        a box dropped on the floor (collisions, friction) -> sim_results_box.csv
Units are simulation steps per second summed over all robots. Each GPU number is the median of
several repeats, with the range recorded.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from gpu_robot_lab.bench import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pendulum", "box"], default="pendulum")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 256, 1024, 4096, 16384])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or ("sim_results.csv" if args.model == "pendulum" else "sim_results_box.csv")

    import jax
    import jax.numpy as jnp
    import mujoco

    from gpu_robot_lab import sim

    device = jax.devices()[0]
    print(f"JAX device: {device.platform} ({device.device_kind}); jax {jax.__version__}; mujoco {mujoco.__version__}")

    if args.model == "pendulum":
        model = sim.make_model()
        data = mujoco.MjData(model)
        n_cpu = 20000
        start = time.perf_counter()
        for _ in range(n_cpu):
            mujoco.mj_step(model, data)
        cpu_rate = n_cpu / (time.perf_counter() - start)
        rollout = sim.make_batched_rollout(model)

        def make_inputs(batch):
            rng = np.random.default_rng(0)
            return (jnp.asarray(rng.uniform(-1, 1, size=(batch, args.steps)).astype(np.float32)),)
    else:
        model = sim.make_box_model()
        qpos, qvel = sim.random_box_starts(1, seed=0)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos[0], qvel[0]
        n_cpu = 5000
        start = time.perf_counter()
        for _ in range(n_cpu):
            mujoco.mj_step(model, data)
        cpu_rate = n_cpu / (time.perf_counter() - start)
        rollout = sim.make_box_rollout(model, args.steps)

        def make_inputs(batch):
            q, v = sim.random_box_starts(batch, seed=0)
            return jnp.asarray(q), jnp.asarray(v)

    print(f"CPU, 1 thread, {args.model}: {cpu_rate:,.0f} steps per second\n")

    rows = []
    for batch in args.batches:
        inputs = make_inputs(batch)
        np.asarray(rollout(*inputs))  # first call compiles, keep it out of the timing
        rates = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            np.asarray(rollout(*inputs))  # converting to numpy waits for the device to finish
            rates.append(batch * args.steps / (time.perf_counter() - start))
        s = summarize(rates)
        row = {"model": args.model, "batch": batch, "device": device.platform,
               "mjx_steps_per_s": round(s.median), "mjx_min": round(s.low), "mjx_max": round(s.high),
               "cpu_steps_per_s": round(cpu_rate), "speedup_vs_cpu_1_thread": round(s.median / cpu_rate, 2)}
        rows.append(row)
        print(row)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
