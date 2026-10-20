"""Train a neural-network policy for the pendulum by evolution strategies, then replay and animate it.

Training uses a curriculum: balance near the top first, then recover from bigger and bigger leans, ending
with a swing-up from the bottom. ``--iterations`` is per stage (5 stages).

Settings that worked in 3 of 3 runs:  python scripts/train_policy.py --population 512 --iterations 30
Runs with 2048 (CPU) and 4096 (GPU) candidates failed (see the README).
Writes results/policy/policy.json, policy_curve.png, policy_swingup.gif and policy_params.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=512, help="candidates per iteration (512 worked in 3 of 3 runs; 2048 and 4096 each failed once)")
    parser.add_argument("--iterations", type=int, default=30, help="iterations per curriculum stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/policy"))
    args = parser.parse_args()

    import jax

    from gpu_robot_lab import policy, sim, viz

    device = jax.devices()[0]
    print(f"JAX device: {device.platform} ({device.device_kind})")
    model = sim.make_model()

    start = time.perf_counter()
    result = policy.train_curriculum(model, population=args.population, iterations_per_stage=args.iterations, seed=args.seed)
    total = len(result.mean_fitness)
    seconds = time.perf_counter() - start
    print(f"Training took {seconds:.1f} s ({args.population} candidates x {total} iterations in {len(policy.CURRICULUM_OFFSETS)} stages)")
    for i in range(args.iterations - 1, total, args.iterations):
        print(f"  iteration {i + 1:>4}: average fitness {result.mean_fitness[i]:+.2f}, best {result.best_fitness[i]:+.2f}")

    # Replay in plain MuJoCo with a plain NumPy network, from starting angles the policy was not trained on.
    tests = {}
    for start_angle in (-0.5, -0.2, 0.0, 0.2, 0.5):
        angles, _ = policy.rollout_policy_cpu(model, result.params, start_angle, steps=500)
        ok = policy.success(angles)
        tests[str(start_angle)] = {"success": ok, "last_second_mean_height": round(float(sim.tip_height(angles[-100:]).mean()), 3)}
        print(f"  replay from start angle {start_angle:+.1f}: {'SUCCESS' if ok else 'failed'} "
              f"(mean tip height over the last second {tests[str(start_angle)]['last_second_mean_height']:+.3f} m, top is {sim.POLE_LENGTH})")
    n_ok = sum(t["success"] for t in tests.values())
    print(f"{n_ok} of {len(tests)} replays reached and held the top")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "policy_params.npy", result.params)
    gif_angles, _ = policy.rollout_policy_cpu(model, result.params, 0.2, steps=500)
    viz.animate_pendulum(gif_angles, args.out / "policy_swingup.gif", dt=sim.DT,
                         title=f"Learned policy ({args.population} candidates, {device.platform})")
    viz.plot_training(result.mean_fitness, result.best_fitness, args.out / "policy_curve.png", stage_length=args.iterations)
    (args.out / "policy.json").write_text(json.dumps({
        "device": f"{device.platform} ({device.device_kind})", "population": args.population,
        "iterations_per_stage": args.iterations, "stages": len(policy.CURRICULUM_OFFSETS), "seed": args.seed, "training_seconds": round(seconds, 1),
        "replays": tests, "replays_successful": n_ok, "replays_total": len(tests),
        "mean_fitness": [round(x, 3) for x in result.mean_fitness],
        "best_fitness": [round(x, 3) for x in result.best_fitness],
    }, indent=2))
    print(f"Saved {args.out}/policy.json, policy_curve.png, policy_swingup.gif, policy_params.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
