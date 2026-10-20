"""Train a neural-network policy for the pendulum by evolution strategies, then replay and animate it.

Run on a GPU runtime for the big population:  python scripts/train_policy.py --population 4096 --iterations 150
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
    parser.add_argument("--population", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/policy"))
    args = parser.parse_args()

    import jax

    from gpu_robot_lab import policy, sim, viz

    device = jax.devices()[0]
    print(f"JAX device: {device.platform} ({device.device_kind})")
    model = sim.make_model()

    start = time.perf_counter()
    result = policy.train_es(model, population=args.population, iterations=args.iterations, seed=args.seed)
    seconds = time.perf_counter() - start
    print(f"Training took {seconds:.1f} s ({args.population} candidates x {args.iterations} iterations)")
    step = max(1, args.iterations // 10)
    for i in range(0, args.iterations, step):
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
    viz.plot_training(result.mean_fitness, result.best_fitness, args.out / "policy_curve.png")
    (args.out / "policy.json").write_text(json.dumps({
        "device": f"{device.platform} ({device.device_kind})", "population": args.population,
        "iterations": args.iterations, "seed": args.seed, "training_seconds": round(seconds, 1),
        "replays": tests, "replays_successful": n_ok, "replays_total": len(tests),
        "mean_fitness": [round(x, 3) for x in result.mean_fitness],
        "best_fitness": [round(x, 3) for x in result.best_fitness],
    }, indent=2))
    print(f"Saved {args.out}/policy.json, policy_curve.png, policy_swingup.gif, policy_params.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
