"""A small simulated robot: a motor-driven pendulum, simulated on the CPU (MuJoCo) and the GPU (MJX).

The pole hangs down at angle 0 and points straight up at angle pi. The motor is too weak to lift
the pole directly (its peak torque is less than gravity's pull at horizontal), so reaching the top
means swinging back and forth to build up energy. That makes a good test: a planner has to find a
whole sequence of motor commands, not just one push.

Needs the ``sim`` extra: ``pip install -e '.[sim]'``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POLE_LENGTH = 0.5  # metres from the hinge to the tip
DT = 0.01  # seconds per simulation step

PENDULUM_XML = f"""
<mujoco model="pendulum">
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{POLE_LENGTH}" size="0.02" mass="0.2"/>
      <geom type="sphere" pos="0 0 -{POLE_LENGTH}" size="0.05" mass="0.8"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="2" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def make_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(PENDULUM_XML)


def tip_height(angle: np.ndarray | float) -> np.ndarray | float:
    """Height of the pole tip above the hinge: -L at the bottom (angle 0), +L at the top (angle pi)."""
    return -POLE_LENGTH * np.cos(angle)


def rollout_cpu(model, controls: np.ndarray, start_angle: float = 0.0) -> np.ndarray:
    """Run one sequence of motor commands on the CPU and return the angle after every step."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[0] = start_angle
    angles = np.empty(len(controls))
    for i, u in enumerate(controls):
        data.ctrl[0] = u
        mujoco.mj_step(model, data)
        angles[i] = data.qpos[0]
    return angles


def make_batched_rollout(model):
    """Return a function that simulates many control sequences at once with MJX (on the GPU if there is one).

    The returned function takes an array of shape (batch, steps) and gives back angles of the same shape.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    mjx_model = mjx.put_model(model)
    initial = mjx.make_data(mjx_model)

    def one(controls):
        def step(data, u):
            data = data.replace(ctrl=jnp.reshape(u, (1,)))
            data = mjx.step(mjx_model, data)
            return data, data.qpos[0]

        _, angles = jax.lax.scan(step, initial, controls)
        return angles

    return jax.jit(jax.vmap(one))


@dataclass(frozen=True)
class PlanResult:
    controls: np.ndarray  # best motor command for every step
    best_scores: list[float]  # best score found after each iteration
    final_heights: list[float]  # tip height at the end of the best plan, after each iteration
    success: bool  # did the best plan end near the top?


def score_plans(angles: np.ndarray) -> np.ndarray:
    """Higher is better: mean tip height over the run plus the final tip height."""
    heights = tip_height(angles)
    return heights.mean(axis=1) + heights[:, -1]


def plan_swing_up(
    model,
    batch: int = 1024,
    iterations: int = 15,
    horizon: int = 300,
    knots: int = 30,
    elite_fraction: float = 0.1,
    seed: int = 0,
) -> PlanResult:
    """Find motor commands that swing the pole up, using the cross-entropy method.

    Each iteration tries ``batch`` random plans at the same time in the simulator, keeps the best
    ``elite_fraction`` of them, and samples the next batch around those. A plan has ``knots``
    values, each held for ``horizon // knots`` steps.
    """
    import jax.numpy as jnp

    if horizon % knots:
        raise ValueError("horizon must be a multiple of knots")
    rollout = make_batched_rollout(model)
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(knots), np.full(knots, 0.8)
    hold = horizon // knots
    n_elite = max(2, int(batch * elite_fraction))

    best_score, best_controls = -np.inf, np.zeros(horizon)
    best_scores, final_heights = [], []
    for _ in range(iterations):
        samples = np.clip(mean + std * rng.standard_normal((batch, knots)), -1.0, 1.0)
        controls = np.repeat(samples, hold, axis=1).astype(np.float32)
        angles = np.asarray(rollout(jnp.asarray(controls)))
        scores = score_plans(angles)
        order = np.argsort(-scores)
        if scores[order[0]] > best_score:
            best_score, best_controls = float(scores[order[0]]), controls[order[0]]
            best_final = float(tip_height(angles[order[0], -1]))
        elite = samples[order[:n_elite]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.05)
        best_scores.append(best_score)
        final_heights.append(best_final)
    return PlanResult(best_controls, best_scores, final_heights, success=final_heights[-1] > 0.9 * POLE_LENGTH)
