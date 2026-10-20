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
    rollout=None,
) -> PlanResult:
    """Find motor commands that swing the pole up, using the cross-entropy method.

    Each iteration tries ``batch`` random plans at the same time in the simulator, keeps the best
    ``elite_fraction`` of them, and samples the next batch around those. A plan has ``knots``
    values, each held for ``horizon // knots`` steps. Pass a ``rollout`` from ``make_batched_rollout`` to
    reuse an already compiled simulator (otherwise one is built, and compiled, on the first iteration).
    """
    import jax.numpy as jnp

    if horizon % knots:
        raise ValueError("horizon must be a multiple of knots")
    rollout = rollout or make_batched_rollout(model)
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


# ---------------------------------------------------------------------------------------------
# A contact-rich test: a box dropped onto the floor. Contacts (collisions, friction) make each
# simulation step much more expensive than the pendulum, so this shows how the CPU and GPU compare
# when the physics is harder. It is not a walking robot.

BOX_DT = 0.005
BOX_START_HEIGHT = 0.6
BOX_HALF_SIZE = 0.1

BOX_XML = f"""
<mujoco model="box">
  <option timestep="{BOX_DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 {BOX_START_HEIGHT}">
      <freejoint/>
      <geom type="box" size="{BOX_HALF_SIZE} {BOX_HALF_SIZE} {BOX_HALF_SIZE}" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_box_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(BOX_XML)


def random_box_starts(batch: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random orientations and small random velocities for ``batch`` boxes: qpos (batch, 7), qvel (batch, 6)."""
    rng = np.random.default_rng(seed)
    quat = rng.standard_normal((batch, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qpos = np.zeros((batch, 7))
    qpos[:, 2] = BOX_START_HEIGHT
    qpos[:, 3:] = quat
    qvel = rng.normal(scale=0.5, size=(batch, 6))
    return qpos.astype(np.float32), qvel.astype(np.float32)


def box_rollout_cpu(model, qpos0: np.ndarray, qvel0: np.ndarray, steps: int) -> float:
    """Drop one box in plain MuJoCo and return its height at the end."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return float(data.qpos[2])


def make_box_rollout(model, steps: int):
    """Return a function (qpos0 (batch, 7), qvel0 (batch, 6)) -> final box heights (batch,), simulated with MJX."""
    import jax
    from mujoco import mjx

    mjx_model = mjx.put_model(model)
    initial = mjx.make_data(mjx_model)

    def one(qpos0, qvel0):
        data = initial.replace(qpos=qpos0, qvel=qvel0)

        def step(d, _):
            return mjx.step(mjx_model, d), None

        data, _ = jax.lax.scan(step, data, None, length=steps)
        return data.qpos[2]

    return jax.jit(jax.vmap(one))
