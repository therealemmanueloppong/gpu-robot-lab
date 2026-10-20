"""A learned feedback policy for the pendulum, trained by evolution strategies on parallel simulations.

The policy is a small neural network that looks at the pole's angle and speed and outputs a motor
command. Unlike the open-loop plan from ``sim.plan_swing_up``, it reacts to what it sees, so it
works from different starting angles. Training tries thousands of slightly different networks at
once in MJX, which is the kind of workload a GPU is good at.

Needs the ``sim`` extra: ``pip install -e '.[sim]'``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import POLE_LENGTH, make_model, tip_height

HIDDEN = 32
N_OBS = 3  # cos(angle), sin(angle), angular speed (scaled)
SPEED_SCALE = 8.0
N_PARAMS = N_OBS * HIDDEN + HIDDEN + HIDDEN + 1


def observation(angle, speed, xp=np):
    return xp.stack([xp.cos(angle), xp.sin(angle), speed / SPEED_SCALE])


def unpack(params, xp=np):
    i = 0
    w1 = params[i : i + N_OBS * HIDDEN].reshape(N_OBS, HIDDEN)
    i += N_OBS * HIDDEN
    b1 = params[i : i + HIDDEN]
    i += HIDDEN
    w2 = params[i : i + HIDDEN].reshape(HIDDEN, 1)
    i += HIDDEN
    b2 = params[i : i + 1]
    return w1, b1, w2, b2


def action(params, obs, xp=np):
    """Motor command in [-1, 1] for one observation."""
    w1, b1, w2, b2 = unpack(params, xp)
    hidden = xp.tanh(obs @ w1 + b1)
    return xp.tanh((hidden @ w2 + b2)[0])


def init_params(rng: np.random.Generator) -> np.ndarray:
    p = rng.standard_normal(N_PARAMS).astype(np.float32) * 0.1
    return p


# Starting angles used in training: four near the bottom (the policy must swing up) and two near the
# top (the policy must hold the pole there, which gives balancing its own direct learning signal).
TRAIN_STARTS = (-0.3, -0.1, 0.1, 0.3, float(np.pi - 0.25), float(np.pi + 0.25))


# Stage 1 of the curriculum: only starts close to the top, so the policy first learns to hold the pole there.
BALANCE_STARTS = (float(np.pi - 0.3), float(np.pi - 0.1), float(np.pi + 0.1), float(np.pi + 0.3))


def make_evaluator(model, horizon: int = 400, starts=TRAIN_STARTS):
    """Return a jitted function mapping a batch of parameter vectors to a fitness for each."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    mjx_model = mjx.put_model(model)
    initial = mjx.make_data(mjx_model)
    start_angles = jnp.asarray(starts, dtype=jnp.float32)

    def run(params, start):
        data = initial.replace(qpos=initial.qpos.at[0].set(start))

        def step(d, _):
            obs = observation(d.qpos[0], d.qvel[0], jnp)
            u = action(params, obs, jnp)
            d = d.replace(ctrl=jnp.reshape(u, (1,)))
            d = mjx.step(mjx_model, d)
            return d, d.qpos[0]

        _, angles = jax.lax.scan(step, data, None, length=horizon)
        return angles

    def fitness(params):
        angles = jax.vmap(lambda s: run(params, s))(start_angles)  # (starts, horizon)
        height = -jnp.cos(angles)  # +1 at the top, -1 at the bottom
        # mean height rewards being up, the last second rewards ending up there and staying, and the highest point
        # reached gives a smooth signal while the pole is only swinging a little (it must pump energy first)
        tail = height[:, -100:].mean(axis=1)  # staying near the top for the last second
        return height.mean() + 2.0 * tail.mean() + height.max(axis=1).mean()

    return jax.jit(jax.vmap(fitness))


@dataclass(frozen=True)
class PolicyResult:
    params: np.ndarray
    mean_fitness: list[float]  # average over the population, per iteration
    best_fitness: list[float]


def train_es(
    model=None,
    population: int = 1024,
    iterations: int = 60,
    sigma: float = 0.2,
    learning_rate: float = 0.05,
    horizon: int = 400,
    seed: int = 0,
    starts=TRAIN_STARTS,
    init=None,
) -> PolicyResult:
    """Evolution strategies with mirrored sampling and an Adam update."""
    import jax.numpy as jnp

    if population % 2:
        raise ValueError("population must be even")
    model = model or make_model()
    evaluate = make_evaluator(model, horizon, starts)
    rng = np.random.default_rng(seed)
    params = init_params(rng) if init is None else np.asarray(init, dtype=np.float32).copy()
    m, v = np.zeros_like(params), np.zeros_like(params)
    mean_fit, best_fit = [], []
    for t in range(1, iterations + 1):
        eps = rng.standard_normal((population // 2, N_PARAMS)).astype(np.float32)
        candidates = np.concatenate([params + sigma * eps, params - sigma * eps])
        fit = np.asarray(evaluate(jnp.asarray(candidates)))
        mean_fit.append(float(fit.mean()))
        best_fit.append(float(fit.max()))
        ranks = np.argsort(np.argsort(fit)).astype(np.float32) / (population - 1) - 0.5  # centered ranks
        weights = ranks[: population // 2] - ranks[population // 2 :]
        grad = (weights[:, None] * eps).sum(axis=0) / (population * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        params = params + learning_rate * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return PolicyResult(params.astype(np.float32), mean_fit, best_fit)


# Curriculum: how far from the top the pole may start, in radians. 0.3 is nearly upright, 3.14 is hanging down.
CURRICULUM_OFFSETS = (0.3, 1.0, 1.8, 2.5, float(np.pi))


def curriculum_starts(max_offset: float) -> tuple[float, ...]:
    """Starting angles up to ``max_offset`` away from the top, on both sides (keeps all easier starts too)."""
    offsets = [o for o in (0.1, 0.3, 0.6, 1.0, 1.4, 1.8, 2.2, 2.5, 2.9, float(np.pi)) if o <= max_offset + 1e-6]
    return tuple(float(np.pi - o) for o in offsets) + tuple(float(np.pi + o) for o in offsets if o < np.pi - 1e-6)


def train_curriculum(
    model=None,
    population: int = 1024,
    iterations_per_stage: int = 40,
    seed: int = 0,
    offsets=CURRICULUM_OFFSETS,
) -> PolicyResult:
    """Learn to balance first, then to recover from larger and larger leans, ending with a swing-up from the bottom.

    Every stage keeps the easier starting angles, so earlier skills are not forgotten. The returned
    history covers all stages in order.
    """
    model = model or make_model()
    params, mean_fit, best_fit = None, [], []
    for stage, offset in enumerate(offsets):
        r = train_es(model, population, iterations_per_stage, seed=seed + stage, starts=curriculum_starts(offset), init=params)
        params = r.params
        mean_fit += r.mean_fitness
        best_fit += r.best_fitness
    return PolicyResult(params, mean_fit, best_fit)


def rollout_policy_cpu(model, params: np.ndarray, start_angle: float = 0.1, steps: int = 400):
    """Run the trained policy in plain MuJoCo with a plain NumPy network. Returns angles and commands."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[0] = start_angle
    angles, commands = np.empty(steps), np.empty(steps)
    p = np.asarray(params, dtype=np.float64)
    for i in range(steps):
        u = float(action(p, observation(data.qpos[0], data.qvel[0]), np))
        data.ctrl[0] = u
        import mujoco as mj

        mj.mj_step(model, data)
        angles[i], commands[i] = data.qpos[0], u
    return angles, commands


def success(angles: np.ndarray, last_seconds_steps: int = 100) -> bool:
    """Did the pole spend the last stretch near the top?"""
    return bool(tip_height(angles[-last_seconds_steps:]).mean() > 0.8 * POLE_LENGTH)
