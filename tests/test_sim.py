"""Tests for the simulated pendulum. Skipped where mujoco, jax or matplotlib are not installed."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("jax")

from gpu_robot_lab import sim


@pytest.fixture(scope="module")
def model():
    return sim.make_model()


def test_tip_height_at_the_bottom_and_top():
    assert sim.tip_height(0.0) == pytest.approx(-sim.POLE_LENGTH)
    assert sim.tip_height(np.pi) == pytest.approx(sim.POLE_LENGTH)


def test_a_pole_hanging_still_stays_down(model):
    angles = sim.rollout_cpu(model, np.zeros(300))
    assert np.abs(angles).max() == 0.0


def test_a_released_pole_swings_back_and_forth(model):
    angles = sim.rollout_cpu(model, np.zeros(400), start_angle=0.5)
    assert angles.min() < -0.2 and angles.max() > 0.2


def test_the_motor_alone_cannot_lift_the_pole_to_the_top(model):
    angles = sim.rollout_cpu(model, np.ones(600))  # full torque the whole time
    assert sim.tip_height(angles).max() < 0.0  # never even reaches the level of the hinge


def test_gpu_simulator_matches_the_cpu_simulator(model):
    controls = np.clip(np.random.default_rng(1).normal(size=(1, 200)), -1, 1).astype(np.float32)
    batched = np.asarray(sim.make_batched_rollout(model)(controls))
    cpu = sim.rollout_cpu(model, controls[0].astype(np.float64))
    assert np.abs(batched[0] - cpu).max() < 1e-4


def test_each_robot_in_a_batch_is_independent(model):
    rng = np.random.default_rng(2)
    controls = np.clip(rng.normal(size=(4, 100)), -1, 1).astype(np.float32)
    rollout = sim.make_batched_rollout(model)
    together = np.asarray(rollout(controls))
    alone = np.asarray(rollout(controls[2:3]))
    assert together.shape == (4, 100)
    assert np.allclose(together[2], alone[0], atol=1e-5)


def test_score_prefers_a_pole_that_ends_up_high():
    low = np.full((1, 10), 0.0)
    high = np.full((1, 10), np.pi)
    assert sim.score_plans(high)[0] > sim.score_plans(low)[0]


def test_planner_validates_its_settings(model):
    with pytest.raises(ValueError):
        sim.plan_swing_up(model, horizon=100, knots=30)


def test_planner_swings_the_pole_up_and_a_cpu_replay_confirms_it(model):
    plan = sim.plan_swing_up(model, batch=512, iterations=15, seed=0)
    assert plan.success
    assert plan.best_scores[-1] >= plan.best_scores[0]  # the best plan never gets worse
    replay = sim.rollout_cpu(model, plan.controls.astype(np.float64))
    assert sim.tip_height(replay[-1]) > 0.9 * sim.POLE_LENGTH
    assert np.abs(plan.controls).max() <= 1.0  # respects the motor limit


def test_gif_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from gpu_robot_lab import viz

    out = viz.animate_pendulum(np.linspace(0, 3, 40), tmp_path / "pole.gif", dt=0.01, every=5)
    assert out.read_bytes()[:3] == b"GIF"
