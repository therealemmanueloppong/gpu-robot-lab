import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("jax")

from gpu_robot_lab import policy, sim


@pytest.fixture(scope="module")
def model():
    return sim.make_model()


def test_parameter_count_matches_the_network_shape():
    params = policy.init_params(np.random.default_rng(0))
    assert params.shape == (policy.N_PARAMS,)
    w1, b1, w2, b2 = policy.unpack(params)
    assert w1.shape == (3, policy.HIDDEN) and b1.shape == (policy.HIDDEN,)
    assert w2.shape == (policy.HIDDEN, 1) and b2.shape == (1,)


def test_the_motor_command_stays_within_its_limits():
    rng = np.random.default_rng(1)
    params = rng.standard_normal(policy.N_PARAMS) * 5  # deliberately huge weights
    for _ in range(50):
        obs = policy.observation(rng.uniform(-7, 7), rng.uniform(-30, 30))
        assert -1.0 <= policy.action(params, obs) <= 1.0


def test_numpy_and_jax_versions_of_the_network_agree():
    import jax.numpy as jnp

    params = policy.init_params(np.random.default_rng(2))
    obs = policy.observation(0.7, -1.3)
    a_np = policy.action(params, obs, np)
    a_jax = policy.action(jnp.asarray(params), policy.observation(jnp.asarray(0.7), jnp.asarray(-1.3), jnp), jnp)
    assert abs(float(a_jax) - float(a_np)) < 1e-5


def test_evaluator_gives_one_finite_fitness_per_candidate(model):
    import jax.numpy as jnp

    evaluate = policy.make_evaluator(model, horizon=120)
    rng = np.random.default_rng(3)
    candidates = np.stack([policy.init_params(rng) for _ in range(6)])
    fitness = np.asarray(evaluate(jnp.asarray(candidates)))
    assert fitness.shape == (6,) and np.all(np.isfinite(fitness))


def test_training_improves_the_average_candidate(model):
    result = policy.train_es(model, population=128, iterations=15, horizon=200, seed=0)
    assert len(result.mean_fitness) == 15 and result.params.shape == (policy.N_PARAMS,)
    assert result.mean_fitness[-1] > result.mean_fitness[0]


def test_population_must_be_even(model):
    with pytest.raises(ValueError):
        policy.train_es(model, population=33, iterations=1)


def test_replay_in_plain_mujoco_returns_angles_and_commands(model):
    params = policy.init_params(np.random.default_rng(4))
    angles, commands = policy.rollout_policy_cpu(model, params, 0.2, steps=50)
    assert angles.shape == commands.shape == (50,)
    assert np.abs(commands).max() <= 1.0


def test_success_needs_the_pole_to_stay_near_the_top():
    assert policy.success(np.full(200, np.pi))
    assert not policy.success(np.zeros(200))


def test_curriculum_starts_grow_from_the_top_and_keep_the_easy_ones():
    easy = policy.curriculum_starts(0.3)
    hard = policy.curriculum_starts(float(np.pi))
    assert set(np.round(easy, 6)) <= set(np.round(hard, 6))
    assert len(hard) > len(easy)
    assert all(abs(a - np.pi) <= 0.3 + 1e-6 for a in easy)
    assert min(hard) <= 1e-6  # the hardest stage includes starting at the very bottom (angle 0)


def test_curriculum_training_returns_the_history_of_every_stage(model):
    result = policy.train_curriculum(model, population=32, iterations_per_stage=2, seed=0)
    assert len(result.mean_fitness) == 2 * len(policy.CURRICULUM_OFFSETS)
