"""The experiment scripts and the contact model, run on the CPU so mistakes show up before any GPU time."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("jax")

from gpu_robot_lab import sim

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def run_script(name, args, cwd):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args], check=False, cwd=cwd, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_box_settles_on_the_floor_in_plain_mujoco():
    model = sim.make_box_model()
    qpos, qvel = sim.random_box_starts(1, seed=0)
    z = sim.box_rollout_cpu(model, qpos[0], qvel[0], steps=400)
    assert 0.0 < z < 0.2  # the box is resting on the floor (a cube of half-size 0.1 sits at most about 0.17 up)


def test_mjx_box_simulation_agrees_with_plain_mujoco():
    model = sim.make_box_model()
    qpos, qvel = sim.random_box_starts(4, seed=1)
    final = np.asarray(sim.make_box_rollout(model, 400)(qpos, qvel))
    cpu = np.array([sim.box_rollout_cpu(model, qpos[i], qvel[i], 400) for i in range(4)])
    assert final.shape == (4,)
    assert np.abs(final - cpu).max() < 0.03  # contacts amplify tiny differences, so the tolerance is looser


def test_planner_with_a_prebuilt_simulator_gives_the_same_plan():
    model = sim.make_model()
    rollout = sim.make_batched_rollout(model)
    a = sim.plan_swing_up(model, batch=64, iterations=3, seed=5)
    b = sim.plan_swing_up(model, batch=64, iterations=3, seed=5, rollout=rollout)
    assert np.allclose(a.best_scores, b.best_scores)


def test_bench_sim_script_runs_for_both_models(tmp_path):
    out = run_script("bench_sim.py", ["--steps", "20", "--batches", "1", "4", "--repeats", "2"], tmp_path)
    assert "speedup_vs_cpu_1_thread" in out and (tmp_path / "sim_results.csv").exists()
    out = run_script("bench_sim.py", ["--model", "box", "--steps", "20", "--batches", "1", "4", "--repeats", "2"], tmp_path)
    assert (tmp_path / "sim_results_box.csv").exists()
    header = (tmp_path / "sim_results_box.csv").read_text().splitlines()[0]
    assert header.startswith("model,batch,device,mjx_steps_per_s,mjx_min,mjx_max")


def test_bench_planner_script_runs(tmp_path):
    run_script("bench_planner.py", ["--batches", "16", "--iterations", "2"], tmp_path)
    assert (tmp_path / "planner_cpu.csv").exists()


def test_train_policy_script_writes_all_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "policy"
    run_script("train_policy.py", ["--population", "32", "--iterations", "2", "--out", str(out)], tmp_path)
    for name in ["policy.json", "policy_curve.png", "policy_swingup.gif", "policy_params.npy"]:
        assert (out / name).exists(), name
