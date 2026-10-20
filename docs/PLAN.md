# GPU Robot Lab plan and status

## Goal

Write GPU kernels from scratch, measure them honestly against PyTorch, and then use GPU-accelerated simulation to train a simulated robot. Everything runs on a free shared GPU, so the project costs nothing.

## Rules for claims

Only claim what was measured and can be reproduced from this repo. Every number in the README names the GPU, the date and the command that produced it. If a kernel is slower than PyTorch, or an experiment fails, the README says so.

## Status: first complete version (2026-10-20)

Stage 1 and stage 2 are finished and measured on a Colab T4. The full results, with the raw logs, are in the README and in `results/full_run_2026-10-20/`.

### Stage 1: kernels in Triton (done)

- Matrix multiply (v1 and an autotuned v2) and row-wise softmax, all checked against PyTorch.
- The first kernel reached about 4% of `torch.matmul`. A 135-setting sweep and compiler output showed it was spilling registers (255 registers, 84 spills). Tuned versions run about 3.4 times faster.
- A launch order that groups tiles gives another 4 to 14%.
- The compiled kernels contain no tensor-core instructions, while PyTorch runs a tensor-core library kernel. This is the likely main reason for the remaining 6 to 10 times gap, but it was not tested by forcing the tensor cores.
- Not done: a fused softmax and a small attention kernel (the original step 4).

### Stage 2: robot simulation (done)

- A simulated pendulum with a motor too weak to lift it. A planner tries thousands of motor plans at once in MuJoCo MJX and finds a swing-up, checked by an independent replay.
- MJX on the GPU simulates up to about 126 times more steps per second than one CPU thread (pendulum). With collisions (a dropped box) the advantage shrinks to 2.75 times at 4,096 boxes.
- The planner itself is 2 to 26 times faster per iteration on the GPU at the same batch size.
- A neural-network controller trained by evolution strategies with a curriculum swings the pole up and holds it. See the README for the exact outcomes, including a failed run.

### Stage 3 (dropped)

A fused policy kernel inside the robot loop was optional and was not attempted.

## If this is continued

1. Force the matrix-multiply kernel onto the tensor cores, or try a newer GPU, and measure the change.
2. Make the learned controller train reliably at larger population sizes.
3. Move from a pendulum to a robot with legs, and from evolution strategies to PPO.
4. Repeat the measurements to tighten the error bars, since a shared GPU varies from run to run.

## Where things run

| What | Where | Cost |
|---|---|---|
| Unit tests, linting, CI | A laptop, a Codespace, GitHub Actions | free |
| Kernel checks, benchmarks, robot experiments | Google Colab with a GPU runtime (`notebooks/all_experiments_colab.ipynb`) | free, with usage limits that change |

## Risks

- Free GPU time is limited and a session can be reclaimed.
- Results on a shared GPU vary from run to run (PyTorch's own speed varied by up to 44% between repeats).
- Triton and simulator versions on Colab change. Versions used: torch 2.11.0, triton 3.6.0, CUDA 13.0, jax 0.11.1, mujoco 3.16.0.
