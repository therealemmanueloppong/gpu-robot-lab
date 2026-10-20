# GPU Robot Lab plan

## Goal

Write GPU kernels from scratch, measure them honestly against PyTorch, and then use GPU-accelerated simulation to train a simulated robot. Everything runs on a free shared GPU, so the project costs nothing.

## Rules for claims

Only claim what was measured and can be reproduced from this repo. Every number in the README names the GPU, the date and the command that produced it. If a kernel is slower than PyTorch, the README says so and explains why.

## Stage 1: kernels in Triton (Python)

1. Tiled matrix multiply, checked against `torch.matmul` for correctness, then timed.
2. Row-wise softmax.
3. Tuning: try different tile sizes and measure the effect of each change, one at a time.
4. Fused softmax and a small attention kernel.

Done when: `scripts/check_kernels.py` passes on a GPU and `scripts/bench_matmul.py` produces a results table.

## Stage 2: robot simulation on the GPU

Status: built, verified on a CPU, and measured on a Colab T4 (up to 129 times the simulation steps per second of one CPU thread at 16,384 robots; swing-up found on the GPU).

What was built: a simulated pendulum whose motor is too weak to lift it directly, so it has to swing. A planner tries thousands of motor plans at once in MuJoCo's GPU-capable simulator (MJX), keeps the best, and repeats (the cross-entropy method). The best plan is replayed in plain MuJoCo as an independent check, then drawn as a GIF. `scripts/bench_sim.py` compares one CPU thread against many robots at once.

Why planning instead of reinforcement learning: it has far fewer moving parts and library versions, so it can be tested on a CPU before any GPU time is used, and it uses the GPU in the same way (thousands of simultaneous simulations). A learned policy remains possible later.

Added after the first GPU run (code and CPU tests done; GPU results pending): repeated measurements, a collisions-and-friction test, a fair CPU versus GPU timing of the planner, and a learned neural-network policy trained by evolution strategies. `notebooks/all_experiments_colab.ipynb` runs them all and downloads one zip.

Still not done: a walking robot, and reinforcement-learning algorithms such as PPO.

## Stage 3 (optional): use a custom kernel in the robot loop

Decide after stage 2, depending on what works on the free tier. The idea is a fused kernel for the policy network, benchmarked in the rollout loop. If the simulator and Triton do not combine cleanly, stage 3 is dropped and the repo stays honest about it.

## Where things run

| What | Where | Cost |
|---|---|---|
| Unit tests, linting, CI | Your laptop, a Codespace, GitHub Actions | free |
| Kernel checks and benchmarks | Google Colab (or Kaggle) with a GPU runtime | free, with usage limits that change |
| Real hardware (Jetson, robot kit) | Not used. Possible later if funded | not needed |

## Risks

- Free GPU time is limited and a session can be reclaimed. Save results to the repo (CSV files) as they are produced.
- The Triton kernels in this scaffold have not yet been run on a GPU. Stage 1 starts with that.
- Triton and simulator versions on Colab change. Pin the versions that worked once they are known.
