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

1. Pick one GPU-accelerated simulator that runs on a free Colab GPU (MuJoCo's GPU version or Brax are the candidates) and confirm it installs first.
2. Train a policy for a simple robot task with reinforcement learning, and record reward curves and a video.
3. Measure simulation speed in steps per second on the GPU compared with the CPU. This is the number that shows why GPUs matter for robotics.

Done when: a trained policy video and a CPU versus GPU speed table are in the repo.

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
