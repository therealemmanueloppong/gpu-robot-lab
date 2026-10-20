# GPU Robot Lab

GPU kernels written from scratch and benchmarked against PyTorch, then used to speed up a simulated robot. Built to run on a free shared GPU, so it costs nothing.

Status: early scaffold. The testing and benchmarking tools are tested on a CPU. **The Triton kernels have not yet been run on a GPU and no performance numbers exist yet.** The first job is to run them on Colab and record real results here.

See [docs/PLAN.md](docs/PLAN.md) for the stages, the rules for what can be claimed, and the risks.

## What is in the repo

- `src/gpu_robot_lab/kernels/` a tiled matrix multiply and a row-wise softmax in Triton
- `src/gpu_robot_lab/bench.py` timing that works correctly for asynchronous GPU code, plus GFLOPS and speedup helpers
- `src/gpu_robot_lab/reference.py` and `checks.py` the CPU ground truth and error checking
- `scripts/check_kernels.py` checks every kernel against PyTorch
- `scripts/bench_matmul.py` times the Triton matmul against `torch.matmul` and writes a CSV
- `notebooks/colab_quickstart.ipynb` runs all of the above on a free Colab GPU

## Run the tests (no GPU needed)

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

GPU tests are skipped automatically when no CUDA GPU is found.

## Run the kernels on a free GPU

1. Open `notebooks/colab_quickstart.ipynb` in Google Colab (File, then Open notebook, then the GitHub tab).
2. Choose Runtime, then Change runtime type, and pick a GPU.
3. Run the cells in order. Expect `ALL PASSED` from the correctness check, then a results table.

Kaggle notebooks also offer a free GPU and work the same way. Free-tier limits change over time.

## Results

None yet. This section will list each measurement with the GPU, the date and the command that produced it.
