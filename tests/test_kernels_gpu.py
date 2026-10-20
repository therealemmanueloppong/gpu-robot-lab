"""GPU tests. They are skipped on machines without a CUDA GPU, torch and triton (like CI)."""
import pytest

from gpu_robot_lab.kernels import gpu_ready

pytestmark = pytest.mark.skipif(not gpu_ready(), reason="needs a CUDA GPU with torch and triton")


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (100, 77, 130), (512, 512, 512)])
def test_triton_matmul_matches_torch(m, n, k):
    import torch

    from gpu_robot_lab.kernels.matmul import matmul

    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    assert torch.allclose(matmul(a, b), a @ b, rtol=1e-2, atol=1e-1)


@pytest.mark.parametrize("rows,cols", [(1, 10), (128, 257), (1024, 1000)])
def test_triton_softmax_matches_torch(rows, cols):
    import torch

    from gpu_robot_lab.kernels.softmax import softmax

    x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
    assert torch.allclose(softmax(x), torch.softmax(x, dim=1), rtol=1e-4, atol=1e-5)
