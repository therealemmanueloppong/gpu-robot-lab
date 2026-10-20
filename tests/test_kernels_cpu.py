from gpu_robot_lab.kernels import gpu_ready


def test_gpu_ready_returns_a_bool_even_without_torch_or_a_gpu():
    assert isinstance(gpu_ready(), bool)
