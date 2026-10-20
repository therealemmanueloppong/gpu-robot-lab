def gpu_ready() -> bool:
    """True when torch, triton and a CUDA GPU are all available (for example on a Colab GPU runtime)."""
    try:
        import torch
        import triton  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())
