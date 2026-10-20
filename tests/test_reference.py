import numpy as np
import pytest

from gpu_robot_lab.reference import matmul_ref, softmax_ref


def test_matmul_ref_matches_a_hand_worked_example():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    assert np.array_equal(matmul_ref(a, b), np.array([[19, 22], [43, 50]], dtype=np.float32))


def test_matmul_ref_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        matmul_ref(np.zeros((2, 3)), np.zeros((4, 2)))


def test_softmax_rows_sum_to_one_and_keep_order():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 0.0, -10.0]])
    out = softmax_ref(x)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > out[0, 1] > out[0, 0]


def test_softmax_is_stable_for_huge_values():
    out = softmax_ref(np.array([[1000.0, 1000.0, 1000.0]]))
    assert np.all(np.isfinite(out)) and np.allclose(out, 1 / 3)


def test_softmax_ignores_a_constant_shift():
    x = np.random.default_rng(0).normal(size=(4, 7))
    assert np.allclose(softmax_ref(x), softmax_ref(x + 123.0))
