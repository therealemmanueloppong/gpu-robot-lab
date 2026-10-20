import numpy as np
import pytest

from gpu_robot_lab.checks import assert_close, max_abs_error


def test_max_abs_error_finds_the_worst_element():
    assert max_abs_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0])) == pytest.approx(0.5)


def test_assert_close_passes_for_small_errors():
    assert_close(np.array([1.0, 2.0]), np.array([1.001, 2.001]))


def test_assert_close_reports_the_error_size():
    with pytest.raises(AssertionError, match="max abs error 1.0"):
        assert_close(np.array([1.0]), np.array([2.0]))


def test_assert_close_rejects_nan_and_wrong_shapes():
    with pytest.raises(AssertionError):
        assert_close(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(AssertionError):
        assert_close(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        max_abs_error(np.zeros(2), np.zeros(3))
