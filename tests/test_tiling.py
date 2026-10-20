import numpy as np
import pytest

from gpu_robot_lab.reference import matmul_ref
from gpu_robot_lab.tiling import naive_loads, tiled_matmul, traffic_reduction


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (8, 8, 8), (100, 77, 130), (65, 63, 31), (64, 64, 64)])
@pytest.mark.parametrize("blocks", [(16, 16, 8), (64, 64, 32), (32, 128, 16)])
def test_tiled_result_matches_the_reference_for_awkward_shapes(m, n, k, blocks):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    c, _ = tiled_matmul(a, b, *blocks)
    assert np.allclose(c, matmul_ref(a, b), rtol=1e-4, atol=1e-4)


def test_program_and_step_counts():
    a, b = np.ones((100, 130), np.float32), np.ones((130, 77), np.float32)
    _, stats = tiled_matmul(a, b, 64, 64, 32)
    assert stats.programs == 2 * 2 and stats.k_steps == 5


def test_loads_formula_for_evenly_divisible_sizes():
    n = 64
    a, b = np.ones((n, n), np.float32), np.ones((n, n), np.float32)
    for bm, bn, bk in [(8, 8, 8), (16, 32, 8), (64, 64, 32)]:
        _, stats = tiled_matmul(a, b, bm, bn, bk)
        assert stats.loads == (bm + bn) * n * (n // bm) * (n // bn)
        assert naive_loads(n, n, n) / stats.loads == pytest.approx(traffic_reduction(bm, bn))


def test_bigger_tiles_read_less():
    assert traffic_reduction(64, 64) == 64.0
    assert traffic_reduction(128, 128) > traffic_reduction(64, 64) > traffic_reduction(16, 16)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        tiled_matmul(np.zeros((2, 3)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        tiled_matmul(np.zeros((2, 2)), np.zeros((2, 2)), 0, 4, 4)


from gpu_robot_lab.tiling import grouped_order, row_major_order, tiles_needed


@pytest.mark.parametrize("shape", [(1, 1), (8, 8), (7, 5), (5, 7), (13, 3), (64, 64)])
@pytest.mark.parametrize("group_m", [1, 2, 4, 8, 100])
def test_grouped_order_visits_every_tile_exactly_once(shape, group_m):
    order = grouped_order(*shape, group_m)
    assert len(order) == shape[0] * shape[1]
    assert set(order) == set(row_major_order(*shape))


def test_group_of_one_is_the_plain_row_major_order():
    assert grouped_order(6, 5, 1) == row_major_order(6, 5)


def test_a_square_group_needs_fewer_loaded_bands_than_row_major():
    wave = 16  # programs imagined to run at the same time
    row = tiles_needed(row_major_order(8, 8)[:wave])
    grouped = tiles_needed(grouped_order(8, 8, 4)[:wave])
    assert (row, grouped) == (10, 8)


def test_group_size_must_be_positive():
    with pytest.raises(ValueError):
        grouped_order(4, 4, 0)
