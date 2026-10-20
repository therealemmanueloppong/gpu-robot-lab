import pytest

from gpu_robot_lab.bench import gflops, matmul_flops, speedup, time_fn


def test_matmul_flops_counts_a_multiply_and_an_add_per_term():
    assert matmul_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_gflops_for_a_known_case():
    # 2e9 operations in 1000 ms is 2 GFLOPS
    assert gflops(2e9, 1000.0) == pytest.approx(2.0)


def test_gflops_rejects_zero_time():
    with pytest.raises(ValueError):
        gflops(1.0, 0.0)


def test_speedup_above_one_means_faster():
    assert speedup(10.0, 5.0) == 2.0
    assert speedup(5.0, 10.0) == 0.5


def test_time_fn_runs_warmup_plus_iterations():
    calls = []
    result = time_fn(lambda: calls.append(1), warmup=3, iters=7)
    assert len(calls) == 10 and result.runs == 7
    assert 0 <= result.min_ms <= result.median_ms


def test_time_fn_syncs_around_every_timed_call():
    syncs = []
    time_fn(lambda: None, warmup=0, iters=4, sync=lambda: syncs.append(1))
    # once after warm-up, then before and after each of the 4 timed calls
    assert len(syncs) == 1 + 2 * 4


def test_time_fn_rejects_bad_settings():
    with pytest.raises(ValueError):
        time_fn(lambda: None, iters=0)
