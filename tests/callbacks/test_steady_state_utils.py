from lit_llms.callbacks.steady_state_utils import (
    _check_atol,
    _check_rtol,
    _check_tols,
    calc_total_time_per_node,
    is_steady_state,
)


def test_is_steady_state():
    assert is_steady_state(1.014, 1 - 0.014, 1, rtol=0.015, atol=0.015)
    assert not is_steady_state(10.14, 10 - 0.14, 10, rtol=0.015, atol=0.015)
    assert is_steady_state(10.14, 10 - 0.14, 10, rtol=0.015)


def test_check_atol():
    assert _check_atol(1, 1, 0.015)
    assert not _check_atol(1, 1.015, 0.001)
    assert _check_atol(1, 1, None)
    assert _check_atol(1, 10, None)


def test_check_rtol():
    assert _check_rtol(1, 1, 0.015)
    assert not _check_rtol(1, 1.015, 0.01)


def test_check_tols():
    assert _check_tols(1, 1.014, 0.015, 0.015)
    assert not _check_tols(1, 1.014, 0.015, 0.01)
    assert _check_tols(1, 1.014, 0.015, None)
    assert not _check_tols(1, 1.014, 0.01, None)
    assert not _check_tols(1, 1.014, 0.01, 0.01)


def test_calc_total_time_per_node():
    assert calc_total_time_per_node(0, 1, 1, 1) == 0
    assert calc_total_time_per_node(1, 1, 1, 0) == 0
    assert calc_total_time_per_node(3600, 1, 1, 1) == 1
    assert calc_total_time_per_node(3600, 1, 1, 2) == 2
    assert calc_total_time_per_node(3600, 1, 2, 1) == 0.5
    assert calc_total_time_per_node(3600, 2, 1, 1) == 0.5
