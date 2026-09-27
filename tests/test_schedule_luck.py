import numpy as np

from rebalance_tranching.schedule_luck import cagr, null_spreads, spread


def test_cagr_annualizes_a_constant_daily_return():
    daily = np.full((252, 1), 0.0004)
    assert np.isclose(cagr(daily)[0], 100 * (1.0004**252 - 1))


def test_identical_schedules_have_no_spread_under_the_null():
    column = np.random.default_rng(1).normal(0.0005, 0.01, 500)
    returns = np.column_stack([column] * 3)
    assert spread(returns) == 0
    assert np.allclose(null_spreads(returns, block=20, draws=20), 0)


def test_null_removes_differences_in_expected_return():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.01, (2000, 1))
    returns = base + np.array([[0.0, 0.001, 0.002]])  # large, persistent differences
    assert spread(returns) > 40
    assert np.median(null_spreads(returns, block=20, draws=50)) < 1
