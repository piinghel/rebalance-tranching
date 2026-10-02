"""Variance accounting and the equal-volatility approximation have distinct roles."""

import numpy as np
import pytest

from rebalance_tranching.ridge_evidence import covariance_checks


def test_uncorrelated_equal_vol_books_reduce_volatility_by_root_three():
    # Orthogonal columns give zero pairwise covariance and identical volatility.
    books = 0.001 + 0.01 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    returns = np.tile(books, (1, 5))
    keys = [(day, offset) for day in range(1, 6) for offset in range(3)]
    for row in covariance_checks(returns, keys):
        assert row["mean_correlation"] == pytest.approx(0, abs=1e-15)
        assert row["combined_volatility"] == pytest.approx(
            row["mean_standalone_volatility"] / np.sqrt(3)
        )
        assert row["equal_volatility_approximation"] == pytest.approx(
            row["combined_volatility"]
        )
        assert row["variance_identity_error"] < 1e-15
        assert row["log_growth_gain_pp"] > 0
        assert row["second_order_log_growth_gain_pp"] == pytest.approx(0.84)
