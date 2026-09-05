"""Calendar accounting and descriptive decomposition have distinct contracts."""

import datetime as dt

import polars as pl
import pytest

from rebalance_tranching.calendar_grid import combine_grid, decompose


def test_equal_notional_combination_preserves_returns_but_sums_orders():
    frame = pl.DataFrame(
        [
            {
                "date": dt.date(2022, 1, 3),
                "weekday": d,
                "offset": o,
                "gross": (d + o) / 1000,
                "net": (d + o) / 1000 - 0.0005,
                "traded_notional": 1.0,
                "order_count": 2,
            }
            for d in range(1, 6)
            for o in range(3)
        ]
    )
    result = combine_grid(frame).lazy().filter(pl.col("sleeves") == 3).collect()
    assert result["gross"].to_list() == pytest.approx(
        [0.002, 0.003, 0.004, 0.005, 0.006]
    )
    assert result["traded_notional"].to_list() == [1.0] * 5
    assert result["order_count"].to_list() == [6] * 5
    with pytest.raises(ValueError):
        combine_grid(frame.slice(1))
    with pytest.raises(ValueError):
        combine_grid(pl.concat([frame, frame.head(1)]))


def test_additive_calendar_effects_have_no_interaction():
    frame = pl.DataFrame(
        [
            {
                "weekday": d,
                "schedules": str(o),
                "sleeves": 1,
                "net_cagr": float(d + 10 * o),
            }
            for d in range(1, 6)
            for o in range(1, 4)
        ]
    )
    ss = decompose(frame)
    assert ss["interaction"] == pytest.approx(0)
    assert ss["offset"] == pytest.approx(1000)
    assert ss["weekday"] == pytest.approx(30)
    assert ss["total"] == pytest.approx(
        sum(ss[k] for k in ("weekday", "offset", "interaction"))
    )
