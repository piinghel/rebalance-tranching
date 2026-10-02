"""Calendar accounting and descriptive decomposition have distinct contracts."""

import datetime as dt

import polars as pl
import pytest

from rebalance_tranching.calendar_grid import (
    combine_grid,
    decompose,
    tranche_comparison,
)


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
    pairs = combine_grid(frame, include_pairs=True).filter(
        (pl.col("sleeves") == 2) & (pl.col("weekday") == 1)
    )
    assert pairs["schedules"].to_list() == ["1+2", "1+3", "2+3"]
    assert pairs["gross"].to_list() == pytest.approx([0.0015, 0.002, 0.0025])
    assert pairs["order_count"].to_list() == [4, 4, 4]
    assert pairs["traded_notional"].to_list() == [1.0] * 3
    second_day = frame.with_columns(
        pl.lit(dt.date(2022, 1, 4)).alias("date"),
        (-pl.col("gross")).alias("gross"),
        (-pl.col("gross") - 0.0005).alias("net"),
    )
    comparison = tranche_comparison(pl.concat([frame, second_day]))
    assert comparison["calendar_count"].to_list() == [15, 15, 5]
    assert comparison["annual_orders"].to_list() == [504.0, 1008.0, 1512.0]
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


def test_weekday_dependent_offset_effect_is_interaction():
    frame = pl.DataFrame(
        [
            {
                "weekday": day,
                "schedules": str(offset + 2),
                "sleeves": 1,
                "net_cagr": float(10 + (day - 3) * offset),
            }
            for day in range(1, 6)
            for offset in (-1, 0, 1)
        ]
    )
    ss = decompose(frame)
    assert ss["weekday"] == pytest.approx(0)
    assert ss["offset"] == pytest.approx(0)
    assert ss["interaction"] == pytest.approx(20)
    assert ss["total"] == pytest.approx(20)
