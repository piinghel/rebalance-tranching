import datetime as dt
import itertools
import math

import polars as pl
import pytest

import rebalance_tranching.analysis as timing


@pytest.fixture
def schedules() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": dt.date(2022, 1, 3) + dt.timedelta(days=day),
                "offset": offset,
                "gross": 0.001 * (day - offset),
                "net": 0.001 * (day - offset) - 0.0001,
            }
            for offset, day in itertools.product(range(3), range(4))
        ]
    )


def test_all_subsets_preserve_linear_return_and_cost_identity(schedules):
    result = timing.mixtures(schedules)
    assert result["schedules"].n_unique() == 7
    blend = result.filter(pl.col("sleeves") == 3).sort("date")
    expected = (
        schedules.group_by("date").agg(pl.col("gross", "net").mean()).sort("date")
    )
    assert blend.select("date", "gross", "net").equals(expected)
    assert (blend["gross"] - blend["net"]).to_list() == pytest.approx([0.0001] * 4)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "nan", "null_date", "ruin"]
)
def test_reject_bad_schedule_observations(schedules, corruption):
    if corruption == "missing":
        schedules = schedules.slice(1)
    elif corruption == "duplicate":
        schedules = pl.concat([schedules, schedules.head(1)])
    elif corruption == "null_date":
        schedules = schedules.with_columns(pl.lit(None, dtype=pl.Date).alias("date"))
    else:
        schedules = schedules.with_columns(
            pl.lit(float("nan") if corruption == "nan" else -1.0).alias("net")
        )
    with pytest.raises(ValueError):
        timing.mixtures(schedules)


def test_initial_loss_counts_toward_drawdown():
    frame = pl.DataFrame(
        {
            "date": [dt.date(2022, 1, 3), dt.date(2022, 1, 4)],
            "gross": [-0.1, 0.02],
            "net": [-0.1, 0.02],
            "schedules": ["1", "1"],
            "sleeves": [1, 1],
        }
    )
    summary = timing.summarize(frame).row(0, named=True)
    assert summary["drawdown"] == pytest.approx(-10)
    assert summary["net_cagr"] == pytest.approx((0.9 * 1.02) ** 126 * 100 - 100)
    assert math.isfinite(summary["sharpe"])


def test_saved_three_sleeve_returns_match_the_funded_daily_mean():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data"
    daily = pl.read_parquet(root / "timing_daily.parquet")
    expected = (
        daily.filter(pl.col("sleeves") == 1)
        .group_by("date")
        .agg(pl.col("gross", "net").mean())
        .sort("date")
    )
    mixture = daily.filter(pl.col("sleeves") == 3).sort("date")
    assert mixture["date"].equals(expected["date"])
    for column in ("gross", "net"):
        assert mixture[column].to_list() == pytest.approx(
            expected[column].to_list(), abs=1e-14
        )
