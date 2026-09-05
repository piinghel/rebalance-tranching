"""Matched 3-by-5 calendar comparisons and descriptive variation decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from rebalance_tranching.analysis import summarize


def combine_grid(daily: pl.DataFrame) -> pl.DataFrame:
    """Preserve notional; sum order counts while averaging returns and turnover."""
    keys = ["date", "weekday", "offset"]
    checks = (
        daily.lazy()
        .select(
            pl.len().alias("rows"),
            pl.struct(keys).n_unique().alias("unique"),
            pl.any_horizontal(pl.col(keys).is_null()).any().alias("null_keys"),
            pl.col("weekday").unique().sort().implode().alias("weekdays"),
            pl.col("offset").unique().sort().implode().alias("offsets"),
            pl.any_horizontal(
                pl.col(c).is_null() | ~pl.col(c).is_finite() | (pl.col(c) <= -1)
                for c in ("gross", "net")
            )
            .any()
            .alias("bad_returns"),
        )
        .collect()
        .row(0, named=True)
    )
    if (
        not checks["rows"]
        or checks["rows"] != checks["unique"]
        or checks["null_keys"]
        or checks["bad_returns"]
        or checks["weekdays"] != [1, 2, 3, 4, 5]
        or checks["offsets"] != [0, 1, 2]
    ):
        raise ValueError(
            "Expected unique finite returns for three offsets and five weekdays"
        )
    if daily.lazy().group_by("date").len().filter(pl.col("len") != 15).collect().height:
        raise ValueError("Every calendar must have exactly matched dates")
    columns = ["date", "weekday", "schedules", "sleeves", "gross", "net"]
    optional = [
        c
        for c in ("trading_cost", "traded_notional", "order_count")
        if c in daily.columns
    ]
    standalone = (
        daily.lazy()
        .with_columns(
            (pl.col("offset") + 1).cast(pl.String).alias("schedules"),
            pl.lit(1).alias("sleeves"),
        )
        .select(*columns, *optional)
    )
    combined = (
        daily.lazy()
        .group_by("date", "weekday")
        .agg(
            pl.col("gross", "net").mean(),
            *[
                pl.col(c).sum() if c == "order_count" else pl.col(c).mean()
                for c in optional
            ],
        )
        .with_columns(pl.lit("1+2+3").alias("schedules"), pl.lit(3).alias("sleeves"))
        .select(*columns, *optional)
    )
    return (
        pl.concat([standalone, combined]).sort("weekday", "schedules", "date").collect()
    )


def grid_metrics(daily: pl.DataFrame) -> pl.DataFrame:
    """Recompute every weekday's portfolios, then return comparable calendar cells."""
    combined = combine_grid(daily)
    return pl.concat(
        [
            summarize(combined.lazy().filter(pl.col("weekday") == weekday).collect())
            .lazy()
            .with_columns(pl.lit(weekday).alias("weekday"))
            .collect()
            for weekday in range(1, 6)
        ]
    )


def decompose(metrics: pl.DataFrame, metric: str = "net_cagr") -> dict[str, float]:
    """Balanced two-way sums of squares; descriptive, without independent-sample tests."""
    residuals = (
        metrics.lazy()
        .filter(pl.col("sleeves") == 1)
        .select("weekday", "schedules", metric)
        .with_columns(
            pl.col(metric).mean().alias("grand"),
            pl.col(metric).mean().over("weekday").alias("weekday_mean"),
            pl.col(metric).mean().over("schedules").alias("offset_mean"),
        )
        .select(
            ((pl.col(metric) - pl.col("grand")) ** 2).sum().alias("total"),
            ((pl.col("weekday_mean") - pl.col("grand")) ** 2).sum().alias("weekday"),
            ((pl.col("offset_mean") - pl.col("grand")) ** 2).sum().alias("offset"),
            (
                (
                    pl.col(metric)
                    - pl.col("weekday_mean")
                    - pl.col("offset_mean")
                    + pl.col("grand")
                )
                ** 2
            )
            .sum()
            .alias("interaction"),
        )
        .collect()
        .row(0, named=True)
    )
    return {str(k): float(v) for k, v in residuals.items()}


def dispersion(metrics: pl.DataFrame) -> pl.DataFrame:
    """Population standard deviation and range across the enumerated calendars."""
    return (
        metrics.lazy()
        .group_by("sleeves")
        .agg(
            pl.col("net_cagr").mean().alias("mean_cagr"),
            (pl.col("net_cagr").max() - pl.col("net_cagr").min()).alias("range_pp"),
            pl.col("net_cagr").std(ddof=0).alias("sd_pp"),
        )
        .sort("sleeves")
        .collect()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = pl.scan_parquet(args.input).collect()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = combine_grid(frame)
    combined.write_parquet(args.output / "calendar_portfolios.parquet")
    for period, expression in (
        ("development", pl.col("date") < pl.date(2022, 1, 1)),
        ("later", pl.col("date") >= pl.date(2022, 1, 1)),
    ):
        metrics = grid_metrics(frame.lazy().filter(expression).collect())
        metrics.write_csv(args.output / f"{period}_metrics.csv")
        metrics.lazy().group_by("weekday", "sleeves").agg(
            *[
                expression
                for column in (
                    "net_arithmetic",
                    "net_cagr",
                    "volatility",
                    "sharpe",
                    "drawdown",
                )
                for expression in (
                    pl.col(column).mean().alias(f"{column}_mean"),
                    pl.col(column).min().alias(f"{column}_min"),
                    pl.col(column).max().alias(f"{column}_max"),
                )
            ]
        ).sort("weekday", "sleeves").collect().write_csv(
            args.output / f"{period}_summary.csv"
        )
        spread = dispersion(metrics)
        spread.write_csv(args.output / f"{period}_dispersion.csv")
        components = decompose(metrics)
        (args.output / f"{period}_decomposition.json").write_text(
            json.dumps(components, indent=2) + "\n"
        )
        standalone = metrics.lazy().filter(pl.col("sleeves") == 1)
        for grouping in ("weekday", "schedules"):
            standalone.group_by(grouping).agg(
                pl.col("net_cagr").mean().alias("mean_cagr"),
                (pl.col("net_cagr").max() - pl.col("net_cagr").min()).alias("range_pp"),
                pl.col("net_cagr").std(ddof=0).alias("sd_pp"),
            ).sort(grouping).collect().write_csv(
                args.output / f"{period}_within_{grouping}.csv"
            )
        if {"traded_notional", "order_count"} <= set(combined.columns):
            combined.lazy().filter(expression).group_by(
                "weekday", "schedules", "sleeves"
            ).agg(
                (pl.col("traded_notional").mean() * 252).alias(
                    "annual_two_way_turnover"
                ),
                (pl.col("order_count").mean() * 252).alias("annual_orders"),
                (
                    pl.col("traded_notional").sum()
                    * 5_000_000
                    / pl.col("order_count").sum()
                ).alias("mean_order_dollars_at_5m"),
                ((pl.col("gross") - pl.col("net")).mean() * 252 * 100).alias(
                    "annual_cost_pp"
                ),
            ).sort("weekday", "sleeves", "schedules").collect().write_csv(
                args.output / f"{period}_trading.csv"
            )
        print(period, spread, components, sep="\n")
    annual = []
    # Complete calendar years only: omit the partial 1998 and 2026 observations.
    for year in range(1999, 2026):
        metrics = grid_metrics(
            frame.lazy().filter(pl.col("date").dt.year() == year).collect()
        )
        annual.append(metrics.lazy().with_columns(pl.lit(year).alias("year")).collect())
    pl.concat(annual).write_csv(args.output / "annual_metrics.csv")


if __name__ == "__main__":
    main()
