"""Compact article evidence; retain the distinction between means and mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from rebalance_tranching.calendar_grid import combine_grid, grid_metrics
from rebalance_tranching.schedule_luck import null_spreads, spread, wide_returns


def evidence(daily: pl.DataFrame) -> dict:
    portfolios = combine_grid(daily)
    result = {}
    for period, expression in [
        ("development", pl.col("date") < pl.date(2022, 1, 1)),
        ("later", pl.col("date") >= pl.date(2022, 1, 1)),
        ("full", pl.lit(True)),
    ]:
        sample = daily.lazy().filter(expression).collect()
        metrics = grid_metrics(sample)
        _, returns, keys = wide_returns(sample)
        observed = spread(returns)
        uncertainty = {}
        for block in [21, 63, 126]:
            null = null_spreads(returns, block, 2000)
            uncertainty[str(block)] = dict(
                median=float(np.median(null)),
                p95=float(np.quantile(null, 0.95)),
                exceedance=float(np.mean(null >= observed)),
            )
        stats = []
        for count in [1, 3]:
            group = metrics.lazy().filter(pl.col("sleeves") == count).collect()
            stats.append(
                dict(
                    sleeves=count,
                    **{
                        column: dict(
                            mean=float(group[column].to_numpy().mean()),
                            min=float(group[column].to_numpy().min()),
                            max=float(group[column].to_numpy().max()),
                        )
                        for column in [
                            "gross_cagr",
                            "net_cagr",
                            "net_arithmetic",
                            "volatility",
                            "sharpe",
                            "drawdown",
                            "arithmetic_cost",
                        ]
                    },
                )
            )
        activity = (
            portfolios.lazy()
            .filter(expression)
            .group_by("weekday", "schedules", "sleeves")
            .agg(
                (pl.col("traded_notional").mean() * 252).alias("annual_turnover"),
                (pl.col("order_count").mean() * 252).alias("annual_orders"),
            )
        )
        activity_summary = (
            activity.group_by("sleeves")
            .agg(pl.col("annual_turnover", "annual_orders").mean())
            .sort("sleeves")
            .collect()
        )
        fixed = {
            str(o + 1): spread(
                returns[:, [i for i, (_, off) in enumerate(keys) if off == o]]
            )
            for o in range(3)
        }
        correlations = []
        for day in range(1, 6):
            values = returns[:, [i for i, (w, _) in enumerate(keys) if w == day]]
            correlations.extend(np.corrcoef(values.T)[np.triu_indices(3, 1)].tolist())
        result[period] = dict(
            start=str(sample["date"].min()),
            end=str(sample["date"].max()),
            days=sample.height // 15,
            metrics=stats,
            standalone_return_spread_pp=observed,
            fixed_start_week_spreads_pp=fixed,
            null=uncertainty,
            activity=activity_summary.to_dicts(),
            mean_same_weekday_correlation=float(np.mean(correlations)),
            calendars=metrics.to_dicts(),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evidence(pl.scan_parquet(args.input).collect())
    args.output.write_text(
        json.dumps(result, indent=2, default=str, allow_nan=False) + "\n"
    )
