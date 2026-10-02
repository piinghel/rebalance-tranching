"""Compact article evidence; retain the distinction between means and mixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from rebalance_tranching.analysis import summarize
from rebalance_tranching.calendar_grid import (
    combine_grid,
    grid_metrics,
    tranche_comparison,
)
from rebalance_tranching.schedule_luck import null_spreads, spread, wide_returns


def covariance_checks(returns: np.ndarray, keys: list[tuple[int, int]]) -> list[dict]:
    """Separate exact variance accounting from equal-vol and log-growth approximations."""
    rows = []
    for weekday in range(1, 6):
        values = returns[:, [i for i, (day, _) in enumerate(keys) if day == weekday]]
        covariance = np.cov(values, rowvar=False, ddof=1) * 252
        volatilities = np.sqrt(np.diag(covariance))
        rho = float(np.corrcoef(values.T)[np.triu_indices(3, 1)].mean())
        blend = values.mean(axis=1)
        blend_variance = float(blend.var(ddof=1) * 252)
        rows.append(
            dict(
                weekday=weekday,
                mean_correlation=rho,
                mean_standalone_volatility=float(volatilities.mean() * 100),
                combined_volatility=float(np.sqrt(blend_variance) * 100),
                equal_volatility_approximation=float(
                    volatilities.mean() * np.sqrt((1 + 2 * rho) / 3) * 100
                ),
                variance_identity_error=float(
                    abs(covariance.sum() / 9 - blend_variance)
                ),
                log_growth_gain_pp=float(
                    (np.log1p(blend).mean() - np.log1p(values).mean()) * 252 * 100
                ),
                second_order_log_growth_gain_pp=float(
                    (np.square(values).mean() - np.square(blend).mean()) * 252 * 50
                ),
            )
        )
    return rows


def all_calendar_metrics(sample: pl.DataFrame) -> dict:
    """Equal fifteenths of executed books, before any cross-calendar order netting."""
    daily = (
        sample.lazy()
        .group_by("date")
        .agg(
            pl.col("gross", "net", "traded_notional").mean(),
            pl.col("order_count").sum(),
        )
        .with_columns(pl.lit("all15").alias("schedules"), pl.lit(15).alias("sleeves"))
        .sort("date")
        .collect()
    )
    activity = (
        daily.lazy()
        .select(
            (pl.col("traded_notional").mean() * 252).alias("annual_turnover"),
            (pl.col("order_count").mean() * 252).alias("annual_orders"),
        )
        .collect()
        .row(0, named=True)
    )
    return {**summarize(daily).row(0, named=True), **activity}


def evidence(daily: pl.DataFrame) -> dict[str, dict[str, object]]:
    portfolios = combine_grid(daily)
    result: dict[str, dict[str, object]] = {}
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
            covariance_checks=covariance_checks(returns, keys),
            all_calendar_blend=all_calendar_metrics(sample),
            tranche_comparison=tranche_comparison(sample).to_dicts(),
            calendars=metrics.to_dicts(),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evidence(pl.scan_parquet(args.input).collect())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=str, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
