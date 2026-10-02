"""Is the spread between rebalance schedules more than luck?

Fifteen schedules (five signal weekdays by three starting weeks) share one
strategy and differ only in trade dates. Under the null that none is better,
their daily net returns have the same expectation. A moving-block bootstrap of
the demeaned daily returns keeps each day's cross-schedule correlation and gives
the spread of annualized returns that luck alone produces. The module also
reports spreads in windows as long as the later period, fixed-starting-week
spreads, yearly spreads and a mixture of all fifteen schedules.

    uv run python -m rebalance_tranching.schedule_luck \\
        --input data/calendar_daily.parquet --output output/historical/luck
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

SESSIONS = 252
BLOCK = 63
DRAWS = 2000
LATER_START = date(2022, 1, 1)


def wide_returns(
    daily: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Dates, a sessions-by-15 net return matrix and its (weekday, offset) columns."""
    frame = (
        daily.with_columns(pl.format("{}_{}", "weekday", "offset").alias("key"))
        .pivot(on="key", index="date", values="net")
        .sort("date")
    )
    keys = sorted((int(k[0]), int(k[2])) for k in frame.columns if k != "date")
    if len(keys) != 15:
        raise ValueError("Expected fifteen schedules")
    matrix = frame.select([f"{w}_{o}" for w, o in keys]).to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Schedules must share complete, finite dates")
    return frame["date"].to_numpy(), matrix, keys


def cagr(returns: np.ndarray) -> np.ndarray:
    """Annualized compounded return in percent, per column."""
    return 100 * (np.exp(np.log1p(returns).sum(axis=0) * SESSIONS / len(returns)) - 1)


def spread(returns: np.ndarray) -> float:
    values = cagr(returns)
    return float(values.max() - values.min())


def null_spreads(
    returns: np.ndarray, block: int, draws: int, seed: int = 0
) -> np.ndarray:
    """Spreads under equal expected returns: demean, then resample whole days in blocks."""
    centred = returns - returns.mean(axis=0) + returns.mean()
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(centred) / block))
    out = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, len(centred) - block + 1, size=blocks)
        rows = (starts[:, None] + np.arange(block)).ravel()[: len(centred)]
        out[i] = spread(centred[rows])
    return out


def combined(returns: np.ndarray, keys: list[tuple[int, int]]) -> np.ndarray:
    """Three-tranche portfolios: the mean of each weekday's three starting weeks."""
    weekdays = sorted({w for w, _ in keys})
    return np.column_stack(
        [
            returns[:, [i for i, (w, _) in enumerate(keys) if w == day]].mean(axis=1)
            for day in weekdays
        ]
    )


def risk(returns: np.ndarray) -> dict[str, float]:
    mean, sd = returns.mean(), returns.std(ddof=1)
    return {
        "cagr": float(cagr(returns[:, None])[0]),
        "volatility": float(100 * sd * np.sqrt(SESSIONS)),
        "sharpe": float(mean / sd * np.sqrt(SESSIONS)),
    }


def evaluate(
    dates: np.ndarray, returns: np.ndarray, keys: list[tuple[int, int]]
) -> dict:
    three = combined(returns, keys)
    observed = spread(returns)
    null = null_spreads(returns, BLOCK, DRAWS)
    later = dates >= np.datetime64(LATER_START)
    window = int(later.sum())
    windows = [
        {
            "start": str(dates[s]),
            "end": str(dates[s + window - 1]),
            "standalone_pp": spread(returns[s : s + window]),
            "combined_pp": spread(three[s : s + window]),
        }
        for s in range(0, len(dates) - window + 1, window)
    ]
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    yearly = {
        int(y): {
            "standalone_pp": spread(returns[years == y]),
            "combined_pp": spread(three[years == y]),
        }
        for y in np.unique(years)
        if (years == y).sum() >= 200
    }
    offsets = sorted({o for _, o in keys})
    fixed = {
        str(o): spread(returns[:, [i for i, (_, off) in enumerate(keys) if off == o]])
        for o in offsets
    }
    return {
        "observed_spread_pp": observed,
        "null": {
            "block": BLOCK,
            "draws": DRAWS,
            "median_pp": float(np.median(null)),
            "p95_pp": float(np.percentile(null, 95)),
            "share_at_least_observed": float((null >= observed).mean()),
        },
        "combined_spread_pp": spread(three),
        "fixed_start_week_spreads_pp": fixed,
        "later_period": {
            "standalone_pp": spread(returns[later]),
            "combined_pp": spread(three[later]),
        },
        "equal_length_windows": windows,
        "yearly": yearly,
        "mixtures": {
            "standalone_mean": {
                k: float(
                    np.mean([risk(returns[:, i])[k] for i in range(returns.shape[1])])
                )
                for k in ("cagr", "volatility", "sharpe")
            },
            "three_tranche_mean": {
                k: float(np.mean([risk(three[:, i])[k] for i in range(three.shape[1])]))
                for k in ("cagr", "volatility", "sharpe")
            },
            "all_fifteen": risk(returns.mean(axis=1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dates, returns, keys = wide_returns(pl.read_parquet(args.input))
    results = evaluate(dates, returns, keys)
    (args.output / "schedule_luck.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
