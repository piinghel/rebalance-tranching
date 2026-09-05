# Rebalance tranching

Code and portfolio-level evidence for
[Combining Rebalance Weeks Reduces Timing Risk](https://piinghel.github.io/quants/2025/05/10/rebalancing-luck.html).

## Start with the sleeves

From the repository root, with Python 3.12 or later and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python -m rebalance_tranching.example
```

The example shows three sleeves over six weeks, each with one-third notional and
its own three-week rebalance cycle. It then combines invented daily portfolio returns
using the same [mixture calculation](rebalance_tranching/analysis.py) as the study.
The example does not simulate stock holdings or claim a historical result.

## Reproduce the results

```bash
uv run python -m rebalance_tranching.analysis
uv run python -m rebalance_tranching.performance
uv run python -m rebalance_tranching.dispersion
```

The first command prints return, volatility, Sharpe and drawdown for all seven
non-empty combinations of the three schedules, separately for Development and Later.
It writes no files. `--input path/to/timing_daily.parquet` selects another daily input.

The second command rebuilds the article's performance chart in light/dark and
desktop/phone layouts. The third retains the supporting combination-dispersion
figure. Both write SVGs to `output/` by default; use `--output` for another directory.
Generated SVGs are ignored by Git.

| File | Purpose |
| --- | --- |
| [analysis.py](rebalance_tranching/analysis.py) | Matched-calendar validation, fixed-notional mixtures and metrics |
| [example.py](rebalance_tranching/example.py) | Six-week schedule and hand-checkable daily mixture |
| [performance.py](rebalance_tranching/performance.py) | Later-period portfolio paths |
| [dispersion.py](rebalance_tranching/dispersion.py) | Return and volatility across all schedule combinations |

## Inputs and conventions

`data/timing_daily.parquet` contains daily gross/net portfolio returns, period,
schedule combination and sleeve count. `data/timing_metrics.csv` contains the
saved period statistics. `data/schedule_returns.csv` is the chart-ready net-return
view; tests reconcile its dates and four series to the daily evidence.
`SOURCE_FILES.json` records their public source snapshots and hashes.

The mixture averages daily P&L per unit of fixed notional, then recomputes its
statistics. It does not average the standalone Sharpes or compounded indices.
Compounded growth is a display index, not a financed account simulation.

Annualization uses 252 sessions and a zero cash rate for Sharpe. Returns, volatility
and drawdowns are reported in percent; daily input returns are decimal fractions.
Development ends in December 2021. January 2022–May 2026 is later, reused evidence.
Paths retain their own volatilities, so compare risk as well as cumulative return.

This repository reproduces mixtures and figures from the included portfolio returns;
it does not reconstruct the stock-selection and execution backtests behind each sleeve.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```

Related study: [portfolio optimization](https://github.com/piinghel/portfolio-optimization-study).
