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
The performance and calendar-grid renderers also export 300 dpi PNGs. Published
figures use the SVGs for sharp text and lines at every screen size. Generated
outputs are ignored by Git.

The complete starting-week × signal-weekday comparison uses the same saved
forecasts and allocation rules for all 15 calendars:

```bash
uv run python -m rebalance_tranching.calendar_grid --input data/calendar_daily.parquet --output output/calendar
uv run python -m rebalance_tranching.grid_figures --input output/calendar --output output
```

This writes daily gross/net returns for the 15 standalone calendars and five
three-tranche portfolios, period and annual metrics, calendar ranges and
population standard deviations, a descriptive offset/weekday/interaction
decomposition, and trading activity. The calendar grid and return/volatility
panels use the full matched period, 22 September 1998–27 May 2026. The Friday
growth chart shows January 2022–May 2026 to make the later divergence visible.
Development and later results remain separate in
the period outputs so the long development history does not hide recent differences.
The decomposition describes this anchored grid; it is not an independent-sample
significance test. Annual comparisons include complete years 1999–2025.

| File | Purpose |
| --- | --- |
| [analysis.py](rebalance_tranching/analysis.py) | Matched-calendar validation, fixed-notional mixtures and metrics |
| [example.py](rebalance_tranching/example.py) | Six-week schedule and hand-checkable daily mixture |
| [performance.py](rebalance_tranching/performance.py) | Later-period fixed best/worst Friday paths and the three-tranche portfolio |
| [dispersion.py](rebalance_tranching/dispersion.py) | Return and volatility across all schedule combinations |
| [calendar_grid.py](rebalance_tranching/calendar_grid.py) | Fifteen calendars, five combined portfolios and matched comparisons |
| [grid_figures.py](rebalance_tranching/grid_figures.py) | Calendar heatmap and aligned return/volatility panels |

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

### Calendar grid

`data/calendar_daily.parquet` has 6,963 matched dates (22 September 1998–27 May
2026) for each of 15 calendars. `weekday` is the ISO signal weekday (1 = Monday,
5 = Friday), and `offset` is 0, 1 or 2. Gross/net returns and `trading_cost` are
daily decimal P&L per unit of fixed notional. `traded_notional` is two-way
executed notional divided by reference capital; `order_count` counts executed
security orders. The cost identity is `gross - net = traded_notional * 0.0005`.

The calendar anchor is the week beginning Monday 31 August 1998. Weekly signal
targets roll forward to the next eligible session (at least 90% universe quote
coverage), duplicate dates are removed, then every third target is selected
at each offset. Orders execute at the next trading-session close. The ledger
records the cost on the first following close-to-close P&L date;
`data/calendar_events.csv` retains both that date and the original signal date.

All calendars reuse the original forecasts, point-in-time universe, stock
selection, sizing rules and gross cap. There is no volatility restoration for
the combined portfolio. Each standalone book uses $5 million reference capital.
The combined $5 million portfolio scales each book's executed positions, P&L
and notional to one third; it does not re-solve integer orders at smaller capital.
Returns, costs and traded notional average across its three books, while order
counts add. No cross-offset execution dates coincide within a weekday, so this
grid claims no netting savings. The 5 bp allowance covers proportional execution
costs and market impact; fixed-ticket charges, borrow and financing are outside
the model. No cost-rate sensitivity is part of this comparison.

The three Friday replays reconcile to the original daily evidence within
0.001 bp per day; the largest numerical difference is 0.000566 bp, confined to
development. Later-period metrics reproduce exactly. Original input files are
retained unchanged. All included data are portfolio-level aggregates.

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
