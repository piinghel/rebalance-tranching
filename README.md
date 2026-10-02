# Rebalance tranching

Code and portfolio-level evidence for
[Reducing Rebalancing Luck](https://piinghel.github.io/quants/rebalancing-luck.html).
Published figures belong in the blog's `assets/rebalancing-luck/` folder; pass
that destination with `ridge_figures --output` when refreshing the article.

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

### Ridge-80 calendar comparison

The Ridge-80 study uses saved `ridge_80_c0p1` forecasts and the B3 allocation,
with 21-session main volatility and a 1.18 volatility calibration multiplier.
`data/ridge80_calendar_daily.parquet` contains the matched portfolio aggregates;
`data/ridge80_calendar_events.csv` contains the signal and ledger dates.
Their hashes and input identities are in `SOURCE_FILES.json`. No security-level
positions, predictions or licensed market data are included.

Reproduce the article calculations and figures from those aggregates:

```sh
uv run python -m rebalance_tranching.calendar_grid --input data/ridge80_calendar_daily.parquet --output output/ridge80
uv run python -m rebalance_tranching.ridge_evidence --input data/ridge80_calendar_daily.parquet --output output/ridge80/article_metrics.json
uv run python -m rebalance_tranching.ridge_figures --input data/ridge80_calendar_daily.parquet --output output/ridge80/figures --blog /path/to/piinghel.github.io
```

The figure exporter reuses `scripts/blog_charts.py` from the
[blog repository](https://github.com/piinghel/piinghel.github.io).
The evidence command retains Development, Later and full-history metrics,
trading activity, covariance identities, an equal-fifteenths combination of all
calendars, and joint-calendar moving-block null comparisons at 21, 63 and
126 sessions (2,000 draws each, seed 0). The all-calendar combination averages
returns and turnover and sums order counts before any cross-book netting.
The null equalizes arithmetic expected
returns; its tail probability does not prove that all calendar effects are luck.
The figure command exports light/dark, desktop/phone SVGs and allowlisted
interactive chart data.
The calendar figure compares return ranges and mean annual order counts for
one, two and three tranches: 15 standalone schedules, 15 same-weekday pairs,
and five weekly-third portfolios. Pairs alternate one- and two-week rebalance
gaps. The ranges describe these enumerated choices, not confidence intervals
or an equal-count statistical comparison. The evidence also retains the five
weekday returns at each fixed starting week for matched comparisons.

#### Replaying the underlying books

`scripts/replay_calendar.py` replays one weekday/offset using the native
portfolio-optimization research runtime. It requires that runtime and the
licensed inputs; they are deliberately not dependencies or data of this public
mixture-analysis package. Each process acquires the shared heavy-compute lock,
records its actual execution, and refuses to overwrite outputs. Run one process
at a time, for weekdays 1–5 and offsets 0–2, using these explicit arguments:

```sh
python scripts/replay_calendar.py \
  --base /path/to/ridge_80_c0p1/backtest/config_resolved.yaml \
  --allocation-dir /path/to/portfolio_optimization/configs/portfolio_management \
  --panel /path/to/normalized/panel.parquet \
  --predictions /path/to/ridge_80_c0p1/backtest/predictions.parquet \
  --registry /path/to/research-registry --database /path/to/registry.sqlite \
  --weekday 5 --offset 0 --calibration 1.18 --volatility-target 0.07 \
  --output /path/to/runs/w5_o0
```

Use a new destination and increment `--attempt` for an actual retry. The
configuration is inherited from `allocation_b3_state_aware_mvo.yaml` and its
parent; the calibration multiplier and forecast volatility target are explicit.
For a separately registered specification, supply its `--experiment-id` and
unique `--run-prefix`; the defaults identify the published Ridge-80 study.
Each standalone calendar is
executed at $5 million reference capital. Mixtures scale these executed books
to thirds: they preserve daily return and proportional-cost arithmetic without
re-solving or re-rounding smaller orders. The collector checks all 15 completed
executions, common inputs, solver status, risk limits, matched dates and costs.

```sh
uv run python -m rebalance_tranching.collect_ridge --input /path/to/runs --output /path/to/aggregates
```

The native runner's optional imports are checked in its separate runtime; the
four package checks below cover the public calculation and rendering modules.

### Historical comparison

The other included inputs retain the earlier Ridge comparison. Their hashes
and source identities remain in `SOURCE_FILES.json`; keep them separate from
the Ridge-80 files above. The shared calculations still accept those inputs:

```bash
uv run python -m rebalance_tranching.analysis
uv run python -m rebalance_tranching.calendar_grid --input data/calendar_daily.parquet --output output/historical
uv run python -m rebalance_tranching.schedule_luck --input data/calendar_daily.parquet --output output/historical/luck
```

The first command prints statistics for all seven non-empty combinations of
three schedules. The calendar command writes combined daily returns, period
and annual metrics, descriptive calendar dispersion, and trading activity.
The last command adds joint moving-block null diagnostics and rolling-window
comparisons. The original figure layouts are recoverable at Git revision
`5e2cb8d`; the maintained renderer is `ridge_figures.py`.

## Code layout

| File | Purpose |
| --- | --- |
| [analysis.py](rebalance_tranching/analysis.py) | Matched-calendar validation, fixed-notional mixtures and metrics |
| [example.py](rebalance_tranching/example.py) | Six-week schedule and hand-checkable daily mixture |
| [calendar_grid.py](rebalance_tranching/calendar_grid.py) | Fifteen calendars, five combined portfolios and matched comparisons |
| [collect_ridge.py](rebalance_tranching/collect_ridge.py) | Validate completed native runs and export matched portfolio aggregates |
| [schedule_luck.py](rebalance_tranching/schedule_luck.py) | Joint-calendar resampling and return-spread diagnostics |
| [ridge_evidence.py](rebalance_tranching/ridge_evidence.py) | Article metrics, trading activity and block-length comparisons |
| [ridge_figures.py](rebalance_tranching/ridge_figures.py) | Both article figures, theme/viewport variants and interactive data |
| [replay_calendar.py](scripts/replay_calendar.py) | Optional native replay and execution registration |

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

Both `data/ridge80_calendar_daily.parquet` and the historical
`data/calendar_daily.parquet` have 6,963 matched dates (22 September 1998–27 May
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
The corresponding `*_calendar_events.csv` or `calendar_events.csv` retains
both that date and the original signal date.

All calendars reuse the original forecasts, point-in-time universe, stock
selection, sizing rules and gross cap. There is no volatility restoration for
the combined portfolio. Each standalone book uses $5 million reference capital.
The combined $5 million portfolio scales each book's executed positions, P&L
and notional to one third; it does not re-solve integer orders at smaller capital.
Returns, costs and traded notional average across its three books, while order
counts add. No cross-offset execution dates coincide within a weekday, so this
grid claims no netting savings. The model charges 5 bp proportionally;
fixed-ticket charges, borrow, financing and explicit market impact are outside
the calculation. No cost-rate sensitivity is part of this comparison.

In the historical inputs, the three Friday replays reconcile to the original
daily evidence within 0.001 bp per day; the largest numerical difference is 0.000566 bp, confined to
development. Later-period metrics reproduce exactly. Original input files are
retained unchanged. All included data are portfolio-level aggregates.

The included portfolio returns reproduce the mixtures and figures. Replaying
the underlying stock books requires the separate runtime and licensed inputs
described above.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run pytest -q
```

Related study: [portfolio optimization](https://github.com/piinghel/portfolio-optimization-study).
