"""Validate complete native calendar replays and export portfolio aggregates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from rebalance_tranching.calendar_grid import combine_grid


def collect(root: Path, output: Path) -> None:
    frames, events, hashes, allocations, sources, solver_checks = [], [], {}, [], [], {}
    zero_quantity_rows = {}
    for weekday in range(1, 6):
        for offset in range(3):
            folder = root / f"w{weekday}_o{offset}"
            record = json.loads((folder / "execution.json").read_text())
            if record["status"] != "completed":
                raise ValueError(f"Incomplete execution: {folder}")
            allocations.append(record["provenance"]["calibration"])
            sources.append(record["provenance"]["inputs"])
            diagnostics = pl.scan_parquet(folder / "optimizer_diagnostics.parquet")
            check = (
                diagnostics.select(
                    pl.len().alias("events"),
                    (
                        pl.col("status").is_null()
                        | ~pl.col("status").is_in(["optimal", "optimal_inaccurate"])
                    )
                    .sum()
                    .alias("unsolved"),
                    (pl.col("status") == "optimal_inaccurate")
                    .sum()
                    .alias("inaccurate"),
                    pl.any_horizontal(
                        pl.col(c).is_null() | ~pl.col(c).is_finite()
                        for c in (
                            "gross",
                            "net_exposure",
                            "beta_exposure",
                            "ex_ante_vol_annual",
                        )
                    )
                    .any()
                    .alias("invalid_risk"),
                    pl.col("gross").max(),
                    pl.col("net_exposure").abs().max(),
                    pl.col("beta_exposure").abs().max(),
                    pl.col("ex_ante_vol_annual").max(),
                )
                .collect()
                .row(0, named=True)
            )
            if (
                not check["events"]
                or check["unsolved"]
                or check["invalid_risk"]
                or check["gross"] > 2.000001
                or check["net_exposure"] > 0.250001
                or check["beta_exposure"] > 0.050001
                or check["ex_ante_vol_annual"] > 0.070001
            ):
                raise ValueError(f"Optimizer audit failed: {folder}: {check}")
            solver_checks[folder.name] = check
            path = folder / "daily.parquet"
            hashes[folder.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            trades = pl.scan_parquet(folder / "trades.parquet")
            zero_quantity_rows[folder.name] = (
                trades.select((pl.col("final_execution_qty") == 0).sum())
                .collect()
                .item()
            )
            executed_counts = trades.group_by("date").agg(
                (pl.col("final_execution_qty") != 0).sum().alias("order_count")
            )
            frames.append(
                pl.scan_parquet(path)
                .drop("order_count")
                .join(executed_counts, on="date", how="left", validate="1:1")
                .with_columns(pl.col("order_count").fill_null(0))
                .collect()
            )
            events.append(
                pl.scan_csv(folder / "events.csv", try_parse_dates=True).with_columns(
                    pl.lit(weekday).alias("weekday"), pl.lit(offset).alias("offset")
                )
            )
    if len(set(allocations)) != 1:
        raise ValueError("Calendars used different risk calibrations")
    if len(set(sources)) != 1:
        raise ValueError("Calendars used different input identities")
    # Start when the last calendar opens its first portfolio. Exclude unequal
    # startup cash periods, but preserve the full carried histories in raw files.
    start = max(
        f.lazy()
        .filter(pl.col("order_count") > 0)
        .select(pl.col("date").min())
        .collect()
        .item()
        for f in frames
    )
    daily = (
        pl.concat(frames)
        .lazy()
        .filter(pl.col("date") >= start)
        .sort("weekday", "offset", "date")
        .collect()
    )
    combine_grid(daily)
    error = (
        daily.lazy()
        .select(
            pl.max_horizontal(
                (pl.col("trading_cost") - pl.col("traded_notional") * 0.0005).abs(),
                (pl.col("gross") - pl.col("net") - pl.col("trading_cost")).abs(),
            ).max()
        )
        .collect()
        .item()
    )
    if error > 1e-12:
        raise ValueError("Cost identity failed")
    event_frame = pl.concat(events).collect()
    collisions = (
        event_frame.lazy()
        .group_by("weekday", "date")
        .agg(pl.col("offset").n_unique().alias("offsets"))
        .filter(pl.col("offsets") > 1)
        .collect()
    )
    output.mkdir(parents=True, exist_ok=False)
    daily.write_parquet(output / "calendar_daily.parquet")
    event_frame.write_csv(output / "calendar_events.csv")
    (output / "audit.json").write_text(
        json.dumps(
            {
                "sha256": hashes,
                "solver_checks": solver_checks,
                "zero_quantity_ledger_rows_excluded": zero_quantity_rows,
                "solver_policy": "Native solver accepts optimal_inaccurate only after post-solve constraint checks (absolute tolerance 1e-6); collector also verifies gross, net, beta and forecast volatility at that tolerance.",
                "start": str(start),
                "end": str(daily["date"].max()),
                "rows": daily.height,
                "risk_calibration_multiplier": allocations[0],
                "cost_max_absolute_error": error,
                "same_weekday_event_collisions": collisions.to_dicts(),
                "standalone_reference_notional": 5_000_000,
                "mixture": "Daily equal fixed-notional replicas; execution orders scaled to thirds, not re-rounded. No volatility restoration or cross-book netting.",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.input, args.output)
