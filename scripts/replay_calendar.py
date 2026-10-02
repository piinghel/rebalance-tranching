"""Replay one calendar with the native research packages and frozen Ridge scores.

Requires the private portfolio-optimization runtime; numerical outputs stay local.
The public package needs only the portfolio-level export from this runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def register(args: argparse.Namespace, record: dict, revision: int = 0) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
        json.dump(record, file)
        file.flush()
        command = [
            str(args.registry),
            "--db",
            str(args.database),
            "--client",
            "codex-rebalancing",
            "put",
            file.name,
            "--key",
            record["id"] + ":" + record["status"],
            "--author",
            "Codex",
        ]
        if revision:
            command += [
                "--expected-revision",
                str(revision),
                "--reason",
                "Record actual calendar execution outcome",
            ]
        return json.loads(subprocess.check_output(command, text=True))


def replay(args: argparse.Namespace) -> None:
    import polars as pl
    from omegaconf import OmegaConf
    from portfolio_optimization import ensemble_replay
    from strategy_backtest.backtest_core import scored_book
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
    end = dt.date(2026, 5, 27)
    configs = ensemble_replay.configuration(
        args.base, args.allocation_dir, end, "allocation_b3_state_aware_mvo"
    )
    # Explicitly selected calibration, otherwise every B3 rule is inherited.
    configs[
        "portfolio_allocation"
    ].risk.volatility.calibration_multiplier = args.calibration
    schedule = configs["tranching"].tranche_backtester.tranche_config
    chosen = schedule.tranche_schedule[args.offset]
    assert chosen.rebalancing_calendar.default_offset == args.offset
    assert chosen.rebalancing_calendar.default_frequency == 3
    chosen.rebalancing_calendar.default_day_nr = args.weekday
    chosen.fraction = 1.0
    schedule.tranche_schedule = [chosen]
    assert schedule.amount == 5_000_000
    assert (
        configs[
            "tranching"
        ].tranche_backtester.allocation_backtester.execution_lag_trading_days
        == 1
    )
    assert dict(configs["tranching"].pnl.transaction_costs_mapping) == {
        "long": 0.0005,
        "short": -0.0005,
    }
    components, configs, normalized, scores, prices, audit = (
        ensemble_replay.load_inputs(configs, args.panel, args.predictions, end)
    )
    resolved = {
        name: OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
        for name, value in configs.items()
    }
    OmegaConf.save(OmegaConf.create(resolved), args.output / "config_resolved.yaml")
    packages = Path(scored_book.__file__).parents[3]
    (args.output / "code_identity.json").write_text(
        json.dumps(
            {
                "packages_commit": subprocess.check_output(
                    ["git", "-C", str(packages), "rev-parse", "HEAD"], text=True
                ).strip(),
                "package_diff_sha256": hashlib.sha256(
                    subprocess.check_output(
                        ["git", "-C", str(packages), "diff", "--", "*.py"]
                    )
                ).hexdigest(),
                "polars": pl.__version__,
                "engine": str(scored_book.__file__),
            },
            indent=2,
        )
        + "\n"
    )
    result = scored_book.run_scored_book(
        scores=scores,
        normalized=normalized,
        prices=prices,
        components=components,
        configs=configs,
        n_top=75,
        n_bottom=75,
        date_column="date",
        asset_column="asset_id_bb_global",
        price_column="px_last",
        logger=logging.getLogger(__name__),
        reporting_end_date=end,
    )
    assert len(result.calendars.books) == 1
    book = next(iter(result.calendars.books.values()))
    trades = book.tranche_results.get_allocations(output_mode="executions")
    trades.write_parquet(args.output / "trades.parquet")
    book.returns.write_parquet(args.output / "returns.parquet")
    diagnostics = book.optimizer_diagnostics
    if diagnostics is None:
        diagnostics = result.optimizer_diagnostics
    if diagnostics is None:
        raise ValueError("Missing optimizer diagnostics")
    diagnostics.write_parquet(args.output / "optimizer_diagnostics.parquet")
    daily = (
        book.returns.lazy()
        .select(
            "date",
            pl.col("long_short_gross").alias("gross"),
            pl.col("long_short_net").alias("net"),
        )
        .join(
            trades.lazy()
            .group_by("date")
            .agg(
                (pl.col("final_execution_value").abs().sum() / 5_000_000).alias(
                    "traded_notional"
                ),
                (pl.col("final_execution_qty") != 0).sum().alias("order_count"),
            ),
            on="date",
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("traded_notional", "order_count").fill_null(0))
        .with_columns(
            (pl.col("gross") - pl.col("net")).alias("trading_cost"),
            pl.lit(args.weekday).alias("weekday"),
            pl.lit(args.offset).alias("offset"),
        )
        .sort("date")
        .collect()
    )
    error = (
        daily.lazy()
        .select(
            (pl.col("trading_cost") - pl.col("traded_notional") * 0.0005).abs().max()
        )
        .collect()
        .item()
    )
    if error > 1e-12:
        raise ValueError(f"Executed-notional costs differ: {error}")
    daily.write_parquet(args.output / "daily.parquet")
    trades.lazy().select("signal_date", "date").unique().sort(
        "signal_date"
    ).collect().write_csv(args.output / "events.csv")
    (args.output / "checks.json").write_text(
        json.dumps(
            {
                "universe": audit,
                "cost_max_absolute_error": error,
                "reference_notional": 5_000_000,
                "rows": daily.height,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "base",
        "allocation-dir",
        "panel",
        "predictions",
        "output",
        "registry",
        "database",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--weekday", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--offset", type=int, choices=range(3), required=True)
    parser.add_argument("--calibration", type=float, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with (args.database.parent / "heavy_compute.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        name = f"w{args.weekday}-o{args.offset}-a{args.attempt}"
        record: dict[str, Any] = dict(
            kind="run",
            id="rebalancing:ridge80:run:" + name,
            project_id="rebalance_tranching",
            experiment_id="rebalancing:exp:ridge80-calendar",
            title="Ridge B3 calendar " + name,
            summary="Frozen-score native calendar replay.",
            execution_kind="replay",
            status="running",
            source_system="native B3 scored book",
            source_run_id=str(args.output.resolve()),
            provenance=dict(
                started_at=dt.datetime.now(dt.UTC).isoformat(),
                finished_at="unknown",
                inputs=json.dumps(
                    {
                        k: identity(getattr(args, k))
                        for k in ["base", "panel", "predictions"]
                    }
                ),
                runner=json.dumps(identity(Path(__file__))),
                calibration=str(args.calibration),
                output=str(args.output),
            ),
            evidence=[dict(record_id="rebalancing:exp:ridge80-calendar", revision=1)],
        )
        receipt = register(args, record)
        try:
            replay(args)
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["failure_reason"] = repr(error)
            raise
        finally:
            record["provenance"]["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
            register(args, record, receipt["revision"])
            (args.output / "execution.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
