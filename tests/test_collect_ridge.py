"""Reject calendar grids whose apparent timing effect has another cause."""

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from rebalance_tranching.collect_ridge import collect


def inputs(root: Path) -> None:
    for weekday in range(1, 6):
        for offset in range(3):
            folder = root / f"w{weekday}_o{offset}"
            folder.mkdir(parents=True)
            (folder / "execution.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provenance": {
                            "calibration": "1.18",
                            "inputs": "same forecasts",
                        },
                    }
                )
            )
            pl.DataFrame(
                {
                    "status": ["optimal"],
                    "gross": [1.8],
                    "net_exposure": [0.2],
                    "beta_exposure": [-0.04],
                    "ex_ante_vol_annual": [0.07],
                }
            ).write_parquet(folder / "optimizer_diagnostics.parquet")
            pl.DataFrame(
                {
                    "date": [dt.date(2022, 1, 3), dt.date(2022, 1, 4)],
                    "weekday": [weekday] * 2,
                    "offset": [offset] * 2,
                    "gross": [0.01, 0.02],
                    "net": [0.0095, 0.02],
                    "traded_notional": [1.0, 0.0],
                    "order_count": [2, 0],
                    "trading_cost": [0.0005, 0.0],
                }
            ).write_parquet(folder / "daily.parquet")
            pl.DataFrame(
                {
                    "date": [dt.date(2022, 1, 3)] * 3,
                    "final_execution_qty": [10, -10, 0],
                }
            ).write_parquet(folder / "trades.parquet")
            pl.DataFrame(
                {
                    "signal_date": [dt.date(2021, 12, 31)],
                    "date": [dt.date(2022, 1, 3)],
                }
            ).write_csv(folder / "events.csv")


@pytest.mark.parametrize(
    "failure", ["different forecasts", "risk breach", "cost mismatch"]
)
def test_collector_rejects_confounded_comparisons(tmp_path, failure):
    root = tmp_path / "runs"
    inputs(root)
    folder = root / "w3_o1"
    if failure == "different forecasts":
        path = folder / "execution.json"
        record = json.loads(path.read_text())
        record["provenance"]["inputs"] = "another model"
        path.write_text(json.dumps(record))
    elif failure == "risk breach":
        path = folder / "optimizer_diagnostics.parquet"
        pl.read_parquet(path).with_columns(
            pl.lit(0.09).alias("ex_ante_vol_annual")
        ).write_parquet(path)
    else:
        path = folder / "daily.parquet"
        pl.read_parquet(path).with_columns(
            pl.lit(0.003).alias("trading_cost")
        ).write_parquet(path)
    with pytest.raises(ValueError):
        collect(root, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_collector_preserves_all_calendars_and_reports_collisions(tmp_path):
    root = tmp_path / "runs"
    inputs(root)
    collect(root, tmp_path / "export")
    daily = pl.read_parquet(tmp_path / "export/calendar_daily.parquet")
    assert daily.height == 30
    assert daily["date"].n_unique() == 2
    assert daily["order_count"].sum() == 30
    audit = json.loads((tmp_path / "export/audit.json").read_text())
    assert len(audit["same_weekday_event_collisions"]) == 5
    assert audit["cost_max_absolute_error"] < 1e-15
