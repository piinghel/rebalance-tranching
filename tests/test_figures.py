"""The figure inputs reconcile to the daily evidence and reject corrupt views."""

from __future__ import annotations

import csv
from pathlib import Path
from xml.etree import ElementTree

import polars as pl
import pytest

from rebalance_tranching import dispersion, performance

DATA = Path(__file__).resolve().parents[1] / "data"


def test_chart_view_matches_daily_evidence():
    chart = pl.scan_csv(DATA / "schedule_returns.csv", try_parse_dates=True).collect()
    daily = pl.scan_parquet(DATA / "timing_daily.parquet")
    for schedule, column in (
        ("1", "week_1"),
        ("2", "week_2"),
        ("3", "week_3"),
        ("1+2+3", "mixture"),
    ):
        source = daily.filter(pl.col("schedules") == schedule).sort("date").collect()
        assert chart["date"].equals(source["date"])
        assert chart["period"].equals(source["period"])
        assert chart[column].to_list() == pytest.approx(
            source["net"].to_list(), abs=1e-14
        )


@pytest.mark.parametrize("corruption", ["duplicate", "nan", "mixture", "period"])
def test_performance_rejects_corrupt_inputs(tmp_path, corruption):
    rows = performance.load_returns(DATA / "schedule_returns.csv")
    if corruption == "duplicate":
        rows.append(rows[-1])
    elif corruption == "nan":
        rows[0]["week_1"] = "nan"
    elif corruption == "mixture":
        rows[0]["mixture"] = "0.99"
    else:
        rows[-1]["period"] = "development"
    path = tmp_path / "returns.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        performance.load_returns(path)


def test_dispersion_rejects_missing_combinations_and_clipped_values(tmp_path):
    rows = dispersion.load_metrics(DATA / "timing_metrics.csv")
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(ValueError, match="seven combinations"):
        dispersion.load_metrics(path)
    rows[0]["volatility"] = "100"
    with pytest.raises(ValueError, match="outside chart axes"):
        dispersion.render(rows, dark=False)


@pytest.mark.parametrize("mobile", [False, True])
def test_performance_exports_matching_theme_layouts(tmp_path, mobile):
    rows = performance.load_returns(DATA / "schedule_returns.csv")
    bounds = []
    for dark in (False, True):
        output = tmp_path / f"performance-{dark}.svg"
        performance.render(rows, output, dark=dark, mobile=mobile)
        svg = ElementTree.parse(output).getroot()
        bounds.append(svg.attrib["viewBox"])
        text = " ".join(svg.itertext())
        assert "Three-tranche" in text and "portfolio" in text
        assert "Net growth index · log scale" in text
    assert bounds[0] == bounds[1]
