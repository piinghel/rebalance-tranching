"""The figure inputs reconcile to the daily evidence and reject corrupt views."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import polars as pl
import pytest

from rebalance_tranching import ridge_figures

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


@pytest.mark.parametrize("mobile", [False, True])
@pytest.mark.parametrize("render", [ridge_figures.performance, ridge_figures.calendars])
def test_current_figures_export_vector_matching_theme_layouts(tmp_path, mobile, render):
    daily = pl.read_parquet(DATA / "ridge80_calendar_daily.parquet")
    bounds = []
    for dark in (False, True):
        output = tmp_path / f"figure-{dark}.svg"
        render(daily, output, dark=dark, mobile=mobile)
        svg = ElementTree.parse(output).getroot()
        bounds.append(svg.attrib["viewBox"])
        text = " ".join(svg.itertext())
        if render is ridge_figures.performance:
            assert "Week 1" in text and "Week 2" in text and "Week 3" in text
            assert "Three" in text and "tranches" in text
        else:
            assert "One schedule" in text and "⅓ each week" in text
            assert "2.65 pp spread" in text and "0.95 pp spread" in text
        assert not list(svg.iter("{http://www.w3.org/2000/svg}image"))
    assert bounds[0] == bounds[1]
