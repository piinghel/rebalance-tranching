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
def test_current_figures_export_vector_matching_theme_layouts(
    tmp_path, mobile, render, data_dir: Path = DATA
):
    daily = pl.read_parquet(data_dir / "ridge80_calendar_daily.parquet")
    bounds = []
    for dark in (False, True):
        output = tmp_path / f"figure-{dark}.svg"
        if render is ridge_figures.performance:
            render(daily, output, dark=dark, mobile=mobile)
        else:
            render(
                ridge_figures.calendar_periods(daily), output, dark=dark, mobile=mobile
            )
        svg = ElementTree.parse(output).getroot()
        bounds.append(svg.attrib["viewBox"])
        text = " ".join(svg.itertext())
        if render is ridge_figures.performance:
            assert "Week 1" in text and "Week 2" in text and "Week 3" in text
            assert "Three" in text and "tranches" in text
        else:
            assert "Annualized net return (%)" in text
            assert "One book" in text and "Three books" in text
            assert "Sep 1998–Dec 2021" in text and "Jan 2022–May 2026" in text
        assert not list(svg.iter("{http://www.w3.org/2000/svg}image"))
    assert bounds[0] == bounds[1]


def test_calendar_figure_preserves_every_choice_and_a_common_scale(
    data_dir: Path = DATA,
):
    daily = pl.scan_parquet(data_dir / "ridge80_calendar_daily.parquet").collect()
    development, later = ridge_figures.calendar_periods(daily)
    assert later["limits"][1] - later["limits"][0] == pytest.approx(
        development["limits"][1] - development["limits"][0]
    )
    for period in (development, later):
        left, right = period["limits"]
        points = period["points"]
        assert len({(point["weekday"], point["schedules"]) for point in points}) == 35
        assert [
            sum(point["sleeves"] == count for point in points) for count in (1, 2, 3)
        ] == [15, 15, 5]
        assert all(left < point["annual_return"] < right for point in points)
    for count, expected in [(1, (5.65, 9.42)), (3, (7.13, 8.08))]:
        values = [
            point["annual_return"]
            for point in later["points"]
            if point["sleeves"] == count
        ]
        assert (min(values), max(values)) == pytest.approx(expected, abs=0.005)
