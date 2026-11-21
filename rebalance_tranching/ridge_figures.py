"""Article figures from matched Ridge calendar returns, with shared theme layouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import TypedDict

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import (
    FixedLocator,
    MaxNLocator,
    NullLocator,
    ScalarFormatter,
)

from rebalance_tranching.calendar_grid import combine_grid, grid_metrics

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "rebalance-tranching",
    }
)


def theme(dark: bool) -> dict[str, str]:
    return dict(
        background="#0d1117" if dark else "#ffffff",
        ink="#dce3eb" if dark else "#27343d",
        grid="#36404a" if dark else "#e2e7eb",
        combined="#58a6ff" if dark else "#185fa9",
        single="#8b949e" if dark else "#6e7781",
        calendar_blend="#76bda5" if dark else "#176b58",
    )


def axes_style(ax: Axes, colors: dict[str, str], *, mobile: bool) -> None:
    ax.set_facecolor(colors["background"])
    ax.tick_params(
        colors=colors["ink"], labelsize=13 if mobile else 11, length=0, pad=8
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_svg(fig: Figure, output: Path) -> None:
    fig.savefig(output, facecolor=fig.get_facecolor(), metadata={"Date": None})
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def performance(daily: pl.DataFrame, output: Path, *, dark: bool, mobile: bool) -> None:
    """Friday schedules and their equal-notional portfolio."""
    colors = theme(dark)
    frame = (
        combine_grid(daily)
        .lazy()
        .filter(pl.col("weekday") == 5, pl.col("date") >= pl.date(2021, 12, 31))
        .sort("date")
        .collect()
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.2) if mobile else (9, 4.5))
    fig.patch.set_facecolor(colors["background"])
    axes_style(ax, colors, mobile=mobile)
    ax.set_yscale("log")
    lines = []
    for key, label, color, dash in [
        ("1", "Week 1", "single", "-"),
        ("2", "Week 2", "single", "--"),
        ("3", "Week 3", "single", "-."),
        ("1+2+3", "Three\ntranches" if mobile else "Three tranches", "combined", "-"),
    ]:
        part = frame.lazy().filter(pl.col("schedules") == key).collect()
        dates = part["date"].to_list()
        # Match the interactive chart: index at the first selected close,
        # excluding the return into that close.
        growth = np.r_[100, 100 * np.cumprod(1 + part["net"].to_numpy()[1:])]
        ax.plot(
            dates,
            growth,
            color=colors[color],
            lw=2.5 if key == "1+2+3" else 1.15,
            ls=dash,
        )
        lines.append((float(growth[-1]), label, colors[color]))
    # Endpoint labels live beyond the data and retain their vertical order.
    lines.sort()
    low, high = np.log(ax.get_ylim())
    gap = (high - low) * (0.11 if mobile else 0.065)
    positions = []
    for value, label, color in lines:
        y = max(np.log(value), positions[-1] + gap) if positions else np.log(value)
        positions.append(y)
        ax.annotate(
            label,
            xy=(dates[-1], value),
            xytext=(1.04, np.exp(y)),
            textcoords=ax.get_yaxis_transform(),
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
            color=color,
            fontsize=13 if mobile else 11,
            va="center",
            annotation_clip=False,
        )
    ax.yaxis.set_major_locator(
        FixedLocator(
            MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(*ax.get_ylim())
        )
    )
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color=colors["grid"], lw=0.6)
    ax.axhline(100, color=colors["single"], lw=0.8, alpha=0.55, zorder=0)
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(
        0,
        1.05,
        "Indexed growth (log scale)",
        transform=ax.transAxes,
        color=colors["ink"],
        fontsize=13 if mobile else 12,
        fontweight="semibold",
    )
    fig.subplots_adjust(
        left=0.13 if mobile else 0.08,
        right=0.78 if mobile else 0.83,
        bottom=0.13,
        top=0.86,
    )
    save_svg(fig, output)


class CalendarPoint(TypedDict):
    weekday: int
    schedules: str
    sleeves: int
    annual_return: float
    position: float


class CalendarPeriod(TypedDict):
    title: str
    limits: tuple[float, float]
    points: list[CalendarPoint]


def calendar_periods(daily: pl.DataFrame) -> list[CalendarPeriod]:
    """One point per observed calendar, with a common horizontal scale."""
    split = pl.date(2022, 1, 1)
    samples = [
        ("Sep 1998–Dec 2021", pl.col("date") < split),
        ("Jan 2022–May 2026", pl.col("date") >= split),
    ]
    frames = [
        grid_metrics(daily.lazy().filter(expression).collect(), include_pairs=True)
        for _, expression in samples
    ]
    returns = [frame["net_cagr"].to_list() for frame in frames]
    bounds = [(min(values), max(values)) for values in returns]
    span = math.ceil(4 * max(high - low + 0.45 for low, high in bounds)) / 4
    panels = []
    for (title, _), frame, (low, high) in zip(samples, frames, bounds, strict=True):
        left = round(2 * (low + high - span)) / 4
        points = []
        for count in (1, 2, 3):
            group = (
                frame.lazy()
                .filter(pl.col("sleeves") == count)
                .select("weekday", "schedules", "net_cagr")
                .sort("net_cagr", "weekday", "schedules")
                .collect()
            )
            # A slight vertical offset keeps similar returns visible on the same row.
            for index, (weekday, schedules, annual_return) in enumerate(
                group.iter_rows()
            ):
                points.append(
                    CalendarPoint(
                        weekday=weekday,
                        schedules=schedules,
                        sleeves=count,
                        annual_return=annual_return,
                        position=3 - count + (0, 0.075, -0.075)[index % 3],
                    )
                )
        panels.append(
            CalendarPeriod(title=title, limits=(left, left + span), points=points)
        )
    return panels


def calendars(
    periods: list[CalendarPeriod], output: Path, *, dark: bool, mobile: bool
) -> None:
    """Show the observed net returns for every calendar choice."""
    colors = theme(dark)
    fig, axs = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(5.2, 6.05) if mobile else (10, 2.95),
        squeeze=False,
    )
    fig.patch.set_facecolor(colors["background"])
    for index, (ax, period) in enumerate(zip(axs.flat, periods, strict=True)):
        axes_style(ax, colors, mobile=mobile)
        for count in (1, 2, 3):
            group = [point for point in period["points"] if point["sleeves"] == count]
            values = [point["annual_return"] for point in group]
            color = colors["calendar_blend" if count == 3 else "single"]
            ax.hlines(3 - count, min(values), max(values), color=color, alpha=0.55)
            ax.scatter(
                values,
                [point["position"] for point in group],
                color=color,
                s=26 if mobile else 23,
                linewidth=0.55,
                edgecolors=colors["background"],
                zorder=3,
            )
        left, right = period["limits"]
        ax.set_xlim(left, right)
        ax.set_xticks(range(math.ceil(left), math.floor(right) + 1))
        ax.set_ylim(-0.48, 2.48)
        ax.set_yticks([2, 1, 0], ["One book", "Two books", "Three books"])
        ax.tick_params(labelleft=mobile or index == 0)
        ax.grid(axis="x", color=colors["grid"], lw=0.65)
        ax.set_title(
            period["title"],
            loc="left",
            color=colors["ink"],
            fontsize=14 if mobile else 12,
            fontweight="semibold",
            pad=11,
        )
    fig.supxlabel(
        "Annualized net return (%)",
        x=0.60 if mobile else 0.54,
        y=0.04 if mobile else 0.07,
        color=colors["ink"],
        fontsize=13 if mobile else 11,
    )
    fig.subplots_adjust(
        left=0.28 if mobile else 0.13,
        right=0.96,
        top=0.89 if mobile else 0.81,
        bottom=0.15 if mobile else 0.28,
        wspace=0.19,
        hspace=0.57,
    )
    save_svg(fig, output)


def interactive(
    daily: pl.DataFrame, periods: list[CalendarPeriod], blog: Path, output: Path
) -> None:
    """Reuse the site's allowlisted exporter and chart interaction design."""
    spec = importlib.util.spec_from_file_location(
        "blog_charts", blog / "scripts/blog_charts.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Site chart exporter is unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    friday = (
        combine_grid(daily).lazy().filter(pl.col("weekday") == 5).sort("date").collect()
    )
    series = []
    for key, label, role, dash in [
        ("1", "Week 1", "short", "solid"),
        ("2", "Week 2", "short", "dash"),
        ("3", "Week 3", "short", "dot"),
        ("1+2+3", "Three tranches", "blend", "solid"),
    ]:
        part = friday.lazy().filter(pl.col("schedules") == key).collect()
        series.append(
            helper.series(
                key,
                label,
                role,
                part["net"].to_list(),
                dash=dash,
                visible=True,
            )
        )
    helper.write_chart(
        output / "ridge-paths.json",
        [str(x) for x in part["date"]],
        series,
        {
            "schedules": dict(
                kind="performance",
                log=True,
                series=["1", "2", "3", "1+2+3"],
                initialRange=["2021-12-31", "2026-05-27"],
                directLabels=True,
                emphasis="1+2+3",
                referenceLevel=100,
                showLegend=False,
                showWindow=False,
                controlsBelow=True,
                unit="Indexed growth (log scale)",
                episodes=[
                    ["Development", str(part["date"][0]), "2021-12-31"],
                    ["Later", "2021-12-31", "2026-05-27"],
                ],
                note="Net of 5 bp per dollar traded. Daily P&L per unit of fixed notional, compounded for display; zero-cash Sharpe.",
            )
        },
    )
    (output / "ridge-calendars.json").write_text(
        json.dumps(
            dict(
                version=1,
                charts={
                    "calendars": dict(kind="calendar-distribution", periods=periods)
                },
            ),
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blog", required=True, type=Path)
    args = parser.parse_args()
    daily = pl.scan_parquet(args.input).collect()
    combine_grid(daily)  # Validate complete, finite, matched calendar inputs.
    periods = calendar_periods(daily)
    args.output.mkdir(parents=True, exist_ok=True)
    for dark in [False, True]:
        for mobile in [False, True]:
            suffix = ("_mobile" if mobile else "") + ("_dark" if dark else "") + ".svg"
            performance(
                daily, args.output / ("ridge-paths" + suffix), dark=dark, mobile=mobile
            )
            calendars(
                periods,
                args.output / ("ridge-calendars" + suffix),
                dark=dark,
                mobile=mobile,
            )
    interactive(daily, periods, args.blog, args.output)


if __name__ == "__main__":
    main()
