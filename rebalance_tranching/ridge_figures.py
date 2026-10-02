"""Article figures from matched Ridge calendar returns, with shared theme layouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import MaxNLocator

from rebalance_tranching.calendar_grid import combine_grid, grid_metrics

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none",
    }
)


def theme(dark: bool) -> dict[str, str]:
    return dict(
        background="#0d1117" if dark else "#ffffff",
        ink="#dce3eb" if dark else "#27343d",
        grid="#36404a" if dark else "#e2e7eb",
        blue="#c9d1d9" if dark else "#24292f",
        gray="#727d88" if dark else "#a3acb5",
        teal="#3987e5" if dark else "#2a78d6",
        orange="#8b949e" if dark else "#6e7781",
    )


def axes_style(ax, colors: dict[str, str], *, mobile: bool) -> None:
    ax.set_facecolor(colors["background"])
    ax.tick_params(
        colors=colors["ink"], labelsize=13 if mobile else 11, length=0, pad=8
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    lines = []
    for key, label, color, dash in [
        ("1", "Week 1", "orange", "-"),
        ("2", "Week 2", "orange", "--"),
        ("3", "Week 3", "orange", "-."),
        ("1+2+3", "Three\ntranches" if mobile else "Three tranches", "teal", "-"),
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
            lw=2.1 if key == "1+2+3" else 1.3,
            ls=dash,
        )
        lines.append((float(growth[-1]), label, colors[color]))
    # Endpoint labels live beyond the data and retain their vertical order.
    lines.sort()
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * (0.11 if mobile else 0.065)
    positions = []
    for value, label, color in lines:
        y = max(value, positions[-1] + gap) if positions else value
        positions.append(y)
        ax.annotate(
            label,
            xy=(dates[-1], value),
            xytext=(1.04, y),
            textcoords=ax.get_yaxis_transform(),
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
            color=color,
            fontsize=13 if mobile else 11,
            va="center",
            annotation_clip=False,
        )
    ax.grid(axis="y", color=colors["grid"], lw=0.6)
    ax.axhline(100, color=colors["grid"], lw=0.8)
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(
        0,
        1.05,
        "Net growth index · 100 at start",
        transform=ax.transAxes,
        color=colors["ink"],
        fontsize=13 if mobile else 12,
        fontweight="semibold",
    )
    fig.subplots_adjust(
        left=0.13 if mobile else 0.08,
        right=0.79 if mobile else 0.87,
        bottom=0.13,
        top=0.86,
    )
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def calendars(daily: pl.DataFrame, output: Path, *, dark: bool, mobile: bool) -> None:
    """Show every calendar on a common return scale, with combined books beside it."""
    colors = theme(dark)
    fig, axs = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(5.2, 8.0) if mobile else (10, 4.9),
        squeeze=False,
    )
    fig.patch.set_facecolor(colors["background"])
    frames = [
        grid_metrics(daily.lazy().filter(expression).collect())
        for expression in [
            pl.col("date") < pl.date(2022, 1, 1),
            pl.col("date") >= pl.date(2022, 1, 1),
        ]
    ]
    low = min(f["net_cagr"].to_numpy().min() for f in frames)
    high = max(f["net_cagr"].to_numpy().max() for f in frames)
    pad = (high - low) * 0.13
    for ax, frame, title in zip(
        axs.flat, frames, ["1998–2021", "2022–May 2026"], strict=True
    ):
        axes_style(ax, colors, mobile=mobile)
        for day in range(1, 6):
            part = frame.lazy().filter(pl.col("weekday") == day).collect()
            singles = (
                part.lazy().filter(pl.col("sleeves") == 1).sort("schedules").collect()
            )
            ax.plot(
                [singles["net_cagr"].min(), singles["net_cagr"].max()],
                [day, day],
                color=colors["gray"],
                alpha=0.45,
                lw=1,
            )
            for row, marker, shift in zip(
                singles.iter_rows(named=True),
                ["o", "s", "^"],
                [-0.11, 0, 0.11],
                strict=True,
            ):
                ax.scatter(
                    row["net_cagr"],
                    day + shift,
                    s=25,
                    marker=marker,
                    color=colors["orange"],
                    zorder=3,
                )
            blend = (
                part.lazy().filter(pl.col("sleeves") == 3).collect()["net_cagr"].item()
            )
            ax.scatter(
                blend, day + 0.27, s=44, marker="D", color=colors["teal"], zorder=4
            )
        ax.set_yticks(
            range(1, 6), ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )
        ax.set_ylim(5.7, 0.5)
        ax.set_xlim(low - pad, high + pad)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color=colors["grid"], lw=0.6)
        ax.set_xlabel(
            "Annualized net return (%)",
            color=colors["ink"],
            fontsize=13 if mobile else 11,
            labelpad=12,
        )
        ax.set_title(
            title,
            loc="left",
            color=colors["ink"],
            fontsize=14 if mobile else 13,
            fontweight="semibold",
            pad=15,
        )
    handles = [
        plt.Line2D(
            [],
            [],
            color=colors["orange"],
            marker=m,
            ls="",
            label=f"Week {i + 1}",
            markersize=5,
        )
        for i, m in enumerate(["o", "s", "^"])
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=colors["teal"],
            marker="D",
            ls="",
            label="Three tranches",
            markersize=6,
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2 if mobile else 4,
        frameon=False,
        labelcolor=colors["ink"],
        fontsize=12.5 if mobile else 10.5,
    )
    fig.subplots_adjust(
        left=0.24 if mobile else 0.11,
        right=0.97,
        top=0.85 if mobile else 0.79,
        bottom=0.08 if mobile else 0.16,
        wspace=0.42,
        hspace=0.45,
    )
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def interactive(daily: pl.DataFrame, blog: Path, output: Path) -> None:
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
        ("1+2+3", "Three tranches", "long", "solid"),
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
                series=["1", "2", "3", "1+2+3"],
                initialRange=["2021-12-31", "2026-05-27"],
                directLabels=True,
                showLegend=False,
                unit="Net growth index · 100 at selected start",
                episodes=[
                    ["Development", str(part["date"][0]), "2021-12-31"],
                    ["Later", "2021-12-31", "2026-05-27"],
                ],
                note="Net of 5 bp per dollar traded. Daily P&L per unit of fixed notional, compounded for display; zero-cash Sharpe.",
            )
        },
    )
    panels = []
    for title, expression in [
        ("1998–2021", pl.col("date") < pl.date(2022, 1, 1)),
        ("2022–May 2026", pl.col("date") >= pl.date(2022, 1, 1)),
    ]:
        frame = grid_metrics(daily.lazy().filter(expression).collect())
        panels.append(
            dict(
                title=title,
                points=frame.lazy()
                .select("weekday", "schedules", "net_cagr")
                .collect()
                .to_dicts(),
            )
        )
    (output / "ridge-calendars.json").write_text(
        json.dumps(
            dict(
                version=1,
                charts={"calendars": dict(kind="calendar-comparison", panels=panels)},
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
    args.output.mkdir(parents=True, exist_ok=True)
    for dark in [False, True]:
        for mobile in [False, True]:
            suffix = ("_mobile" if mobile else "") + ("_dark" if dark else "") + ".svg"
            performance(
                daily, args.output / ("ridge-paths" + suffix), dark=dark, mobile=mobile
            )
            calendars(
                daily,
                args.output / ("ridge-calendars" + suffix),
                dark=dark,
                mobile=mobile,
            )
    interactive(daily, args.blog, args.output)


if __name__ == "__main__":
    main()
