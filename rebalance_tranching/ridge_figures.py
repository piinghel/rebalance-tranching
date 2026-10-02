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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, ScalarFormatter

from rebalance_tranching.calendar_grid import combine_grid, tranche_comparison

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
        range="#727d88" if dark else "#a3acb5",
        combined="#3987e5" if dark else "#2a78d6",
        single="#8b949e" if dark else "#6e7781",
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
            lw=2.1 if key == "1+2+3" else 1.3,
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
    ax.axhline(100, color=colors["grid"], lw=0.8)
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(
        0,
        1.05,
        "Net growth · log scale",
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
    save_svg(fig, output)


def calendars(daily: pl.DataFrame, output: Path, *, dark: bool, mobile: bool) -> None:
    """Show the observed dispersion/trading trade-off through weekly thirds."""
    colors = theme(dark)
    frames = [
        tranche_comparison(daily.lazy().filter(expression).collect())
        for expression in [
            pl.col("date") < pl.date(2022, 1, 1),
            pl.col("date") >= pl.date(2022, 1, 1),
        ]
    ]
    fig, axs = plt.subplots(
        2 if mobile else 1,
        1 if mobile else 2,
        figsize=(5.2, 7.6) if mobile else (10, 4.2),
        squeeze=False,
    )
    fig.patch.set_facecolor(colors["background"])
    for ax, metric, title in zip(
        axs.flat,
        ["spread_pp", "annual_orders"],
        ["Calendar spread (pp)", "Orders per year"],
        strict=True,
    ):
        axes_style(ax, colors, mobile=mobile)
        for i, (frame, period, color, dash) in enumerate(
            zip(
                frames,
                ["1998–2021", "2022–May 2026"],
                [colors["single"], colors["combined"]],
                ["--", "-"],
                strict=True,
            )
        ):
            values = frame[metric].to_numpy()
            other = frames[1 - i][metric].to_numpy()
            ax.plot(
                [1, 2, 3],
                values,
                color=color,
                ls=dash,
                marker="o",
                lw=1.8,
                label=period,
            )
            for count, value, comparison in zip([1, 2, 3], values, other, strict=True):
                above = value >= comparison
                ax.annotate(
                    f"{value:.2f}" if metric == "spread_pp" else f"{value:,.0f}",
                    (count, value),
                    xytext=(0, 8 if above else -10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if above else "top",
                    color=color,
                    fontsize=12 if mobile else 11,
                )
        maximum = max(float(f[metric].to_numpy().max()) for f in frames)
        ax.set_ylim(0, maximum * 1.18)
        ax.set_xlim(0.75, 3.25)
        ax.set_xticks([1, 2, 3])
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.grid(axis="y", color=colors["grid"], lw=0.6)
        ax.set_xlabel(
            "Number of tranches", color=colors["ink"], fontsize=12, labelpad=10
        )
        ax.set_title(
            title,
            loc="left",
            color=colors["ink"],
            fontsize=14 if mobile else 13,
            pad=15,
        )
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        labelcolor=colors["ink"],
        fontsize=12 if mobile else 11,
    )
    fig.subplots_adjust(
        left=0.13 if mobile else 0.07,
        right=0.96,
        top=0.88 if mobile else 0.79,
        bottom=0.09 if mobile else 0.17,
        wspace=0.33,
        hspace=0.65,
    )
    save_svg(fig, output)


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
                log=True,
                series=["1", "2", "3", "1+2+3"],
                initialRange=["2021-12-31", "2026-05-27"],
                directLabels=True,
                showLegend=False,
                unit="Net growth · log scale",
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
        frame = tranche_comparison(daily.lazy().filter(expression).collect())
        panels.append(
            dict(
                title=title,
                points=frame.to_dicts(),
            )
        )
    (output / "ridge-calendars.json").write_text(
        json.dumps(
            dict(
                version=1,
                charts={"calendars": dict(kind="tranche-tradeoff", periods=panels)},
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
