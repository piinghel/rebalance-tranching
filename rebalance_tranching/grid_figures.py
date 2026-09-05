"""Render the calendar comparison from saved metrics, in both site themes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def save(fig: plt.Figure, path: Path) -> None:
    svg = path.with_suffix(".svg")
    fig.savefig(svg, transparent=True, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def cell(metrics: pl.DataFrame, weekday: int, schedule: str, metric: str) -> float:
    return (
        metrics.lazy()
        .filter((pl.col("weekday") == weekday) & (pl.col("schedules") == schedule))
        .select(metric)
        .collect()
        .item()
    )


def render(metrics: pl.DataFrame, output: Path, *, dark: bool, mobile: bool) -> None:
    ink, muted, grid, accent, background = (
        ("#d2d9df", "#a2afbc", "#343f49", "#79b9f0", "#0d1117")
        if dark
        else ("#33404b", "#8796a4", "#dbe1e3", "#174b78", "#ffffff")
    )
    suffix = f"{'_mobile' if mobile else ''}{'_dark' if dark else ''}"
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    with plt.rc_context(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "svg.fonttype": "none",
            "svg.hashsalt": "calendar-grid",
            "figure.facecolor": background,
            "axes.facecolor": background,
        }
    ):
        values = np.array(
            [
                [cell(metrics, d, str(o), "net_cagr") for d in range(1, 6)]
                for o in range(1, 4)
            ]
        )
        combined = [cell(metrics, d, "1+2+3", "net_cagr") for d in range(1, 6)]
        cmap = LinearSegmentedColormap.from_list(
            "calendar", ["#263743", "#7295b1"] if dark else ["#f0f3f5", "#7295b1"]
        )
        fig, ax = plt.subplots(figsize=(4.6, 3.7) if mobile else (8.5, 3.8))
        ax.pcolormesh(
            np.arange(6) - 0.5,
            np.arange(4) - 0.5,
            values,
            cmap=cmap,
            vmin=values.min(),
            vmax=values.max(),
            rasterized=False,
        )
        for row in range(3):
            for column in range(5):
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="#f5f7f9" if dark else "#243340",
                )
        for column, value in enumerate(combined):
            ax.text(
                column,
                3.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=accent,
                fontweight="bold",
            )
        ax.axhline(2.95, color=grid, linewidth=0.8)
        ax.set_xticks(range(5), [d[:3] for d in weekdays] if mobile else weekdays)
        ax.xaxis.tick_top()
        ax.set_yticks([0, 1, 2, 3.5], ["Week 1", "Week 2", "Week 3", "Combined"])
        ax.set_ylim(4, -0.5)
        ax.tick_params(length=0, pad=9)
        ax.set_title(
            "Annualized net return (%) · 2022–May 2026",
            loc="left",
            color=ink,
            fontsize=11.5 if mobile else 12,
            pad=35,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(
            left=0.21 if mobile else 0.16, right=0.99, top=0.72, bottom=0.05
        )
        save(fig, output / f"calendar-grid{suffix}")

        fig, axes = plt.subplots(
            2 if mobile else 1,
            1 if mobile else 2,
            figsize=(4.6, 7.7) if mobile else (9, 4.6),
        )
        for ax, metric, title in zip(
            axes,
            ("net_cagr", "volatility"),
            ("Annualized net return (%)", "Annualized volatility (%)"),
            strict=True,
        ):
            for d in range(1, 6):
                singles = np.array(
                    [cell(metrics, d, str(o), metric) for o in range(1, 4)]
                )
                value = cell(metrics, d, "1+2+3", metric)
                ax.plot(
                    [singles.min(), singles.max()], [d, d], color=muted, linewidth=0.7
                )
                # Fixed row offsets reveal near-coincident means; x stays exact.
                ax.scatter(
                    singles,
                    [d] * 3,
                    color=muted,
                    s=24,
                    label="Individual offsets" if d == 1 else None,
                    zorder=3,
                )
                ax.scatter(
                    singles.mean(),
                    d - 0.16,
                    marker="D",
                    s=40,
                    facecolor=background,
                    edgecolor=ink,
                    label="Standalone mean" if d == 1 else None,
                    zorder=4,
                )
                ax.scatter(
                    value,
                    d + 0.16,
                    marker="s",
                    s=38,
                    color=accent,
                    label="Three-tranche portfolio" if d == 1 else None,
                    zorder=5,
                )
            means = (
                metrics.lazy().group_by("sleeves").agg(pl.col(metric).mean()).collect()
            )
            before = means.filter(pl.col("sleeves") == 1)[metric].item()
            after = means.filter(pl.col("sleeves") == 3)[metric].item()
            ax.set_title(title, loc="left", color=ink, fontsize=12, pad=32)
            ax.text(
                0,
                1.025,
                f"Mean: {before:.2f}% → {after:.2f}%",
                transform=ax.transAxes,
                color=ink,
                fontsize=11,
                va="bottom",
            )
            ax.set_yticks(range(1, 6), weekdays)
            ax.set_ylim(5.55, 0.45)
            ax.grid(axis="x", color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=0, pad=8)
            ax.margins(x=0.10)
            for spine in ax.spines.values():
                spine.set_visible(False)
        if not mobile:
            axes[1].set_yticklabels([])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1 if mobile else 3,
            frameon=False,
            fontsize=11.5 if mobile else 10.5,
            labelcolor=ink,
            bbox_to_anchor=(0.55 if mobile else 0.51, 0.005),
        )
        fig.subplots_adjust(
            left=0.26 if mobile else 0.15,
            right=0.98,
            top=0.90 if mobile else 0.82,
            bottom=0.16 if mobile else 0.20,
            hspace=0.43,
            wspace=0.14,
        )
        save(fig, output / f"calendar-return-risk{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pl.scan_csv(
        args.input / "later_metrics.csv", schema_overrides={"schedules": pl.String}
    ).collect()
    for dark in (False, True):
        for mobile in (False, True):
            render(metrics, args.output, dark=dark, mobile=mobile)


if __name__ == "__main__":
    main()
