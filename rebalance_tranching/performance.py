"""Show fixed best/worst Friday calendars and their three-tranche portfolio."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, StrMethodFormatter

matplotlib.use("Agg")


def load_returns(path: Path) -> list[dict[str, str]]:
    """Validate the chart-ready view, including the equal-notional identity."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (
        not rows
        or not {"date", "period", "week_1", "week_2", "week_3", "mixture"}
        <= rows[0].keys()
    ):
        raise ValueError("Expected dated schedule returns and a mixture")
    dates = [date.fromisoformat(row["date"]) for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Schedule dates must be ordered and unique")
    if sum(row["period"] == "later" for row in rows) < 2:
        raise ValueError("Expected at least two later-period observations")
    for row, observation_date in zip(rows, dates, strict=True):
        expected_period = (
            "later" if observation_date >= date(2022, 1, 1) else "development"
        )
        if row["period"] != expected_period:
            raise ValueError("Reporting period disagrees with observation date")
        values = [float(row[key]) for key in ("week_1", "week_2", "week_3", "mixture")]
        if not all(math.isfinite(value) and value > -1 for value in values):
            raise ValueError("Expected finite daily returns above -1")
        if not math.isclose(values[3], sum(values[:3]) / 3, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                "Mixture must equal the mean of the three schedule returns"
            )
    return rows


def render(
    rows: list[dict[str, str]], output: Path, *, dark: bool, mobile: bool = False
) -> None:
    ink, grid = ("#c9d1d9", "#30363d") if dark else ("#33404b", "#dbe1e3")
    muted, accent = ("#a7b4c1", "#79b9f0") if dark else ("#8796a4", "#174b78")
    with plt.rc_context(
        {
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": "timing-performance",
        }
    ):
        fig, ax = plt.subplots(figsize=(5.2, 5.6) if mobile else (9.0, 4.8))
        fig.patch.set_alpha(0)
        sample = [row for row in rows if row["period"] == "later"]
        observations = [date.fromisoformat(row["date"]) for row in sample]
        dates = np.array(
            [observations[0] - timedelta(days=1), *observations], dtype="datetime64[D]"
        )
        paths, cagrs = [], []
        for key in ("week_1", "week_2", "week_3", "mixture"):
            returns = np.array([float(row[key]) for row in sample])
            growth = np.r_[1.0, np.cumprod(1 + returns)]
            paths.append(growth)
            cagrs.append((growth[-1] ** (252 / len(returns)) - 1) * 100)
        # Select once over the entire period; never switch calendars along a path.
        best, worst = int(np.argmax(cagrs[:3])), int(np.argmin(cagrs[:3]))
        ax.fill_between(dates, paths[worst], paths[best], color=muted, alpha=0.20)
        for index in (best, worst, 3):
            color = accent if index == 3 else muted
            growth = paths[index]
            ax.plot(dates, growth, color=color, linewidth=2.1 if index == 3 else 0.9)
            label = (
                "Three-tranche\nportfolio"
                if index == 3
                else f"{'Best' if index == best else 'Worst'}: Week {index + 1}"
            )
            label_y = 0.96 if index == best else 0.60 if index == worst else 0.78
            ax.annotate(
                f"{label}\n{cagrs[index]:.2f}%",
                xy=(dates[-1], growth[-1]),
                xycoords="data",
                xytext=(1.035, label_y),
                textcoords=ax.transAxes,
                va="center",
                fontsize=12,
                color=color,
                fontweight="medium" if index == 3 else "normal",
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 0.7,
                    "connectionstyle": "angle,angleA=0,angleB=90,rad=2",
                },
                annotation_clip=False,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([1, 1.2, 1.4, 1.6]))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(
            "Net growth index · log scale",
            loc="left",
            color=ink,
            fontsize=12,
            pad=14,
        )
        ax.text(
            0.02,
            0.94,
            f"Return spread: {max(cagrs[:3]) - min(cagrs[:3]):.2f} pp",
            transform=ax.transAxes,
            color=ink,
            fontsize=12,
            va="top",
        )
        years = [2024, 2026] if mobile else [2023, 2024, 2025, 2026]
        ax.xaxis.set_major_locator(
            FixedLocator(
                mdates.date2num(
                    [observations[0], *[date(year, 1, 1) for year in years]]
                )
            )
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", which="major", color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.patch.set_alpha(0)
        ax.tick_params(axis="both", which="both", colors=ink, length=0, pad=7)
        ax.margins(x=0, y=0.06)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.subplots_adjust(
            left=0.09 if mobile else 0.065,
            right=0.69 if mobile else 0.79,
            top=0.90,
            bottom=0.10,
        )
        fig.savefig(output, transparent=True, metadata={"Date": None})
        output.write_text(
            "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
        )
        fig.savefig(
            output.with_suffix(".png"),
            dpi=300,
            facecolor="#0d1117" if dark else "#ffffff",
        )
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=root / "data/schedule_returns.csv"
    )
    parser.add_argument("--output", type=Path, default=root / "output")
    arguments = parser.parse_args()
    rows = load_returns(arguments.input)
    arguments.output.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        for mobile in (False, True):
            render(
                rows,
                arguments.output
                / f"schedule-performance{'_mobile' if mobile else ''}{'_dark' if dark else ''}.svg",
                dark=dark,
                mobile=mobile,
            )


if __name__ == "__main__":
    main()
