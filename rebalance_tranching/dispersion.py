"""Render schedule dispersion and diversification from the published metrics CSV."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import math
from pathlib import Path


def load_metrics(path: Path) -> list[dict[str, str]]:
    """Require all seven unique combinations in each reporting period."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "+".join(map(str, members))
        for n in (1, 2, 3)
        for members in itertools.combinations((1, 2, 3), n)
    }
    if len(rows) != 14:
        raise ValueError("Expected seven combinations in each of two periods")
    for period in ("development", "later"):
        sample = [row for row in rows if row["period"] == period]
        if len(sample) != 7 or {row["schedules"] for row in sample} != expected:
            raise ValueError(f"Missing or duplicate combinations in {period}")
        if len({(row["start"], row["end"], row["days"]) for row in sample}) != 1:
            raise ValueError(f"Unmatched reporting windows in {period}")
        for row in sample:
            if int(row["sleeves"]) != len(row["schedules"].split("+")):
                raise ValueError("Sleeve count disagrees with combination")
            if not all(
                math.isfinite(float(row[key])) for key in ("net_cagr", "volatility")
            ):
                raise ValueError("Non-finite chart metric")
    return rows


def render(rows: list[dict[str, str]], *, dark: bool) -> str:
    """Use position scales, retaining every combination and a common risk axis."""
    ink, muted, grid, accent = (
        ("#c9d1d9", "#9ba5af", "#30363d", "#7fa4c4")
        if dark
        else ("#33404b", "#81909c", "#dbe1e3", "#4f7396")
    )
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 880" role="img">',
        "<title>Return dispersion and volatility for all combinations of three rebalance weeks</title>",
        '<g font-family="DejaVu Sans, sans-serif">',
    ]

    def text(
        x: float, y: float, value: str, *, anchor: str = "start", bold: bool = False
    ) -> None:
        elements.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{ink}" font-size="21" font-weight="{600 if bold else 400}" text-anchor="{anchor}">{html.escape(value)}</text>'
        )

    def line(
        x1: float, y1: float, x2: float, y2: float, color: str, dashed: bool = False
    ) -> None:
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        elements.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.5"{dash}/>'
        )

    for column, (period, title) in enumerate(
        (("development", "Development · 1998–2021"), ("later", "Later · 2022–May 2026"))
    ):
        left, right = 70 + column * 560, 520 + column * 560
        sample = [row for row in rows if row["period"] == period]
        text(left, 30, title, bold=True)
        for metric, label, top, bottom, ticks in (
            (
                "net_cagr",
                "Net geometric return (% / year)",
                112,
                342,
                (11.5, 12, 12.5, 13) if column == 0 else (5, 6, 7, 8, 9, 10.5),
            ),
            ("volatility", "Annualized volatility (%)", 510, 740, (7, 8, 9, 10)),
        ):
            text(left, top - 32, label)
            lo, hi = ticks[0], ticks[-1]

            def ypos(
                value: float,
                lo: float = lo,
                hi: float = hi,
                top: float = top,
                bottom: float = bottom,
                context: str = f"{period} {metric}",
            ) -> float:
                if not lo <= value <= hi:
                    raise ValueError(f"{context}={value} falls outside chart axes")
                return bottom - (value - lo) * (bottom - top) / (hi - lo)

            for tick in ticks:
                y = ypos(tick)
                line(left, y, right, y, grid)
                text(left - 12, y + 7, f"{tick:g}", anchor="end")
            mean_single = (
                sum(float(row[metric]) for row in sample if row["sleeves"] == "1") / 3
            )
            if metric == "volatility":
                line(
                    left,
                    ypos(mean_single),
                    right,
                    ypos(mean_single),
                    muted,
                    dashed=True,
                )
                text(
                    left + 4, ypos(mean_single) - 13, f"Mean single: {mean_single:.2f}%"
                )
            for count in (1, 2, 3):
                group = [row for row in sample if int(row["sleeves"]) == count]
                x = left + 55 + (count - 1) * 155
                ys = [ypos(float(row[metric])) for row in group]
                line(x, min(ys), x, max(ys), muted)
                for index, y in enumerate(ys):
                    jitter = (index - (len(ys) - 1) / 2) * 14
                    if count == 3:
                        elements.append(
                            f'<path d="M{x:.1f},{y - 8:.1f} l8,14 h-16 Z" fill="{accent}"/>'
                        )
                    else:
                        elements.append(
                            f'<circle cx="{x + jitter:.1f}" cy="{y:.1f}" r="5.5" fill="{muted}"/>'
                        )
                text(x, bottom + 32, str(count), anchor="middle")
                if count == 3 and metric == "volatility":
                    value = float(group[0][metric])
                    reduction = 100 * (1 - value / mean_single)
                    text(x, ys[0] + 31, f"{value:.2f}%", anchor="middle", bold=True)
                    text(
                        left,
                        bottom + 88,
                        f"All three: {reduction:.1f}% less volatility",
                        bold=True,
                    )
            text((left + right) / 2, bottom + 61, "Number of sleeves", anchor="middle")
    text(
        560,
        873,
        "Dots: individual weeks and pairs · triangle: all three weeks",
        anchor="middle",
    )
    return "\n".join(elements + ["</g></svg>"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, default=root / "data/timing_metrics.csv"
    )
    parser.add_argument("--output", type=Path, default=root / "output")
    args = parser.parse_args()
    rows = load_metrics(args.metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    for dark in (False, True):
        (args.output / f"timing-dispersion{'_dark' if dark else ''}.svg").write_text(
            render(rows, dark=dark), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
