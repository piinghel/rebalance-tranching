"""Show the schedule rotation and combine invented daily portfolio returns."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from rebalance_tranching import analysis


def main() -> None:
    print("Each sleeve has one-third notional; X marks its rebalance.")
    print("Sleeve   W1 W2 W3 W4 W5 W6")
    for offset, sleeve in enumerate("ABC"):
        marks = "  ".join("X" if week % 3 == offset else "." for week in range(6))
        print(f"{sleeve}        {marks}")

    # These are made-up portfolio returns, not returns derived from the marks.
    # The schedule shows WHEN sleeves trade; the input series shows their P&L.
    returns = ((0.03, -0.02, 0.01), (-0.01, 0.02, 0.00), (0.01, -0.01, 0.02))
    standalone = pl.DataFrame(
        [
            {
                "date": date(2026, 1, 5) + timedelta(days=day),
                "offset": offset,
                "gross": gross,
                "net": gross - 0.0001,
            }
            for offset, path in enumerate(returns)
            for day, gross in enumerate(path)
        ]
    )
    mixture = (
        analysis.mixtures(standalone)
        .lazy()
        .filter(pl.col("sleeves") == 3)
        .select("date", "gross", "net")
        .collect()
    )
    print("Synthetic three-sleeve daily P&L per unit of total fixed notional:")
    print(mixture)
    print("Gross first-day mixture: (3% - 1% + 1%) / 3 = 1%.")
    print("The illustrative 1 bp daily drag is not an execution-cost simulation.")


if __name__ == "__main__":
    main()
