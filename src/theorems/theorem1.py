"""
theorem1.py
===========
Theorem 1: The 2-for-1.

Contains the data-collection, visualisation, and documentation-generation
logic for Theorem 1.  Results are persisted as ``theorem1_sweep.csv`` under
``data/processed/`` so that plots can be reproduced without re-running the
full data-collection pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.theorems.utils import load_sweep_csv, write_sweep_csv

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
CSV_FILENAME = "theorem1_sweep.csv"
FIGURE_FILENAME = "two_for_one_ev_curve.svg"
DOC_FILENAME = "theorem1_two_for_one.md"

# Default win rate used when a bucket has no historical observations
_DEFAULT_WIN_RATE = 0.5
_TIME_WINDOW_S = 1

# Consistent aesthetics
FIGURE_DPI = 150
FONT_FAMILY = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 1 (2-for-1) historical win rates and save to CSV.

    Filters the historical log for tied games and groups possessions by
    whether the team shot ('shoot') or held the ball.  Saves a
    ``theorem1_sweep.csv`` to *out_dir*.

    Parameters
    ----------
    out_dir       : directory where the CSV will be written.
    processed_dir : directory containing ``transitions.parquet`` (defaults to
                    *out_dir*).

    Returns
    -------
    Path to the saved CSV file.
    """
    if processed_dir is None:
        processed_dir = out_dir

    from src.collect_data import _load_historical_log

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 1 (2-for-1) historical win rates…")

    rows: List[Dict] = []

    tied = (
        df[(df["score_differential"] == 0) & (df["possession"] == 1)]
        if not df.empty
        else df
    )

    for sec in range(10, 65, 2):
        if tied.empty:
            rows.append(
                {
                    "seconds_remaining": sec,
                    "ev_rush": _DEFAULT_WIN_RATE,
                    "ev_normal": _DEFAULT_WIN_RATE,
                    "ev_gain": 0.0,
                    "rush_is_optimal": False,
                }
            )
            continue

        window = tied[
            tied["seconds_remaining"].between(
                sec - _TIME_WINDOW_S, sec + _TIME_WINDOW_S
            )
        ]
        rush_outcomes = window.loc[window["action_taken"] == "shoot", "game_outcome"]
        hold_outcomes = window.loc[window["action_taken"] != "shoot", "game_outcome"]

        ev_rush = (
            float(rush_outcomes.mean()) if len(rush_outcomes) > 0 else _DEFAULT_WIN_RATE
        )
        ev_normal = (
            float(hold_outcomes.mean()) if len(hold_outcomes) > 0 else _DEFAULT_WIN_RATE
        )
        ev_gain = ev_rush - ev_normal

        rows.append(
            {
                "seconds_remaining": sec,
                "ev_rush": round(ev_rush, 4),
                "ev_normal": round(ev_normal, 4),
                "ev_gain": round(ev_gain, 4),
                "rush_is_optimal": ev_gain > 0,
            }
        )

    out_path = out_dir / CSV_FILENAME
    write_sweep_csv(
        out_path,
        rows,
        fieldnames=[
            "seconds_remaining",
            "ev_rush",
            "ev_normal",
            "ev_gain",
            "rush_is_optimal",
        ],
    )
    logger.info("Saved Theorem 1 sweep to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot(
    processed_dir: Path,
    images_dir: Path,
) -> Path:
    """
    Generate the 2-for-1 win-percentage curve from ``theorem1_sweep.csv``.

    Parameters
    ----------
    processed_dir : directory containing ``theorem1_sweep.csv``.
    images_dir    : directory where the SVG will be saved.

    Returns
    -------
    Path to the saved SVG file.
    """
    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    out_path = images_dir / FIGURE_FILENAME

    seconds = [r["seconds_remaining"] for r in sweep]
    ev_rush = [r["ev_rush"] for r in sweep]
    ev_normal = [r["ev_normal"] for r in sweep]
    ev_gain = [r["ev_gain"] for r in sweep]

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(seconds, ev_rush, color="#E63946", linewidth=2.2, label="Rush (shoot now)")
    ax1.plot(
        seconds,
        ev_normal,
        color="#457B9D",
        linewidth=2.2,
        label="Normal (full possession)",
    )
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Historical Win Percentage")
    ax1.set_title(
        "Theorem 1: The 2-for-1\n"
        "Historical Win Percentage: Rush Shot vs. Full Possession",
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    gain_arr = np.array(ev_gain)
    colors = np.where(gain_arr >= 0, "#2DC653", "#E63946")
    ax2.bar(seconds, gain_arr, color=colors, width=1.6, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seconds Remaining in Possession")
    ax2.set_ylabel("Historical Win % Gain from Rushing (pp)")
    ax2.set_title("Win % Gain: Rush - Normal  (green = rushing is better)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Theorem 1 EV curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# Theorem 1: The 2-for-1

## Claim

> **Based on NBA play-by-play data from 2019--2024, rushing a shot in tied
> games shows a positive win-rate signal around 18--22 seconds remaining.
> The effect is noisy and no single threshold reliably separates when
> rushing helps from when it hurts.**

---

## How We Measure It

We filter the historical play-by-play log for tied games **where the home team has possession** and group each possession by strategy:

- **Rush (shoot):** The possessing team takes a shot attempt.
- **Normal (hold):** The possessing team holds the ball (any non-shooting action).

We calculate the **historical win percentage** for each group — the fraction
of games where the home team went on to win given that choice.

---

## Results

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.svg)

### Key Findings

{key_findings}

### Historical Data Summary

Data from 5 NBA seasons (2019--2024):

| Scenario | Rush Win % | Hold Win % | Win % Gain | Better Strategy |
|----------|-----------|-----------|------------|----------------|
| 32 s, tied | {ev_rush_32} | {ev_normal_32} | {ev_gain_32} | {optimal_32} |
| 40 s, tied | {ev_rush_40} | {ev_normal_40} | {ev_gain_40} | {optimal_40} |
| 20 s, tied | {ev_rush_20} | {ev_normal_20} | {ev_gain_20} | {optimal_20} |

> *Values are historical win percentages from NBA play-by-play data, 2019--2024.*

---

## Conclusion

{conclusion}
"""


def generate_doc(
    processed_dir: Path,
    docs_dir: Path,
) -> Path:
    """
    Generate the Theorem 1 Markdown documentation from ``theorem1_sweep.csv``.

    Parameters
    ----------
    processed_dir : directory containing ``theorem1_sweep.csv``.
    docs_dir      : directory where the Markdown file will be written.

    Returns
    -------
    Path to the written Markdown file.
    """
    from src.generate_docs import (
        _fmt_ev,
        _gain_label,
        _find_sweep_entry,
        _largest_window,
        _consecutive_positive_windows,
    )

    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)

    e32 = _find_sweep_entry(sweep, 32)
    e40 = _find_sweep_entry(sweep, 40)
    e20 = _find_sweep_entry(sweep, 20)

    main_window = _largest_window(sweep)
    window_low, window_high = main_window

    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    all_secs = [e["seconds_remaining"] for e in sorted_sweep]
    sweep_min, sweep_max = min(all_secs), max(all_secs)
    window_covers_full_range = (
        main_window[0] == sweep_min and main_window[1] == sweep_max
    )

    if window_low == 0 and window_high == 0:
        conclusion = (
            "**No consistent rushing advantage found** in the historical data. "
            "Normal possession performs at least as well across all analyzed time "
            "buckets. Do not sacrifice shot quality based on the clock alone."
        )
    elif window_covers_full_range:
        conclusion = (
            f"**Rushing shows a modest advantage across the full analyzed range "
            f"({sweep_min}--{sweep_max} s)**, but the gain is small and the results "
            "are noisy. Shot quality matters more than the exact clock value."
        )
    else:
        conclusion = (
            f"**The 2-for-1 shows a positive signal around the {window_low}--"
            f"{window_high} s range**, but results are noisy across individual "
            "time buckets. Favour rushing when a good shot is available in this "
            "window, but do not sacrifice shot quality for a specific clock value."
        )

    from src.generate_docs import _build_theorem1_key_findings

    key_findings = _build_theorem1_key_findings(sweep)

    def _optimal_label(gain: float) -> str:
        return "Rush ✓" if gain > 0 else "Normal ✓"

    content = _TEMPLATE.format(
        key_findings=key_findings,
        conclusion=conclusion,
        ev_rush_32=_fmt_ev(e32["ev_rush"]),
        ev_normal_32=_fmt_ev(e32["ev_normal"]),
        ev_gain_32=_gain_label(e32["ev_gain"]),
        optimal_32=_optimal_label(e32["ev_gain"]),
        ev_rush_40=_fmt_ev(e40["ev_rush"]),
        ev_normal_40=_fmt_ev(e40["ev_normal"]),
        ev_gain_40=_gain_label(e40["ev_gain"]),
        optimal_40=_optimal_label(e40["ev_gain"]),
        ev_rush_20=_fmt_ev(e20["ev_rush"]),
        ev_normal_20=_fmt_ev(e20["ev_normal"]),
        ev_gain_20=_gain_label(e20["ev_gain"]),
        optimal_20=_optimal_label(e20["ev_gain"]),
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 1 doc to %s", out_path)
    return out_path
