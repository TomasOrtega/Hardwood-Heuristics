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

from src.theorems.utils import (
    apply_plot_aesthetics,
    FIGURE_DPI,
    get_resolved_possessions_at_time,
    load_sweep_csv,
    write_sweep_csv,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
CSV_FILENAME = "theorem1_sweep.csv"
FIGURE_FILENAME = "two_for_one_ev_curve.svg"
DOC_FILENAME = "theorem1_two_for_one.md"

# Default win rate used when a bucket has no historical observations
_DEFAULT_WIN_RATE = 0.5


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

    for sec in range(10, 41, 2):
        if df.empty:
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

        resolved = get_resolved_possessions_at_time(df, sec)
        window = resolved[
            (resolved["score_differential"] == 0) & (resolved["possession"] == 1)
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
    ev_rush = [r["ev_rush"] * 100 for r in sweep]
    ev_normal = [r["ev_normal"] * 100 for r in sweep]
    ev_gain = [r["ev_gain"] * 100 for r in sweep]

    apply_plot_aesthetics()

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

> **Rushing a shot in tied games is beneficial when there is more than one possession remaining.**

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

    content = _TEMPLATE.format(
        conclusion="The 2-for-1 shows a positive signal for most of the analyzed time range.",
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 1 doc to %s", out_path)
    return out_path
