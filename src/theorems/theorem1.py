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

RUSH_THRESHOLD_SECONDS = 5


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 1 (2-for-1) historical win rates and save to CSV.

    At each clock value from 30--40 seconds, filters for tied games and keeps
    possessions whose next event is a shot by the team with the ball. Shots
    within five seconds are classified as rush attempts; later shots are
    classified as normal possessions.

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

    for sec in range(30, 41, 2):
        resolved = get_resolved_possessions_at_time(df, sec)
        window = resolved[resolved["score_differential"] == 0].copy()
        window = window[
            (window["action_taken"] == "shoot")
            & (window["action_team"] == window["possession"])
        ].copy()
        window["team_won"] = np.where(
            window["possession"] == 1,
            window["game_outcome"],
            1 - window["game_outcome"],
        )
        rush_outcomes = window.loc[
            window["action_delay"] <= RUSH_THRESHOLD_SECONDS,
            "team_won",
        ]
        normal_outcomes = window.loc[
            window["action_delay"] > RUSH_THRESHOLD_SECONDS,
            "team_won",
        ]

        ev_rush = (
            float(rush_outcomes.mean()) if not rush_outcomes.empty else float("nan")
        )
        ev_normal = (
            float(normal_outcomes.mean())
            if not normal_outcomes.empty
            else float("nan")
        )
        ev_gain = ev_rush - ev_normal

        rows.append(
            {
                "seconds_remaining": sec,
                "ev_rush": round(ev_rush, 4),
                "ev_normal": round(ev_normal, 4),
                "ev_gain": round(ev_gain, 4),
                "n_rush": len(rush_outcomes),
                "n_normal": len(normal_outcomes),
                "rush_is_optimal": bool(np.isfinite(ev_gain) and ev_gain > 0),
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
            "n_rush",
            "n_normal",
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
    ax2.set_xlabel("Seconds Remaining in Period")
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

> **Does taking a quick shot in a tied game improve the chance of winning when
> 30--40 seconds remain?**

---

## How We Measure It

At each clock value from 30 to 40 seconds, we filter for tied games and include
both home and away possessions whose next logged event is a shot by the team
with the ball. We group those possessions by timing:

- **Rush:** The shot occurs within five seconds of the target clock.
- **Normal:** The shot occurs more than five seconds later.

The saved sweep reports the possessing team's historical win percentage and
the number of observations in each group. It is a descriptive association,
not a causal estimate; team quality and game context are not adjusted for.

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

    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    comparable = [
        row
        for row in sweep
        if row["n_rush"] > 0
        and row["n_normal"] > 0
        and np.isfinite(row["ev_gain"])
    ]
    positive = sum(row["ev_gain"] > 0 for row in comparable)
    conclusion = (
        f"Rushing had the higher observed win rate at {positive} of "
        f"{len(comparable)} comparable clock points. The result is mixed and "
        "should not be interpreted as a causal effect."
        if comparable
        else "There are not enough observations to compare the strategies."
    )
    content = _TEMPLATE.format(conclusion=conclusion)

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 1 doc to %s", out_path)
    return out_path
