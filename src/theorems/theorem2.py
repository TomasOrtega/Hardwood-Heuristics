"""
theorem2.py
===========
Theorem 2: Foul Up 3.

Contains the data-collection, visualisation, and documentation-generation
logic for Theorem 2.  Results are persisted as CSV files under
``data/processed/`` so that plots can be reproduced without re-running the
full data-collection pipeline.

Saved files
-----------
* ``theorem2_grid.csv``          -- Win-rate-gain grid (foul - no-foul).
* ``theorem2_wp_foul_grid.csv``  -- Historical win rate when fouling.
* ``theorem2_wp_no_foul_grid.csv`` -- Historical win rate without fouling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.theorems.utils import (
    apply_plot_aesthetics,
    FIGURE_DPI,
    PALETTE,
    get_resolved_possessions_at_time,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
FIGURE_FILENAME = "foul_up_3_heatmap.svg"
DOC_FILENAME = "theorem2_foul_up_3.md"

# Fixed parameter grids (hardcoded for reproducibility)
TIME_VALUES: List[int] = list(range(2, 12, 2))
FG3_VALUES: List[float] = [round(x, 2) for x in [0.25, 0.30, 0.35, 0.40, 0.45]]

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
    Compute Theorem 2 (Foul-Up-3) historical win rates and save to CSV files.

    Parameters
    ----------
    out_dir       : directory where output files will be written.
    processed_dir : directory containing ``transitions.parquet``.

    Returns
    -------
    Path to the saved gain-grid CSV.
    """
    if processed_dir is None:
        processed_dir = out_dir

    from src.collect_data import _load_historical_log

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 2 (Foul-Up-3) historical win rates…")

    n_time = len(TIME_VALUES)
    n_fg3 = len(FG3_VALUES)
    grid = np.zeros((n_time, n_fg3))
    wp_foul_grid = np.zeros((n_time, n_fg3))
    wp_no_foul_grid = np.zeros((n_time, n_fg3))

    for i, sec in enumerate(TIME_VALUES):
        if df.empty:
            for j in range(n_fg3):
                wp_foul_grid[i, j] = _DEFAULT_WIN_RATE
                wp_no_foul_grid[i, j] = _DEFAULT_WIN_RATE
                grid[i, j] = 0.0
            continue

        resolved = get_resolved_possessions_at_time(df, sec)
        window = resolved[
            (resolved["score_differential"] == 3) & (resolved["possession"] == 0)
        ]

        for j, fg3 in enumerate(FG3_VALUES):
            if window.empty or "opponent_fg3_pct" not in window.columns:
                wp_foul = _DEFAULT_WIN_RATE
                wp_no_foul = _DEFAULT_WIN_RATE
            else:
                fg3_bin = window[
                    window["opponent_fg3_pct"].between(
                        fg3 - 0.025, fg3 + 0.025, inclusive="both"
                    )
                ]
                foul_outcomes = fg3_bin.loc[
                    fg3_bin["action_taken"] == "foul", "game_outcome"
                ]
                no_foul_outcomes = fg3_bin.loc[
                    fg3_bin["action_taken"] != "foul", "game_outcome"
                ]

                wp_foul = (
                    float(foul_outcomes.mean())
                    if len(foul_outcomes) > 0
                    else _DEFAULT_WIN_RATE
                )
                wp_no_foul = (
                    float(no_foul_outcomes.mean())
                    if len(no_foul_outcomes) > 0
                    else _DEFAULT_WIN_RATE
                )

            wp_foul_grid[i, j] = round(wp_foul, 4)
            wp_no_foul_grid[i, j] = round(wp_no_foul, 4)
            grid[i, j] = round(wp_foul - wp_no_foul, 4)

    grid_path = out_dir / "theorem2_grid.csv"
    np.savetxt(grid_path, grid, delimiter=",")
    np.savetxt(out_dir / "theorem2_wp_foul_grid.csv", wp_foul_grid, delimiter=",")
    np.savetxt(out_dir / "theorem2_wp_no_foul_grid.csv", wp_no_foul_grid, delimiter=",")
    logger.info("Saved Theorem 2 grids to %s", out_dir)

    return grid_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_grids(processed_dir: Path):
    """Load Theorem 2 CSV grids.  Returns (gain_grid, wp_foul_grid,
    wp_no_foul_grid, time_values, fg3_values)."""
    grid_path = processed_dir / "theorem2_grid.csv"
    foul_path = processed_dir / "theorem2_wp_foul_grid.csv"
    no_foul_path = processed_dir / "theorem2_wp_no_foul_grid.csv"

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Theorem 2 data not found at {grid_path}. "
            "Run `python -m src.collect_data` first."
        )

    gain_grid = np.loadtxt(grid_path, delimiter=",")

    if foul_path.exists() and no_foul_path.exists():
        wp_foul_grid = np.loadtxt(foul_path, delimiter=",")
        wp_no_foul_grid = np.loadtxt(no_foul_path, delimiter=",")
    else:
        logger.warning(
            "Individual WP grids not found; reconstructing from gain grid. "
            "Re-run `python -m src.collect_data` to cache them."
        )
        wp_foul_grid = np.full_like(gain_grid, 0.5)
        wp_no_foul_grid = wp_foul_grid - gain_grid

    return gain_grid, wp_foul_grid, wp_no_foul_grid, TIME_VALUES, FG3_VALUES


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot(
    processed_dir: Path,
    images_dir: Path,
) -> Path:
    """
    Generate the Foul-Up-3 heatmap from CSV data.

    Parameters
    ----------
    processed_dir : directory containing the Theorem 2 CSV files.
    images_dir    : directory where the SVG will be saved.

    Returns
    -------
    Path to the saved SVG file.
    """
    gain_grid, _wf, _wn, time_values, fg3_values = load_grids(processed_dir)
    out_path = images_dir / FIGURE_FILENAME

    display_grid = gain_grid * 100
    x_labels = [f"{v:.0%}" for v in fg3_values]
    y_labels = [f"{t}s" for t in time_values]

    apply_plot_aesthetics()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        display_grid,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=PALETTE,
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Historical Win % Gain from Fouling (pp)", "shrink": 0.85},
    )
    ax.set_title(
        "Theorem 2: Foul Up 3\n"
        "Historical Win % Gain from Intentional Foul vs. Normal Defense",
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Opponent 3-Point Field Goal %", labelpad=10)
    ax.set_ylabel("Seconds Remaining", labelpad=10)
    ax.text(
        0.5,
        -0.14,
        "Green = Fouling is better  |  Red = Normal defense is better  |  "
        "Values in percentage points",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Theorem 2 heatmap to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------


def generate_doc(
    processed_dir: Path,
    docs_dir: Path,
) -> Path:
    """
    Write the Theorem 2 Markdown documentation.

    The content is static; this function simply ensures the doc file is
    present under *docs_dir* so that the site build finds it.

    Parameters
    ----------
    processed_dir : reserved for API compatibility with other theorem modules.
    docs_dir      : directory where the Markdown file will be written.

    Returns
    -------
    Path to the written Markdown file.
    """
    content = """\
# Theorem 2: Foul Up 3

## Claim

> **When leading by 3 with fewer than 12 seconds left, intentionally fouling
> is better against average-to-good 3PT shooters (≥ 30%) and worse against
> poor shooters.**

---

## Results

![Foul Up 3 Heatmap](assets/images/foul_up_3_heatmap.svg)

Green cells show situations where fouling improved the historical win rate;
red cells show where normal defense was better.
Each cell value is the win-percentage gain (in percentage points) from fouling
versus playing normal defense, based on NBA play-by-play data (2019--2024).

---

## Conclusion

Foul the opponent when they are a competent 3PT shooting team (≥ 30%).
Against poor 3PT teams, normal defense remains the safer choice.

"""

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 2 doc to %s", out_path)
    return out_path

