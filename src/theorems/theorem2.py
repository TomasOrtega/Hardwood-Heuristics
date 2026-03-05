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
* ``theorem2_metadata.json``     -- Parameter labels (time_values, fg3_pct_values).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
FIGURE_FILENAME = "foul_up_3_heatmap.svg"
DOC_FILENAME = "theorem2_foul_up_3.md"

# Default win rate used when a bucket has no historical observations
_DEFAULT_WIN_RATE = 0.5
_TIME_WINDOW_S = 1

# Consistent aesthetics
PALETTE = "RdYlGn"
FIGURE_DPI = 150
FONT_FAMILY = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Compute Theorem 2 (Foul-Up-3) historical win rates and save to CSV files.

    Parameters
    ----------
    out_dir       : directory where output files will be written.
    processed_dir : directory containing ``transitions.parquet``.

    Returns
    -------
    Tuple of (grid_csv_path, metadata_json_path).
    """
    if processed_dir is None:
        processed_dir = out_dir

    from src.collect_data import _load_historical_log

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 2 (Foul-Up-3) historical win rates…")

    time_values = list(range(2, 12, 2))
    fg3_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]

    n_time = len(time_values)
    n_fg3 = len(fg3_values)
    grid = np.zeros((n_time, n_fg3))
    wp_foul_grid = np.zeros((n_time, n_fg3))
    wp_no_foul_grid = np.zeros((n_time, n_fg3))

    if not df.empty:
        mask = (
            (
                ((df["score_differential"] == 3) & (df["possession"] == 0))
                | ((df["score_differential"] == -3) & (df["possession"] == 1))
            )
            & (df["seconds_remaining"] < 12)
        )
        filtered = df[mask]
    else:
        filtered = df

    for i, sec in enumerate(time_values):
        if filtered.empty:
            window = filtered
        else:
            window = filtered[
                filtered["seconds_remaining"].between(
                    sec - _TIME_WINDOW_S, sec + _TIME_WINDOW_S
                )
            ]

        for j, fg3 in enumerate(fg3_values):
            if window.empty or "opponent_fg3_pct" not in window.columns:
                wp_foul = _DEFAULT_WIN_RATE
                wp_no_foul = _DEFAULT_WIN_RATE
            else:
                fg3_bin = window[
                    window["opponent_fg3_pct"].between(fg3 - 0.01, fg3 + 0.01, inclusive="both")
                ]
                foul_outcomes = fg3_bin.loc[fg3_bin["action_taken"] == "foul", "game_outcome"]
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

    meta_path = out_dir / "theorem2_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(
            {"time_values": time_values, "fg3_pct_values": fg3_values}, fh, indent=2
        )
    logger.info("Saved Theorem 2 metadata to %s", meta_path)

    return grid_path, meta_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_grids(processed_dir: Path):
    """Load Theorem 2 CSV grids and metadata.  Returns (gain_grid, wp_foul_grid,
    wp_no_foul_grid, time_values, fg3_values)."""
    grid_path = processed_dir / "theorem2_grid.csv"
    foul_path = processed_dir / "theorem2_wp_foul_grid.csv"
    no_foul_path = processed_dir / "theorem2_wp_no_foul_grid.csv"
    meta_path = processed_dir / "theorem2_metadata.json"

    for p in (grid_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Theorem 2 data not found at {p}. "
                "Run `python -m src.collect_data` first."
            )

    with open(meta_path) as fh:
        meta = json.load(fh)

    time_values: List[int] = meta["time_values"]
    fg3_values: List[float] = meta["fg3_pct_values"]
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

    return gain_grid, wp_foul_grid, wp_no_foul_grid, time_values, fg3_values


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

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

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
    Generate the Theorem 2 Markdown documentation from CSV data.

    Parameters
    ----------
    processed_dir : directory containing the Theorem 2 CSV files.
    docs_dir      : directory where the Markdown file will be written.

    Returns
    -------
    Path to the written Markdown file.
    """
    from src.generate_docs import (
        _fmt_ev,
        _fmt_gain,
        _build_theorem2_key_findings,
        _build_theorem2_conclusion,
    )

    gain_grid, wp_foul_grid, wp_no_foul_grid, time_values, fg3_values = load_grids(
        processed_dir
    )

    def _cell(sec: int, fg3: float):
        i = time_values.index(sec)
        j = min(range(len(fg3_values)), key=lambda k: abs(fg3_values[k] - fg3))
        return (
            float(wp_foul_grid[i, j]),
            float(wp_no_foul_grid[i, j]),
            float(gain_grid[i, j]),
        )

    wf_8_28, wn_8_28, wg_8_28 = _cell(8, 0.28)
    wf_8_36, wn_8_36, wg_8_36 = _cell(8, 0.36)
    wf_8_44, wn_8_44, wg_8_44 = _cell(8, 0.44)
    wf_4_36, wn_4_36, wg_4_36 = _cell(4, 0.36)

    threshold_low = max(0.0, 2.0 / 3.0 - 0.01)
    key_findings = _build_theorem2_key_findings(gain_grid, time_values, fg3_values)
    conclusion = _build_theorem2_conclusion(
        gain_grid, time_values, fg3_values, threshold_low
    )

    _TEMPLATE = """\
# Theorem 2: Foul Up 3

## Claim

> **Based on NBA play-by-play data from 2019--2024, intentionally fouling
> when leading by 3 with fewer than 12 seconds remaining shows mixed
> results — outcomes depend on time remaining and are not consistently
> better in this historical sample.**

---

## How We Measure It

We filter the historical play-by-play log for:

- Home team defending (away team has the ball)
- Home team leads by exactly 3 points
- Fewer than 12 seconds remain

We group possessions by **Foul** (intentional) vs. **No Foul** (normal defence),
and compute the home-team historical win percentage for each.

---

## Results

![Foul Up 3 Heatmap](assets/images/foul_up_3_heatmap.svg)

The heatmap shows the historical win % gain from fouling (green = fouling better,
red = normal defence better) across time remaining and opponent 3PT%.

### Key Findings

{key_findings}

### Historical Data Summary

Data from 5 NBA seasons (2019--2024):

| Seconds | Opp 3PT% | Foul Win % | No-Foul Win % | Win % Gain |
|---------|----------|-----------|---------------|------------|
| 8 s | 28 % | {wp_foul_8_28} | {wp_no_foul_8_28} | {wp_gain_8_28} |
| 8 s | 36 % | {wp_foul_8_36} | {wp_no_foul_8_36} | {wp_gain_8_36} |
| 8 s | 44 % | {wp_foul_8_44} | {wp_no_foul_8_44} | {wp_gain_8_44} |
| 4 s | 36 % | {wp_foul_4_36} | {wp_no_foul_4_36} | {wp_gain_4_36} |

> *Values are historical win percentages from NBA play-by-play data, 2019--2024.*

---

## Sensitivity Analysis

Results vary by both **time remaining** and **opponent 3PT%** — possessions
are now segmented into narrow 3PT% bins (±1 pp) so each cell reflects games
where the opponent shot within that range.

Analyzed range ({fg3_min:.0%}--{fg3_max:.0%} opponent 3PT%):
win % gain from fouling ranges from {min_gain_pp:.1f} pp to +{max_gain_pp:.1f} pp.

---

## Conclusion

{conclusion}
"""

    content = _TEMPLATE.format(
        fg3_min=min(fg3_values),
        fg3_max=max(fg3_values),
        min_gain_pp=float(gain_grid.min() * 100),
        max_gain_pp=float(gain_grid.max() * 100),
        key_findings=key_findings,
        conclusion=conclusion,
        wp_foul_8_28=_fmt_ev(wf_8_28),
        wp_no_foul_8_28=_fmt_ev(wn_8_28),
        wp_gain_8_28=_fmt_gain(wg_8_28, pp=True),
        wp_foul_8_36=_fmt_ev(wf_8_36),
        wp_no_foul_8_36=_fmt_ev(wn_8_36),
        wp_gain_8_36=_fmt_gain(wg_8_36, pp=True),
        wp_foul_8_44=_fmt_ev(wf_8_44),
        wp_no_foul_8_44=_fmt_ev(wn_8_44),
        wp_gain_8_44=_fmt_gain(wg_8_44, pp=True),
        wp_foul_4_36=_fmt_ev(wf_4_36),
        wp_no_foul_4_36=_fmt_ev(wn_4_36),
        wp_gain_4_36=_fmt_gain(wg_4_36, pp=True),
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 2 doc to %s", out_path)
    return out_path
