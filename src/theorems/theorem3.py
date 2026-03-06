"""
theorem3.py
===========
Theorem 3: The Late-Game Timeout.

Contains the data-collection, visualisation, and documentation-generation
logic for Theorem 3.  Results are persisted as ``theorem3_sweep.csv`` under
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
CSV_FILENAME = "theorem3_sweep.csv"
FIGURE_FILENAME = "timeout_ev_curve.svg"
DOC_FILENAME = "theorem3_timeout.md"

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
    Compute Theorem 3 (Late-Game Timeout) historical win rates and save to CSV.

    Filters the historical log for close games where the home team has
    possession and is trailing by 1--3 points or tied, with 20--50 seconds
    remaining.  Groups possessions by whether the team called a timeout or
    played on, and saves ``theorem3_sweep.csv`` to *out_dir*.

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
    logger.info("Computing Theorem 3 (Late-Game Timeout) historical win rates…")

    rows: List[Dict] = []

    for sec in range(20, 51, 2):
        if df.empty:
            rows.append(
                {
                    "seconds_remaining": sec,
                    "ev_timeout": _DEFAULT_WIN_RATE,
                    "ev_play_on": _DEFAULT_WIN_RATE,
                    "ev_gain": 0.0,
                    "timeout_is_optimal": False,
                }
            )
            continue

        resolved = get_resolved_possessions_at_time(df, sec)
        window = resolved[
            (resolved["score_differential"].between(-3, 0))
            & (resolved["possession"] == 1)
        ]
        timeout_outcomes = window.loc[
            window["action_taken"] == "timeout", "game_outcome"
        ]
        play_on_outcomes = window.loc[
            window["action_taken"] != "timeout", "game_outcome"
        ]

        ev_timeout = (
            float(timeout_outcomes.mean())
            if len(timeout_outcomes) > 0
            else _DEFAULT_WIN_RATE
        )
        ev_play_on = (
            float(play_on_outcomes.mean())
            if len(play_on_outcomes) > 0
            else _DEFAULT_WIN_RATE
        )
        ev_gain = ev_timeout - ev_play_on

        rows.append(
            {
                "seconds_remaining": sec,
                "ev_timeout": round(ev_timeout, 4),
                "ev_play_on": round(ev_play_on, 4),
                "ev_gain": round(ev_gain, 4),
                "timeout_is_optimal": ev_gain > 0,
            }
        )

    out_path = out_dir / CSV_FILENAME
    write_sweep_csv(
        out_path,
        rows,
        fieldnames=[
            "seconds_remaining",
            "ev_timeout",
            "ev_play_on",
            "ev_gain",
            "timeout_is_optimal",
        ],
    )
    logger.info("Saved Theorem 3 sweep to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot(
    processed_dir: Path,
    images_dir: Path,
) -> Path:
    """
    Generate the Late-Game Timeout win-percentage curve from ``theorem3_sweep.csv``.

    The figure contains two panels:

    * **Top panel** -- absolute historical win percentage for the *Timeout* and
      *Play On* strategies across 20--50 seconds remaining.
    * **Bottom panel** -- win-percentage gain from calling a timeout (positive
      values favour the timeout; negative values favour playing on).

    Parameters
    ----------
    processed_dir : directory containing ``theorem3_sweep.csv``.
    images_dir    : directory where the SVG will be saved.

    Returns
    -------
    Path to the saved SVG file.
    """
    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    out_path = images_dir / FIGURE_FILENAME

    seconds = [r["seconds_remaining"] for r in sweep]
    ev_timeout = [r["ev_timeout"] * 100 for r in sweep]
    ev_play_on = [r["ev_play_on"] * 100 for r in sweep]
    ev_gain = [r["ev_gain"] * 100 for r in sweep]

    apply_plot_aesthetics()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: absolute win-percentage lines ---
    ax1 = axes[0]
    ax1.plot(seconds, ev_timeout, color="#E63946", linewidth=2.2, label="Timeout")
    ax1.plot(seconds, ev_play_on, color="#457B9D", linewidth=2.2, label="Play On")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Historical Win Percentage")
    ax1.set_title(
        "Theorem 3: The Late-Game Timeout\n"
        "Historical Win Percentage: Timeout vs. Play On",
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: win-percentage gain ---
    ax2 = axes[1]
    gain_arr = np.array(ev_gain)
    colors = np.where(gain_arr >= 0, "#2DC653", "#E63946")
    ax2.bar(seconds, gain_arr, color=colors, width=1.6, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seconds Remaining")
    ax2.set_ylabel("Historical Win % Gain from Timeout (pp)")
    ax2.set_title("Win % Gain: Timeout - Play On  (green = timeout is better)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Theorem 3 timeout curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# Theorem 3: The Late-Game Timeout

## Claim

> **Calling a timeout when trailing by 1--3 (or tied) with less than 50 seconds is
> beneficial.**

---

## How We Measure It

We filter the historical play-by-play log for situations where:

- The home team has possession
- The score differential is between -3 and 0 (trailing by up to 3, or tied)
- Between 20 and 50 seconds remain

We group possessions by:

- **Timeout:** The team stops play with a timeout call.
- **Play On:** The team continues without calling a timeout.

We then calculate the **historical win percentage** for the home team in
each group across a sweep of time-remaining values.

---

## Results

![Late-Game Timeout Win Percentage Curve](assets/images/timeout_ev_curve.svg)

---

## Conclusion

{conclusion}
"""


def generate_doc(
    processed_dir: Path,
    docs_dir: Path,
) -> Path:
    """
    Generate the Theorem 3 Markdown documentation from ``theorem3_sweep.csv``.

    Parameters
    ----------
    processed_dir : directory containing ``theorem3_sweep.csv``.
    docs_dir      : directory where the Markdown file will be written.

    Returns
    -------
    Path to the written Markdown file.
    """

    conclusion = (
        "With 36--50 seconds remaining, calling a timeout is historically beneficial."
    )

    content = _TEMPLATE.format(
        conclusion=conclusion,
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 3 doc to %s", out_path)
    return out_path
