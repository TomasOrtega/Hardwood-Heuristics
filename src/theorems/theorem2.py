"""
theorem2.py
===========
Theorem 2: Foul Up 3.

Contains the data-collection, visualisation, and documentation-generation
logic for Theorem 2.  Results are persisted as CSV files under
``data/processed/`` so that plots can be reproduced without re-running the
full data-collection pipeline.

Saved file
----------
* ``theorem2_sweep.csv`` -- Historical win-rate comparison by time remaining.
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
    get_resolved_possessions_at_times,
    load_sweep_csv,
    summarize_binary_comparison,
    write_sweep_csv,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
CSV_FILENAME = "theorem2_sweep.csv"
FIGURE_FILENAME = "foul_up_3_curve.svg"
DOC_FILENAME = "theorem2_foul_up_3.md"

TIME_VALUES: List[int] = list(range(2, 12, 2))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 2 (Foul-Up-3) historical win rates and save to CSV.

    Includes both home and away teams leading by exactly three while the
    trailing opponent has possession. A defensive foul is compared with a
    normal-defense possession whose first action is a shot or turnover.

    Parameters
    ----------
    out_dir       : directory where output files will be written.
    processed_dir : directory containing ``transitions.parquet``.

    Returns
    -------
    Path to the saved time-sweep CSV.
    """
    if processed_dir is None:
        processed_dir = out_dir

    from src.collect_data import _load_historical_log

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 2 (Foul-Up-3) historical win rates…")

    resolved = get_resolved_possessions_at_times(df, TIME_VALUES)
    window = resolved[
        ((resolved["score_differential"] == 3) & (resolved["possession"] == 0))
        | (
            (resolved["score_differential"] == -3)
            & (resolved["possession"] == 1)
        )
    ].copy()
    window["leader"] = np.where(window["score_differential"] > 0, 1, 0)
    window["leader_won"] = np.where(
        window["leader"] == 1,
        window["game_outcome"],
        1 - window["game_outcome"],
    )
    foul = (window["action_taken"] == "foul") & (
        window["action_team"] == window["leader"]
    )
    defend = window["action_taken"].isin(["shoot", "turnover"]) & (
        window["action_team"] == window["possession"]
    )
    window["strategy"] = np.select(
        [foul, defend],
        ["foul", "defend"],
        default=None,
    )

    comparison = summarize_binary_comparison(
        window,
        target_values=TIME_VALUES,
        group_col="strategy",
        outcome_col="leader_won",
        groups=("foul", "defend"),
        seed=2,
    )
    rows: List[Dict] = []
    for result in comparison:
        gain = result["difference"]
        rows.append(
            {
                "seconds_remaining": result["target_seconds"],
                "ev_foul": round(result["foul"], 4),
                "ev_foul_ci_low": round(result["foul_ci_low"], 4),
                "ev_foul_ci_high": round(result["foul_ci_high"], 4),
                "ev_defend": round(result["defend"], 4),
                "ev_defend_ci_low": round(result["defend_ci_low"], 4),
                "ev_defend_ci_high": round(result["defend_ci_high"], 4),
                "ev_gain": round(gain, 4),
                "ev_gain_ci_low": round(result["difference_ci_low"], 4),
                "ev_gain_ci_high": round(result["difference_ci_high"], 4),
                "n_foul": result["n_foul"],
                "n_defend": result["n_defend"],
                "foul_is_better": bool(np.isfinite(gain) and gain > 0),
            }
        )

    out_path = out_dir / CSV_FILENAME
    write_sweep_csv(
        out_path,
        rows,
        fieldnames=[
            "seconds_remaining",
            "ev_foul",
            "ev_foul_ci_low",
            "ev_foul_ci_high",
            "ev_defend",
            "ev_defend_ci_low",
            "ev_defend_ci_high",
            "ev_gain",
            "ev_gain_ci_low",
            "ev_gain_ci_high",
            "n_foul",
            "n_defend",
            "foul_is_better",
        ],
    )
    logger.info("Saved Theorem 2 sweep to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot(
    processed_dir: Path,
    images_dir: Path,
) -> Path:
    """
    Generate the Foul-Up-3 win-rate curve from CSV data.

    Parameters
    ----------
    processed_dir : directory containing ``theorem2_sweep.csv``.
    images_dir    : directory where the SVG will be saved.

    Returns
    -------
    Path to the saved SVG file.
    """
    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    out_path = images_dir / FIGURE_FILENAME

    seconds = [row["seconds_remaining"] for row in sweep]
    ev_foul = [row["ev_foul"] * 100 for row in sweep]
    ev_foul_low = [row["ev_foul_ci_low"] * 100 for row in sweep]
    ev_foul_high = [row["ev_foul_ci_high"] * 100 for row in sweep]
    ev_defend = [row["ev_defend"] * 100 for row in sweep]
    ev_defend_low = [row["ev_defend_ci_low"] * 100 for row in sweep]
    ev_defend_high = [row["ev_defend_ci_high"] * 100 for row in sweep]
    ev_gain = [row["ev_gain"] * 100 for row in sweep]
    ev_gain_low = np.array([row["ev_gain_ci_low"] * 100 for row in sweep])
    ev_gain_high = np.array([row["ev_gain_ci_high"] * 100 for row in sweep])

    apply_plot_aesthetics()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1 = axes[0]
    ax1.plot(seconds, ev_foul, color="#E63946", linewidth=2.2, label="Foul")
    ax1.fill_between(
        seconds,
        ev_foul_low,
        ev_foul_high,
        color="#E63946",
        alpha=0.15,
        label="95% game-clustered interval",
    )
    ax1.plot(
        seconds,
        ev_defend,
        color="#457B9D",
        linewidth=2.2,
        label="Defend",
    )
    ax1.fill_between(
        seconds,
        ev_defend_low,
        ev_defend_high,
        color="#457B9D",
        alpha=0.15,
    )
    ax1.set_ylabel("Historical Win Percentage")
    ax1.set_title(
        "Theorem 2: Foul Up 3\n"
        "Historical Win Percentage: Intentional Foul vs. Normal Defense",
        fontweight="bold",
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    gain_arr = np.array(ev_gain)
    colors = np.where(gain_arr >= 0, "#2DC653", "#E63946")
    ax2.bar(seconds, gain_arr, color=colors, width=1.4, alpha=0.85)
    finite = np.isfinite(gain_arr) & np.isfinite(ev_gain_low) & np.isfinite(
        ev_gain_high
    )
    ax2.errorbar(
        np.asarray(seconds)[finite],
        gain_arr[finite],
        yerr=np.vstack(
            [
                np.maximum(0, gain_arr[finite] - ev_gain_low[finite]),
                np.maximum(0, ev_gain_high[finite] - gain_arr[finite]),
            ]
        ),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seconds Remaining")
    ax2.set_ylabel("Win % Gain from Fouling (pp)")
    ax2.set_title("Foul - Defend (green = higher observed foul win rate)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Theorem 2 curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------


def generate_doc(
    processed_dir: Path,
    docs_dir: Path,
) -> Path:
    """
    Write Theorem 2 documentation derived from its sweep data.

    Parameters
    ----------
    processed_dir : directory containing ``theorem2_sweep.csv``.
    docs_dir      : directory where the Markdown file will be written.

    Returns
    -------
    Path to the written Markdown file.
    """
    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    comparable = [
        row
        for row in sweep
        if row["n_foul"] > 0
        and row["n_defend"] > 0
        and np.isfinite(row["ev_gain"])
    ]
    positive = sum(row["ev_gain"] > 0 for row in comparable)
    clearly_positive = sum(row["ev_gain_ci_low"] > 0 for row in comparable)
    conclusion = (
        f"Fouling had the higher observed win rate at {positive} of "
        f"{len(comparable)} comparable clock points. The 95% interval excluded "
        f"zero in favor of fouling at {clearly_positive} points. Sample sizes "
        "are small, so the data do not establish that fouling causes better "
        "outcomes."
        if comparable
        else "There are not enough observations to compare the strategies."
    )
    content = f"""\
# Theorem 2: Foul Up 3

## Claim

> **When leading by three with fewer than 12 seconds left, is intentionally
> fouling better than defending normally?**

---

## How We Measure It

For both home and away teams, we select situations where the leading team is
up exactly three and the trailing team has the ball. A sample is retained when
the first logged action after the target clock is:

- **Foul:** The leading team commits the foul.
- **Defend:** The trailing offense shoots or turns the ball over before a foul.

Empty groups remain missing rather than being assigned a 50% win rate. The
saved data include observation counts and pointwise 95% uncertainty intervals.
Each interval uses the wider limits from a game-cluster bootstrap and
Wilson/Newcombe finite-sample bounds. Resampling games as blocks keeps repeated
clock values and overtime periods from the same game together. This is a
descriptive comparison and does not adjust for why coaches chose to foul.

---

## Results

![Foul Up 3 Curve](assets/images/foul_up_3_curve.svg)

---

## Conclusion

{conclusion}

"""

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 2 doc to %s", out_path)
    return out_path
