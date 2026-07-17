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
    get_resolved_possessions_at_times,
    load_sweep_csv,
    summarize_binary_comparison,
    write_sweep_csv,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Output file names
CSV_FILENAME = "theorem3_sweep.csv"
FIGURE_FILENAME = "timeout_ev_curve.svg"
DOC_FILENAME = "theorem3_timeout.md"
TIME_VALUES = list(range(20, 51, 2))

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 3 (Late-Game Timeout) historical win rates and save to CSV.

    Filters the historical log for home or away teams with possession while
    trailing by 1--3 points or tied. The first action is classified as a
    timeout only when called by the possessing team, or as playing on when
    that team shoots or turns the ball over.

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

    resolved = get_resolved_possessions_at_times(df, TIME_VALUES)
    resolved["possessing_margin"] = np.where(
        resolved["possession"] == 1,
        resolved["score_differential"],
        -resolved["score_differential"],
    )
    window = resolved[resolved["possessing_margin"].between(-3, 0)].copy()
    window["team_won"] = np.where(
        window["possession"] == 1,
        window["game_outcome"],
        1 - window["game_outcome"],
    )
    timeout = (window["action_taken"] == "timeout") & (
        window["action_team"] == window["possession"]
    )
    play_on = window["action_taken"].isin(["shoot", "turnover"]) & (
        window["action_team"] == window["possession"]
    )
    window["strategy"] = np.select(
        [timeout, play_on],
        ["timeout", "play_on"],
        default=None,
    )

    comparison = summarize_binary_comparison(
        window,
        target_values=TIME_VALUES,
        group_col="strategy",
        outcome_col="team_won",
        groups=("timeout", "play_on"),
        seed=3,
    )
    rows: List[Dict] = []
    for result in comparison:
        gain = result["difference"]
        rows.append(
            {
                "seconds_remaining": result["target_seconds"],
                "ev_timeout": round(result["timeout"], 4),
                "ev_timeout_ci_low": round(result["timeout_ci_low"], 4),
                "ev_timeout_ci_high": round(result["timeout_ci_high"], 4),
                "ev_play_on": round(result["play_on"], 4),
                "ev_play_on_ci_low": round(result["play_on_ci_low"], 4),
                "ev_play_on_ci_high": round(result["play_on_ci_high"], 4),
                "ev_gain": round(gain, 4),
                "ev_gain_ci_low": round(result["difference_ci_low"], 4),
                "ev_gain_ci_high": round(result["difference_ci_high"], 4),
                "n_timeout": result["n_timeout"],
                "n_play_on": result["n_play_on"],
                "timeout_is_optimal": bool(np.isfinite(gain) and gain > 0),
            }
        )

    out_path = out_dir / CSV_FILENAME
    write_sweep_csv(
        out_path,
        rows,
        fieldnames=[
            "seconds_remaining",
            "ev_timeout",
            "ev_timeout_ci_low",
            "ev_timeout_ci_high",
            "ev_play_on",
            "ev_play_on_ci_low",
            "ev_play_on_ci_high",
            "ev_gain",
            "ev_gain_ci_low",
            "ev_gain_ci_high",
            "n_timeout",
            "n_play_on",
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
    ev_timeout_low = [r["ev_timeout_ci_low"] * 100 for r in sweep]
    ev_timeout_high = [r["ev_timeout_ci_high"] * 100 for r in sweep]
    ev_play_on = [r["ev_play_on"] * 100 for r in sweep]
    ev_play_on_low = [r["ev_play_on_ci_low"] * 100 for r in sweep]
    ev_play_on_high = [r["ev_play_on_ci_high"] * 100 for r in sweep]
    ev_gain = [r["ev_gain"] * 100 for r in sweep]
    ev_gain_low = np.array([r["ev_gain_ci_low"] * 100 for r in sweep])
    ev_gain_high = np.array([r["ev_gain_ci_high"] * 100 for r in sweep])

    apply_plot_aesthetics()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: absolute win-percentage lines ---
    ax1 = axes[0]
    ax1.plot(seconds, ev_timeout, color="#E63946", linewidth=2.2, label="Timeout")
    ax1.fill_between(
        seconds,
        ev_timeout_low,
        ev_timeout_high,
        color="#E63946",
        alpha=0.15,
        label="95% game-clustered interval",
    )
    ax1.plot(seconds, ev_play_on, color="#457B9D", linewidth=2.2, label="Play On")
    ax1.fill_between(
        seconds,
        ev_play_on_low,
        ev_play_on_high,
        color="#457B9D",
        alpha=0.15,
    )
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

> **When trailing by up to three points (or tied), is calling a timeout better
> than playing on with 20--50 seconds remaining?**

---

## How We Measure It

We filter the historical play-by-play log for situations where:

- Either the home or away team has possession
- That team is trailing by up to 3 points, or tied
- Between 20 and 50 seconds remain

We group possessions by:

- **Timeout:** The team stops play with a timeout call.
- **Play On:** The team's first action is a shot or turnover.

Other first actions are excluded. We calculate the possessing team's historical
win percentage and save observation counts and pointwise 95% uncertainty
intervals for each group. Each interval uses the wider limits from a
game-cluster bootstrap and Wilson/Newcombe finite-sample bounds. Games are
resampled as blocks so repeated clock values and overtime periods from one game
remain together. This is descriptive: timeout availability, team quality, and
why a coach stopped play are not controlled for.

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

    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)
    comparable = [
        row
        for row in sweep
        if row["n_timeout"] > 0
        and row["n_play_on"] > 0
        and np.isfinite(row["ev_gain"])
    ]
    positive = sum(row["ev_gain"] > 0 for row in comparable)
    clearly_positive = sum(row["ev_gain_ci_low"] > 0 for row in comparable)
    conclusion = (
        f"Calling timeout had the higher observed win rate at {positive} of "
        f"{len(comparable)} comparable clock points. The 95% interval excluded "
        f"zero in favor of timeout at {clearly_positive} points. This "
        "association does not show that the timeout itself caused the "
        "difference."
        if comparable
        else "There are not enough observations to compare the strategies."
    )

    content = _TEMPLATE.format(
        conclusion=conclusion,
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 3 doc to %s", out_path)
    return out_path
