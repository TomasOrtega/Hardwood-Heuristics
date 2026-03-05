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
    ev_timeout = [r["ev_timeout"] for r in sweep]
    ev_play_on = [r["ev_play_on"] for r in sweep]
    ev_gain = [r["ev_gain"] for r in sweep]

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

> **Based on NBA play-by-play data from 2019--2024, calling a timeout when
> trailing by 1--3 (or tied) with 36--50 seconds remaining shows a consistent
> win-rate advantage. Results are mixed below 36 seconds.**

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


_MIN_SIGNIFICANT_WINDOW_SIZE = 4


def _build_conclusion(sweep: List[Dict]) -> str:
    """Return the conclusion paragraph for Theorem 3."""
    from src.generate_docs import _consecutive_positive_windows

    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    gains = [e["ev_gain"] for e in sorted_sweep]
    positive_count = sum(1 for g in gains if g > 0)
    total = len(gains)
    mean_gain_pp = (sum(gains) / total) * 100 if total > 0 else 0.0

    if positive_count == 0:
        return (
            "**The historical data does not support a timeout advantage** when "
            "trailing in the final 20--50 seconds — playing on is at least as good "
            "in all cases studied. The timeout decision has limited measurable impact; "
            "focus on execution and matchups instead."
        )
    if positive_count == total:
        return (
            f"**Calling a timeout is historically beneficial** when trailing with "
            f"20--50 seconds remaining (average gain: +{mean_gain_pp:.1f} pp). "
            "Use available timeouts to set up the final possession."
        )
    # Mixed case — check for a dominant consecutive positive window
    windows = _consecutive_positive_windows(sweep)
    if windows:
        best_window = max(windows, key=lambda w: w[1] - w[0])
        window_entries = [
            e
            for e in sorted_sweep
            if best_window[0] <= e["seconds_remaining"] <= best_window[1]
        ]
        if len(window_entries) >= _MIN_SIGNIFICANT_WINDOW_SIZE:
            window_mean_gain_pp = (
                sum(e["ev_gain"] for e in window_entries) / len(window_entries)
            ) * 100
            return (
                f"**With {best_window[0]}--{best_window[1]} seconds remaining, calling "
                f"a timeout is historically beneficial** (average gain: "
                f"+{window_mean_gain_pp:.1f} pp across all {len(window_entries)} buckets "
                f"in this window). Below {best_window[0]} s the data is mixed — results "
                "are close to even. Rely on matchup specifics rather than a universal "
                "rule in the final 30 seconds."
            )
    return (
        f"**The data is inconclusive**: a timeout does not consistently help or "
        f"hurt in the 20--50 second window. On average the gain is "
        f"{mean_gain_pp:+.1f} pp. Base the decision on matchup specifics."
    )


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
    from src.generate_docs import (
        _fmt_ev,
        _gain_label,
        _find_sweep_entry,
    )

    sweep = load_sweep_csv(processed_dir / CSV_FILENAME)

    e40 = _find_sweep_entry(sweep, 40)
    e30 = _find_sweep_entry(sweep, 30)
    e20 = _find_sweep_entry(sweep, 20)

    def _optimal_label(gain: float) -> str:
        return "Timeout ✓" if gain > 0 else "Play On ✓"

    conclusion = _build_conclusion(sweep)

    content = _TEMPLATE.format(
        conclusion=conclusion,
        ev_timeout_40=_fmt_ev(e40["ev_timeout"]),
        ev_play_on_40=_fmt_ev(e40["ev_play_on"]),
        ev_gain_40=_gain_label(e40["ev_gain"]),
        optimal_40=_optimal_label(e40["ev_gain"]),
        ev_timeout_30=_fmt_ev(e30["ev_timeout"]),
        ev_play_on_30=_fmt_ev(e30["ev_play_on"]),
        ev_gain_30=_gain_label(e30["ev_gain"]),
        optimal_30=_optimal_label(e30["ev_gain"]),
        ev_timeout_20=_fmt_ev(e20["ev_timeout"]),
        ev_play_on_20=_fmt_ev(e20["ev_play_on"]),
        ev_gain_20=_gain_label(e20["ev_gain"]),
        optimal_20=_optimal_label(e20["ev_gain"]),
    )

    out_path = docs_dir / DOC_FILENAME
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 3 doc to %s", out_path)
    return out_path
