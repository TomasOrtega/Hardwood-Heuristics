"""
collect_data.py
===============
Aggregates historical NBA play-by-play data to compute actual win rates
for each strategic choice described by the Folk Theorems.

Run this script once to compute and save the historical win-rate results that
``src/visualizations.py`` needs.  Subsequent runs of the visualization script
will load the saved files instead of re-computing them.

Adding a new theorem
--------------------
1. Define the historical filter criteria (score differential, time remaining,
   possession).
2. Define the action to test (e.g. ``action_taken == 'timeout'``).
3. Add a ``_collect_<key>`` function below that filters the dataframe, groups
   by the action, calculates ``game_outcome.mean()``, and dumps summary stats.
4. Register it in :data:`_COLLECTORS`.
5. Add corresponding plotting and documentation generation functions.

Saved artefacts (written to ``data/processed/``)
-------------------------------------------------
* ``theorem1_sweep.json``          – Historical win-rate sweep for Theorem 1 (2-for-1).
* ``theorem2_grid.csv``            – Win-rate-gain grid for Theorem 2 (Foul-Up-3).
* ``theorem2_wp_foul_grid.csv``    – Historical win rate when fouling (per cell).
* ``theorem2_wp_no_foul_grid.csv`` – Historical win rate without fouling (per cell).
* ``theorem2_metadata.json``       – Parameter labels (time_values, fg3_pct_values).
* ``theorem3_sweep.json``          – Historical win-rate sweep for Theorem 3 (Late-Game Timeout).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Default win rate used when a bucket has no historical observations
_DEFAULT_WIN_RATE = 0.5

# Window size (in seconds) around each target time bucket when querying the
# historical log.  A ±1 s window is used to capture sufficient observations
# while keeping adjacent buckets roughly independent.
_TIME_WINDOW_S = 1


def _load_historical_log(processed_dir: Path) -> pd.DataFrame:
    """
    Load the historical possession log from ``transitions.parquet``.

    Falls back to a synthetic dataset if the file does not exist (useful for
    offline development and CI environments without scraped data).
    """
    transitions_path = processed_dir / "transitions.parquet"
    if transitions_path.exists():
        logger.info("Loading historical log from %s", transitions_path)
        return pd.read_parquet(transitions_path)

    logger.warning(
        "No historical data found at %s; using synthetic fallback for development.",
        transitions_path,
    )
    from src.data_pipeline import build_synthetic_transitions
    return build_synthetic_transitions()


# ---------------------------------------------------------------------------
# Per-theorem collection functions
# ---------------------------------------------------------------------------
def _collect_theorem1(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 1 (2-for-1) historical win rates and save to JSON.

    Filters the historical log for tied games (score_differential == 0)
    and groups possessions by whether the team shot ('shoot') or held
    ('hold' / other action).  Calculates the mean game_outcome (historical
    win percentage) for each 2-second time bucket from 10 to 64 seconds.
    """
    if processed_dir is None:
        processed_dir = out_dir

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 1 (2-for-1) historical win rates...")

    sweep = []
    # Tied games only
    tied = df[df["score_differential"] == 0] if not df.empty else df

    for sec in range(10, 65, 2):
        if tied.empty:
            sweep.append({
                "seconds_remaining": sec,
                "ev_rush": _DEFAULT_WIN_RATE,
                "ev_normal": _DEFAULT_WIN_RATE,
                "ev_gain": 0.0,
                "rush_is_optimal": False,
            })
            continue

        # Include a ±_TIME_WINDOW_S second window around the target second value
        window = tied[tied["seconds_remaining"].between(sec - _TIME_WINDOW_S, sec + _TIME_WINDOW_S)]
        rush_outcomes = window.loc[window["action_taken"] == "shoot", "game_outcome"]
        # All non-shoot actions (hold, timeout, rebound, etc.) are treated as
        # "normal possession" — i.e., the team did not rush a shot.
        hold_outcomes = window.loc[window["action_taken"] != "shoot", "game_outcome"]

        ev_rush = float(rush_outcomes.mean()) if len(rush_outcomes) > 0 else _DEFAULT_WIN_RATE
        ev_normal = float(hold_outcomes.mean()) if len(hold_outcomes) > 0 else _DEFAULT_WIN_RATE
        ev_gain = ev_rush - ev_normal

        sweep.append({
            "seconds_remaining": sec,
            "ev_rush": round(ev_rush, 4),
            "ev_normal": round(ev_normal, 4),
            "ev_gain": round(ev_gain, 4),
            "rush_is_optimal": ev_gain > 0,
        })

    out_path = out_dir / "theorem1_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info("Saved Theorem 1 sweep to %s", out_path)
    return out_path


def _collect_theorem2(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Compute Theorem 2 (Foul-Up-3) historical win rates and save grids.

    Filters the historical log for situations where the home team is
    defending (away possession), leads by exactly 3 points, and fewer than
    12 seconds remain.  Groups by foul vs. no-foul and calculates the mean
    game_outcome for each time bucket.

    The win rates are broadcast uniformly across the opponent 3PT% axis
    (fg3_pct_values) since the historical log does not track the opponent's
    3PT shooting percentage at the moment of decision.
    """
    if processed_dir is None:
        processed_dir = out_dir

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 2 (Foul-Up-3) historical win rates...")

    time_values = list(range(2, 12, 2))
    fg3_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]

    n_time = len(time_values)
    n_fg3 = len(fg3_values)
    grid = np.zeros((n_time, n_fg3))
    wp_foul_grid = np.zeros((n_time, n_fg3))
    wp_no_foul_grid = np.zeros((n_time, n_fg3))

    # Filter: home defending (away has ball), home up by 3, < 12 seconds
    if not df.empty:
        mask = (
            (df["score_differential"] == 3)
            & (df["possession"] == 0)
            & (df["seconds_remaining"] < 12)
        )
        filtered = df[mask]
    else:
        filtered = df

    for i, sec in enumerate(time_values):
        if filtered.empty:
            wp_foul = _DEFAULT_WIN_RATE
            wp_no_foul = _DEFAULT_WIN_RATE
        else:
            window = filtered[filtered["seconds_remaining"].between(sec - _TIME_WINDOW_S, sec + _TIME_WINDOW_S)]
            foul_outcomes = window.loc[window["action_taken"] == "foul", "game_outcome"]
            no_foul_outcomes = window.loc[window["action_taken"] != "foul", "game_outcome"]

            wp_foul = (
                float(foul_outcomes.mean()) if len(foul_outcomes) > 0 else _DEFAULT_WIN_RATE
            )
            wp_no_foul = (
                float(no_foul_outcomes.mean()) if len(no_foul_outcomes) > 0 else _DEFAULT_WIN_RATE
            )

        # Broadcast across the fg3_pct axis
        for j in range(n_fg3):
            wp_foul_grid[i, j] = round(wp_foul, 4)
            wp_no_foul_grid[i, j] = round(wp_no_foul, 4)
            grid[i, j] = round(wp_foul - wp_no_foul, 4)

    grid_path = out_dir / "theorem2_grid.csv"
    np.savetxt(grid_path, grid, delimiter=",")
    logger.info("Saved Theorem 2 gain grid to %s", grid_path)

    np.savetxt(out_dir / "theorem2_wp_foul_grid.csv", wp_foul_grid, delimiter=",")
    np.savetxt(out_dir / "theorem2_wp_no_foul_grid.csv", wp_no_foul_grid, delimiter=",")
    logger.info("Saved Theorem 2 individual WP grids to %s", out_dir)

    meta_path = out_dir / "theorem2_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"time_values": time_values, "fg3_pct_values": fg3_values}, f, indent=2)
    logger.info("Saved Theorem 2 metadata to %s", meta_path)

    return grid_path, meta_path


def _collect_theorem3(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 3 (Late-Game Timeout) historical win rates and save to JSON.

    Filters the historical log for close games where the home team has
    possession and is trailing by 1–3 points or tied, with 20–50 seconds
    remaining.  Groups possessions by whether the team called a timeout
    ('timeout') or played on (any other action).  Calculates the mean
    game_outcome (historical win percentage for the home team) for each
    2-second time bucket.
    """
    if processed_dir is None:
        processed_dir = out_dir

    df = _load_historical_log(processed_dir)
    logger.info("Computing Theorem 3 (Late-Game Timeout) historical win rates...")

    sweep = []

    # Close games: home team trailing (−3 to 0) or tied, home has possession
    if not df.empty:
        mask = (
            (df["score_differential"].between(-3, 0))
            & (df["possession"] == 1)
        )
        close = df[mask]
    else:
        close = df

    for sec in range(20, 51, 2):
        if close.empty:
            sweep.append({
                "seconds_remaining": sec,
                "ev_timeout": _DEFAULT_WIN_RATE,
                "ev_play_on": _DEFAULT_WIN_RATE,
                "ev_gain": 0.0,
                "timeout_is_optimal": False,
            })
            continue

        window = close[close["seconds_remaining"].between(sec - _TIME_WINDOW_S, sec + _TIME_WINDOW_S)]
        timeout_outcomes = window.loc[window["action_taken"] == "timeout", "game_outcome"]
        play_on_outcomes = window.loc[window["action_taken"] != "timeout", "game_outcome"]

        ev_timeout = float(timeout_outcomes.mean()) if len(timeout_outcomes) > 0 else _DEFAULT_WIN_RATE
        ev_play_on = float(play_on_outcomes.mean()) if len(play_on_outcomes) > 0 else _DEFAULT_WIN_RATE
        ev_gain = ev_timeout - ev_play_on

        sweep.append({
            "seconds_remaining": sec,
            "ev_timeout": round(ev_timeout, 4),
            "ev_play_on": round(ev_play_on, 4),
            "ev_gain": round(ev_gain, 4),
            "timeout_is_optimal": ev_gain > 0,
        })

    out_path = out_dir / "theorem3_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info("Saved Theorem 3 sweep to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Registry – maps theorem key to collection function.
# Add new theorems here; each function must accept (out_dir, processed_dir).
# ---------------------------------------------------------------------------
_COLLECTORS: Dict[str, Callable[..., Any]] = {
    "theorem1": _collect_theorem1,
    "theorem2": _collect_theorem2,
    "theorem3": _collect_theorem3,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def collect_theorem1(out_dir: Path = PROCESSED_DIR) -> Path:
    """Public wrapper kept for backwards compatibility."""
    return _collect_theorem1(out_dir)


def collect_theorem2(out_dir: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    """Public wrapper kept for backwards compatibility."""
    return _collect_theorem2(out_dir)


def collect_theorem3(out_dir: Path = PROCESSED_DIR) -> Path:
    """Public wrapper for Theorem 3 (Late-Game Timeout)."""
    return _collect_theorem3(out_dir)


def collect_all(out_dir: Path = PROCESSED_DIR) -> None:
    """
    Run all registered data-collection steps and save results to *out_dir*.

    Reads the historical possession log from ``data/processed/transitions.parquet``
    if it exists; otherwise falls back to a synthetic dataset.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, collector in _COLLECTORS.items():
        logger.info("Collecting data for %s...", key)
        collector(out_dir, processed_dir=out_dir)

    logger.info("All data collected and saved to %s", out_dir)

    from src.generate_docs import generate_all_docs
    generate_all_docs(processed_dir=out_dir)
    logger.info("Theorem documentation regenerated from analysis results.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
    print("Data collection complete. Run `python -m src.visualizations` to generate plots.")
