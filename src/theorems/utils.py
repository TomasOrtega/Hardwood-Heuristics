"""
utils.py
========
Shared utilities for theorem modules.

Provides pandas-based CSV helpers used by theorem1, theorem3, and any future
theorem that persists results as a row-per-entry CSV file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Total seconds in the final period tracked by the historical log.
_PERIOD_SECONDS = 180

PALETTE = "RdYlGn"
FIGURE_DPI = 150
FONT_FAMILY = "DejaVu Sans"


def apply_plot_aesthetics() -> None:
    """Apply standard matplotlib formatting for project visualizations."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
        }
    )


def write_sweep_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Write a list of row dicts to *path* as a CSV using pandas.

    Parameters
    ----------
    path       : destination file path.
    rows       : list of dicts, one per CSV row.
    fieldnames : column order for the CSV header.
    """
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)


def load_sweep_csv(csv_path: Path) -> List[Dict]:
    """Load a sweep CSV and return a list of row dicts with Python-native types.

    pandas automatically infers integer and float columns.  String columns
    whose values are exclusively ``"True"`` / ``"False"`` (case-insensitive)
    are converted to Python :class:`bool`.  All numpy scalar types are cast to
    their Python equivalents so that callers receive plain ``int``, ``float``,
    and ``bool`` values.

    Parameters
    ----------
    csv_path : path to the CSV file to read.

    Returns
    -------
    List of dicts with Python-native scalar types.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data not found at {csv_path}. " "Run `python -m src.collect_data` first."
        )

    df = pd.read_csv(csv_path)

    # Convert "True"/"False" string columns to proper Python booleans.
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.strip().str.lower()
        if cleaned.isin(["true", "false"]).all():
            df[col] = cleaned == "true"

    rows: List[Dict] = []
    for record in df.to_dict(orient="records"):
        rows.append({k: _to_python(v) for k, v in record.items()})
    return rows


def get_resolved_possessions_at_time(df: pd.DataFrame, target_sec: int) -> pd.DataFrame:
    """Return one resolved possession row per game at *target_sec*.

    Uses ``pd.merge_asof`` (temporal join) to avoid the survivorship bias
    introduced by a fixed time-window filter.  Play-by-play data only logs
    discrete events, so a window filter silently misses possessions where the
    clock ran without an event.  ``merge_asof`` instead:

    * looks *backward* to get the most recent game state (score, possession,
      opponent 3PT%) at or just before *target_sec*, and
    * looks *forward* to get the very first action taken at or after
      *target_sec*.

    Parameters
    ----------
    df         : historical possession log (output of ``_load_historical_log``).
    target_sec : seconds remaining at which to resolve each game's state.

    Returns
    -------
    DataFrame with one row per game that has a resolved action, containing
    columns: ``game_id``, ``elapsed_time``, ``score_differential``,
    ``possession``, ``opponent_fg3_pct``, ``action_taken``, ``game_outcome``.
    Games with no logged action at or after *target_sec* are excluded.
    """
    work = df.copy()
    work["elapsed_time"] = _PERIOD_SECONDS - work["seconds_remaining"]
    work = work.sort_values("elapsed_time").reset_index(drop=True)

    target_elapsed = _PERIOD_SECONDS - target_sec

    target_df = pd.DataFrame(
        {
            "game_id": work["game_id"].unique(),
            "elapsed_time": target_elapsed,
        }
    ).sort_values("elapsed_time")

    state_cols = [
        "game_id",
        "elapsed_time",
        "score_differential",
        "possession",
        "opponent_fg3_pct",
    ]
    backward = pd.merge_asof(
        target_df,
        work[state_cols],
        on="elapsed_time",
        by="game_id",
        direction="backward",
    )

    action_cols = ["game_id", "elapsed_time", "action_taken", "game_outcome"]
    forward = pd.merge_asof(
        target_df,
        work[action_cols],
        on="elapsed_time",
        by="game_id",
        direction="forward",
    )

    result = backward.merge(
        forward[["game_id", "action_taken", "game_outcome"]],
        on="game_id",
        how="left",
    )

    result = result.dropna(subset=["action_taken"])
    return result


def _to_python(val):
    """Convert a numpy scalar to its Python-native equivalent."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val
