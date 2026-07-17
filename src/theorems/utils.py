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
    """Return one resolved decision per game-period at *target_sec*.

    Uses ``pd.merge_asof`` (temporal join) to avoid the survivorship bias
    introduced by a fixed time-window filter.  Play-by-play data only logs
    discrete events, so a window filter silently misses possessions where the
    clock ran without an event.  ``merge_asof`` instead:

    * looks *backward* to get the score immediately before *target_sec*, and
    * looks *forward* to get the possessing team and first action at or after
      *target_sec*.

    Regulation and overtime periods are separate decision opportunities.

    Parameters
    ----------
    df         : historical possession log (output of ``_load_historical_log``).
    target_sec : seconds remaining at which to resolve each game's state.

    Returns
    -------
    DataFrame with one row per game-period that has both a prior score and a
    subsequent action. ``action_delay`` is the number of seconds from the
    target clock to that action.
    """
    work = df.copy()
    if "period" not in work.columns:
        work["period"] = 4
    if "event_num" not in work.columns:
        work["event_num"] = np.arange(len(work))
    if "action_team" not in work.columns:
        work["action_team"] = pd.NA

    work["elapsed_time"] = _PERIOD_SECONDS - work["seconds_remaining"]
    group_cols = ["game_id", "period"]
    work = work.sort_values(
        ["elapsed_time", *group_cols, "event_num"],
        kind="stable",
    ).reset_index(drop=True)

    target_elapsed = _PERIOD_SECONDS - target_sec
    target_df = work[group_cols].drop_duplicates()
    target_df["elapsed_time"] = target_elapsed
    target_df = target_df.sort_values(
        ["elapsed_time", *group_cols],
        kind="stable",
    )

    state_cols = [*group_cols, "elapsed_time", "score_differential"]
    backward = pd.merge_asof(
        target_df,
        work[state_cols],
        on="elapsed_time",
        by=group_cols,
        direction="backward",
        allow_exact_matches=False,
    )

    action_cols = [
        *group_cols,
        "elapsed_time",
        "event_num",
        "possession",
        "action_taken",
        "action_team",
        "game_outcome",
    ]
    actions = work[action_cols].copy()
    actions["action_elapsed_time"] = actions["elapsed_time"]
    forward = pd.merge_asof(
        target_df,
        actions,
        on="elapsed_time",
        by=group_cols,
        direction="forward",
    )

    result = backward.merge(
        forward.drop(columns="score_differential", errors="ignore"),
        on=[*group_cols, "elapsed_time"],
        how="inner",
    )
    result["action_delay"] = result["action_elapsed_time"] - result["elapsed_time"]
    result["action_seconds_remaining"] = (
        _PERIOD_SECONDS - result["action_elapsed_time"]
    )
    result = result.dropna(subset=["score_differential", "action_taken"])
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
