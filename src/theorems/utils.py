"""
utils.py
========
Shared utilities for theorem modules.

Provides pandas-based CSV helpers used by theorem1, theorem3, and any future
theorem that persists results as a row-per-entry CSV file.
"""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Sequence, Tuple

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


def get_resolved_possessions_at_times(
    df: pd.DataFrame,
    target_secs: Sequence[int],
) -> pd.DataFrame:
    """Return one resolved decision per game-period and target clock.

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
    df          : historical possession log (output of ``_load_historical_log``).
    target_secs : clock values at which to resolve each game's state.

    Returns
    -------
    DataFrame with one row per game-period and target clock that has both a
    prior score and a subsequent action. ``action_delay`` is the number of
    seconds from the target clock to that action.
    """
    target_values = list(dict.fromkeys(int(sec) for sec in target_secs))
    if not target_values:
        return pd.DataFrame()

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

    target_df = work[group_cols].drop_duplicates().merge(
        pd.DataFrame({"target_seconds": target_values}),
        how="cross",
    )
    target_df["elapsed_time"] = _PERIOD_SECONDS - target_df["target_seconds"]
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
        on=[*group_cols, "target_seconds", "elapsed_time"],
        how="inner",
    )
    result["action_delay"] = result["action_elapsed_time"] - result["elapsed_time"]
    result["action_seconds_remaining"] = (
        _PERIOD_SECONDS - result["action_elapsed_time"]
    )
    result = result.dropna(subset=["score_differential", "action_taken"])
    return result.sort_values(
        ["target_seconds", *group_cols],
        kind="stable",
    ).reset_index(drop=True)


def get_resolved_possessions_at_time(df: pd.DataFrame, target_sec: int) -> pd.DataFrame:
    """Return one resolved decision per game-period at *target_sec*."""
    return get_resolved_possessions_at_times(df, [target_sec]).drop(
        columns="target_seconds"
    )


def _wilson_interval(
    successes: float,
    observations: float,
    confidence: float,
) -> Tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if observations <= 0:
        return float("nan"), float("nan")

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = successes / observations
    denominator = 1 + z**2 / observations
    center = (rate + z**2 / (2 * observations)) / denominator
    half_width = (
        z
        * np.sqrt(
            rate * (1 - rate) / observations
            + z**2 / (4 * observations**2)
        )
        / denominator
    )
    return max(0, center - half_width), min(1, center + half_width)


def summarize_binary_comparison(
    samples: pd.DataFrame,
    target_values: Sequence[int],
    group_col: str,
    outcome_col: str,
    groups: Tuple[str, str],
    *,
    target_col: str = "target_seconds",
    cluster_col: str = "game_id",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> List[Dict]:
    """Summarize two binary-outcome groups with game-clustered intervals.

    Game IDs are resampled as blocks, preserving dependence between repeated
    clock values and overtime periods from the same game.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    targets = list(dict.fromkeys(int(value) for value in target_values))
    first, second = groups
    work = samples.loc[
        samples[group_col].isin(groups),
        [cluster_col, target_col, group_col, outcome_col],
    ].dropna()
    target_index = {value: index for index, value in enumerate(targets)}
    work = work.loc[work[target_col].isin(target_index)].copy()

    n_targets = len(targets)
    rates = np.full((n_targets, 2), np.nan)
    counts = np.zeros((n_targets, 2), dtype=int)
    ci_low = np.full((n_targets, 2), np.nan)
    ci_high = np.full((n_targets, 2), np.nan)
    difference_ci_low = np.full(n_targets, np.nan)
    difference_ci_high = np.full(n_targets, np.nan)

    if not work.empty:
        outcomes = pd.to_numeric(work[outcome_col])
        if not outcomes.isin([0, 1]).all():
            raise ValueError(f"{outcome_col} must contain only binary outcomes")

        group_index = {first: 0, second: 1}
        outcomes = pd.to_numeric(work[outcome_col]).to_numpy(dtype=float)

        cluster_codes, cluster_values = pd.factorize(work[cluster_col], sort=False)
        target_codes = work[target_col].map(target_index).to_numpy(dtype=int)
        group_codes = work[group_col].map(group_index).to_numpy(dtype=int)

        shape = (len(cluster_values), n_targets, 2)
        successes = np.zeros(shape, dtype=float)
        observations = np.zeros(shape, dtype=float)
        np.add.at(
            successes,
            (cluster_codes, target_codes, group_codes),
            outcomes,
        )
        np.add.at(
            observations,
            (cluster_codes, target_codes, group_codes),
            1,
        )

        total_successes = successes.sum(axis=0)
        total_observations = observations.sum(axis=0)
        counts = total_observations.astype(int)
        rates = np.divide(
            total_successes,
            total_observations,
            out=np.full_like(total_successes, np.nan),
            where=total_observations > 0,
        )

        rng = np.random.default_rng(seed)
        probabilities = np.full(len(cluster_values), 1 / len(cluster_values))
        bootstrap_rates = np.full((n_resamples, n_targets, 2), np.nan)
        batch_size = min(100, n_resamples)
        for start in range(0, n_resamples, batch_size):
            stop = min(start + batch_size, n_resamples)
            weights = rng.multinomial(
                len(cluster_values),
                probabilities,
                size=stop - start,
            )
            sampled_successes = np.einsum(
                "bc,ctg->btg",
                weights,
                successes,
                optimize=True,
            )
            sampled_observations = np.einsum(
                "bc,ctg->btg",
                weights,
                observations,
                optimize=True,
            )
            bootstrap_rates[start:stop] = np.divide(
                sampled_successes,
                sampled_observations,
                out=np.full_like(sampled_successes, np.nan),
                where=sampled_observations > 0,
            )

        alpha = (1 - confidence) / 2
        for target_index_value in range(n_targets):
            for group_index_value in range(2):
                values = bootstrap_rates[:, target_index_value, group_index_value]
                values = values[np.isfinite(values)]
                if values.size:
                    ci_low[target_index_value, group_index_value] = np.quantile(
                        values,
                        alpha,
                    )
                    ci_high[target_index_value, group_index_value] = np.quantile(
                        values,
                        1 - alpha,
                    )

            differences = (
                bootstrap_rates[:, target_index_value, 0]
                - bootstrap_rates[:, target_index_value, 1]
            )
            differences = differences[np.isfinite(differences)]
            if differences.size:
                difference_ci_low[target_index_value] = np.quantile(
                    differences,
                    alpha,
                )
                difference_ci_high[target_index_value] = np.quantile(
                    differences,
                    1 - alpha,
                )

        for target_index_value in range(n_targets):
            score_intervals = []
            for group_index_value in range(2):
                score_low, score_high = _wilson_interval(
                    total_successes[target_index_value, group_index_value],
                    total_observations[target_index_value, group_index_value],
                    confidence,
                )
                score_intervals.append((score_low, score_high))
                if np.isfinite(score_low):
                    ci_low[target_index_value, group_index_value] = np.nanmin(
                        [ci_low[target_index_value, group_index_value], score_low]
                    )
                    ci_high[target_index_value, group_index_value] = np.nanmax(
                        [ci_high[target_index_value, group_index_value], score_high]
                    )

            if np.isfinite(rates[target_index_value]).all():
                first_low, first_high = score_intervals[0]
                second_low, second_high = score_intervals[1]
                difference = (
                    rates[target_index_value, 0] - rates[target_index_value, 1]
                )
                score_difference_low = difference - np.sqrt(
                    (rates[target_index_value, 0] - first_low) ** 2
                    + (second_high - rates[target_index_value, 1]) ** 2
                )
                score_difference_high = difference + np.sqrt(
                    (first_high - rates[target_index_value, 0]) ** 2
                    + (rates[target_index_value, 1] - second_low) ** 2
                )
                difference_ci_low[target_index_value] = max(
                    -1,
                    np.nanmin(
                        [
                            difference_ci_low[target_index_value],
                            score_difference_low,
                        ]
                    ),
                )
                difference_ci_high[target_index_value] = min(
                    1,
                    np.nanmax(
                        [
                            difference_ci_high[target_index_value],
                            score_difference_high,
                        ]
                    ),
                )

    rows: List[Dict] = []
    for index, target in enumerate(targets):
        rows.append(
            {
                target_col: target,
                first: rates[index, 0],
                f"{first}_ci_low": ci_low[index, 0],
                f"{first}_ci_high": ci_high[index, 0],
                f"n_{first}": int(counts[index, 0]),
                second: rates[index, 1],
                f"{second}_ci_low": ci_low[index, 1],
                f"{second}_ci_high": ci_high[index, 1],
                f"n_{second}": int(counts[index, 1]),
                "difference": rates[index, 0] - rates[index, 1],
                "difference_ci_low": difference_ci_low[index],
                "difference_ci_high": difference_ci_high[index],
            }
        )
    return rows


def _to_python(val):
    """Convert a numpy scalar to its Python-native equivalent."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val
