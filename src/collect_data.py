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
1. Create ``src/theorems/theorem<N>.py`` with a ``collect(out_dir, processed_dir)``
   function that filters the dataframe, saves results to a CSV, and returns
   the output path.
2. Register the new module in :data:`_COLLECTORS`.
3. Add corresponding ``plot`` and ``generate_doc`` functions in the same module,
   and register them in ``visualizations._PLOTTERS`` and
   ``generate_docs._DOC_GENERATORS``.

Saved artefacts (written to ``data/processed/``)
-------------------------------------------------
* ``theorem1_sweep.csv``           – Historical win-rate sweep for Theorem 1 (2-for-1).
* ``theorem2_grid.csv``            – Win-rate-gain grid for Theorem 2 (Foul-Up-3).
* ``theorem2_wp_foul_grid.csv``    – Historical win rate when fouling (per cell).
* ``theorem2_wp_no_foul_grid.csv`` – Historical win rate without fouling (per cell).
* ``theorem2_metadata.json``       – Parameter labels (time_values, fg3_pct_values).
* ``theorem3_sweep.csv``           – Historical win-rate sweep for Theorem 3 (Late-Game Timeout).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
# Per-theorem collection functions (delegate to theorem modules)
# ---------------------------------------------------------------------------

def _collect_theorem1(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 1 (2-for-1) historical win rates and save to CSV.

    Delegates to :func:`src.theorems.theorem1.collect`.
    """
    from src.theorems.theorem1 import collect
    return collect(out_dir, processed_dir)


def _collect_theorem2(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Compute Theorem 2 (Foul-Up-3) historical win rates and save grids.

    Delegates to :func:`src.theorems.theorem2.collect`.
    """
    from src.theorems.theorem2 import collect
    return collect(out_dir, processed_dir)


def _collect_theorem3(
    out_dir: Path,
    processed_dir: Optional[Path] = None,
) -> Path:
    """
    Compute Theorem 3 (Late-Game Timeout) historical win rates and save to CSV.

    Delegates to :func:`src.theorems.theorem3.collect`.
    """
    from src.theorems.theorem3 import collect
    return collect(out_dir, processed_dir)


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
