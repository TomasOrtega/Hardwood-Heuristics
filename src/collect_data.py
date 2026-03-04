"""
collect_data.py
===============
One-time data-collection script for the NBA Folk Theorems project.

Run this script once to compute and save the MDP sweep results that
``src/visualizations.py`` needs.  Subsequent runs of the visualization
script will load the saved files instead of re-computing them.

Adding a new theorem
--------------------
1. Implement the theorem class in ``src/mdp_engine.py`` and add it to
   :data:`~src.mdp_engine.THEOREM_REGISTRY`.
2. Add a ``_collect_<key>`` function below.
3. Register it in :data:`_COLLECTORS`.

Saved artefacts (written to ``data/processed/``)
-------------------------------------------------
* ``theorem1_sweep.json``  – EV-gain sweep results for Theorem 1 (2-for-1).
* ``theorem2_grid.npy``    – Win-probability-gain grid for Theorem 2 (Foul-Up-3).
* ``theorem2_metadata.json`` – Parameter labels (time_values, fg3_pct_values).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Per-theorem collection functions
# ---------------------------------------------------------------------------
def _collect_theorem1(
    out_dir: Path,
    model: Optional[object] = None,
) -> Path:
    """Compute and save the Theorem 1 (2-for-1) sweep results."""
    from src.mdp_engine import Theorem1TwoForOne

    logger.info("Computing Theorem 1 sweep…")
    kwargs = {"model": model} if model is not None else {}
    t1 = Theorem1TwoForOne(**kwargs)
    sweep = t1.sweep_time(time_range=list(range(10, 65, 2)))

    out_path = out_dir / "theorem1_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info("Saved Theorem 1 sweep to %s", out_path)
    return out_path


def _collect_theorem2(
    out_dir: Path,
    model: Optional[object] = None,
) -> tuple[Path, Path]:
    """Compute and save the Theorem 2 (Foul-Up-3) sweep grid."""
    from src.mdp_engine import Theorem2FoulUp3

    logger.info("Computing Theorem 2 sweep…")
    time_values = list(range(2, 12, 2))
    fg3_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]

    # Theorem2FoulUp3 uses opp_ft_pct / opp_fg3_pct, not a generic model arg;
    # if a model was provided, extract its ft_pct for this theorem.
    ft_pct_kwarg = {}
    if model is not None and hasattr(model, "ft_pct"):
        ft_pct_kwarg = {"opp_ft_pct": model.ft_pct}

    t2 = Theorem2FoulUp3(**ft_pct_kwarg)
    grid = t2.sweep(time_values=time_values, fg3_pct_values=fg3_values)

    grid_path = out_dir / "theorem2_grid.npy"
    np.save(grid_path, grid)
    logger.info("Saved Theorem 2 grid to %s", grid_path)

    meta_path = out_dir / "theorem2_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"time_values": time_values, "fg3_pct_values": fg3_values}, f, indent=2)
    logger.info("Saved Theorem 2 metadata to %s", meta_path)

    return grid_path, meta_path


# ---------------------------------------------------------------------------
# Registry – maps theorem key → collection function.
# Add new theorems here; each function must accept (out_dir, model=None).
# ---------------------------------------------------------------------------
_COLLECTORS: Dict[str, Callable[..., Any]] = {
    "theorem1": _collect_theorem1,
    "theorem2": _collect_theorem2,
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


def collect_all(out_dir: Path = PROCESSED_DIR) -> None:
    """
    Run all registered data-collection steps and save results to *out_dir*.

    If scraped NBA transition data exists in ``data/processed/transitions.parquet``
    the empirical shooting statistics are used to calibrate the transition model;
    otherwise the built-in league-average defaults are used.
    """
    from src.data_pipeline import load_empirical_params
    from src.mdp_engine import TransitionModel

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a TransitionModel calibrated to scraped data when available.
    params = load_empirical_params(out_dir)
    model: Optional[TransitionModel] = (
        TransitionModel.from_data(**params) if params else None
    )
    if model is not None:
        logger.info("Using empirically calibrated TransitionModel: %s", params)
    else:
        logger.info("No scraped data found; using default TransitionModel parameters.")

    for key, collector in _COLLECTORS.items():
        logger.info("Collecting data for %s…", key)
        collector(out_dir, model=model)

    logger.info("All data collected and saved to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
    print("Data collection complete. Run `python -m src.visualizations` to generate plots.")

