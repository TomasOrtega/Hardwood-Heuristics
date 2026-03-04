"""
collect_data.py
===============
One-time data-collection script for the NBA Folk Theorems project.

Run this script once to compute and save the MDP sweep results that
``src/visualizations.py`` needs.  Subsequent runs of the visualization
script will load the saved files instead of re-computing them.

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

import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def collect_theorem1(out_dir: Path = PROCESSED_DIR) -> Path:
    """Compute and save the Theorem 1 (2-for-1) sweep results."""
    from src.mdp_engine import Theorem1TwoForOne

    logger.info("Computing Theorem 1 sweep…")
    t1 = Theorem1TwoForOne()
    sweep = t1.sweep_time(time_range=list(range(10, 65, 2)))

    out_path = out_dir / "theorem1_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info("Saved Theorem 1 sweep to %s", out_path)
    return out_path


def collect_theorem2(out_dir: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    """Compute and save the Theorem 2 (Foul-Up-3) sweep grid."""
    from src.mdp_engine import Theorem2FoulUp3

    logger.info("Computing Theorem 2 sweep…")
    time_values = list(range(2, 12, 2))
    fg3_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]

    t2 = Theorem2FoulUp3()
    grid = t2.sweep(time_values=time_values, fg3_pct_values=fg3_values)

    grid_path = out_dir / "theorem2_grid.npy"
    np.save(grid_path, grid)
    logger.info("Saved Theorem 2 grid to %s", grid_path)

    meta_path = out_dir / "theorem2_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"time_values": time_values, "fg3_pct_values": fg3_values}, f, indent=2)
    logger.info("Saved Theorem 2 metadata to %s", meta_path)

    return grid_path, meta_path


def collect_all(out_dir: Path = PROCESSED_DIR) -> None:
    """Run all data-collection steps and save results to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    collect_theorem1(out_dir)
    collect_theorem2(out_dir)
    logger.info("All data collected and saved to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all()
    print("Data collection complete. Run `python -m src.visualizations` to generate plots.")
