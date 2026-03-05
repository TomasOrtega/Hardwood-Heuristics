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

import numpy as np
import pandas as pd


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
            f"Data not found at {csv_path}. "
            "Run `python -m src.collect_data` first."
        )

    df = pd.read_csv(csv_path)

    # Convert "True"/"False" string columns to proper Python booleans.
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.strip().str.lower()
        if cleaned.isin(["true", "false"]).all():
            df[col] = cleaned == "true"

    rows: List[Dict] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                k: _to_python(v)
                for k, v in record.items()
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
