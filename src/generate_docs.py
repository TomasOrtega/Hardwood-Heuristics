"""
generate_docs.py
================
Regenerates the theorem Markdown documentation files with numbers derived
directly from the historical win-rate results stored in ``data/processed/``.

Call this module after running ``python -m src.collect_data`` (which invokes
:func:`generate_all_docs` automatically), or run it standalone::

    python -m src.generate_docs

Adding a new theorem
--------------------
1. Implement a ``_generate_<key>_doc`` helper below.
2. Register it in :data:`_DOC_GENERATORS`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Minimum number of consecutive positive-gain time buckets required to treat
# a run as a *dominant* timeout window worth highlighting in Theorem 3.
_MIN_SIGNIFICANT_WINDOW_SIZE = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_ev(value: float) -> str:
    """Format a win-percentage value to 2 decimal places."""
    return f"{value:.2f}"


def _fmt_gain(value: float, pp: bool = False) -> str:
    """Format a gain value with sign and optional 'pp' suffix."""
    suffix = " pp" if pp else ""
    if value > 0:
        return f"**+{value * 100:.1f}{suffix}**"
    return f"{value * 100:.1f}{suffix}"


def _gain_label(value: float) -> str:
    """Format a raw win-probability gain (0--1 scale) with bold for positive values."""
    if value > 0:
        return f"**+{value:.2f}**"
    return f"{value:.2f}"


def _find_sweep_entry(sweep: List[Dict], seconds: int) -> Dict:
    """Return the sweep dict entry for the given seconds value."""
    for entry in sweep:
        if entry["seconds_remaining"] == seconds:
            return entry
    raise KeyError(f"No sweep entry for {seconds} seconds remaining")


def _consecutive_positive_windows(
    sweep: List[Dict],
) -> List[tuple[int, int]]:
    """
    Return a list of (low_sec, high_sec) ranges where ev_gain > 0,
    identifying consecutive blocks in the sorted sweep.
    """
    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    windows: List[tuple[int, int]] = []
    in_window = False
    window_start = 0
    for entry in sorted_sweep:
        sec = entry["seconds_remaining"]
        if entry["ev_gain"] > 0 and not in_window:
            in_window = True
            window_start = sec
        elif entry["ev_gain"] <= 0 and in_window:
            in_window = False
            windows.append((window_start, prev_sec))  # type: ignore[possibly-undefined]
        prev_sec = sec
    if in_window:
        windows.append((window_start, prev_sec))  # type: ignore[possibly-undefined]
    return windows


def _largest_window(sweep: List[Dict]) -> tuple[int, int]:
    """Return the largest consecutive window where ev_gain > 0."""
    windows = _consecutive_positive_windows(sweep)
    if not windows:
        return (0, 0)
    return max(windows, key=lambda w: w[1] - w[0])


# ---------------------------------------------------------------------------
# Theorem 1 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem1_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    from src.theorems.theorem1 import generate_doc
    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Theorem 2 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem2_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Write the static Theorem 2 Markdown file."""
    from src.theorems.theorem2 import generate_doc

    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Theorem 3 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem3_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    from src.theorems.theorem3 import generate_doc
    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOC_GENERATORS: Dict = {
    "theorem1": _generate_theorem1_doc,
    "theorem2": _generate_theorem2_doc,
    "theorem3": _generate_theorem3_doc,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all_docs(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> List[Path]:
    """
    Regenerate all theorem Markdown files from historical analysis results.

    Reads pre-computed sweep data from *processed_dir* and writes updated
    Markdown files to *docs_dir*.  Call after ``collect_all()`` so that the
    processed data is up-to-date.

    Returns
    -------
    List of Path objects for all generated Markdown files.
    """
    paths: List[Path] = []
    for key, generator in _DOC_GENERATORS.items():
        logger.info("Generating doc for %s...", key)
        paths.append(generator(processed_dir=processed_dir, docs_dir=docs_dir))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generated = generate_all_docs()
    for p in generated:
        print(f"Written: {p}")
