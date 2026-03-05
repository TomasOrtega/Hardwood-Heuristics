"""
Unit tests for theorem visualization functions.

These tests verify that plot functions produce saved files without errors,
using small synthetic inputs to keep runtime short.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Theorem 3: Late-Game Timeout EV curve (via CSV)
# ---------------------------------------------------------------------------


def _make_theorem3_csv(tmp_path: Path, n: int = 8) -> Path:
    """Write a synthetic theorem3_sweep.csv and return the path."""
    seconds = list(range(20, 20 + n * 2, 2))
    rng = np.random.default_rng(99)
    ev_timeout = rng.uniform(0.43, 0.57, size=n).tolist()
    ev_play_on = rng.uniform(0.43, 0.57, size=n).tolist()
    rows = [
        {
            "seconds_remaining": s,
            "ev_timeout": round(t, 4),
            "ev_play_on": round(p, 4),
            "ev_gain": round(t - p, 4),
            "timeout_is_optimal": t > p,
        }
        for s, t, p in zip(seconds, ev_timeout, ev_play_on)
    ]
    csv_path = tmp_path / "theorem3_sweep.csv"
    fieldnames = [
        "seconds_remaining",
        "ev_timeout",
        "ev_play_on",
        "ev_gain",
        "timeout_is_optimal",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestPlotTheorem3TimeoutCurve:
    def test_file_created(self, tmp_path):
        _make_theorem3_csv(tmp_path)
        from src.theorems.theorem3 import plot as t3_plot

        out = tmp_path / "timeout_ev_curve.svg"
        result = t3_plot(processed_dir=tmp_path, images_dir=tmp_path)
        assert result.exists()
        assert result.stat().st_size > 1000

    def test_returns_path(self, tmp_path):
        _make_theorem3_csv(tmp_path)
        from src.theorems.theorem3 import plot as t3_plot

        result = t3_plot(processed_dir=tmp_path, images_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.name == "timeout_ev_curve.svg"

    def test_missing_csv_raises(self, tmp_path):
        """Plotting without data should raise FileNotFoundError."""
        from src.theorems.theorem3 import plot as t3_plot

        with pytest.raises(FileNotFoundError):
            t3_plot(processed_dir=tmp_path, images_dir=tmp_path)
