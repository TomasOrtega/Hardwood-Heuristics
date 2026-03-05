"""
Unit tests for visualizations.py.

These tests verify that plot functions produce saved files without errors,
using small synthetic inputs to keep runtime short.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.visualizations import plot_foul_up_3_heatmap, plot_two_for_one_ev_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _small_grid(rows: int = 3, cols: int = 4) -> np.ndarray:
    """Return a small grid of WP-gain values centred around 0."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.15, 0.15, size=(rows, cols))


# ---------------------------------------------------------------------------
# plot_foul_up_3_heatmap
# ---------------------------------------------------------------------------
class TestPlotFoulUp3Heatmap:
    def test_file_created(self, tmp_path):
        grid = _small_grid(3, 4)
        out = tmp_path / "foul_heatmap.svg"
        result = plot_foul_up_3_heatmap(
            grid=grid,
            time_values=[2, 4, 6],
            fg3_pct_values=[0.30, 0.34, 0.38, 0.42],
            out_path=out,
        )
        assert result.exists()
        assert result.stat().st_size > 1000   # non-trivial PNG

    def test_returns_path(self, tmp_path):
        grid = _small_grid(2, 3)
        out = tmp_path / "out.svg"
        result = plot_foul_up_3_heatmap(
            grid=grid,
            time_values=[4, 8],
            fg3_pct_values=[0.30, 0.36, 0.42],
            out_path=out,
        )
        assert isinstance(result, Path)

    def test_default_output_path(self, tmp_path, monkeypatch):
        """Without specifying out_path, file should land in IMAGES_DIR."""
        import src.visualizations as viz
        monkeypatch.setattr(viz, "IMAGES_DIR", tmp_path)
        grid = _small_grid(2, 2)
        result = plot_foul_up_3_heatmap(
            grid=grid,
            time_values=[4, 8],
            fg3_pct_values=[0.30, 0.42],
        )
        assert result.exists()

    def test_all_positive_grid(self, tmp_path):
        """Fully positive grid should still render without errors."""
        grid = np.full((3, 3), 0.10)
        out = tmp_path / "pos.svg"
        result = plot_foul_up_3_heatmap(
            grid=grid,
            time_values=[2, 4, 6],
            fg3_pct_values=[0.30, 0.36, 0.42],
            out_path=out,
        )
        assert result.exists()

    def test_all_negative_grid(self, tmp_path):
        grid = np.full((3, 3), -0.10)
        out = tmp_path / "neg.svg"
        result = plot_foul_up_3_heatmap(
            grid=grid,
            time_values=[2, 4, 6],
            fg3_pct_values=[0.30, 0.36, 0.42],
            out_path=out,
        )
        assert result.exists()


# ---------------------------------------------------------------------------
# plot_two_for_one_ev_curve
# ---------------------------------------------------------------------------
def _make_sweep(n: int = 10) -> list:
    seconds = list(range(10, 10 + n * 2, 2))
    rng = np.random.default_rng(7)
    ev_rush   = rng.uniform(0.45, 0.60, size=n).tolist()
    ev_normal = rng.uniform(0.43, 0.58, size=n).tolist()
    return [
        {
            "seconds_remaining": s,
            "ev_rush":   r,
            "ev_normal": n_,
            "ev_gain":   r - n_,
        }
        for s, r, n_ in zip(seconds, ev_rush, ev_normal)
    ]


class TestPlotTwoForOneEvCurve:
    def test_file_created(self, tmp_path):
        out = tmp_path / "ev_curve.svg"
        result = plot_two_for_one_ev_curve(sweep_results=_make_sweep(), out_path=out)
        assert result.exists()
        assert result.stat().st_size > 1000

    def test_returns_path(self, tmp_path):
        out = tmp_path / "ev.svg"
        result = plot_two_for_one_ev_curve(sweep_results=_make_sweep(5), out_path=out)
        assert isinstance(result, Path)

    def test_default_output_path(self, tmp_path, monkeypatch):
        import src.visualizations as viz
        monkeypatch.setattr(viz, "IMAGES_DIR", tmp_path)
        result = plot_two_for_one_ev_curve(sweep_results=_make_sweep(6))
        assert result.exists()

    def test_with_crossover(self, tmp_path):
        """Ensure crossover annotation doesn't break the plot."""
        sweep = _make_sweep(15)
        # Force a sign flip in ev_gain
        for i in range(8):
            sweep[i]["ev_gain"] = -0.01 - i * 0.001
        for i in range(8, 15):
            sweep[i]["ev_gain"] = 0.01 + (i - 8) * 0.001
        out = tmp_path / "crossover.svg"
        result = plot_two_for_one_ev_curve(sweep_results=sweep, out_path=out)
        assert result.exists()

    def test_no_crossover(self, tmp_path):
        """All-negative gain: no crossover annotation, should still render."""
        sweep = _make_sweep(8)
        for r in sweep:
            r["ev_gain"] = -0.05
        out = tmp_path / "no_cross.svg"
        result = plot_two_for_one_ev_curve(sweep_results=sweep, out_path=out)
        assert result.exists()

    def test_single_point(self, tmp_path):
        sweep = [{"seconds_remaining": 30, "ev_rush": 0.55, "ev_normal": 0.52, "ev_gain": 0.03}]
        out = tmp_path / "single.svg"
        result = plot_two_for_one_ev_curve(sweep_results=sweep, out_path=out)
        assert result.exists()
