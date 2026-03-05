"""
Unit tests for visualizations.py.

These tests verify that plot functions produce saved files without errors,
using small synthetic inputs to keep runtime short.
"""

from __future__ import annotations

import csv

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
            "ev_timeout":        round(t, 4),
            "ev_play_on":        round(p, 4),
            "ev_gain":           round(t - p, 4),
            "timeout_is_optimal": t > p,
        }
        for s, t, p in zip(seconds, ev_timeout, ev_play_on)
    ]
    csv_path = tmp_path / "theorem3_sweep.csv"
    fieldnames = ["seconds_remaining", "ev_timeout", "ev_play_on",
                  "ev_gain", "timeout_is_optimal"]
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

    def test_with_crossover(self, tmp_path):
        """Ensure a sign flip in ev_gain doesn't break the plot."""
        csv_path = _make_theorem3_csv(tmp_path, n=10)
        # Rewrite with forced sign flip
        rows = []
        for i, sec in enumerate(range(20, 40, 2)):
            gain = -0.02 if i < 5 else 0.02
            rows.append({
                "seconds_remaining": sec,
                "ev_timeout":        0.50 + gain,
                "ev_play_on":        0.50,
                "ev_gain":           gain,
                "timeout_is_optimal": gain > 0,
            })
        import csv as _csv
        fieldnames = ["seconds_remaining", "ev_timeout", "ev_play_on",
                      "ev_gain", "timeout_is_optimal"]
        with open(csv_path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        from src.theorems.theorem3 import plot as t3_plot
        result = t3_plot(processed_dir=tmp_path, images_dir=tmp_path)
        assert result.exists()

    def test_via_visualizations_registry(self, tmp_path):
        """Theorem 3 should be present in the _PLOTTERS registry."""
        import src.visualizations as viz
        assert "theorem3" in viz._PLOTTERS

    def test_missing_csv_raises(self, tmp_path):
        """Plotting without data should raise FileNotFoundError."""
        from src.theorems.theorem3 import plot as t3_plot
        with pytest.raises(FileNotFoundError):
            t3_plot(processed_dir=tmp_path, images_dir=tmp_path)
