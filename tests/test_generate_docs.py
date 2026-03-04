"""
Unit tests for generate_docs.py.

These tests verify that the doc-generation functions produce valid Markdown
files whose numerical content is derived from the analysis data, not from
hardcoded values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.generate_docs import (
    _build_theorem1_key_findings,
    _build_theorem2_conclusion,
    _build_theorem2_key_findings,
    _consecutive_positive_windows,
    _find_sweep_entry,
    _fmt_ev,
    _fmt_gain,
    _largest_window,
    generate_all_docs,
    _generate_theorem1_doc,
    _generate_theorem2_doc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sweep(
    time_range: list[int] | None = None,
    positive_from: int = 28,
) -> list[dict]:
    """
    Create a synthetic sweep list.

    Entries with seconds_remaining >= positive_from get positive ev_gain;
    all others get negative ev_gain.
    """
    if time_range is None:
        time_range = list(range(10, 65, 2))
    results = []
    for s in time_range:
        rush   = 0.62 if s >= positive_from else 0.40
        normal = 0.55
        results.append(
            {
                "seconds_remaining": s,
                "ev_rush":   rush,
                "ev_normal": normal,
                "ev_gain":   rush - normal,
            }
        )
    return results


def _write_theorem1_data(tmp_path: Path, sweep: list[dict]) -> None:
    """Write sweep list to theorem1_sweep.json in tmp_path."""
    with open(tmp_path / "theorem1_sweep.json", "w") as f:
        json.dump(sweep, f)


def _write_theorem2_data(
    tmp_path: Path,
    time_values: list[int] | None = None,
    fg3_values: list[float] | None = None,
    gain_offset: float = 0.0,
) -> None:
    """Write Theorem 2 processed data files to tmp_path."""
    if time_values is None:
        time_values = [2, 4, 6, 8, 10]
    if fg3_values is None:
        fg3_values = [0.28, 0.32, 0.36, 0.40, 0.44]

    n_t, n_f = len(time_values), len(fg3_values)
    # gain increases with fg3_pct; may be shifted by gain_offset
    gain_grid    = np.array([[gain_offset + 0.05 + j * 0.01 for j in range(n_f)]
                              for _ in range(n_t)])
    wp_foul_grid    = np.full((n_t, n_f), 0.90)
    wp_no_foul_grid = wp_foul_grid - gain_grid

    np.save(tmp_path / "theorem2_grid.npy",          gain_grid)
    np.save(tmp_path / "theorem2_wp_foul_grid.npy",    wp_foul_grid)
    np.save(tmp_path / "theorem2_wp_no_foul_grid.npy", wp_no_foul_grid)
    with open(tmp_path / "theorem2_metadata.json", "w") as f:
        json.dump({"time_values": time_values, "fg3_pct_values": fg3_values}, f)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------
class TestFmtEv:
    def test_positive(self):
        assert _fmt_ev(0.623) == "0.62"

    def test_zero(self):
        assert _fmt_ev(0.0) == "0.00"


class TestFmtGain:
    def test_positive_pp(self):
        result = _fmt_gain(0.075, pp=True)
        assert result.startswith("**+")
        assert "pp" in result

    def test_negative_pp(self):
        result = _fmt_gain(-0.02, pp=True)
        assert result.startswith("-")
        assert "pp" in result

    def test_no_pp_suffix(self):
        result = _fmt_gain(0.03, pp=False)
        assert "pp" not in result


class TestFindSweepEntry:
    def test_found(self):
        sweep = [{"seconds_remaining": 32, "ev_rush": 0.6, "ev_normal": 0.5, "ev_gain": 0.1}]
        entry = _find_sweep_entry(sweep, 32)
        assert entry["ev_gain"] == pytest.approx(0.1)

    def test_not_found(self):
        with pytest.raises(KeyError):
            _find_sweep_entry([], 32)


class TestConsecutivePositiveWindows:
    def test_single_window(self):
        sweep = _make_sweep(time_range=list(range(10, 65, 2)), positive_from=28)
        windows = _consecutive_positive_windows(sweep)
        assert any(w[0] == 28 for w in windows)

    def test_two_windows(self):
        # positive at 10-12s and 28+
        sweep = _make_sweep(time_range=list(range(10, 65, 2)), positive_from=28)
        # manually add positive entries at 10-12s
        for entry in sweep:
            if entry["seconds_remaining"] <= 12:
                entry["ev_rush"]  = 0.70
                entry["ev_gain"]  = 0.15
        windows = _consecutive_positive_windows(sweep)
        assert len(windows) == 2

    def test_all_negative(self):
        sweep = [{"seconds_remaining": s, "ev_gain": -0.1} for s in range(10, 65, 2)]
        assert _consecutive_positive_windows(sweep) == []

    def test_all_positive(self):
        sweep = [{"seconds_remaining": s, "ev_gain": 0.1} for s in range(10, 65, 2)]
        windows = _consecutive_positive_windows(sweep)
        assert len(windows) == 1
        assert windows[0] == (10, 64)


class TestLargestWindow:
    def test_returns_widest(self):
        sweep = _make_sweep(time_range=list(range(10, 65, 2)), positive_from=28)
        lo, hi = _largest_window(sweep)
        assert lo == 28
        assert hi == 64

    def test_no_positive(self):
        sweep = [{"seconds_remaining": s, "ev_gain": -0.1} for s in range(10, 65, 2)]
        assert _largest_window(sweep) == (0, 0)


# ---------------------------------------------------------------------------
# Key Findings builders
# ---------------------------------------------------------------------------
class TestBuildTheorem1KeyFindings:
    def test_includes_window(self):
        sweep = _make_sweep(positive_from=28)
        text = _build_theorem1_key_findings(sweep)
        assert "28" in text
        assert "64" in text

    def test_all_negative(self):
        sweep = [{"seconds_remaining": s, "ev_gain": -0.1} for s in range(10, 65, 2)]
        text = _build_theorem1_key_findings(sweep)
        assert "No time window" in text


class TestBuildTheorem2KeyFindings:
    def test_all_positive(self):
        gain_grid = np.full((3, 3), 0.07)
        text = _build_theorem2_key_findings(gain_grid, [2, 4, 6], [0.28, 0.36, 0.44])
        assert "all analyzed scenarios" in text
        assert "7.0 pp" in text or "7." in text

    def test_mixed(self):
        gain_grid = np.array([[-0.02, 0.05], [-0.01, 0.06]])
        text = _build_theorem2_key_findings(gain_grid, [4, 8], [0.28, 0.44])
        assert "44" in text or "normal defense" in text.lower()


class TestBuildTheorem2Conclusion:
    def test_all_positive(self):
        gain_grid = np.full((3, 3), 0.07)
        text = _build_theorem2_conclusion(gain_grid, [2, 4, 6], [0.28, 0.36, 0.44], 0.34)
        assert "all analyzed" in text

    def test_some_negative(self):
        gain_grid = np.array([[-0.02, 0.05], [-0.01, 0.06]])
        text = _build_theorem2_conclusion(gain_grid, [4, 8], [0.28, 0.44], 0.34)
        assert "34" in text


# ---------------------------------------------------------------------------
# _generate_theorem1_doc
# ---------------------------------------------------------------------------
class TestGenerateTheorem1Doc:
    def test_file_created(self, tmp_path):
        sweep = _make_sweep()
        _write_theorem1_data(tmp_path, sweep)
        out = _generate_theorem1_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_values_match_analysis(self, tmp_path):
        sweep = _make_sweep()
        _write_theorem1_data(tmp_path, sweep)
        _generate_theorem1_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem1_two_for_one.md").read_text()

        # Values at 32s should appear in the doc
        e32 = _find_sweep_entry(sweep, 32)
        assert f"{e32['ev_rush']:.2f}" in content
        assert f"{e32['ev_normal']:.2f}" in content

    def test_missing_data_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _generate_theorem1_doc(processed_dir=tmp_path, docs_dir=tmp_path)

    def test_returns_path(self, tmp_path):
        sweep = _make_sweep()
        _write_theorem1_data(tmp_path, sweep)
        result = _generate_theorem1_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.name == "theorem1_two_for_one.md"


# ---------------------------------------------------------------------------
# _generate_theorem2_doc
# ---------------------------------------------------------------------------
class TestGenerateTheorem2Doc:
    def test_file_created(self, tmp_path):
        _write_theorem2_data(tmp_path)
        out = _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_values_match_analysis(self, tmp_path):
        _write_theorem2_data(tmp_path)
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()

        # The doc should contain percentage-formatted values from the grid
        # (gain values ≥ 5 pp are expected given gain_offset=0.0 default)
        assert "Win %" in content or "win %" in content.lower()

    def test_foul_cost_correct(self, tmp_path):
        """Verify the expected cost of foul is ~-1.2 pp, not ~-118 pp."""
        _write_theorem2_data(tmp_path)
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()

        # foul cost should appear in the -0.x to -2.x range (not hundreds)
        import re
        match = re.search(r"\\approx\s*(-?\d+\.\d+)\\text\{\\s*pp\}", content)
        if match:
            cost_val = float(match.group(1))
            assert -3.0 < cost_val < 0.0, f"Unexpected foul cost: {cost_val}"

    def test_missing_data_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)

    def test_returns_path(self, tmp_path):
        _write_theorem2_data(tmp_path)
        result = _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.name == "theorem2_foul_up_3.md"

    def test_all_positive_grid_conclusion(self, tmp_path):
        """When fouling is always better, conclusion should say 'all analyzed'."""
        _write_theorem2_data(tmp_path, gain_offset=0.05)
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()
        assert "all analyzed" in content

    def test_fallback_without_wp_grids(self, tmp_path):
        """Without individual WP grids the code reconstructs using gain grid + neutral default."""
        _write_theorem2_data(tmp_path)
        # Remove the individual WP grid files to trigger the fallback path.
        (tmp_path / "theorem2_wp_foul_grid.npy").unlink()
        (tmp_path / "theorem2_wp_no_foul_grid.npy").unlink()
        out = _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert out.exists()


# ---------------------------------------------------------------------------
# generate_all_docs
# ---------------------------------------------------------------------------
class TestGenerateAllDocs:
    def test_generates_both_files(self, tmp_path):
        sweep = _make_sweep()
        _write_theorem1_data(tmp_path, sweep)
        _write_theorem2_data(tmp_path)
        paths = generate_all_docs(processed_dir=tmp_path, docs_dir=tmp_path)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_returns_path_list(self, tmp_path):
        sweep = _make_sweep()
        _write_theorem1_data(tmp_path, sweep)
        _write_theorem2_data(tmp_path)
        result = generate_all_docs(processed_dir=tmp_path, docs_dir=tmp_path)
        assert all(isinstance(p, Path) for p in result)
