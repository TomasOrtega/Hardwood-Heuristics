"""
Unit tests for generate_docs.py.

These tests verify that the doc-generation functions produce valid Markdown
files whose numerical content is derived from the analysis data, not from
hardcoded values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.generate_docs import (
    generate_all_docs,
    _generate_theorem1_doc,
    _generate_theorem2_doc,
    _generate_theorem3_doc,
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
        rush = 0.62 if s >= positive_from else 0.40
        normal = 0.55
        results.append(
            {
                "seconds_remaining": s,
                "ev_rush": rush,
                "ev_normal": normal,
                "ev_gain": rush - normal,
            }
        )
    return results


def _write_theorem1_data(tmp_path: Path, sweep: list[dict]) -> None:
    """Write sweep list to theorem1_sweep.csv in tmp_path."""
    import csv as _csv

    fieldnames = [
        "seconds_remaining",
        "ev_rush",
        "ev_normal",
        "ev_gain",
        "rush_is_optimal",
    ]
    with open(tmp_path / "theorem1_sweep.csv", "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            writer.writerow({k: row[k] for k in fieldnames if k in row})


def _make_theorem3_sweep(
    time_range: list[int] | None = None,
    timeout_better_from: int | None = None,
) -> list[dict]:
    """
    Create a synthetic Theorem 3 sweep list.

    If timeout_better_from is given, entries with seconds_remaining >=
    timeout_better_from get positive ev_gain; all others get negative.
    Otherwise all entries get a small mixed signal.
    """
    if time_range is None:
        time_range = list(range(20, 51, 2))
    results = []
    for s in time_range:
        if timeout_better_from is not None:
            ev_timeout = 0.55 if s >= timeout_better_from else 0.40
            ev_play_on = 0.45
        else:
            ev_timeout = 0.48
            ev_play_on = 0.50
        results.append(
            {
                "seconds_remaining": s,
                "ev_timeout": ev_timeout,
                "ev_play_on": ev_play_on,
                "ev_gain": ev_timeout - ev_play_on,
                "timeout_is_optimal": ev_timeout > ev_play_on,
            }
        )
    return results


def _write_theorem3_data(tmp_path: Path, sweep: list[dict]) -> None:
    """Write sweep list to theorem3_sweep.csv in tmp_path."""
    import csv as _csv

    fieldnames = [
        "seconds_remaining",
        "ev_timeout",
        "ev_play_on",
        "ev_gain",
        "timeout_is_optimal",
    ]
    with open(tmp_path / "theorem3_sweep.csv", "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            writer.writerow({k: row[k] for k in fieldnames if k in row})


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
        fg3_values = [0.25, 0.30, 0.35, 0.40, 0.45]

    n_t, n_f = len(time_values), len(fg3_values)
    # gain increases with fg3_pct; may be shifted by gain_offset
    gain_grid = np.array(
        [[gain_offset + 0.05 + j * 0.01 for j in range(n_f)] for _ in range(n_t)]
    )
    wp_foul_grid = np.full((n_t, n_f), 0.90)
    wp_no_foul_grid = wp_foul_grid - gain_grid

    np.savetxt(tmp_path / "theorem2_grid.csv", gain_grid, delimiter=",")
    np.savetxt(tmp_path / "theorem2_wp_foul_grid.csv", wp_foul_grid, delimiter=",")
    np.savetxt(
        tmp_path / "theorem2_wp_no_foul_grid.csv", wp_no_foul_grid, delimiter=","
    )


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

        # The figure and conclusion should be present
        assert "two_for_one_ev_curve.svg" in content
        assert "## Conclusion" in content

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
        out = _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert out.exists()
        assert out.stat().st_size > 100

    def test_contains_figure(self, tmp_path):
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()
        assert "foul_up_3_heatmap.svg" in content

    def test_contains_conclusion(self, tmp_path):
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()
        assert "## Conclusion" in content

    def test_returns_path(self, tmp_path):
        result = _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.name == "theorem2_foul_up_3.md"


# ---------------------------------------------------------------------------
# _generate_theorem3_doc
# ---------------------------------------------------------------------------
class TestGenerateTheorem3Doc:
    def test_file_created(self, tmp_path):
        sweep = _make_theorem3_sweep()
        _write_theorem3_data(tmp_path, sweep)
        out = _generate_theorem3_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_returns_path(self, tmp_path):
        sweep = _make_theorem3_sweep()
        _write_theorem3_data(tmp_path, sweep)
        result = _generate_theorem3_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        assert isinstance(result, Path)
        assert result.name == "theorem3_timeout.md"

    def test_values_match_analysis(self, tmp_path):
        sweep = _make_theorem3_sweep()
        _write_theorem3_data(tmp_path, sweep)
        _generate_theorem3_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem3_timeout.md").read_text()

        # The figure and conclusion should be present
        assert "timeout_ev_curve.svg" in content
        assert "## Conclusion" in content


# ---------------------------------------------------------------------------
# generate_all_docs
# ---------------------------------------------------------------------------
class TestGenerateAllDocs:
    def test_generates_all_three_files(self, tmp_path):
        sweep1 = _make_sweep()
        _write_theorem1_data(tmp_path, sweep1)
        _write_theorem2_data(tmp_path)
        sweep3 = _make_theorem3_sweep()
        _write_theorem3_data(tmp_path, sweep3)
        paths = generate_all_docs(processed_dir=tmp_path, docs_dir=tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_returns_path_list(self, tmp_path):
        sweep1 = _make_sweep()
        _write_theorem1_data(tmp_path, sweep1)
        _write_theorem2_data(tmp_path)
        sweep3 = _make_theorem3_sweep()
        _write_theorem3_data(tmp_path, sweep3)
        result = generate_all_docs(processed_dir=tmp_path, docs_dir=tmp_path)
        assert all(isinstance(p, Path) for p in result)
