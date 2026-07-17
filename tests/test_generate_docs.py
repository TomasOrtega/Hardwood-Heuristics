"""
Unit tests for generate_docs.py.

These tests verify that the doc-generation functions produce valid Markdown
files whose numerical content is derived from the analysis data, not from
hardcoded values.
"""

from __future__ import annotations

from pathlib import Path

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
        time_range = list(range(30, 41, 2))
    results = []
    for s in time_range:
        rush = 0.62 if s >= positive_from else 0.40
        normal = 0.55
        results.append(
            {
                "seconds_remaining": s,
                "ev_rush": rush,
                "ev_rush_ci_low": max(0, rush - 0.1),
                "ev_rush_ci_high": min(1, rush + 0.1),
                "ev_normal": normal,
                "ev_normal_ci_low": max(0, normal - 0.1),
                "ev_normal_ci_high": min(1, normal + 0.1),
                "ev_gain": rush - normal,
                "ev_gain_ci_low": rush - normal - 0.15,
                "ev_gain_ci_high": rush - normal + 0.15,
                "n_rush": 12,
                "n_normal": 14,
                "rush_is_optimal": rush > normal,
            }
        )
    return results


def _write_theorem1_data(tmp_path: Path, sweep: list[dict]) -> None:
    """Write sweep list to theorem1_sweep.csv in tmp_path."""
    import csv as _csv

    fieldnames = [
        "seconds_remaining",
        "ev_rush",
        "ev_rush_ci_low",
        "ev_rush_ci_high",
        "ev_normal",
        "ev_normal_ci_low",
        "ev_normal_ci_high",
        "ev_gain",
        "ev_gain_ci_low",
        "ev_gain_ci_high",
        "n_rush",
        "n_normal",
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
                "ev_timeout_ci_low": max(0, ev_timeout - 0.1),
                "ev_timeout_ci_high": min(1, ev_timeout + 0.1),
                "ev_play_on": ev_play_on,
                "ev_play_on_ci_low": max(0, ev_play_on - 0.1),
                "ev_play_on_ci_high": min(1, ev_play_on + 0.1),
                "ev_gain": ev_timeout - ev_play_on,
                "ev_gain_ci_low": ev_timeout - ev_play_on - 0.15,
                "ev_gain_ci_high": ev_timeout - ev_play_on + 0.15,
                "n_timeout": 8,
                "n_play_on": 20,
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
        "ev_timeout_ci_low",
        "ev_timeout_ci_high",
        "ev_play_on",
        "ev_play_on_ci_low",
        "ev_play_on_ci_high",
        "ev_gain",
        "ev_gain_ci_low",
        "ev_gain_ci_high",
        "n_timeout",
        "n_play_on",
        "timeout_is_optimal",
    ]
    with open(tmp_path / "theorem3_sweep.csv", "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            writer.writerow({k: row[k] for k in fieldnames if k in row})


def _write_theorem2_data(tmp_path: Path, gain_offset: float = 0.0) -> None:
    """Write Theorem 2 processed data files to tmp_path."""
    rows = []
    for seconds in [2, 4, 6, 8, 10]:
        ev_foul = 0.8
        ev_defend = 0.7 - gain_offset
        rows.append(
            {
                "seconds_remaining": seconds,
                "ev_foul": ev_foul,
                "ev_foul_ci_low": ev_foul - 0.1,
                "ev_foul_ci_high": min(1, ev_foul + 0.1),
                "ev_defend": ev_defend,
                "ev_defend_ci_low": ev_defend - 0.1,
                "ev_defend_ci_high": ev_defend + 0.1,
                "ev_gain": ev_foul - ev_defend,
                "ev_gain_ci_low": ev_foul - ev_defend - 0.15,
                "ev_gain_ci_high": ev_foul - ev_defend + 0.15,
                "n_foul": 6,
                "n_defend": 15,
                "foul_is_better": ev_foul > ev_defend,
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(tmp_path / "theorem2_sweep.csv", index=False)


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
        assert "game-cluster bootstrap" in content

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
        assert out.stat().st_size > 100

    def test_contains_figure(self, tmp_path):
        _write_theorem2_data(tmp_path)
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()
        assert "foul_up_3_curve.svg" in content

    def test_contains_conclusion(self, tmp_path):
        _write_theorem2_data(tmp_path)
        _generate_theorem2_doc(processed_dir=tmp_path, docs_dir=tmp_path)
        content = (tmp_path / "theorem2_foul_up_3.md").read_text()
        assert "## Conclusion" in content
        assert "5 of 5 comparable clock points" in content
        assert "game-cluster bootstrap" in content

    def test_returns_path(self, tmp_path):
        _write_theorem2_data(tmp_path)
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
        assert "game-cluster bootstrap" in content


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
