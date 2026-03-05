"""
Unit tests for collect_data.py.

Tests verify that each collector produces valid output files from both
synthetic and empty data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.collect_data import (
    _collect_theorem1,
    _collect_theorem2,
    _collect_theorem3,
    collect_theorem3,
)
from src.data_pipeline import build_synthetic_transitions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_transitions(tmp_path: Path, n: int = 1000) -> None:
    """Write synthetic transitions.parquet to tmp_path."""
    df = build_synthetic_transitions(n_samples=n, seed=7)
    df.to_parquet(tmp_path / "transitions.parquet", index=False)


def _read_sweep_csv(path: Path):
    """Read a theorem sweep CSV and return a list of dicts."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# _collect_theorem3
# ---------------------------------------------------------------------------


class TestCollectTheorem3:
    def test_creates_json_file(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        assert out.exists()
        assert out.name == "theorem3_sweep.csv"

    def test_sweep_has_expected_keys(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        assert len(sweep) > 0
        for entry in sweep:
            assert "seconds_remaining" in entry
            assert "ev_timeout" in entry
            assert "ev_play_on" in entry
            assert "ev_gain" in entry
            assert "timeout_is_optimal" in entry

    def test_sweep_covers_20_to_50_seconds(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        secs = [int(e["seconds_remaining"]) for e in sweep]
        assert min(secs) == 20
        assert max(secs) == 50

    def test_win_rates_bounded(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        for entry in sweep:
            assert 0.0 <= float(entry["ev_timeout"]) <= 1.0
            assert 0.0 <= float(entry["ev_play_on"]) <= 1.0

    def test_ev_gain_matches_difference(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        for entry in sweep:
            expected = round(float(entry["ev_timeout"]) - float(entry["ev_play_on"]), 4)
            assert abs(float(entry["ev_gain"]) - expected) < 1e-9

    def test_fallback_with_no_transitions(self, tmp_path):
        """Without transitions.parquet uses synthetic fallback — should not error."""
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        assert out.exists()

    def test_public_wrapper(self, tmp_path):
        """collect_theorem3 public API delegates to _collect_theorem3."""
        _write_synthetic_transitions(tmp_path)
        from src.collect_data import PROCESSED_DIR

        # Redirect to tmp_path by calling private directly via public alias
        out = collect_theorem3(out_dir=tmp_path)
        assert out.exists()
