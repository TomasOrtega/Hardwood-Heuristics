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
from src.theorems.utils import get_resolved_possessions_at_time

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


def _write_transitions(tmp_path: Path, rows: list[dict]) -> None:
    defaults = {
        "season": "2022-23",
        "period": 4,
    }
    records = [{**defaults, **row} for row in rows]
    pd.DataFrame(records).to_parquet(
        tmp_path / "transitions.parquet",
        index=False,
    )


def test_resolver_keeps_regulation_and_overtime_separate():
    df = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "period": 4,
                "event_num": 1,
                "seconds_remaining": 40,
                "score_differential": 1,
                "possession": 1,
                "action_taken": "other",
                "action_team": 1,
                "game_outcome": 1,
            },
            {
                "game_id": "g1",
                "period": 4,
                "event_num": 2,
                "seconds_remaining": 20,
                "score_differential": 1,
                "possession": 0,
                "action_taken": "rebound",
                "action_team": 0,
                "game_outcome": 1,
            },
            {
                "game_id": "g1",
                "period": 5,
                "event_num": 3,
                "seconds_remaining": 40,
                "score_differential": -2,
                "possession": 0,
                "action_taken": "other",
                "action_team": 0,
                "game_outcome": 1,
            },
            {
                "game_id": "g1",
                "period": 5,
                "event_num": 4,
                "seconds_remaining": 20,
                "score_differential": -2,
                "possession": 1,
                "action_taken": "shoot",
                "action_team": 1,
                "game_outcome": 1,
            },
        ]
    )

    resolved = get_resolved_possessions_at_time(df, 30).set_index("period")

    assert len(resolved) == 2
    assert resolved.loc[4, "score_differential"] == 1
    assert resolved.loc[4, "action_taken"] == "rebound"
    assert resolved.loc[5, "score_differential"] == -2
    assert resolved.loc[5, "action_taken"] == "shoot"


# ---------------------------------------------------------------------------
# _collect_theorem1
# ---------------------------------------------------------------------------


class TestCollectTheorem1:
    def test_creates_csv_file(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem1(out_dir=tmp_path, processed_dir=tmp_path)
        assert out.exists()
        assert out.name == "theorem1_sweep.csv"

    def test_sweep_covers_30_to_40_seconds(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem1(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        secs = [int(e["seconds_remaining"]) for e in sweep]
        assert min(secs) == 30
        assert max(secs) == 40

    def test_compares_quick_and_deliberate_shots_for_both_teams(self, tmp_path):
        _write_transitions(
            tmp_path,
            [
                {
                    "game_id": "rush",
                    "event_num": 1,
                    "seconds_remaining": 40,
                    "score_differential": 0,
                    "possession": 1,
                    "action_taken": "other",
                    "action_team": 1,
                    "game_outcome": 1,
                },
                {
                    "game_id": "rush",
                    "event_num": 2,
                    "seconds_remaining": 34,
                    "score_differential": 0,
                    "possession": 1,
                    "action_taken": "shoot",
                    "action_team": 1,
                    "game_outcome": 1,
                },
                {
                    "game_id": "normal",
                    "event_num": 1,
                    "seconds_remaining": 40,
                    "score_differential": 0,
                    "possession": 0,
                    "action_taken": "other",
                    "action_team": 0,
                    "game_outcome": 1,
                },
                {
                    "game_id": "normal",
                    "event_num": 2,
                    "seconds_remaining": 28,
                    "score_differential": 0,
                    "possession": 0,
                    "action_taken": "shoot",
                    "action_team": 0,
                    "game_outcome": 1,
                },
            ],
        )

        out = _collect_theorem1(out_dir=tmp_path, processed_dir=tmp_path)
        row = pd.read_csv(out).set_index("seconds_remaining").loc[36]

        assert row["n_rush"] == 1
        assert row["n_normal"] == 1
        assert row["ev_rush"] == 1
        assert row["ev_normal"] == 0


class TestCollectTheorem2:
    def test_compares_defensive_foul_with_normal_defense(self, tmp_path):
        _write_transitions(
            tmp_path,
            [
                {
                    "game_id": "foul",
                    "event_num": 1,
                    "seconds_remaining": 12,
                    "score_differential": 3,
                    "possession": 0,
                    "action_taken": "other",
                    "action_team": 0,
                    "game_outcome": 1,
                },
                {
                    "game_id": "foul",
                    "event_num": 2,
                    "seconds_remaining": 8,
                    "score_differential": 3,
                    "possession": 0,
                    "action_taken": "foul",
                    "action_team": 1,
                    "game_outcome": 1,
                },
                {
                    "game_id": "defend",
                    "event_num": 1,
                    "seconds_remaining": 12,
                    "score_differential": -3,
                    "possession": 1,
                    "action_taken": "other",
                    "action_team": 1,
                    "game_outcome": 0,
                },
                {
                    "game_id": "defend",
                    "event_num": 2,
                    "seconds_remaining": 8,
                    "score_differential": -3,
                    "possession": 1,
                    "action_taken": "shoot",
                    "action_team": 1,
                    "game_outcome": 0,
                },
                {
                    "game_id": "ignored",
                    "event_num": 1,
                    "seconds_remaining": 12,
                    "score_differential": 3,
                    "possession": 0,
                    "action_taken": "other",
                    "action_team": 0,
                    "game_outcome": 1,
                },
                {
                    "game_id": "ignored",
                    "event_num": 2,
                    "seconds_remaining": 8,
                    "score_differential": 3,
                    "possession": 0,
                    "action_taken": "free_throw",
                    "action_team": 0,
                    "game_outcome": 1,
                },
            ],
        )

        out = _collect_theorem2(out_dir=tmp_path, processed_dir=tmp_path)
        row = pd.read_csv(out).set_index("seconds_remaining").loc[10]

        assert out.name == "theorem2_sweep.csv"
        assert row["n_foul"] == 1
        assert row["n_defend"] == 1
        assert row["ev_foul"] == 1
        assert row["ev_defend"] == 1


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
            assert "n_timeout" in entry
            assert "n_play_on" in entry

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
            if entry["ev_timeout"]:
                assert 0.0 <= float(entry["ev_timeout"]) <= 1.0
            else:
                assert int(entry["n_timeout"]) == 0
            if entry["ev_play_on"]:
                assert 0.0 <= float(entry["ev_play_on"]) <= 1.0
            else:
                assert int(entry["n_play_on"]) == 0

    def test_ev_gain_matches_difference(self, tmp_path):
        _write_synthetic_transitions(tmp_path)
        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        sweep = _read_sweep_csv(out)
        for entry in sweep:
            if not entry["ev_timeout"] or not entry["ev_play_on"]:
                assert not entry["ev_gain"]
                continue
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

    def test_compares_only_possessing_team_timeout_or_live_ball_action(
        self, tmp_path
    ):
        _write_transitions(
            tmp_path,
            [
                {
                    "game_id": "timeout",
                    "event_num": 1,
                    "seconds_remaining": 32,
                    "score_differential": -2,
                    "possession": 1,
                    "action_taken": "other",
                    "action_team": 1,
                    "game_outcome": 1,
                },
                {
                    "game_id": "timeout",
                    "event_num": 2,
                    "seconds_remaining": 28,
                    "score_differential": -2,
                    "possession": 1,
                    "action_taken": "timeout",
                    "action_team": 1,
                    "game_outcome": 1,
                },
                {
                    "game_id": "play_on",
                    "event_num": 1,
                    "seconds_remaining": 32,
                    "score_differential": 2,
                    "possession": 0,
                    "action_taken": "other",
                    "action_team": 0,
                    "game_outcome": 1,
                },
                {
                    "game_id": "play_on",
                    "event_num": 2,
                    "seconds_remaining": 28,
                    "score_differential": 2,
                    "possession": 0,
                    "action_taken": "turnover",
                    "action_team": 0,
                    "game_outcome": 1,
                },
            ],
        )

        out = _collect_theorem3(out_dir=tmp_path, processed_dir=tmp_path)
        row = pd.read_csv(out).set_index("seconds_remaining").loc[30]

        assert row["n_timeout"] == 1
        assert row["n_play_on"] == 1
        assert row["ev_timeout"] == 1
        assert row["ev_play_on"] == 0
