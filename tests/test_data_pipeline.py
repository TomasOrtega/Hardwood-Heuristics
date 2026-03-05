"""
Unit tests for data_pipeline.py.

NBA API calls are mocked so tests run without network access.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.data_pipeline import (
    _period_clock_to_seconds,
    GameState,
    NBAPlayByPlayScraper,
    PlayByPlayParser,
    build_synthetic_transitions,
    FINAL_PERIOD_SECONDS,
    MAX_FOULS_TO_GIVE,
)


# ---------------------------------------------------------------------------
# _period_clock_to_seconds
# ---------------------------------------------------------------------------
class TestPeriodClockToSeconds:
    def test_iso_format_full(self):
        assert _period_clock_to_seconds("PT02M35.00S") == 155

    def test_iso_format_seconds_only(self):
        assert _period_clock_to_seconds("PT00M08.70S") == 8

    def test_mm_ss_format(self):
        assert _period_clock_to_seconds("1:45") == 105

    def test_zero(self):
        assert _period_clock_to_seconds("PT00M00.00S") == 0

    def test_empty_string(self):
        assert _period_clock_to_seconds("") == 0

    def test_three_minutes(self):
        assert _period_clock_to_seconds("PT03M00.00S") == 180


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------
class TestGameState:
    def test_as_tuple_round_trip(self):
        gs = GameState(
            score_differential=3, seconds_remaining=45, possession=1, fouls_to_give=2
        )
        assert GameState.from_tuple(gs.as_tuple()) == gs

    def test_from_tuple(self):
        gs = GameState.from_tuple((-2, 30, 0, 1))
        assert gs.score_differential == -2
        assert gs.seconds_remaining == 30
        assert gs.possession == 0
        assert gs.fouls_to_give == 1


# ---------------------------------------------------------------------------
# NBAPlayByPlayScraper (mocked)
# ---------------------------------------------------------------------------
class TestNBAPlayByPlayScraper:
    def test_default_seasons(self):
        scraper = NBAPlayByPlayScraper()
        assert len(scraper.seasons) == 5
        assert "2023-24" in scraper.seasons

    def test_fetch_season_game_ids_uses_db(self, tmp_path):
        """fetch_season_game_ids should parse GAME_ID from the SQLite database."""
        import sqlite3

        db_path = tmp_path / "nba.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE play_by_play ("
            "game_id TEXT, period INTEGER, pctimestring TEXT, "
            "eventmsgtype INTEGER, score TEXT, "
            "homedescription TEXT, visitordescription TEXT)"
        )
        conn.execute(
            "INSERT INTO play_by_play VALUES "
            "('0022300001', 4, 'PT00M30.00S', 1, '95 - 92', '', '')"
        )
        conn.execute(
            "INSERT INTO play_by_play VALUES "
            "('0022300002', 4, 'PT00M20.00S', 2, '95 - 92', '', '')"
        )
        conn.commit()
        conn.close()

        scraper = NBAPlayByPlayScraper(seasons=["2023-24"], raw_dir=tmp_path)
        ids = scraper.fetch_season_game_ids("2023-24")

        assert "0022300001" in ids
        assert "0022300002" in ids

    def test_fetch_play_by_play_uses_cache(self, tmp_path):
        """fetch_play_by_play should return cached file if it exists."""
        game_id = "0022300999"
        cached_df = pd.DataFrame({"GAME_ID": [game_id], "PERIOD": [4]})
        cache_path = tmp_path / f"{game_id}.parquet"
        cached_df.to_parquet(cache_path)

        scraper = NBAPlayByPlayScraper(raw_dir=tmp_path)
        result = scraper.fetch_play_by_play(game_id)
        assert len(result) == 1
        assert result.iloc[0]["GAME_ID"] == game_id


# ---------------------------------------------------------------------------
# PlayByPlayParser
# ---------------------------------------------------------------------------
def _make_raw_pbp(n: int = 20) -> pd.DataFrame:
    """Build a minimal valid raw PBP DataFrame for testing."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "GAME_ID": ["0022300001"] * n,
            "PERIOD": [4] * n,
            "PCTIMESTRING": [f"PT00M{i:02d}.00S" for i in range(n, 0, -1)],
            "EVENTMSGTYPE": rng.choice([1, 2, 5, 6], size=n).tolist(),
            "SCORE": ["95 - 92"] * n,
            "HOMEDESCRIPTION": [""] * n,
            "VISITORDESCRIPTION": [""] * n,
        }
    )


class TestPlayByPlayParser:
    def test_parse_returns_dataframe(self, tmp_path):
        raw = _make_raw_pbp(30)
        # Make sure some events have small seconds_remaining (≤180)
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(raw)
        assert isinstance(result, pd.DataFrame)

    def test_parse_keeps_only_final_3_minutes(self, tmp_path):
        raw = _make_raw_pbp(30)
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(raw)
        if not result.empty:
            assert result["seconds_remaining"].max() <= FINAL_PERIOD_SECONDS

    def test_parse_missing_columns_raises(self, tmp_path):
        bad_df = pd.DataFrame({"GAME_ID": [1], "PERIOD": [4]})
        parser = PlayByPlayParser(processed_dir=tmp_path)
        with pytest.raises(ValueError, match="missing columns"):
            parser.parse(bad_df)

    def test_parse_empty_dataframe(self, tmp_path):
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(pd.DataFrame())
        assert result.empty

    def test_score_parsing(self, tmp_path):
        raw = _make_raw_pbp(10)
        raw["SCORE"] = "100 - 97"
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(raw)
        if not result.empty:
            # Home leads by 3
            assert (result["score_differential"] == 3).all()

    def test_parse_output_columns(self, tmp_path):
        """parse() should return the new flat historical log columns."""
        raw = _make_raw_pbp(30)
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(raw)
        if not result.empty:
            required = {
                "game_id",
                "season",
                "seconds_remaining",
                "score_differential",
                "possession",
                "fouls_to_give",
                "action_taken",
                "game_outcome",
                "opponent_fg3_pct",
            }
            assert required.issubset(set(result.columns))

    def test_parse_game_outcome_binary(self, tmp_path):
        """game_outcome should be 0 or 1."""
        raw = _make_raw_pbp(30)
        parser = PlayByPlayParser(processed_dir=tmp_path)
        result = parser.parse(raw)
        if not result.empty:
            assert set(result["game_outcome"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# build_synthetic_transitions
# ---------------------------------------------------------------------------
class TestBuildSyntheticTransitions:
    def test_returns_dataframe(self):
        df = build_synthetic_transitions(n_samples=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_expected_columns(self):
        df = build_synthetic_transitions(n_samples=50)
        required = {
            "game_id",
            "season",
            "seconds_remaining",
            "score_differential",
            "possession",
            "fouls_to_give",
            "action_taken",
            "game_outcome",
            "opponent_fg3_pct",
        }
        assert required.issubset(set(df.columns))

    def test_fouls_bounded(self):
        df = build_synthetic_transitions(n_samples=500)
        assert (df["fouls_to_give"] <= MAX_FOULS_TO_GIVE).all()
        assert (df["fouls_to_give"] >= 0).all()

    def test_possession_binary(self):
        df = build_synthetic_transitions(n_samples=200)
        assert set(df["possession"].unique()).issubset({0, 1})

    def test_game_outcome_binary(self):
        df = build_synthetic_transitions(n_samples=200)
        assert set(df["game_outcome"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = build_synthetic_transitions(seed=99)
        df2 = build_synthetic_transitions(seed=99)
        pd.testing.assert_frame_equal(df1, df2)
