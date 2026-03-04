"""
Unit tests for scrape_nba_data.py.

NBA API calls are mocked so tests run without network access.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.scrape_nba_data import scrape, DEFAULT_SEASONS, _parse_args


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.seasons is None
        assert args.dry_run is False
        assert args.log_level == "INFO"

    def test_seasons_flag(self):
        args = _parse_args(["--seasons", "2022-23", "2023-24"])
        assert args.seasons == ["2022-23", "2023-24"]

    def test_dry_run_flag(self):
        args = _parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_out_dir_flag(self, tmp_path):
        args = _parse_args(["--out-dir", str(tmp_path)])
        assert args.out_dir == tmp_path

    def test_raw_dir_flag(self, tmp_path):
        args = _parse_args(["--raw-dir", str(tmp_path)])
        assert args.raw_dir == tmp_path

    def test_log_level_flag(self):
        args = _parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# DEFAULT_SEASONS
# ---------------------------------------------------------------------------
class TestDefaultSeasons:
    def test_default_seasons_count(self):
        assert len(DEFAULT_SEASONS) == 5

    def test_default_seasons_includes_recent(self):
        assert "2023-24" in DEFAULT_SEASONS
        assert "2019-20" in DEFAULT_SEASONS


# ---------------------------------------------------------------------------
# scrape() – mocked
# ---------------------------------------------------------------------------
def _make_raw_pbp(n: int = 30) -> pd.DataFrame:
    """Minimal raw PBP DataFrame accepted by PlayByPlayParser."""
    import numpy as np

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


class TestScrape:
    @staticmethod
    def _get_passed_seasons(call_kwargs) -> list[str] | None:
        """Extract the seasons argument from a mock call."""
        seasons = call_kwargs.kwargs.get("seasons")
        if seasons is None and call_kwargs.args:
            seasons = call_kwargs.args[0]
        return seasons

    def test_scrape_returns_path(self, tmp_path):
        """scrape() should always return a Path object."""
        raw_pbp = _make_raw_pbp(30)
        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = raw_pbp

        with patch("src.scrape_nba_data.NBAPlayByPlayScraper", return_value=mock_scraper):
            result = scrape(
                seasons=["2023-24"],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert isinstance(result, Path)
        assert result.name == "transitions.parquet"

    def test_scrape_writes_transitions_parquet(self, tmp_path):
        """scrape() should write transitions.parquet when data is available."""
        raw_pbp = _make_raw_pbp(30)
        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = raw_pbp

        with patch("src.scrape_nba_data.NBAPlayByPlayScraper", return_value=mock_scraper):
            out = scrape(
                seasons=["2023-24"],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert out.exists()
        df = pd.read_parquet(out)
        assert not df.empty

    def test_scrape_empty_raw_data(self, tmp_path):
        """scrape() should handle an empty raw DataFrame gracefully."""
        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = pd.DataFrame()

        with patch("src.scrape_nba_data.NBAPlayByPlayScraper", return_value=mock_scraper):
            out = scrape(
                seasons=["2023-24"],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        assert isinstance(out, Path)
        assert not out.exists()

    def test_scrape_uses_provided_seasons(self, tmp_path):
        """scrape() should pass the given seasons to NBAPlayByPlayScraper."""
        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = pd.DataFrame()

        with patch(
            "src.scrape_nba_data.NBAPlayByPlayScraper", return_value=mock_scraper
        ) as mock_cls:
            scrape(
                seasons=["2022-23"],
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        mock_cls.assert_called_once()
        passed_seasons = self._get_passed_seasons(mock_cls.call_args)
        assert passed_seasons == ["2022-23"]

    def test_scrape_dry_run_no_parquet_files(self, tmp_path):
        """dry-run with no cached files should return without writing output."""
        out = scrape(
            seasons=["2023-24"],
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            dry_run=True,
        )
        assert isinstance(out, Path)
        assert not out.exists()

    def test_scrape_dry_run_uses_cached_files(self, tmp_path):
        """dry-run should load existing parquet files from raw_dir."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        processed_dir = tmp_path / "processed"

        raw_pbp = _make_raw_pbp(30)
        raw_pbp.to_parquet(raw_dir / "test_game.parquet", index=False)

        out = scrape(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            dry_run=True,
        )

        assert isinstance(out, Path)

    def test_scrape_default_seasons_used_when_none(self, tmp_path):
        """When seasons=None, DEFAULT_SEASONS should be used."""
        mock_scraper = MagicMock()
        mock_scraper.fetch_all.return_value = pd.DataFrame()

        with patch(
            "src.scrape_nba_data.NBAPlayByPlayScraper", return_value=mock_scraper
        ) as mock_cls:
            scrape(
                seasons=None,
                raw_dir=tmp_path / "raw",
                processed_dir=tmp_path / "processed",
            )

        mock_cls.assert_called_once()
        passed_seasons = self._get_passed_seasons(mock_cls.call_args)
        assert passed_seasons == DEFAULT_SEASONS
