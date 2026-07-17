"""
data_pipeline.py
================
Loads NBA play-by-play data from the Kaggle basketball dataset
(wyattowalsh/basketball), cleans and parses each event, then converts
the chronological log into a flat historical possession log.

Output columns
--------------
    game_id             : string  -- unique game identifier
    season              : string  -- e.g. "2023-24"
    period              : int     -- 4 = fourth quarter, 5+ = overtime
    event_num           : int     -- source event sequence within the game
    seconds_remaining   : int     -- seconds left in the period [0, 180]
    score_differential  : int     -- home minus away score (clipped to ±30)
    possession          : int     -- 0 = away team, 1 = home team
    action_taken        : string  -- "shoot", "foul", "timeout", "turnover",
                                    "rebound", "free_throw", or "other"
    action_team         : int?    -- 0 = away, 1 = home, null = unknown
    game_outcome        : int     -- 1 = home team won, 0 = away team won
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FINAL_PERIOD_SECONDS = 180  # last 3 minutes of regulation / OT
SCORE_DIFF_CLIP = 30  # clip extreme leads for analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _period_clock_to_seconds(clock_str: str) -> int:
    """Convert 'PT02M35.00S' or 'MM:SS' style strings to total seconds."""
    if not clock_str:
        return 0
    clock_str = str(clock_str).strip()
    # ISO 8601 duration as returned by PlayByPlayV2 ('PT02M35.00S')
    if clock_str.startswith("PT"):
        clock_str = clock_str[2:]
        minutes = 0.0
        seconds = 0.0
        if "M" in clock_str:
            parts = clock_str.split("M")
            minutes = float(parts[0])
            clock_str = parts[1]
        if "S" in clock_str:
            seconds = float(clock_str.replace("S", ""))
        return int(minutes * 60 + seconds)
    # legacy 'MM:SS' format
    if ":" in clock_str:
        parts = clock_str.split(":")
        return int(parts[0]) * 60 + int(float(parts[1]))
    return 0


def _season_from_game_id(game_id: str) -> str:
    """Derive a season string (e.g. '2023-24') from a 10-digit NBA game ID."""
    _prefix_to_season: Dict[str, str] = {
        "00219": "2019-20",
        "00220": "2020-21",
        "00221": "2021-22",
        "00222": "2022-23",
        "00223": "2023-24",
    }
    prefix = str(game_id)[:5]
    return _prefix_to_season.get(prefix, "")


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class NBAPlayByPlayScraper:
    """
    Loads NBA play-by-play data from the Kaggle basketball dataset
    (wyattowalsh/basketball).

    The dataset must be available as ``basketball.sqlite`` in *raw_dir*.
    If the file is absent it is downloaded automatically via the Kaggle API
    (requires ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables or
    a ``~/.kaggle/kaggle.json`` credentials file).

    Parameters
    ----------
    seasons : list of str
        Season strings, e.g. ``["2022-23", "2023-24"]``.
    raw_dir : Path
        Directory where ``basketball.sqlite`` is cached.
    """

    KAGGLE_DATASET = "wyattowalsh/basketball"
    DB_FILENAME = "nba.sqlite"

    # Map season string → 5-char regular-season game_id prefix
    _SEASON_PREFIXES: Dict[str, str] = {
        "2019-20": "00219",
        "2020-21": "00220",
        "2021-22": "00221",
        "2022-23": "00222",
        "2023-24": "00223",
    }

    # Column mapping from SQLite names to the uppercase names expected by
    # PlayByPlayParser
    _COLUMN_MAP: Dict[str, str] = {
        "game_id": "GAME_ID",
        "eventnum": "EVENTNUM",
        "period": "PERIOD",
        "pctimestring": "PCTIMESTRING",
        "eventmsgtype": "EVENTMSGTYPE",
        "score": "SCORE",
        "homedescription": "HOMEDESCRIPTION",
        "visitordescription": "VISITORDESCRIPTION",
        "player1_team_id": "PLAYER1_TEAM_ID",
        "team_id_home": "HOME_TEAM_ID",
        "team_id_away": "AWAY_TEAM_ID",
    }

    _CACHE_COLUMNS = {
        "GAME_ID",
        "EVENTNUM",
        "PERIOD",
        "PCTIMESTRING",
        "EVENTMSGTYPE",
        "SCORE",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "PLAYER1_TEAM_ID",
        "HOME_TEAM_ID",
        "AWAY_TEAM_ID",
        "SEASON",
    }

    def __init__(
        self,
        seasons: Optional[List[str]] = None,
        raw_dir: Path = RAW_DIR,
    ) -> None:
        if seasons is None:
            seasons = [
                "2019-20",
                "2020-21",
                "2021-22",
                "2022-23",
                "2023-24",
            ]
        self.seasons = seasons
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_season_game_ids(self, season: str) -> List[str]:
        """Return all regular-season game IDs for *season* from the database."""
        import sqlite3

        prefix = self._SEASON_PREFIXES.get(season, "")
        if not prefix:
            logger.warning("Unknown season '%s'; returning empty list.", season)
            return []

        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query(
                "SELECT DISTINCT game_id FROM play_by_play WHERE game_id LIKE ?",
                conn,
                params=(f"{prefix}%",),
            )
        finally:
            conn.close()

        ids: List[str] = df["game_id"].tolist()
        logger.info("Season %s: %d games found.", season, len(ids))
        return ids

    def fetch_play_by_play(self, game_id: str) -> pd.DataFrame:
        """Fetch raw play-by-play for a single game, with caching."""
        cache_path = self.raw_dir / f"{game_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        import sqlite3

        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query(
                "SELECT game_id, eventnum, period, pctimestring, eventmsgtype, "
                "score, homedescription, visitordescription, player1_team_id "
                "FROM play_by_play WHERE game_id = ?",
                conn,
                params=(game_id,),
            )
        finally:
            conn.close()

        df = df.rename(columns=self._COLUMN_MAP)
        df.to_parquet(cache_path, index=False)
        return df

    def fetch_all(self) -> pd.DataFrame:
        """Fetch PBP for all configured seasons and return combined DataFrame."""
        prefixes = [
            self._SEASON_PREFIXES[s] for s in self.seasons if s in self._SEASON_PREFIXES
        ]
        if not prefixes:
            logger.warning("No valid season prefixes found; returning empty DataFrame.")
            return pd.DataFrame()

        combined_path = self.raw_dir / "all_seasons_pbp.parquet"
        manifest_path = self.raw_dir / "all_seasons_pbp.json"
        if combined_path.exists():
            cached = pd.read_parquet(combined_path)
            cached_prefixes = set()
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cached_prefixes = set(manifest.get("requested_prefixes", []))
            if self._CACHE_COLUMNS.issubset(cached.columns) and set(prefixes).issubset(
                cached_prefixes
            ):
                logger.info("Loading cached combined PBP from %s", combined_path)
                return cached.loc[
                    cached["GAME_ID"].astype(str).str[:5].isin(prefixes)
                ].copy()
            logger.info("Cached PBP schema or season coverage is stale; rebuilding it.")

        import sqlite3

        db_path = self._get_db_path()

        where_clause = " OR ".join(["game_id LIKE ?" for _ in prefixes])
        params = [f"{p}%" for p in prefixes]

        query = (
            "SELECT game_id, eventnum, period, pctimestring, eventmsgtype, score, "
            "homedescription, visitordescription, player1_team_id "
            f"FROM play_by_play WHERE ({where_clause})"
        )
        game_query = (
            "SELECT game_id, MAX(team_id_home) AS team_id_home, "
            "MAX(team_id_away) AS team_id_away FROM game "
            f"WHERE ({where_clause}) GROUP BY game_id"
        )

        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query(query, conn, params=params)
            games = pd.read_sql_query(game_query, conn, params=params)
        finally:
            conn.close()

        if df.empty:
            logger.warning("No play-by-play data found for the specified seasons.")
            return pd.DataFrame()

        df = df.merge(games, on="game_id", how="left")
        df = df.rename(columns=self._COLUMN_MAP)

        prefix_to_season = {v: k for k, v in self._SEASON_PREFIXES.items()}
        df["SEASON"] = df["GAME_ID"].str[:5].map(prefix_to_season).fillna("")

        df.to_parquet(combined_path, index=False)
        manifest_path.write_text(
            json.dumps({"requested_prefixes": prefixes}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d events to %s", len(df), combined_path)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_db_path(self) -> Path:
        """Return the path to ``basketball.sqlite``, downloading if absent."""
        db_path = self.raw_dir / self.DB_FILENAME
        if not db_path.exists():
            self._download_dataset(db_path)
        return db_path

    def _download_dataset(self, db_path: Path) -> None:
        """Download the Kaggle dataset to *raw_dir*."""
        try:
            import kaggle  # type: ignore[import-untyped]

            kaggle.api.authenticate()
            logger.info("Downloading Kaggle dataset '%s'…", self.KAGGLE_DATASET)
            kaggle.api.dataset_download_files(
                self.KAGGLE_DATASET,
                path=str(self.raw_dir),
                unzip=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Kaggle dataset '{self.KAGGLE_DATASET}'. "
                f"Set KAGGLE_USERNAME and KAGGLE_KEY environment variables (or "
                f"place a kaggle.json credentials file in ~/.kaggle/), or "
                f"manually copy '{self.DB_FILENAME}' into '{self.raw_dir}'. "
                f"Error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Parser / Feature Engineering
# ---------------------------------------------------------------------------
class PlayByPlayParser:
    """
    Converts raw PBP rows into a flat historical possession log, keeping only
    events from the final ``FINAL_PERIOD_SECONDS`` seconds of the 4th quarter
    or any overtime period.

    Parameters
    ----------
    processed_dir : Path
        Directory for processed Parquet outputs.
    """

    _SHOT_MADE_TYPE = 1
    _SHOT_MISSED_TYPE = 2
    _FREE_THROW_TYPE = 3
    _REBOUND_TYPE = 4
    _TURNOVER_TYPE = 5
    _FOUL_TYPE = 6
    _VIOLATION_TYPE = 7
    _SUBSTITUTION_TYPE = 8
    _TIMEOUT_TYPE = 9
    _JUMP_BALL_TYPE = 10
    _EJECTION_TYPE = 11
    _PERIOD_BEGIN_TYPE = 12
    _PERIOD_END_TYPE = 13

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw PBP into a flat historical possession log."""
        if raw_df.empty:
            return pd.DataFrame()

        required_cols = {
            "GAME_ID",
            "PERIOD",
            "PCTIMESTRING",
            "EVENTMSGTYPE",
            "SCORE",
            "HOMEDESCRIPTION",
            "VISITORDESCRIPTION",
        }
        missing = required_cols - set(raw_df.columns)
        if missing:
            raise ValueError(f"Raw DataFrame missing columns: {missing}")

        raw_df = raw_df.copy()
        if "EVENTNUM" not in raw_df.columns:
            raw_df["EVENTNUM"] = np.arange(len(raw_df))
        raw_df = raw_df.sort_values(
            ["GAME_ID", "PERIOD", "EVENTNUM"],
            kind="stable",
        ).reset_index(drop=True)
        raw_df = self._add_seconds_remaining(raw_df)
        raw_df = self._add_score_columns(raw_df)

        rows: List[Dict] = []
        # Group by the FULL raw_df to trace chronological possession perfectly
        for game_id, game_df in raw_df.groupby("GAME_ID"):
            # Skip games that don't have any late-game data
            has_late_game = (
                (game_df["PERIOD"] >= 4)
                & (game_df["seconds_remaining"] <= FINAL_PERIOD_SECONDS)
            ).any()
            if not has_late_game:
                continue

            game_outcome = self._compute_game_outcome(game_df)
            if "SEASON" in raw_df.columns:
                season = str(raw_df.loc[raw_df["GAME_ID"] == game_id, "SEASON"].iloc[0])
            else:
                season = _season_from_game_id(str(game_id))

            game_rows = self._process_game(str(game_id), season, game_outcome, game_df)
            rows.extend(game_rows)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        result["action_team"] = result["action_team"].astype("Int8")
        out_path = self.processed_dir / "transitions.parquet"
        result.to_parquet(out_path, index=False)
        logger.info("Saved %d possession records to %s", len(result), out_path)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _add_seconds_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        df["seconds_remaining"] = df["PCTIMESTRING"].apply(_period_clock_to_seconds)
        return df

    def _add_score_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse 'VVV - HHH' SCORE strings into away/home integers."""

        def parse_score(s):
            if pd.isna(s) or str(s).strip() == "":
                return np.nan, np.nan
            parts = str(s).split(" - ")
            if len(parts) != 2:
                return np.nan, np.nan
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return np.nan, np.nan

        scores = df["SCORE"].apply(parse_score)
        df["away_score"] = [s[0] for s in scores]
        df["home_score"] = [s[1] for s in scores]
        df["home_score"] = df.groupby("GAME_ID")["home_score"].ffill()
        df["away_score"] = df.groupby("GAME_ID")["away_score"].ffill()
        df["home_score"] = df["home_score"].fillna(0)
        df["away_score"] = df["away_score"].fillna(0)
        df["score_differential"] = (
            (df["home_score"] - df["away_score"])
            .clip(-SCORE_DIFF_CLIP, SCORE_DIFF_CLIP)
            .astype(int)
        )
        return df

    def _compute_game_outcome(self, game_df: pd.DataFrame) -> int:
        """
        Return 1 if the home team won, 0 if the away team won.

        Uses the latest tracked score from the final regulation or overtime
        period as a proxy for the final game score.
        """
        final_period = game_df["PERIOD"].max()
        final_period_df = game_df.loc[game_df["PERIOD"] == final_period]
        final_seconds = final_period_df["seconds_remaining"].min()
        final_row = final_period_df.loc[
            final_period_df["seconds_remaining"] == final_seconds
        ].iloc[-1]
        final_diff = int(final_row.get("score_differential", 0))
        return 1 if final_diff > 0 else 0

    @staticmethod
    def _normalize_team_id(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).removesuffix(".0")

    @staticmethod
    def _description(value) -> str:
        return "" if pd.isna(value) else str(value)

    def _infer_action_team(self, row: pd.Series) -> Optional[int]:
        player_team = self._normalize_team_id(row.get("PLAYER1_TEAM_ID"))
        home_team = self._normalize_team_id(row.get("HOME_TEAM_ID"))
        away_team = self._normalize_team_id(row.get("AWAY_TEAM_ID"))
        if player_team and player_team == home_team:
            return 1
        if player_team and player_team == away_team:
            return 0

        home_desc = self._description(row.get("HOMEDESCRIPTION")).strip()
        away_desc = self._description(row.get("VISITORDESCRIPTION")).strip()
        if home_desc and not away_desc:
            return 1
        if away_desc and not home_desc:
            return 0
        return None

    def _possession_before_event(
        self,
        row: pd.Series,
        current_possession: int,
        action_team: Optional[int],
    ) -> int:
        if action_team is None:
            return current_possession

        etype = int(row.get("EVENTMSGTYPE", 0))
        if etype in {
            self._SHOT_MADE_TYPE,
            self._SHOT_MISSED_TYPE,
            self._FREE_THROW_TYPE,
            self._TURNOVER_TYPE,
            self._TIMEOUT_TYPE,
        }:
            return action_team
        if etype == self._FOUL_TYPE:
            desc = (
                self._description(row.get("HOMEDESCRIPTION"))
                + self._description(row.get("VISITORDESCRIPTION"))
            ).upper()
            return action_team if "OFFENSIVE" in desc else 1 - action_team
        return current_possession

    def _infer_possession(
        self,
        row: pd.Series,
        current_possession: int,
        action_team: Optional[int],
    ) -> int:
        """Heuristic: determine which team has the ball after this event."""
        etype = int(row.get("EVENTMSGTYPE", 0))
        home_desc = self._description(row.get("HOMEDESCRIPTION")).upper()
        away_desc = self._description(row.get("VISITORDESCRIPTION")).upper()

        if etype == self._SHOT_MADE_TYPE:
            return 1 - action_team if action_team is not None else 1 - current_possession
        if etype == self._SHOT_MISSED_TYPE:
            return action_team if action_team is not None else current_possession
        if etype == self._REBOUND_TYPE:
            if action_team is not None:
                return action_team
            if home_desc:
                return 1  # home rebound → home possession
            if away_desc:
                return 0
            return current_possession
        if etype == self._TURNOVER_TYPE:
            return 1 - action_team if action_team is not None else 1 - current_possession
        if etype == self._FREE_THROW_TYPE:
            desc = home_desc + away_desc
            if "1 OF 1" in desc or "2 OF 2" in desc or "3 OF 3" in desc:
                if "MISS" in desc:
                    return action_team if action_team is not None else current_possession
                return (
                    1 - action_team
                    if action_team is not None
                    else 1 - current_possession
                )
        if etype == self._VIOLATION_TYPE:
            desc = home_desc + away_desc
            # Kicked ball and defensive violations retain possession.
            if "KICKED BALL" in desc or "DEFENSIVE" in desc:
                return current_possession
            return 1 - current_possession

        return current_possession

    def _classify_action(self, row: pd.Series) -> str:
        """Map an event type to a strategic action label."""
        etype = int(row.get("EVENTMSGTYPE", 0))
        mapping = {
            self._SHOT_MADE_TYPE: "shoot",
            self._SHOT_MISSED_TYPE: "shoot",
            self._FREE_THROW_TYPE: "free_throw",
            self._REBOUND_TYPE: "rebound",
            self._TURNOVER_TYPE: "turnover",
            self._FOUL_TYPE: "foul",
            self._TIMEOUT_TYPE: "timeout",
        }
        return mapping.get(etype, "other")

    def _process_game(
        self,
        game_id: str,
        season: str,
        game_outcome: int,
        df: pd.DataFrame,
    ) -> List[Dict]:
        rows: List[Dict] = []
        possession = (
            1  # Start with home team (will self-correct within seconds of tip-off)
        )
        events = df.sort_values(["PERIOD", "EVENTNUM"], kind="stable").reset_index(
            drop=True
        )

        for _, row in events.iterrows():
            period = int(row["PERIOD"])
            sec = int(row["seconds_remaining"])
            score_diff = int(row["score_differential"])
            action_taken = self._classify_action(row)
            action_team = self._infer_action_team(row)
            event_possession = self._possession_before_event(
                row,
                possession,
                action_team,
            )

            # Update state variables chronologically for the entire game
            next_possession = self._infer_possession(
                row,
                event_possession,
                action_team,
            )
            # Only record the possession if it falls into our target time window
            if period >= 4 and sec <= FINAL_PERIOD_SECONDS:
                rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "period": period,
                        "event_num": int(row["EVENTNUM"]),
                        "seconds_remaining": sec,
                        "score_differential": score_diff,
                        "possession": event_possession,
                        "action_taken": action_taken,
                        "action_team": action_team,
                        "game_outcome": game_outcome,
                    }
                )

            possession = next_possession

        return rows


# ---------------------------------------------------------------------------
# Convenience function for testing / offline development
# ---------------------------------------------------------------------------
def build_synthetic_transitions(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic historical possession log with realistic NBA distributions.

    Used for unit tests and offline development when the live API is unavailable.
    The game_outcome is assigned deterministically based on the score_differential
    at the time of the event (positive → home win).
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    score_diffs = rng.integers(-10, 11, size=n)
    seconds = rng.integers(0, 181, size=n)
    possessions = rng.integers(0, 2, size=n)
    actions = rng.choice(
        [
            "shoot",
            "shoot",
            "free_throw",
            "rebound",
            "turnover",
            "foul",
            "timeout",
            "other",
        ],
        size=n,
        p=[0.22, 0.28, 0.15, 0.15, 0.08, 0.07, 0.03, 0.02],
    )
    seasons = rng.choice(
        ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"],
        size=n,
    )
    game_outcomes = np.where(score_diffs > 0, 1, 0)
    action_teams = possessions.copy()
    action_teams[actions == "foul"] = 1 - possessions[actions == "foul"]

    return pd.DataFrame(
        {
            "game_id": [f"synthetic_{i // 50}" for i in range(n)],
            "season": seasons,
            "period": np.full(n, 4),
            "event_num": np.arange(n),
            "seconds_remaining": seconds,
            "score_differential": score_diffs,
            "possession": possessions,
            "action_taken": actions,
            "action_team": action_teams,
            "game_outcome": game_outcomes,
        }
    )
