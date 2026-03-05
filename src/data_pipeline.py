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
    seconds_remaining   : int     -- seconds left in the period [0, 180]
    score_differential  : int     -- home minus away score (clipped to ±30)
    possession          : int     -- 0 = away team, 1 = home team
    fouls_to_give       : int     -- [0, 2]
    action_taken        : string  -- "shoot", "foul", "timeout", "turnover",
                                    "rebound", "free_throw", or "other"
    game_outcome        : int     -- 1 = home team won, 0 = away team won
    opponent_fg3_pct    : float   -- away (visiting) team's 3PT FG% for that
                                    game, derived from play-by-play events
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
MAX_FOULS_TO_GIVE = 2
MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # exponential back-off multiplier


# ---------------------------------------------------------------------------
# Data-classes (kept for backwards compatibility)
# ---------------------------------------------------------------------------
@dataclass
class GameState:
    """Discrete state for a late-game possession."""

    score_differential: int  # home - away, clipped to ±SCORE_DIFF_CLIP
    seconds_remaining: int  # seconds left in the period [0, 180]
    possession: int  # 0 = away team, 1 = home team
    fouls_to_give: int  # [0, 2]

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (
            self.score_differential,
            self.seconds_remaining,
            self.possession,
            self.fouls_to_give,
        )

    @staticmethod
    def from_tuple(t: Tuple[int, int, int, int]) -> "GameState":
        return GameState(*t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _retry_with_backoff(fn, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Call *fn* with exponential back-off on exception."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = BACKOFF_BASE**attempt + 0.1 * attempt
            logger.warning(
                "Attempt %d failed (%s). Retrying in %.1fs…", attempt + 1, exc, wait
            )
            time.sleep(wait)


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
        "period": "PERIOD",
        "pctimestring": "PCTIMESTRING",
        "eventmsgtype": "EVENTMSGTYPE",
        "score": "SCORE",
        "homedescription": "HOMEDESCRIPTION",
        "visitordescription": "VISITORDESCRIPTION",
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
                "SELECT game_id, period, pctimestring, eventmsgtype, score, "
                "homedescription, visitordescription "
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
        combined_path = self.raw_dir / "all_seasons_pbp.parquet"
        if combined_path.exists():
            logger.info("Loading cached combined PBP from %s", combined_path)
            return pd.read_parquet(combined_path)

        import sqlite3

        db_path = self._get_db_path()

        prefixes = [
            self._SEASON_PREFIXES[s] for s in self.seasons if s in self._SEASON_PREFIXES
        ]
        if not prefixes:
            logger.warning("No valid season prefixes found; returning empty DataFrame.")
            return pd.DataFrame()

        where_clause = " OR ".join(["game_id LIKE ?" for _ in prefixes])
        params = [f"{p}%" for p in prefixes]

        query = (
            "SELECT game_id, period, pctimestring, eventmsgtype, score, "
            "homedescription, visitordescription "
            f"FROM play_by_play WHERE ({where_clause})"
        )

        conn = sqlite3.connect(str(db_path))
        try:
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

        if df.empty:
            logger.warning("No play-by-play data found for the specified seasons.")
            return pd.DataFrame()

        df = df.rename(columns=self._COLUMN_MAP)

        prefix_to_season = {v: k for k, v in self._SEASON_PREFIXES.items()}
        df["SEASON"] = df["GAME_ID"].str[:5].map(prefix_to_season).fillna("")

        df.to_parquet(combined_path, index=False)
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
        raw_df = self._add_seconds_remaining(raw_df)
        raw_df = self._add_score_columns(raw_df)

        fg3_pct_by_game: Dict[str, float] = {}
        for game_id, full_game_df in raw_df.groupby("GAME_ID"):
            visitor_desc = full_game_df["VISITORDESCRIPTION"].fillna("").str.upper()
            is_shot_made = full_game_df["EVENTMSGTYPE"] == self._SHOT_MADE_TYPE
            is_shot_missed = full_game_df["EVENTMSGTYPE"] == self._SHOT_MISSED_TYPE
            is_3pt = visitor_desc.str.contains("3PT", na=False)
            fg3_made = int((is_shot_made & is_3pt).sum())
            fg3_attempted = int(((is_shot_made | is_shot_missed) & is_3pt).sum())
            fg3_pct_by_game[str(game_id)] = (
                round(fg3_made / fg3_attempted, 4) if fg3_attempted > 0 else 0.33
            )

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

            opponent_fg3_pct = fg3_pct_by_game.get(str(game_id), 0.33)
            game_rows = self._process_game(
                str(game_id), season, game_outcome, game_df, opponent_fg3_pct
            )
            rows.extend(game_rows)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
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
        """Parse 'HHH - VVV' SCORE strings into home/away integers."""

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
        df["home_score"] = [s[0] for s in scores]
        df["away_score"] = [s[1] for s in scores]
        df["home_score"] = df["home_score"].ffill()
        df["away_score"] = df["away_score"].ffill()
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

        Uses the score at the minimum seconds_remaining row (the latest
        tracked event) as a proxy for the final game score.
        """
        final_row = game_df.loc[game_df["seconds_remaining"].idxmin()]
        final_diff = int(final_row.get("score_differential", 0))
        return 1 if final_diff > 0 else 0

    def _infer_possession(self, row: pd.Series, current_possession: int) -> int:
        """Heuristic: determine which team has the ball after this event."""
        etype = int(row.get("EVENTMSGTYPE", 0))
        home_desc = str(row.get("HOMEDESCRIPTION", "") or "").upper()
        away_desc = str(row.get("VISITORDESCRIPTION", "") or "").upper()

        if etype == self._SHOT_MADE_TYPE:
            return 1 - current_possession
        if etype == self._SHOT_MISSED_TYPE:
            return current_possession  # assume same team until rebound
        if etype == self._REBOUND_TYPE:
            if home_desc:
                return 1  # home rebound → home possession
            if away_desc:
                return 0
            return current_possession
        if etype == self._TURNOVER_TYPE:
            return 1 - current_possession
        if etype == self._FREE_THROW_TYPE:
            desc = home_desc + away_desc
            if "1 OF 1" in desc or "2 OF 2" in desc or "3 OF 3" in desc:
                if "MISS" in desc:
                    return current_possession
                return 1 - current_possession
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

    def _fouls_remaining(self, row: pd.Series, possession: int) -> int:
        """Very coarse approximation: foul count from description keywords."""
        desc = str(row.get("HOMEDESCRIPTION", "") or "") + str(
            row.get("VISITORDESCRIPTION", "") or ""
        )
        if "LOOSE BALL" in desc.upper() or "OFFENSIVE" in desc.upper():
            return min(MAX_FOULS_TO_GIVE, 2)
        return 1

    def _process_game(
        self,
        game_id: str,
        season: str,
        game_outcome: int,
        df: pd.DataFrame,
        opponent_fg3_pct: float,
    ) -> List[Dict]:
        rows: List[Dict] = []
        possession = (
            1  # Start with home team (will self-correct within seconds of tip-off)
        )
        fouls_to_give = 1

        # Sort chronologically: Period ascending, Seconds Remaining descending
        events = df.sort_values(
            ["PERIOD", "seconds_remaining"], ascending=[True, False]
        ).reset_index(drop=True)

        for _, row in events.iterrows():
            period = int(row["PERIOD"])
            sec = int(row["seconds_remaining"])
            score_diff = int(row["score_differential"])
            action_taken = self._classify_action(row)

            # Update state variables chronologically for the entire game
            next_possession = self._infer_possession(row, possession)
            next_fouls = self._fouls_remaining(row, possession)

            # Only record the possession if it falls into our target time window
            if period >= 4 and sec <= FINAL_PERIOD_SECONDS:
                rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "seconds_remaining": sec,
                        "score_differential": score_diff,
                        "possession": possession,
                        "fouls_to_give": fouls_to_give,
                        "action_taken": action_taken,
                        "game_outcome": game_outcome,
                        "opponent_fg3_pct": opponent_fg3_pct,
                    }
                )

            possession = next_possession
            fouls_to_give = next_fouls

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
    fouls = rng.integers(0, 3, size=n)
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
    opponent_fg3_pct = rng.uniform(0.25, 0.45, size=n).round(4)

    return pd.DataFrame(
        {
            "game_id": [f"synthetic_{i // 50}" for i in range(n)],
            "season": seasons,
            "seconds_remaining": seconds,
            "score_differential": score_diffs,
            "possession": possessions,
            "fouls_to_give": np.minimum(MAX_FOULS_TO_GIVE, fouls),
            "action_taken": actions,
            "game_outcome": game_outcomes,
            "opponent_fg3_pct": opponent_fg3_pct,
        }
    )


# ---------------------------------------------------------------------------
# Empirical parameter extraction (kept for backwards compatibility)
# ---------------------------------------------------------------------------
def compute_shooting_stats(transitions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic shooting statistics from a historical possession log.

    Returns
    -------
    dict with keys:
        ``shoot_rate`` -- fraction of events that are shot attempts
        ``foul_rate``  -- fraction of events that are fouls
    """
    if transitions_df.empty:
        return {}

    stats: Dict[str, float] = {}
    total = len(transitions_df)
    if total > 0:
        actions = transitions_df.get("action_taken", pd.Series(dtype=str))
        stats["shoot_rate"] = float((actions == "shoot").sum() / total)
        stats["foul_rate"] = float((actions == "foul").sum() / total)
    return stats


def load_empirical_params(processed_dir: Path = PROCESSED_DIR) -> Dict[str, float]:
    """
    Load ``transitions.parquet`` from *processed_dir* and return basic
    shooting statistics.

    Returns an empty dict if the file does not exist.
    """
    transitions_path = processed_dir / "transitions.parquet"
    if not transitions_path.exists():
        logger.debug("transitions.parquet not found; no empirical params available.")
        return {}
    try:
        df = pd.read_parquet(transitions_path)
        params = compute_shooting_stats(df)
        logger.info("Loaded empirical params from %s: %s", transitions_path, params)
        return params
    except Exception as exc:
        logger.warning(
            "Could not load empirical params from %s: %s", transitions_path, exc
        )
        return {}
