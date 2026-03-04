"""
data_pipeline.py
================
Scrapes NBA play-by-play data via the ``nba_api`` library, cleans and parses
each event, then converts the chronological log into a discrete MDP state
space and empirical transition-probability matrices.

State representation
--------------------
A game state is the 4-tuple:
    s = (score_differential, seconds_remaining, possession, fouls_to_give)

where
    score_differential  : integer in [-30, 30]  (home minus away, clipped)
    seconds_remaining   : integer in [0, 180]   (final 3 minutes only)
    possession          : 0 = away, 1 = home
    fouls_to_give       : integer in [0, 2]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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

FINAL_PERIOD_SECONDS = 180          # last 3 minutes of regulation / OT
SCORE_DIFF_CLIP = 30                # clip extreme leads for state compression
MAX_FOULS_TO_GIVE = 2
RATE_LIMIT_SLEEP = 0.65             # seconds between API calls
MAX_RETRIES = 5
BACKOFF_BASE = 2.0                  # exponential back-off multiplier


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class GameState:
    """Discrete MDP state for a late-game possession."""

    score_differential: int   # home − away, clipped to ±SCORE_DIFF_CLIP
    seconds_remaining: int    # seconds left in the period [0, 180]
    possession: int           # 0 = away team, 1 = home team
    fouls_to_give: int        # [0, 2]

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


@dataclass
class Transition:
    """A single observed (state, action) → next_state transition."""

    state: GameState
    action: str       # e.g. "shoot", "foul", "hold"
    next_state: GameState
    reward: float     # +1.0 = won game, −1.0 = lost game, 0.0 = in-progress


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
            wait = BACKOFF_BASE ** attempt + 0.1 * attempt
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


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------
class NBAPlayByPlayScraper:
    """
    Fetches play-by-play data from the NBA Stats API for a range of seasons.

    Parameters
    ----------
    seasons : list of str
        Season strings understood by nba_api, e.g. ``["2022-23", "2023-24"]``.
    raw_dir : Path
        Directory where raw Parquet files are cached.
    """

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
        """Return all regular-season game IDs for *season*."""
        from nba_api.stats.endpoints import leaguegamefinder

        logger.info("Fetching game IDs for season %s…", season)
        finder = _retry_with_backoff(
            leaguegamefinder.LeagueGameFinder,
            season_nullable=season,
            season_type_nullable="Regular Season",
        )
        time.sleep(RATE_LIMIT_SLEEP)
        df = finder.get_data_frames()[0]
        ids: List[str] = df["GAME_ID"].unique().tolist()
        logger.info("  → %d games found.", len(ids))
        return ids

    def fetch_play_by_play(self, game_id: str) -> pd.DataFrame:
        """Fetch raw play-by-play for a single game, with caching."""
        cache_path = self.raw_dir / f"{game_id}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        from nba_api.stats.endpoints import playbyplayv2

        pbp = _retry_with_backoff(
            playbyplayv2.PlayByPlayV2,
            game_id=game_id,
        )
        time.sleep(RATE_LIMIT_SLEEP)
        df = pbp.get_data_frames()[0]
        df.to_parquet(cache_path, index=False)
        return df

    def fetch_all(self) -> pd.DataFrame:
        """Fetch PBP for all configured seasons and return combined DataFrame."""
        combined_path = self.raw_dir / "all_seasons_pbp.parquet"
        if combined_path.exists():
            logger.info("Loading cached combined PBP from %s", combined_path)
            return pd.read_parquet(combined_path)

        frames: List[pd.DataFrame] = []
        for season in self.seasons:
            game_ids = self.fetch_season_game_ids(season)
            for gid in game_ids:
                try:
                    df = self.fetch_play_by_play(gid)
                    df["SEASON"] = season
                    frames.append(df)
                except Exception as exc:
                    logger.error("Failed to fetch game %s: %s", gid, exc)

        if not frames:
            logger.warning("No frames fetched; returning empty DataFrame.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(combined_path, index=False)
        return combined


# ---------------------------------------------------------------------------
# Parser / Feature Engineering
# ---------------------------------------------------------------------------
class PlayByPlayParser:
    """
    Converts raw PBP rows into (state, action, next_state, reward) transitions,
    keeping only events from the final ``FINAL_PERIOD_SECONDS`` seconds of the
    4th quarter or any overtime period.

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
        """
        Parse raw PBP into a tidy state-transition DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            game_id, period, seconds_remaining, score_differential,
            possession, fouls_to_give, action, next_score_diff,
            next_seconds_remaining, next_possession, next_fouls_to_give, reward
        """
        if raw_df.empty:
            return pd.DataFrame()

        required_cols = {
            "GAME_ID", "PERIOD", "PCTIMESTRING", "EVENTMSGTYPE",
            "SCORE", "HOMEDESCRIPTION", "VISITORDESCRIPTION",
        }
        missing = required_cols - set(raw_df.columns)
        if missing:
            raise ValueError(f"Raw DataFrame missing columns: {missing}")

        raw_df = raw_df.copy()
        raw_df = self._add_seconds_remaining(raw_df)
        raw_df = self._add_score_columns(raw_df)

        # Keep only final 3 minutes of 4th quarter or OT periods
        late_game = raw_df[
            (raw_df["PERIOD"] >= 4) & (raw_df["seconds_remaining"] <= FINAL_PERIOD_SECONDS)
        ].copy()

        if late_game.empty:
            return pd.DataFrame()

        rows: List[Dict] = []
        for game_id, game_df in late_game.groupby("GAME_ID"):
            game_rows = self._process_game(str(game_id), game_df)
            rows.extend(game_rows)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        out_path = self.processed_dir / "transitions.parquet"
        result.to_parquet(out_path, index=False)
        logger.info("Saved %d transitions to %s", len(result), out_path)
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
            df["home_score"] - df["away_score"]
        ).clip(-SCORE_DIFF_CLIP, SCORE_DIFF_CLIP).astype(int)
        return df

    def _infer_possession(self, row: pd.Series, current_possession: int) -> int:
        """Heuristic: determine which team has the ball after this event."""
        etype = int(row.get("EVENTMSGTYPE", 0))
        home_desc = str(row.get("HOMEDESCRIPTION", "") or "")
        away_desc = str(row.get("VISITORDESCRIPTION", "") or "")

        if etype == self._SHOT_MADE_TYPE:
            # Possession flips after a make
            return 1 - current_possession
        if etype == self._SHOT_MISSED_TYPE:
            # Possession changes if the *other* team rebounds (simplified heuristic)
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
            # If last free throw made, possession flips; missed → check rebound
            desc = home_desc + away_desc
            if "1 of 1" in desc or "2 of 2" in desc or "3 of 3" in desc:
                if "MISS" in desc.upper():
                    return current_possession
                return 1 - current_possession
        return current_possession

    def _classify_action(self, row: pd.Series) -> str:
        """Map an event type to a strategic action label."""
        etype = int(row.get("EVENTMSGTYPE", 0))
        mapping = {
            self._SHOT_MADE_TYPE: "shoot_make",
            self._SHOT_MISSED_TYPE: "shoot_miss",
            self._FREE_THROW_TYPE: "free_throw",
            self._REBOUND_TYPE: "rebound",
            self._TURNOVER_TYPE: "turnover",
            self._FOUL_TYPE: "foul",
            self._TIMEOUT_TYPE: "timeout",
            self._PERIOD_END_TYPE: "period_end",
        }
        return mapping.get(etype, "other")

    def _fouls_remaining(self, row: pd.Series, possession: int) -> int:
        """Very coarse approximation: foul count from description keywords."""
        desc = str(row.get("HOMEDESCRIPTION", "") or "") + str(
            row.get("VISITORDESCRIPTION", "") or ""
        )
        if "LOOSE BALL" in desc.upper() or "OFFENSIVE" in desc.upper():
            return min(MAX_FOULS_TO_GIVE, 2)
        # Hard-code a default; real implementation would track per-team fouls
        return 1

    def _process_game(self, game_id: str, df: pd.DataFrame) -> List[Dict]:
        rows: List[Dict] = []
        possession = 1  # start with home team (simplification)
        fouls_to_give = 1

        events = df.sort_values("seconds_remaining", ascending=False).reset_index(drop=True)

        for idx, row in events.iterrows():
            sec = int(row["seconds_remaining"])
            score_diff = int(row["score_differential"])
            action = self._classify_action(row)
            next_possession = self._infer_possession(row, possession)
            next_fouls = self._fouls_remaining(row, possession)

            # Look ahead for next state values
            if idx + 1 < len(events):
                next_row = events.iloc[idx + 1]
                next_sec = int(next_row["seconds_remaining"])
                next_diff = int(next_row["score_differential"])
            else:
                next_sec = 0
                next_diff = score_diff

            reward = 0.0
            if action == "period_end":
                reward = 1.0 if next_diff > 0 else (-1.0 if next_diff < 0 else 0.0)

            rows.append(
                {
                    "game_id": game_id,
                    "period": int(row.get("PERIOD", 4)),
                    "seconds_remaining": sec,
                    "score_differential": score_diff,
                    "possession": possession,
                    "fouls_to_give": fouls_to_give,
                    "action": action,
                    "next_score_diff": next_diff,
                    "next_seconds_remaining": next_sec,
                    "next_possession": next_possession,
                    "next_fouls_to_give": next_fouls,
                    "reward": reward,
                }
            )
            possession = next_possession
            fouls_to_give = next_fouls

        return rows


# ---------------------------------------------------------------------------
# Transition Matrix Builder
# ---------------------------------------------------------------------------
class TransitionMatrixBuilder:
    """
    Builds empirical transition probability matrices P(s'|s,a) from a
    tidy transitions DataFrame.

    The state space is a grid of all combinations of:
        score_differential  : {-10, …, 10}  (condensed range for tractability)
        seconds_remaining   : {0, 5, 10, …, 180}  (binned to 5-second slots)
        possession          : {0, 1}
        fouls_to_give       : {0, 1, 2}
    """

    SCORE_RANGE = list(range(-10, 11))           # 21 values
    TIME_BINS   = list(range(0, 185, 5))          # 0, 5, …, 180  (37 values)
    POSSESSIONS = [0, 1]
    FOULS_RANGE = [0, 1, 2]

    def __init__(self) -> None:
        self._state_index: Dict[Tuple, int] = {}
        self._all_states: List[Tuple] = []
        self._build_state_index()

    def _build_state_index(self) -> None:
        idx = 0
        for sd in self.SCORE_RANGE:
            for sec in self.TIME_BINS:
                for pos in self.POSSESSIONS:
                    for ftg in self.FOULS_RANGE:
                        key = (sd, sec, pos, ftg)
                        self._state_index[key] = idx
                        self._all_states.append(key)
                        idx += 1

    @property
    def n_states(self) -> int:
        return len(self._all_states)

    def _bin_seconds(self, sec: int) -> int:
        sec = max(0, min(sec, 180))
        return int(round(sec / 5) * 5)

    def _clip_score(self, sd: int) -> int:
        return max(-10, min(10, sd))

    def _state_key(self, row: pd.Series, prefix: str = "") -> Tuple:
        sd = self._clip_score(int(row[f"{prefix}score_diff" if prefix else "score_differential"]))
        sec = self._bin_seconds(
            int(row[f"{prefix}seconds_remaining"])
        )
        pos = int(row[f"{prefix}possession"])
        ftg = min(MAX_FOULS_TO_GIVE, int(row[f"{prefix}fouls_to_give"]))
        return (sd, sec, pos, ftg)

    def build(
        self, transitions_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Build transition matrices, one per action.

        Returns
        -------
        dict mapping action_name → np.ndarray of shape (n_states, n_states),
        where entry [i, j] is the empirical probability of transitioning from
        state i to state j under that action.
        """
        if transitions_df.empty:
            return {}

        actions = transitions_df["action"].unique().tolist()
        counts: Dict[str, np.ndarray] = {
            a: np.zeros((self.n_states, self.n_states), dtype=np.float64)
            for a in actions
        }

        for _, row in transitions_df.iterrows():
            action = str(row["action"])
            s_key = (
                self._clip_score(int(row["score_differential"])),
                self._bin_seconds(int(row["seconds_remaining"])),
                int(row["possession"]) % 2,
                min(MAX_FOULS_TO_GIVE, int(row["fouls_to_give"])),
            )
            sp_key = (
                self._clip_score(int(row["next_score_diff"])),
                self._bin_seconds(int(row["next_seconds_remaining"])),
                int(row["next_possession"]) % 2,
                min(MAX_FOULS_TO_GIVE, int(row["next_fouls_to_give"])),
            )
            if s_key in self._state_index and sp_key in self._state_index:
                i = self._state_index[s_key]
                j = self._state_index[sp_key]
                counts[action][i, j] += 1

        # Normalise rows to probabilities
        matrices: Dict[str, np.ndarray] = {}
        for action, mat in counts.items():
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid division by zero
            matrices[action] = mat / row_sums

        return matrices

    def state_index(self) -> Dict[Tuple, int]:
        return dict(self._state_index)

    def all_states(self) -> List[Tuple]:
        return list(self._all_states)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def build_synthetic_transitions(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic transitions DataFrame with realistic NBA distributions.

    Used for unit tests and offline development when the live API is unavailable.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    score_diffs = rng.integers(-10, 11, size=n)
    seconds = rng.integers(0, 181, size=n)
    possessions = rng.integers(0, 2, size=n)
    fouls = rng.integers(0, 3, size=n)
    actions = rng.choice(
        ["shoot_make", "shoot_miss", "free_throw", "rebound", "turnover", "foul", "timeout"],
        size=n,
        p=[0.22, 0.28, 0.15, 0.15, 0.08, 0.07, 0.05],
    )

    # Next state
    delta_score = np.where(
        actions == "shoot_make",
        rng.choice([2, 3], size=n, p=[0.7, 0.3]),
        0,
    )
    next_score_diffs = np.clip(score_diffs + delta_score, -10, 10)
    next_seconds = np.maximum(0, seconds - rng.integers(4, 25, size=n))
    next_possessions = np.where(
        np.isin(actions, ["shoot_make", "turnover"]),
        1 - possessions,
        possessions,
    )
    next_fouls = np.minimum(MAX_FOULS_TO_GIVE, fouls)
    rewards = np.where(seconds == 0, np.sign(score_diffs).astype(float), 0.0)

    return pd.DataFrame(
        {
            "game_id": [f"synthetic_{i}" for i in range(n)],
            "period": 4,
            "seconds_remaining": seconds,
            "score_differential": score_diffs,
            "possession": possessions,
            "fouls_to_give": fouls,
            "action": actions,
            "next_score_diff": next_score_diffs,
            "next_seconds_remaining": next_seconds,
            "next_possession": next_possessions,
            "next_fouls_to_give": next_fouls,
            "reward": rewards,
        }
    )


# ---------------------------------------------------------------------------
# Empirical parameter extraction
# ---------------------------------------------------------------------------
def compute_shooting_stats(transitions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute empirical shooting and possession statistics from a transitions DataFrame.

    Parameters
    ----------
    transitions_df : pd.DataFrame
        Output of :class:`PlayByPlayParser` (or :func:`build_synthetic_transitions`).

    Returns
    -------
    dict with keys:
        ``fg_pct``        – overall field-goal percentage (makes / attempts)
        ``ft_pct``        – free-throw percentage (estimated from FT events)
        ``turnover_prob`` – turnover probability per possession
    """
    if transitions_df.empty:
        return {}

    stats: Dict[str, float] = {}

    actions = transitions_df["action"]
    makes  = (actions == "shoot_make").sum()
    misses = (actions == "shoot_miss").sum()
    shots  = makes + misses
    if shots > 0:
        stats["fg_pct"] = float(makes / shots)

    turnovers = (actions == "turnover").sum()
    possessions = shots + turnovers + (actions == "foul").sum()
    if possessions > 0:
        stats["turnover_prob"] = float(turnovers / possessions)

    return stats


def load_empirical_params(processed_dir: Path = PROCESSED_DIR) -> Dict[str, float]:
    """
    Load ``transitions.parquet`` from *processed_dir* and return empirical
    shooting statistics suitable for passing to
    :meth:`~src.mdp_engine.TransitionModel.from_data`.

    Returns an empty dict if the file does not exist (callers should fall back
    to hardcoded defaults).
    """
    transitions_path = processed_dir / "transitions.parquet"
    if not transitions_path.exists():
        logger.debug("transitions.parquet not found; using default model parameters.")
        return {}
    try:
        df = pd.read_parquet(transitions_path)
        params = compute_shooting_stats(df)
        logger.info("Loaded empirical params from %s: %s", transitions_path, params)
        return params
    except Exception as exc:
        logger.warning("Could not load empirical params from %s: %s", transitions_path, exc)
        return {}

