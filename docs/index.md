# NBA Folk Theorems

> *We talking about analytics?*

## Introduction

**NBA Folk Theorems** is an empirical research project that examines the final three
minutes of NBA games to *quantitatively* test common basketball strategies coaches
and analysts have debated for decades.
We call each one of these strategies a "theorem" — a statement about the optimal strategic choice in a given game situation.

We use five seasons of NBA play-by-play data (2019--24) and to compute actual historical win percentages for each strategic choice.

Conclusions are based strictly on what *actually happened* in real games.

---

## How It Works

### Data Collection

Based on Wyatt Walsh's [NBA Database](https://www.kaggle.com/datasets/wyattowalsh/basketball).
For every play-by-play event in the final 3 minutes of the 4th quarter (or overtime),
we record:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season` | NBA season (e.g. 2023-24) |
| `seconds_remaining` | Seconds left in the period |
| `score_differential` | Home minus away score |
| `possession` | 0 = away team, 1 = home team |
| `fouls_to_give` | Fouls available before bonus |
| `action_taken` | Strategic action (shoot, foul, hold, …) |
| `game_outcome` | 1 = home team won, 0 = away team won |

### Analysis Method

Each theorem targets a specific late-game decision point.  We evaluate it by:

1. **Filter** — restrict `transitions.parquet` to the exact game situation
   (score margin, possession, time window, fouls-to-give) that defines the
   theorem's scenario.
2. **Temporal join** — for each target clock value *t*, apply a backward-forward
   temporal join (`merge_asof`) on each game independently:
   - *Look backward*: record the most recent game state (score, possession,
     opponent 3PT%) at or just before *t*.
   - *Look forward*: record the first action taken at or after *t*.
   This avoids the survivorship bias of a simple window filter, which silently
   drops possessions where the clock ran without any logged play-by-play event.
3. **Split by action** — separate possessions into the two (or more) strategic
   choices being compared (e.g., *rush* vs. *hold*, *foul* vs. *no-foul*,
   *timeout* vs. *no timeout*).
4. **Win-rate estimate** — compute `mean(game_outcome)` for each action group,
   yielding an empirical home-team win percentage.  When a bucket has no
   observations the win rate defaults to 0.50 (coin-flip prior).
5. **Compare** — report the win-percentage *gain* (action A minus action B) at
   each time value and identify when, if ever, one strategy is consistently
   better.

---

## Reproducing the Analysis

```bash
# Install dependencies
uv sync

# Run tests
pytest tests/ -v

# Download play-by-play data (requires Kaggle API key, only needs to be done once)
python -m src.scrape_nba_data

# Aggregate historical win rates (requires transitions.parquet)
python -m src.collect_data

# Generate plots
python -m src.generate_plots

# Build documentation
zensical serve
```
