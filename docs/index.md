# NBA Folk Theorems

> *We talking about analytics?*

## Introduction

**NBA Folk Theorems** is an empirical research project that examines the final three
minutes of NBA games to *quantitatively* test common basketball strategies coaches
and analysts have debated for decades.
We call each one of these strategies a "theorem" — a statement about the optimal strategic choice in a given game situation.

We use four seasons of NBA play-by-play data (2019--23) to compute historical
win percentages for each strategic choice.

The results describe what happened in these games. They are associations, not
causal estimates of what would have happened under a different coaching choice.

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
| `period` | Fourth quarter or overtime period |
| `event_num` | Source event sequence within the game |
| `seconds_remaining` | Seconds left in the period |
| `score_differential` | Home minus away score |
| `possession` | 0 = away team, 1 = home team |
| `action_taken` | Logged action (shoot, foul, timeout, …) |
| `action_team` | 0 = away team, 1 = home team |
| `game_outcome` | 1 = home team won, 0 = away team won |

### Analysis Method

Each theorem targets a specific late-game decision point.  We evaluate it by:

1. **Temporal join** — for each target clock value *t*, apply a backward-forward
   temporal join (`merge_asof`) independently to every game-period:
   - *Look backward*: record the score immediately before *t*.
   - *Look forward*: record the possessing team and first action at or after
     *t*.
   This avoids the survivorship bias of a simple window filter, which silently
   drops possessions when the clock runs without an event.
2. **Normalize perspective** — express the outcome from the decision-making
   team's perspective, allowing both home and away possessions to contribute.
3. **Filter and classify** — retain only the score, possession, action owner,
   and timing conditions that define each theorem's two strategies.
4. **Estimate** — compute the observed win percentage and sample count for each
   group. Empty groups remain missing; the analysis does not invent a prior.
5. **Compare** — report action A minus action B at each clock value. Because
   coaching choices are not randomized, these comparisons remain descriptive.

---

## Reproducing the Analysis

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Download play-by-play data (requires Kaggle API key, only needs to be done once)
uv run python -m src.scrape_nba_data

# Aggregate historical win rates (requires transitions.parquet)
uv run python -m src.collect_data

# Generate plots
uv run python -m src.generate_plots

# Build documentation
uv run zensical serve
```
