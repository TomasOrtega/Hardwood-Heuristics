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

For each theorem we:

1. **Filter** the historical log to the relevant game situation.
2. **Group** possessions by the strategic action taken.
3. **Aggregate** — compute `mean(game_outcome)` for each group (historical win %).

---

## Getting Started

```bash
# Install dependencies (pip)
pip install -e ".[dev]"

# Install dependencies (uv — faster)
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Aggregate historical win rates (requires transitions.parquet)
python -m src.collect_data

# Generate plots
python -m src.visualizations

# Build documentation
zensical serve
```
