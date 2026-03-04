# NBA Folk Theorems

> *"The plural of anecdote is not data — but the plural of possession is history."*

## Introduction

**NBA Folk Theorems** is an empirical research project that investigates the final three
minutes of NBA games to *quantitatively* prove or disprove common basketball strategies
that coaches and analysts have debated for decades.

We scrape five seasons of NBA play-by-play data (2019–24), convert the raw event log
into a flat historical possession log, and use **pandas aggregations** to compute actual
historical win percentages for each strategic choice.

Conclusions are based strictly on what *actually happened* in real games — not on
theoretical probability models.

---

## How It Works

### Data Collection

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

1. **Filter** the historical log to the relevant game situation (e.g., tied games with 24–40 s left).
2. **Group** possessions by the strategic action taken (e.g., `shoot` vs. other).
3. **Aggregate** — compute `mean(game_outcome)` for each group, which is the historical win percentage for the home team.

Conclusions reflect what percentage of the time teams *won* after making each choice.

### Shooting Statistics (2019–24 Reference)

| Statistic | Value |
|-----------|-------|
| 2PT field-goal % | 52 % |
| 3PT field-goal % | 36 % |
| Free-throw % | 77 % |
| Turnover rate | 12 % |

---

## Repository Structure

```
Hardwood-Heuristics/
├── src/
│   ├── data_pipeline.py     # scraper, parser, flat possession log builder
│   ├── collect_data.py      # pandas groupby aggregations for each theorem
│   └── visualizations.py   # publication-ready historical win % plots
├── tests/                   # pytest unit tests
├── data/
│   ├── raw/                 # cached per-game Parquet files
│   └── processed/           # tidy possession log and aggregated results
├── notebooks/               # exploratory Jupyter notebooks
├── docs/                    # this site (built by MkDocs)
│   └── assets/images/       # generated plots
├── mkdocs.yml
└── pyproject.toml
```

---

## Theorems Investigated

| # | Name | Claim |
|---|------|-------|
| [1](theorem1_two_for_one.md) | The 2-for-1 | Rushing a shot with ~32 s left to gain an extra possession historically increases win rate |
| [2](theorem2_foul_up_3.md)   | Foul Up 3   | Intentionally fouling when leading by 3 with < 12 s left historically increases win rate vs. normal defence |

---

## Getting Started

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Aggregate historical win rates (requires transitions.parquet)
python -m src.collect_data

# Generate plots
python -m src.visualizations

# Build documentation
mkdocs serve
```
