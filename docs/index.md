# NBA Folk Theorems

> *"The plural of anecdote is not data — but the plural of possession is a Markov chain."*

## Introduction

**NBA Folk Theorems** is an empirical research project that models the final three
minutes of NBA games as a **Markov Decision Process (MDP)** in order to
*quantitatively* prove or disprove common basketball strategies that coaches and
analysts have debated for decades.

We scrape five seasons of NBA play-by-play data (2019–24) via the official
`nba_api` library, convert the raw event log into a discrete state space, and
solve the resulting MDP with **backward induction** to compute optimal policies
and expected win probabilities.

---

## Mathematical Framework

### State Space

A game state is the 4-tuple:

$$s = (\Delta, \tau, \pi, f)$$

where:

| Symbol | Domain | Description |
|--------|--------|-------------|
| $\Delta$ | $\mathbb{Z} \cap [-10, 10]$ | Score differential (home − away) |
| $\tau$ | $\{0, 5, 10, \ldots, 180\}$ | Seconds remaining (5-second bins) |
| $\pi$ | $\{0, 1\}$ | Possession (0 = away, 1 = home) |
| $f$ | $\{0, 1, 2\}$ | Fouls to give |

### Action Space

At each state the decision-maker chooses from three abstract actions:

$$\mathcal{A} = \{\texttt{shoot},\ \texttt{foul},\ \texttt{hold}\}$$

### Transition Probabilities

The empirical transition probability $P(s' \mid s, a)$ is estimated by counting
the observed frequencies in the play-by-play corpus:

$$\hat{P}(s' \mid s, a) = \frac{N(s, a, s')}{\sum_{s''} N(s, a, s'')}$$

For states with insufficient data we fall back to an **analytic model**
calibrated to league-average shooting percentages (2019–24):

| Statistic | Value |
|-----------|-------|
| 2PT field-goal % | 52 % |
| 3PT field-goal % | 36 % |
| Free-throw % | 77 % |
| Turnover rate | 12 % |

### Bellman Equation (Finite Horizon)

Because the game has a hard time limit, we solve a **finite-horizon MDP** using
**backward induction**. Let $T = 0$ be the terminal (clock-zero) time step:

$$V^*_T(s) = R_{\text{terminal}}(s) = \begin{cases} +1 & \Delta > 0 \text{ (home win)} \\ -1 & \Delta < 0 \text{ (away win)} \\ 0 & \Delta = 0 \text{ (OT)} \end{cases}$$

For earlier time steps, the home team maximises and the away team minimises:

$$V^*_t(s) = \begin{cases}
\displaystyle\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\bigl[R(s,a,s') + V^*_{t+1}(s')\bigr] & \text{if } \pi = 1 \\[10pt]
\displaystyle\min_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\bigl[R(s,a,s') + V^*_{t+1}(s')\bigr] & \text{if } \pi = 0
\end{cases}$$

### Optimal Policy

The optimal policy $\pi^*$ selects the action that achieves the extremum:

$$\pi^*(s) = \arg\!\max_{a} \sum_{s'} P(s' \mid s, a)\bigl[R(s,a,s') + V^*_{t+1}(s')\bigr] \quad (\pi = 1)$$

---

## Repository Structure

```
Hardwood-Heuristics/
├── src/
│   ├── data_pipeline.py     # scraper, parser, transition-matrix builder
│   ├── mdp_engine.py        # backward-induction solver + theorem simulators
│   └── visualizations.py   # publication-ready plots
├── tests/                   # pytest unit tests
├── data/
│   ├── raw/                 # cached per-game Parquet files
│   └── processed/           # tidy transition tables
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
| [1](theorem1_two_for_one.md) | The 2-for-1 | Rushing a shot with ~32 s left to gain an extra possession increases expected win probability |
| [2](theorem2_foul_up_3.md)   | Foul Up 3   | Intentionally fouling when leading by 3 with < 10 s left increases win probability versus playing normal defence |

---

## Getting Started

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate plots
python -m src.visualizations

# Build documentation
mkdocs serve
```
