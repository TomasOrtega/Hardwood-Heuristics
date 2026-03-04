# Hardwood Heuristics

**Evaluating NBA folk theorems using Markov Decision Processes and empirical EV optimization.**

While basketball broadcasts are full of unverified rules of thumb ("always foul up 3", "push for the 2-for-1"), this project treats late-game basketball strategy as a formal verification problem. 

By modeling the final three minutes of an NBA game as a finite-horizon Markov Decision Process (MDP), this repository provides an empirical mathematical engine to prove or disprove conventional sports wisdom using expected value (EV) optimization and backward induction.

---

## 📖 Full Documentation & Research Findings

The full methodology, Bellman equations, state-space definitions, and interactive visualizations are hosted on our GitHub Pages site:

**[Read the Hardwood Heuristics Documentation Here](https://hardwood-heuristics.github.io/Hardwood-Heuristics/)**

---

## 🚀 Quick Start & Installation

This project is built as a standard, production-ready Python package and uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Clone the repository**
```bash
the usual
```

**2. Install uv** (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Install the package and dependencies**
```bash
uv sync --extra dev
```

---

## 💻 How to Run the Code

The repository is modularized into data collection, MDP solving, and visualization generation.
Data is collected **once** and saved to `data/processed/`, so subsequent visualization runs are fast.

**Step 1 – Collect and save data (run once)**
```bash
uv run python -m src.collect_data
```
This computes the MDP sweep results and writes them to `data/processed/`.

**Step 2 – Generate the Research Visualizations**
```bash
uv run python -m src.visualizations
```
Loads the pre-saved data and outputs heatmaps and EV curves to `docs/assets/images/`.

**Run the Test Suite**
```bash
uv run pytest tests/ -v
```

**Run the Local Documentation Server**
```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs serve
```
Navigate to `http://127.0.0.1:8000/` in your browser to view the site.

---

## 🏗️ Repository Structure

* `src/collect_data.py`: One-time data-collection script. Runs the MDP sweeps and saves results to `data/processed/`.
* `src/data_pipeline.py`: Object-oriented scraper using `nba_api`, featuring exponential backoff, caching, and a parser to convert raw event strings into discrete MDP state transitions.
* `src/mdp_engine.py`: The mathematical core. Implements a finite-horizon MDP solver using backward induction and defines the canonical simulation harnesses for the theorems.
* `src/visualizations.py`: Output generators using `matplotlib` and `seaborn` to create publication-ready decision boundary plots. Loads pre-saved data from `data/processed/` when available.
* `tests/`: Unit tests for the data pipeline, transition builders, and Bellman solvers.
* `docs/`: The markdown source files for the MkDocs static website.

