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

This project is built as a standard, production-ready Python package. 

**1. Clone the repository**
```bash
the usual
```

**2. Install the package and dependencies**
It is recommended to use a virtual environment. The installation includes development tools like `pytest` and `mkdocs`.

```bash
pip install -e ".[dev]"

```

---

## 💻 How to Run the Code

The repository is modularized into data ingestion, MDP solving, and visualization generation.

**Generate the Research Visualizations**
The quickest way to see the engine in action is to run the visualization script. This initializes the MDP solvers for the specific theorems, runs the value iteration over the state space, and outputs the resulting heatmaps and EV curves to the `docs/assets/images/` directory.

```bash
python -m src.visualizations

```

**Run the Test Suite**
The codebase includes a comprehensive test suite using synthetic data generation to ensure the transition matrices and MDP solvers behave deterministically.

```bash
pytest tests/ -v

```

**Run the Local Documentation Server**
If you want to view or edit the MkDocs website locally before deploying:

```bash
mkdocs serve

```

Navigate to `http://127.0.0.1:8000/` in your browser to view the site.

---

## 🏗️ Repository Structure

* `src/data_pipeline.py`: Object-oriented scraper using `nba_api`, featuring exponential backoff, caching, and a parser to convert raw event strings into discrete MDP state transitions.
* `src/mdp_engine.py`: The mathematical core. Implements a finite-horizon MDP solver using backward induction and defines the canonical simulation harnesses for the theorems.
* `src/visualizations.py`: Output generators using `matplotlib` and `seaborn` to create publication-ready decision boundary plots.
* `tests/`: Unit tests for the data pipeline, transition builders, and Bellman solvers.
* `docs/`: The markdown source files for the MkDocs static website.


