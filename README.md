# Hardwood Heuristics

**Evaluating NBA folk theorems using historical play-by-play data and empirical win-rate analysis.**

While basketball broadcasts are full of unverified rules of thumb ("always foul up 3", "push for the 2-for-1"), this project treats late-game basketball strategy as a data analysis problem.

By aggregating five seasons of NBA play-by-play data (2019–24) and calculating actual historical win rates for each strategic choice, this repository provides an empirical framework to prove or disprove conventional sports wisdom — based strictly on what actually happened in real games.

---

## 📖 Full Documentation & Research Findings

The full methodology, data definitions, and interactive visualizations are hosted on our GitHub Pages site:

**[Read the Hardwood Heuristics Documentation Here](https://hardwood-heuristics.github.io/Hardwood-Heuristics/)**

---

## 🚀 Quick Start & Installation

This project is built as a standard, production-ready Python package and uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Clone the repository**
```bash
git clone https://github.com/TomasOrtega/Hardwood-Heuristics.git
cd Hardwood-Heuristics
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

The repository is modularized into data ingestion, historical aggregation, and visualization generation.
Data is collected **once** and saved to `data/processed/`, so subsequent visualization runs are fast.

**Step 1 – Load real NBA play-by-play data (run once)**
```bash
uv run python -m src.scrape_nba_data
```
Loads play-by-play data from the [Kaggle basketball dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball) for the last 5 seasons and saves
a flat historical possession log to `data/processed/transitions.parquet`.  The dataset
is read from `basketball.sqlite` in `data/raw/` and downloaded automatically
via the Kaggle API if absent (requires `KAGGLE_USERNAME` and `KAGGLE_KEY`
environment variables or a `~/.kaggle/kaggle.json` credentials file).

```bash
# Scrape specific seasons only
uv run python -m src.scrape_nba_data --seasons 2022-23 2023-24

# Dry-run: parse only locally cached raw files (no network calls)
uv run python -m src.scrape_nba_data --dry-run
```

**Step 2 – Aggregate historical win rates (run once)**
```bash
uv run python -m src.collect_data
```
Filters the historical log by game situation, groups by action taken, and calculates
actual win percentages for each strategy. Writes results to `data/processed/`.

**Step 3 – Generate the Research Visualizations**
```bash
uv run python -m src.visualizations
```
Loads the pre-saved data and outputs heatmaps and win-percentage curves to `docs/assets/images/`.

**Run the Test Suite**
```bash
uv run pytest tests/ -v
```

**Run the Local Documentation Server**

This project uses the [zensical](https://github.com/zensical/zensical) MkDocs theme. To start the local docs server:
```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs serve
```
Navigate to `http://127.0.0.1:8000/` in your browser to view the site.

To build the static site without serving:
```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs build
```
The generated site is written to the `site/` directory.

---

## 🏗️ Repository Structure

* `src/scrape_nba_data.py`: Standalone scraping script. Fetches real NBA play-by-play data, parses it into a flat historical possession log, and saves `data/processed/transitions.parquet`.
* `src/collect_data.py`: One-time aggregation script. Filters the historical log by game situation and computes historical win percentages using pandas `groupby`. Saves results to `data/processed/`.
* `src/data_pipeline.py`: Object-oriented data loader using the Kaggle basketball dataset, featuring caching and a parser to convert raw event strings into a flat historical possession log.
* `src/visualizations.py`: Output generators using `matplotlib` and `seaborn` to create publication-ready historical win-percentage plots. Loads pre-saved data from `data/processed/` when available.
* `tests/`: Unit tests for the data pipeline and visualization functions.
* `docs/`: The markdown source files for the MkDocs static website.

