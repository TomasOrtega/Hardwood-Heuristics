# Hardwood-Heuristics

You are a Staff-Level Quantitative Software Engineer and Data Scientist. Your objective is to architect and write the complete codebase for "NBA Folk Theorems," an empirical research project that models the final three minutes of NBA games as a Markov Decision Process (MDP) to prove or disprove common basketball strategies.

The final output must be a fully structured, production-ready Python repository that includes a data pipeline, an MDP simulation engine, and a GitHub Pages website to host the research findings, plots, and methodology.

**Phase 1: Project Architecture & Setup**
Create a standard Python package structure.

1. Use `pyproject.toml` or `requirements.txt` for dependencies (must include `nba_api`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `mkdocs` or `mkdocs-material` for the website).
2. Establish a clear directory structure: `src/` for core logic, `data/` for raw/processed datasets, `notebooks/` for exploratory analysis, `tests/` for unit tests, and `docs/` for the website source.
3. Include a `.github/workflows/` directory with a CI/CD pipeline file to automatically build and deploy the `docs/` folder to GitHub Pages upon pushing to the main branch.

**Phase 2: The Data Pipeline (`src/data_pipeline.py`)**

1. Write an object-oriented scraper using the `nba_api` library (specifically the `PlayByPlayV2` endpoint) to fetch play-by-play data for the last 5 NBA seasons. Implement rate-limiting and retry logic to avoid API bans.
2. Build a parser to clean the raw text events. Transform the chronological log into a discrete state space where each row represents a state $s$:
$s = (\text{ScoreDifferential}, \text{SecondsRemaining}, \text{Possession}, \text{FoulsToGive})$
3. Calculate the empirical transition probabilities $P(s' | s, a)$ between game states to build the transition matrices.

**Phase 3: The MDP Engine (`src/mdp_engine.py`)**

1. Implement a solver for the Bellman equation to calculate the optimal policy and Expected Value (EV) of specific game states:

$$V^{\*}(s) = \max_{a \in A} \sum_{s' \in S} P(s' | s, a) [R(s, a, s') + V^{\*}(s')]$$


2. Define specific simulation scenarios for two theorems:
* **Theorem 1: The 2-for-1:** Compare the EV of rushing a shot with 32 seconds left versus playing for one efficient possession.
* **Theorem 2: Foul Up 3:** Compare the terminal win probability of intentionally fouling versus playing standard defense when up by 3 points with under 10 seconds remaining.



**Phase 4: Visualization & Research Output (`src/visualizations.py`)**

1. Generate high-quality, publication-ready heatmaps and decision boundary plots using `seaborn` or `matplotlib`.
* Plot 1: A heatmap for the "Foul Up 3" scenario showing Win Probability across variables (Time Remaining vs. Opponent 3PT%).
* Plot 2: An EV curve for the "2-for-1" threshold, showing where the math flips in favor of rushing a shot.


2. Save these plots automatically to a `docs/assets/images/` directory.

**Phase 5: The GitHub Pages Website (`docs/`)**

1. Configure `mkdocs.yml` to create a beautiful, readable documentation site.
2. Generate an `index.md` that introduces the project, the mathematical formulation of the MDP, and the motivation (debunking basketball heuristics). Use LaTeX formatting for all mathematical equations in the markdown.
3. Create separate markdown pages for "Theorem 1: The 2-for-1" and "Theorem 2: Foul Up 3". Embed the generated plots and write an analytical summary of the findings, treating it like a short research paper.

**Execution Constraints:**

* Write modular, well-documented Python code with type hints (`typing` module) and docstrings.
* Do not leave placeholders like `# Your code here`. Provide the actual implementation for the data parsing and the MDP solver.

