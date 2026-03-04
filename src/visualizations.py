"""
visualizations.py
=================
Generates publication-ready plots for the NBA Folk Theorems project.

Outputs are saved to ``docs/assets/images/``.

Plots
-----
* ``foul_up_3_heatmap.png``   – Win-probability gain (foul vs. no-foul) as a
  heatmap over Time Remaining × Opponent 3PT%.
* ``two_for_one_ev_curve.png`` – Expected-value gain from rushing a shot vs.
  taking a full possession across a range of seconds-remaining values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend for headless environments

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).parent.parent / "docs" / "assets" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Consistent aesthetics across all figures
PALETTE = "RdYlGn"
FIGURE_DPI = 150
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    }
)


# ---------------------------------------------------------------------------
# Plot 1: Foul-Up-3 heatmap
# ---------------------------------------------------------------------------
def plot_foul_up_3_heatmap(
    grid: np.ndarray,
    time_values: List[int],
    fg3_pct_values: List[float],
    out_path: Optional[Path] = None,
) -> Path:
    """
    Create a heatmap showing home-team win-probability *gain* from fouling
    intentionally when up by 3, as a function of time remaining and the
    opponent's 3-point shooting percentage.

    Parameters
    ----------
    grid          : 2-D array of shape (len(time_values), len(fg3_pct_values))
                    where each cell is WP_foul − WP_no_foul.
    time_values   : seconds-remaining labels for the y-axis (rows).
    fg3_pct_values: opponent 3PT% labels for the x-axis (columns).
    out_path      : destination file path (default: docs/assets/images/…).

    Returns
    -------
    Path to the saved PNG file.
    """
    if out_path is None:
        out_path = IMAGES_DIR / "foul_up_3_heatmap.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to percentage for display
    display_grid = grid * 100  # WP gain as percentage points

    x_labels = [f"{v:.0%}" for v in fg3_pct_values]
    y_labels = [f"{t}s" for t in time_values]

    sns.heatmap(
        display_grid,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=PALETTE,
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "WP Gain from Fouling (pp)", "shrink": 0.85},
    )

    ax.set_title(
        "Theorem 2: Foul Up 3\nWin-Probability Gain from Intentional Foul vs. Normal Defense",
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Opponent 3-Point Field Goal %", labelpad=10)
    ax.set_ylabel("Seconds Remaining", labelpad=10)

    # Annotate the decision boundary
    ax.text(
        0.5,
        -0.14,
        "Green = Fouling is better  |  Red = Normal defense is better  |  "
        "Values in percentage points",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved foul-up-3 heatmap to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: 2-for-1 EV curve
# ---------------------------------------------------------------------------
def plot_two_for_one_ev_curve(
    sweep_results: List[dict],
    out_path: Optional[Path] = None,
) -> Path:
    """
    Create an EV-gain curve showing the advantage of rushing a shot vs.
    taking a full possession at various seconds-remaining values.

    Parameters
    ----------
    sweep_results : list of dicts with keys:
                    ``seconds_remaining``, ``ev_rush``, ``ev_normal``, ``ev_gain``
    out_path      : destination file path.

    Returns
    -------
    Path to the saved PNG file.
    """
    if out_path is None:
        out_path = IMAGES_DIR / "two_for_one_ev_curve.png"

    seconds = [r["seconds_remaining"] for r in sweep_results]
    ev_rush   = [r["ev_rush"]   for r in sweep_results]
    ev_normal = [r["ev_normal"] for r in sweep_results]
    ev_gain   = [r["ev_gain"]   for r in sweep_results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: absolute EV lines ---
    ax1 = axes[0]
    ax1.plot(seconds, ev_rush,   color="#E63946", linewidth=2.2, label="Rush (shoot now)")
    ax1.plot(seconds, ev_normal, color="#457B9D", linewidth=2.2, label="Normal (full possession)")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_ylabel("Expected Win Probability (home)")
    ax1.set_title(
        "Theorem 1: The 2-for-1\nExpected Win Probability: Rush Shot vs. Full Possession",
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: EV gain ---
    ax2 = axes[1]
    gain_arr = np.array(ev_gain)
    colors = np.where(gain_arr >= 0, "#2DC653", "#E63946")
    ax2.bar(seconds, gain_arr, color=colors, width=1.6, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Seconds Remaining in Possession")
    ax2.set_ylabel("EV Gain from Rushing (pp)")
    ax2.set_title("EV Gain: Rush − Normal  (green = rushing is better)")
    ax2.grid(True, alpha=0.3, axis="y")

    # Mark the crossover point
    crossover_seconds: Optional[int] = None
    for i in range(1, len(ev_gain)):
        if ev_gain[i - 1] < 0 and ev_gain[i] >= 0:
            crossover_seconds = seconds[i]
            break
        if ev_gain[i - 1] >= 0 and ev_gain[i] < 0:
            crossover_seconds = seconds[i - 1]
            break

    if crossover_seconds is not None:
        ax2.axvline(
            crossover_seconds,
            color="black",
            linewidth=1.4,
            linestyle=":",
            alpha=0.7,
            label=f"Crossover ≈ {crossover_seconds}s",
        )
        ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 2-for-1 EV curve to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: run both plots
# ---------------------------------------------------------------------------
def generate_all_plots() -> List[Path]:
    """
    Run both MDP theorems and save all plots to ``docs/assets/images/``.

    Pre-computed sweep data is loaded from ``data/processed/`` when available
    (i.e. after running ``python -m src.collect_data``).  If the saved files
    are not found the sweeps are recomputed on the fly.

    Returns
    -------
    List of Path objects for all generated image files.
    """
    saved_paths: List[Path] = []

    # --- Theorem 2: Foul Up 3 heatmap ---
    t2_grid_path = PROCESSED_DIR / "theorem2_grid.npy"
    t2_meta_path = PROCESSED_DIR / "theorem2_metadata.json"
    if t2_grid_path.exists() and t2_meta_path.exists():
        logger.info("Loading pre-computed Theorem 2 data from %s", t2_grid_path)
        grid = np.load(t2_grid_path)
        with open(t2_meta_path) as f:
            meta = json.load(f)
        time_values = meta["time_values"]
        fg3_values = meta["fg3_pct_values"]
    else:
        from src.mdp_engine import Theorem2FoulUp3
        logger.info("Computing Theorem 2 sweep…")
        time_values = list(range(2, 12, 2))
        fg3_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]
        t2 = Theorem2FoulUp3()
        grid = t2.sweep(time_values=time_values, fg3_pct_values=fg3_values)
    saved_paths.append(plot_foul_up_3_heatmap(grid, time_values, fg3_values))

    # --- Theorem 1: 2-for-1 EV curve ---
    t1_sweep_path = PROCESSED_DIR / "theorem1_sweep.json"
    if t1_sweep_path.exists():
        logger.info("Loading pre-computed Theorem 1 data from %s", t1_sweep_path)
        with open(t1_sweep_path) as f:
            sweep = json.load(f)
    else:
        from src.mdp_engine import Theorem1TwoForOne
        logger.info("Computing Theorem 1 sweep…")
        t1 = Theorem1TwoForOne()
        sweep = t1.sweep_time(time_range=list(range(10, 65, 2)))
    saved_paths.append(plot_two_for_one_ev_curve(sweep))

    return saved_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_all_plots()
    for p in paths:
        print(f"Saved: {p}")
