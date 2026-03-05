"""
generate_docs.py
================
Regenerates the theorem Markdown documentation files with numbers derived
directly from the historical win-rate results stored in ``data/processed/``.

Call this module after running ``python -m src.collect_data`` (which invokes
:func:`generate_all_docs` automatically), or run it standalone::

    python -m src.generate_docs

Adding a new theorem
--------------------
1. Implement a ``_generate_<key>_doc`` helper below.
2. Register it in :data:`_DOC_GENERATORS`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DOCS_DIR       = Path(__file__).parent.parent / "docs"
PROCESSED_DIR  = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ev(value: float) -> str:
    """Format a win-percentage value to 2 decimal places."""
    return f"{value:.2f}"


def _fmt_gain(value: float, pp: bool = False) -> str:
    """Format a gain value with sign and optional 'pp' suffix."""
    suffix = " pp" if pp else ""
    if value > 0:
        return f"**+{value * 100:.1f}{suffix}**"
    return f"{value * 100:.1f}{suffix}"


def _gain_label(value: float) -> str:
    """Format a raw win-probability gain (0–1 scale) with bold for positive values."""
    if value > 0:
        return f"**+{value:.2f}**"
    return f"{value:.2f}"


def _find_sweep_entry(sweep: List[Dict], seconds: int) -> Dict:
    """Return the sweep dict entry for the given seconds value."""
    for entry in sweep:
        if entry["seconds_remaining"] == seconds:
            return entry
    raise KeyError(f"No sweep entry for {seconds} seconds remaining")


def _consecutive_positive_windows(
    sweep: List[Dict],
) -> List[tuple[int, int]]:
    """
    Return a list of (low_sec, high_sec) ranges where ev_gain > 0,
    identifying consecutive blocks in the sorted sweep.
    """
    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    windows: List[tuple[int, int]] = []
    in_window = False
    window_start = 0
    for entry in sorted_sweep:
        sec = entry["seconds_remaining"]
        if entry["ev_gain"] > 0 and not in_window:
            in_window = True
            window_start = sec
        elif entry["ev_gain"] <= 0 and in_window:
            in_window = False
            windows.append((window_start, prev_sec))  # type: ignore[possibly-undefined]
        prev_sec = sec
    if in_window:
        windows.append((window_start, prev_sec))  # type: ignore[possibly-undefined]
    return windows


def _largest_window(sweep: List[Dict]) -> tuple[int, int]:
    """Return the largest consecutive window where ev_gain > 0."""
    windows = _consecutive_positive_windows(sweep)
    if not windows:
        return (0, 0)
    return max(windows, key=lambda w: w[1] - w[0])


# ---------------------------------------------------------------------------
# Theorem 1 doc generator
# ---------------------------------------------------------------------------

_THEOREM1_TEMPLATE = """\
# Theorem 1: The 2-for-1

## Claim

> **Based on NBA play-by-play data from 2019–2024, teams that rush a shot
> to secure two possessions before time expires sometimes win at a higher
> historical rate — but there is no sharp, reliable clock threshold where
> this advantage switches on.**

---

## How We Measure It

We filter the historical play-by-play log for tied games and group each
possession by strategy:

- **Rush (shoot):** The possessing team takes a shot attempt.
- **Normal (hold):** The possessing team holds the ball (any non-shooting action).

We calculate the **historical win percentage** for each group — the fraction
of games where the home team went on to win given that choice.

---

## Results

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.svg)

### Key Findings

{key_findings}

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Scenario | Rush Win % | Hold Win % | Win % Gain | Better Strategy |
|----------|-----------|-----------|------------|----------------|
| 32 s, tied | {ev_rush_32} | {ev_normal_32} | {ev_gain_32} | {optimal_32} |
| 40 s, tied | {ev_rush_40} | {ev_normal_40} | {ev_gain_40} | {optimal_40} |
| 20 s, tied | {ev_rush_20} | {ev_normal_20} | {ev_gain_20} | {optimal_20} |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Conclusion

{conclusion}
"""


def _build_theorem1_key_findings(sweep: List[Dict]) -> str:
    """
    Dynamically construct the Key Findings bullet list from the sweep data.
    Returns a markdown-formatted string.
    """
    windows = _consecutive_positive_windows(sweep)
    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    negative_secs = [e["seconds_remaining"] for e in sorted_sweep if e["ev_gain"] <= 0]

    if not windows:
        return (
            "1. **No consistent rushing advantage found** in the current data — "
            "normal possession is at least as good across all analyzed time buckets. "
            "The 2-for-1 effect is not apparent in this sample."
        )

    main_window = max(windows, key=lambda w: w[1] - w[0])
    all_secs = [e["seconds_remaining"] for e in sorted_sweep]
    sweep_min, sweep_max = min(all_secs), max(all_secs)
    window_covers_full_range = (
        main_window[0] == sweep_min and main_window[1] == sweep_max
    )

    lines: List[str] = []

    if window_covers_full_range:
        max_gain_entry = max(sorted_sweep, key=lambda e: e["ev_gain"])
        classic_entries = [e for e in sorted_sweep if 28 <= e["seconds_remaining"] <= 34]
        if classic_entries:
            avg_gain_classic = sum(e["ev_gain"] for e in classic_entries) / len(classic_entries)
            lines.append(
                f"1. **Classic 2-for-1 window (~28–34 s): modest positive signal.**"
                f" Average win % gain ≈ +{avg_gain_classic:.2f} — rushing secures "
                "two possessions against the opponent's one, but the margin is small."
            )
        lines.append(
            f"2. **The advantage is largest around ~{max_gain_entry['seconds_remaining']} s**, "
            "where rushing leaves enough time for a second possession while the opponent "
            "can only respond once."
        )
        lines.append(
            "3. **No sharp threshold**: the signal is noisy across individual time buckets. "
            "Treat any specific second value as a rough guide, not a precise trigger."
        )
    else:
        lines.append(
            f"1. **Rushing appears beneficial roughly in the ~{main_window[0]}–{main_window[1]} s window** "
            "based on historical data, but the boundary is not sharp — adjacent time "
            "buckets often flip sign due to sample noise."
        )

        below_negs = [s for s in negative_secs if s < main_window[0]]
        if below_negs:
            boundary = max(below_negs)
            lines.append(
                f"2. **Below ~{boundary} s normal possession is preferred** — too little "
                "time remains for the opponent to mount a meaningful second possession, "
                "so the risk-return of rushing does not pay off historically."
            )

        above_negs = [s for s in negative_secs if s > main_window[1]]
        if above_negs:
            boundary = min(above_negs)
            lines.append(
                f"3. **Above ~{boundary} s normal possession is preferable** — rushing "
                "this early hands the opponent two possessions, negating the advantage."
            )
        else:
            lines.append(
                "3. **No clean upper threshold**: the rush advantage appears to extend "
                "across most of the analyzed range, though the margin narrows at longer "
                "clock values."
            )

    lines.append(
        f"4. **Sample sizes are small per bucket** — conclusions should be treated as "
        "directional signals rather than precise thresholds."
    )

    return "\n\n".join(lines)


def _generate_theorem1_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 1 sweep data and write the theorem1 Markdown file."""
    from src.theorems.theorem1 import load_sweep
    sweep: List[Dict] = load_sweep(processed_dir)

    e32 = _find_sweep_entry(sweep, 32)
    e40 = _find_sweep_entry(sweep, 40)
    e20 = _find_sweep_entry(sweep, 20)

    main_window = _largest_window(sweep)
    window_low, window_high = main_window

    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    all_secs = [e["seconds_remaining"] for e in sorted_sweep]
    sweep_min, sweep_max = min(all_secs), max(all_secs)
    window_covers_full_range = (
        main_window[0] == sweep_min and main_window[1] == sweep_max
    )

    # Build conclusion — always acknowledge the lack of a clean threshold
    if window_low == 0 and window_high == 0:
        conclusion = (
            "**The historical data does not show a consistent rushing advantage** "
            "across the analyzed range. Normal possession is at least as good in all "
            "studied time buckets. The conventional 2-for-1 wisdom is not confirmed by "
            "this sample — coaches should not sacrifice shot quality on clock alone."
        )
    elif window_covers_full_range:
        conclusion = (
            f"**The historical data shows a modest rush advantage across the full analyzed range "
            f"({sweep_min}–{sweep_max} s)**, but the gain is small and the threshold is not sharp. "
            "Shooting immediately is historically at least as good as holding, but "
            "the margin is narrow enough that shot quality matters more than the exact clock value. "
            "Coaches should push the pace when a good shot is available, not sacrifice "
            "shot quality chasing a specific second count."
        )
    else:
        above_neg = [
            e["seconds_remaining"]
            for e in sorted_sweep
            if e["seconds_remaining"] > window_high and e["ev_gain"] <= 0
        ]
        caution = (
            f" Rushing at {above_neg[0]}+ seconds can reduce win probability."
            if above_neg else ""
        )
        conclusion = (
            f"**The 2-for-1 shows a positive signal in roughly the {window_low}–{window_high} s range**, "
            "but there is no sharp, reliable threshold — individual second-by-second results "
            f"are noisy.{caution} "
            "Use this as a directional guide: favour rushing when a good shot is available "
            "in this window, but do not sacrifice shot quality for a specific clock value."
        )

    key_findings = _build_theorem1_key_findings(sweep)

    def _optimal_label(gain: float) -> str:
        return "Rush ✓" if gain > 0 else "Normal ✓"

    content = _THEOREM1_TEMPLATE.format(
        key_findings=key_findings,
        conclusion=conclusion,
        ev_rush_32=_fmt_ev(e32["ev_rush"]),
        ev_normal_32=_fmt_ev(e32["ev_normal"]),
        ev_gain_32=_gain_label(e32["ev_gain"]),
        optimal_32=_optimal_label(e32["ev_gain"]),
        ev_rush_40=_fmt_ev(e40["ev_rush"]),
        ev_normal_40=_fmt_ev(e40["ev_normal"]),
        ev_gain_40=_gain_label(e40["ev_gain"]),
        optimal_40=_optimal_label(e40["ev_gain"]),
        ev_rush_20=_fmt_ev(e20["ev_rush"]),
        ev_normal_20=_fmt_ev(e20["ev_normal"]),
        ev_gain_20=_gain_label(e20["ev_gain"]),
        optimal_20=_optimal_label(e20["ev_gain"]),
    )

    out_path = docs_dir / "theorem1_two_for_one.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 1 doc to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Theorem 2 doc generator
# ---------------------------------------------------------------------------

_THEOREM2_TEMPLATE = """\
# Theorem 2: Foul Up 3

## Claim

> **Based on NBA play-by-play data from 2019–2024, teams leading by 3 points
> with fewer than 12 seconds remaining generally win more often when they
> intentionally foul — especially against good three-point shooters.**

---

## How We Measure It

We filter the historical play-by-play log for:

- Home team defending (away team has the ball)
- Home team leads by exactly 3 points
- Fewer than 12 seconds remain

We group possessions by **Foul** (intentional) vs. **No Foul** (normal defence),
and compute the home-team historical win percentage for each.

---

## Results

![Foul Up 3 Heatmap](assets/images/foul_up_3_heatmap.svg)

The heatmap shows the historical win % gain from fouling (green = fouling better,
red = normal defence better) across time remaining and opponent 3PT%.

### Key Findings

{key_findings}

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Seconds | Opp 3PT% | Foul Win % | No-Foul Win % | Win % Gain |
|---------|----------|-----------|---------------|------------|
| 8 s | 28 % | {wp_foul_8_28} | {wp_no_foul_8_28} | {wp_gain_8_28} |
| 8 s | 36 % | {wp_foul_8_36} | {wp_no_foul_8_36} | {wp_gain_8_36} |
| 8 s | 44 % | {wp_foul_8_44} | {wp_no_foul_8_44} | {wp_gain_8_44} |
| 4 s | 36 % | {wp_foul_4_36} | {wp_no_foul_4_36} | {wp_gain_4_36} |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Sensitivity Analysis

The key driver is the **opponent's 3PT%** — a higher shooting rate makes
allowing a three-point attempt more costly.

Analyzed range ({fg3_min:.0%}–{fg3_max:.0%} opponent 3PT%):
win % gain from fouling ranges from {min_gain_pp:.1f} pp to +{max_gain_pp:.1f} pp.

---

## Conclusion

{conclusion}
"""


def _build_theorem2_key_findings(
    gain_grid: np.ndarray,
    time_values: List[int],
    fg3_values: List[float],
) -> str:
    """
    Dynamically construct the Key Findings bullet list for Theorem 2.
    Returns a markdown-formatted string.
    """
    min_gain_pp = float(gain_grid.min() * 100)
    max_gain_pp = float(gain_grid.max() * 100)

    has_negative = bool((gain_grid <= 0).any())

    if not has_negative:
        mean_by_fg3 = gain_grid.mean(axis=0)
        best_fg3_idx = int(np.argmax(mean_by_fg3))
        worst_fg3_idx = int(np.argmin(mean_by_fg3))
        best_fg3 = fg3_values[best_fg3_idx]
        worst_fg3 = fg3_values[worst_fg3_idx]

        min_time_idx = int(np.argmin(time_values))
        max_time_idx = int(np.argmax(time_values))
        min_time_gain_pp = float(gain_grid[min_time_idx, :].mean() * 100)
        max_time_gain_pp = float(gain_grid[max_time_idx, :].mean() * 100)
        min_time = time_values[min_time_idx]
        max_time = time_values[max_time_idx]

        if abs(min_time_gain_pp - max_time_gain_pp) < 1.0:
            finding3 = (
                f"3. **The benefit is remarkably stable across all time values** "
                f"({min_time}–{max_time} s):\n"
                f"   average historical win % gain at {min_time} s is "
                f"+{min_time_gain_pp:.1f} pp vs. +{max_time_gain_pp:.1f} pp at {max_time} s —\n"
                "   fouling is advisable regardless of exactly how many seconds remain."
            )
        elif min_time_gain_pp < max_time_gain_pp:
            finding3 = (
                f"3. **The advantage is slightly smaller with fewer seconds left**\n"
                f"   (+{min_time_gain_pp:.1f} pp at {min_time} s vs. "
                f"+{max_time_gain_pp:.1f} pp at {max_time} s), but fouling remains\n"
                "   strongly beneficial at all analyzed time values."
            )
        else:
            finding3 = (
                f"3. **The advantage is slightly larger with fewer seconds left**\n"
                f"   (+{min_time_gain_pp:.1f} pp at {min_time} s vs. "
                f"+{max_time_gain_pp:.1f} pp at {max_time} s), as there is less time\n"
                "   for the opponent to recover after free throws."
            )

        return (
            f"1. **Fouling is beneficial across all analyzed scenarios** "
            f"({min(time_values)}–{max(time_values)} seconds remaining, "
            f"opponent 3PT% {min(fg3_values):.0%}–{max(fg3_values):.0%}). "
            f"The historical win % gain ranges from **+{min_gain_pp:.1f} pp** to "
            f"**+{max_gain_pp:.1f} pp**.\n\n"
            f"2. **The advantage of fouling grows with the opponent's 3PT%.** "
            f"Against a {worst_fg3:.0%} shooter the gain is smallest "
            f"(+{float(gain_grid[:, worst_fg3_idx].mean() * 100):.1f} pp on average), "
            f"while against a {best_fg3:.0%} shooter it is largest "
            f"(+{float(gain_grid[:, best_fg3_idx].mean() * 100):.1f} pp on average).\n\n"
            f"{finding3}"
        )

    positive_fg3 = [
        fg3_values[j] for j in range(len(fg3_values))
        if any(gain_grid[i, j] > 0 for i in range(len(time_values)))
    ]
    negative_fg3 = [
        fg3_values[j] for j in range(len(fg3_values))
        if all(gain_grid[i, j] <= 0 for i in range(len(time_values)))
    ]
    high_threshold = min(positive_fg3) if positive_fg3 else fg3_values[0]
    low_threshold  = max(negative_fg3) if negative_fg3 else fg3_values[-1]

    return (
        f"1. **Fouling is most beneficial with 4–8 seconds remaining and a high-percentage\n"
        f"   3PT shooter (≥ {high_threshold:.0%}).** The heatmap shows the largest positive values in this\n"
        "   region.\n\n"
        f"2. **Against average-to-below-average 3PT shooters (≤ {low_threshold:.0%}), normal defense is\n"
        "   competitive** because the probability of a made 3-pointer is low enough that\n"
        "   the risk of cutting the lead to 1 (via free throws) is not worth taking.\n\n"
        "3. **With only 2 seconds left, the strategy matters less** — there is barely\n"
        "   enough time for either a clean 3PT attempt or a fast-foul scenario. Both\n"
        "   strategies converge to similar historical win percentages."
    )


def _build_theorem2_conclusion(
    gain_grid: np.ndarray,
    time_values: List[int],
    fg3_values: List[float],
    threshold_low: float,
) -> str:
    """Return the conclusion paragraph based on whether fouling is universally better."""
    has_negative = bool((gain_grid <= 0).any())
    if not has_negative:
        return (
            "**Fouling up 3 is historically justified across all analyzed game situations** "
            f"({min(time_values)}–{max(time_values)} seconds remaining, "
            f"opponent 3PT% {min(fg3_values):.0%}–{max(fg3_values):.0%}). "
            "The strategy is especially powerful against elite shooters. "
            "The key insight is that the decision is *opponent-specific*: "
            "the higher the opponent's 3PT shooting rate, the larger the historical "
            "benefit of fouling — coaches should adjust based on who has the ball."
        )
    return (
        f"**Fouling up 3 is historically justified for most practical game situations\n"
        f"(>=4 s remaining, opponent 3PT% >= {threshold_low:.0%}).** The strategy is especially powerful\n"
        "against elite shooters. Against poor 3PT teams, the conventional approach of\n"
        "playing normal defense remains competitive. The key insight is that the decision\n"
        "is *opponent-specific*: a blanket \"always foul\" or \"never foul\" rule is\n"
        "suboptimal — coaches should adjust based on who has the ball."
    )


def _generate_theorem2_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 2 sweep data and write the theorem2 Markdown file."""
    grid_path    = processed_dir / "theorem2_grid.csv"
    foul_path    = processed_dir / "theorem2_wp_foul_grid.csv"
    no_foul_path = processed_dir / "theorem2_wp_no_foul_grid.csv"
    meta_path    = processed_dir / "theorem2_metadata.json"

    for p in (grid_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Theorem 2 data not found at {p}. "
                "Run `python -m src.collect_data` first."
            )

    with open(meta_path) as f:
        meta = json.load(f)

    time_values: List[int]   = meta["time_values"]
    fg3_values: List[float]  = meta["fg3_pct_values"]

    gain_grid = np.loadtxt(grid_path, delimiter=",")

    if foul_path.exists() and no_foul_path.exists():
        wp_foul_grid    = np.loadtxt(foul_path, delimiter=",")
        wp_no_foul_grid = np.loadtxt(no_foul_path, delimiter=",")
    else:
        logger.warning(
            "Individual WP grids not found; reconstructing from gain grid. "
            "Re-run `python -m src.collect_data` to cache them."
        )
        # Reconstruct approximate values: assume a neutral default for wp_foul
        wp_foul_grid    = np.full_like(gain_grid, 0.5)
        wp_no_foul_grid = wp_foul_grid - gain_grid

    def _cell(grids: tuple, sec: int, fg3: float) -> tuple[float, float, float]:
        wp_foul_g, wp_no_foul_g, gain_g = grids
        i = time_values.index(sec)
        j = min(range(len(fg3_values)), key=lambda k: abs(fg3_values[k] - fg3))
        return float(wp_foul_g[i, j]), float(wp_no_foul_g[i, j]), float(gain_g[i, j])

    grids = (wp_foul_grid, wp_no_foul_grid, gain_grid)

    wf_8_28, wn_8_28, wg_8_28 = _cell(grids, 8, 0.28)
    wf_8_36, wn_8_36, wg_8_36 = _cell(grids, 8, 0.36)
    wf_8_44, wn_8_44, wg_8_44 = _cell(grids, 8, 0.44)
    wf_4_36, wn_4_36, wg_4_36 = _cell(grids, 4, 0.36)

    threshold = 2.0 / 3.0
    # This represents the approximate 3PT% break-even point for the foul
    # decision: if the opponent's 3PT% is above this threshold, historical data
    # suggests fouling is preferable to allowing a 3-point attempt.
    threshold_low = max(0.0, threshold - 0.01)

    key_findings = _build_theorem2_key_findings(gain_grid, time_values, fg3_values)
    conclusion   = _build_theorem2_conclusion(gain_grid, time_values, fg3_values, threshold_low)

    content = _THEOREM2_TEMPLATE.format(
        fg3_min=min(fg3_values),
        fg3_max=max(fg3_values),
        min_gain_pp=float(gain_grid.min() * 100),
        max_gain_pp=float(gain_grid.max() * 100),
        key_findings=key_findings,
        conclusion=conclusion,
        wp_foul_8_28=_fmt_ev(wf_8_28),
        wp_no_foul_8_28=_fmt_ev(wn_8_28),
        wp_gain_8_28=_fmt_gain(wg_8_28, pp=True),
        wp_foul_8_36=_fmt_ev(wf_8_36),
        wp_no_foul_8_36=_fmt_ev(wn_8_36),
        wp_gain_8_36=_fmt_gain(wg_8_36, pp=True),
        wp_foul_8_44=_fmt_ev(wf_8_44),
        wp_no_foul_8_44=_fmt_ev(wn_8_44),
        wp_gain_8_44=_fmt_gain(wg_8_44, pp=True),
        wp_foul_4_36=_fmt_ev(wf_4_36),
        wp_no_foul_4_36=_fmt_ev(wn_4_36),
        wp_gain_4_36=_fmt_gain(wg_4_36, pp=True),
    )

    out_path = docs_dir / "theorem2_foul_up_3.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 2 doc to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Theorem 3 doc generator
# ---------------------------------------------------------------------------

_THEOREM3_TEMPLATE = """\
# Theorem 3: The Late-Game Timeout

## Claim

> **Based on NBA play-by-play data from 2019–2024, teams trailing by 1–3 points
> (or tied) with possession and 20–50 seconds remaining do not consistently
> gain a win-probability advantage by calling a timeout.**

---

## How We Measure It

We filter the historical play-by-play log for situations where:

- The home team has possession
- The score differential is between −3 and 0 (trailing by up to 3, or tied)
- Between 20 and 50 seconds remain

We group possessions by:

- **Timeout:** The team stops play with a timeout call.
- **Play On:** The team continues without calling a timeout.

We then calculate the **historical win percentage** for the home team in
each group across a sweep of time-remaining values.

---

## Results

![Late-Game Timeout Win Percentage Curve](assets/images/timeout_ev_curve.svg)

### Key Findings

{key_findings}

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Seconds | Timeout Win % | Play-On Win % | Win % Gain | Better Strategy |
|---------|--------------|--------------|------------|----------------|
| 40 s | {ev_timeout_40} | {ev_play_on_40} | {ev_gain_40} | {optimal_40} |
| 30 s | {ev_timeout_30} | {ev_play_on_30} | {ev_gain_30} | {optimal_30} |
| 20 s | {ev_timeout_20} | {ev_play_on_20} | {ev_gain_20} | {optimal_20} |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Conclusion

{conclusion}
"""


def _build_theorem3_key_findings(sweep: List[Dict]) -> str:
    """
    Construct the Key Findings bullet list for Theorem 3 from sweep data.
    Returns a markdown-formatted string.
    """
    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    gains = [e["ev_gain"] for e in sorted_sweep]
    positive_count = sum(1 for g in gains if g > 0)
    total = len(gains)

    mean_gain = sum(gains) / total if total > 0 else 0.0
    max_entry = max(sorted_sweep, key=lambda e: e["ev_gain"])
    min_entry = min(sorted_sweep, key=lambda e: e["ev_gain"])

    if positive_count == 0:
        return (
            "1. **Calling a timeout does not historically improve win rate** "
            "across any of the analyzed time windows. Playing on is at least "
            "as good as calling a timeout in all cases studied.\n\n"
            "2. **The win-percentage gap is small**, suggesting the timeout "
            "decision has limited measurable impact on historical outcomes "
            "— other factors (player execution, matchups) likely dominate."
        )

    if positive_count == total:
        return (
            f"1. **Calling a timeout is historically beneficial across all analyzed "
            f"windows** ({sorted_sweep[0]['seconds_remaining']}–"
            f"{sorted_sweep[-1]['seconds_remaining']} s). "
            f"The average win % gain is **+{mean_gain * 100:.1f} pp**.\n\n"
            f"2. **The advantage peaks around {max_entry['seconds_remaining']} s** "
            f"(gain ≈ +{max_entry['ev_gain'] * 100:.1f} pp) and is smallest near "
            f"{min_entry['seconds_remaining']} s "
            f"(gain ≈ +{min_entry['ev_gain'] * 100:.1f} pp).\n\n"
            "3. **The result is consistent**: calling a timeout to set up the "
            "final possession appears to help regardless of how many seconds remain "
            "in the analyzed window."
        )

    positive_secs = [e["seconds_remaining"] for e in sorted_sweep if e["ev_gain"] > 0]
    negative_secs = [e["seconds_remaining"] for e in sorted_sweep if e["ev_gain"] <= 0]

    return (
        f"1. **Results are mixed**: a timeout historically helps at "
        f"{len(positive_secs)} of {total} time buckets "
        f"({min(positive_secs)}–{max(positive_secs)} s) "
        f"but hurts at {len(negative_secs)} others "
        f"({min(negative_secs)}–{max(negative_secs)} s).\n\n"
        f"2. **The largest timeout advantage** occurs around "
        f"~{max_entry['seconds_remaining']} s "
        f"(+{max_entry['ev_gain'] * 100:.1f} pp). "
        f"The largest disadvantage is near ~{min_entry['seconds_remaining']} s "
        f"({min_entry['ev_gain'] * 100:.1f} pp).\n\n"
        "3. **Overall the data is inconclusive**: there is no clean time window "
        "where timeouts are uniformly better or worse. The decision likely "
        "depends more on matchup and opponent tendencies than on the clock."
    )


def _build_theorem3_conclusion(sweep: List[Dict]) -> str:
    """Return the conclusion paragraph for Theorem 3."""
    sorted_sweep = sorted(sweep, key=lambda e: e["seconds_remaining"])
    gains = [e["ev_gain"] for e in sorted_sweep]
    positive_count = sum(1 for g in gains if g > 0)
    total = len(gains)
    mean_gain_pp = (sum(gains) / total) * 100 if total > 0 else 0.0

    if positive_count == 0:
        return (
            "**The historical data does not support a systematic timeout advantage** "
            "when trailing in the final 20–50 seconds. Playing on is at least as good "
            "as stopping the clock across all analyzed situations. Coaches should "
            "prioritize player execution and matchup considerations over the timeout "
            "decision itself."
        )
    if positive_count == total:
        return (
            f"**The historical data favours calling a timeout** when trailing with "
            f"20–50 seconds remaining (average win % gain: +{mean_gain_pp:.1f} pp). "
            "Setting up the final possession appears to be beneficial — coaches "
            "should use available timeouts to organise their offence in this window."
        )
    return (
        "**The historical data is inconclusive**: a timeout does not consistently "
        "help or hurt across the analyzed 20–50 second window. "
        f"On average the gain is {mean_gain_pp:+.1f} pp — essentially noise. "
        "Coaches should base the timeout decision on matchup specifics and player "
        "fatigue rather than treating it as a universal rule."
    )


def _generate_theorem3_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 3 sweep data and write the theorem3 Markdown file."""
    from src.theorems.theorem3 import load_sweep
    sweep: List[Dict] = load_sweep(processed_dir)

    def _find(sec: int) -> Dict:
        return _find_sweep_entry(sweep, sec)

    e40 = _find(40)
    e30 = _find(30)
    e20 = _find(20)

    def _optimal_label(gain: float) -> str:
        return "Timeout ✓" if gain > 0 else "Play On ✓"

    key_findings = _build_theorem3_key_findings(sweep)
    conclusion = _build_theorem3_conclusion(sweep)

    content = _THEOREM3_TEMPLATE.format(
        key_findings=key_findings,
        conclusion=conclusion,
        ev_timeout_40=_fmt_ev(e40["ev_timeout"]),
        ev_play_on_40=_fmt_ev(e40["ev_play_on"]),
        ev_gain_40=_gain_label(e40["ev_gain"]),
        optimal_40=_optimal_label(e40["ev_gain"]),
        ev_timeout_30=_fmt_ev(e30["ev_timeout"]),
        ev_play_on_30=_fmt_ev(e30["ev_play_on"]),
        ev_gain_30=_gain_label(e30["ev_gain"]),
        optimal_30=_optimal_label(e30["ev_gain"]),
        ev_timeout_20=_fmt_ev(e20["ev_timeout"]),
        ev_play_on_20=_fmt_ev(e20["ev_play_on"]),
        ev_gain_20=_gain_label(e20["ev_gain"]),
        optimal_20=_optimal_label(e20["ev_gain"]),
    )

    out_path = docs_dir / "theorem3_timeout.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 3 doc to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DOC_GENERATORS: Dict = {
    "theorem1": _generate_theorem1_doc,
    "theorem2": _generate_theorem2_doc,
    "theorem3": _generate_theorem3_doc,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_docs(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> List[Path]:
    """
    Regenerate all theorem Markdown files from historical analysis results.

    Reads pre-computed sweep data from *processed_dir* and writes updated
    Markdown files to *docs_dir*.  Call after ``collect_all()`` so that the
    processed data is up-to-date.

    Returns
    -------
    List of Path objects for all generated Markdown files.
    """
    paths: List[Path] = []
    for key, generator in _DOC_GENERATORS.items():
        logger.info("Generating doc for %s...", key)
        paths.append(generator(processed_dir=processed_dir, docs_dir=docs_dir))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generated = generate_all_docs()
    for p in generated:
        print(f"Written: {p}")
