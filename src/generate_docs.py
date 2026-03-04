"""
generate_docs.py
================
Regenerates the theorem Markdown documentation files with numbers derived
directly from the MDP analysis results stored in ``data/processed/``.

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

from src.mdp_engine import FG2_PCT, FG3_PCT, FT_PCT, TURNOVER_PROB, FOUL_DRAWN_PROB

logger = logging.getLogger(__name__)

DOCS_DIR       = Path(__file__).parent.parent / "docs"
PROCESSED_DIR  = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ev(value: float) -> str:
    """Format an EV / win-probability value to 2 decimal places."""
    return f"{value:.2f}"


def _fmt_gain(value: float, pp: bool = False) -> str:
    """Format a gain value with sign and optional 'pp' suffix."""
    suffix = " pp" if pp else ""
    if value > 0:
        return f"**+{value * 100:.1f}{suffix}**"
    return f"{value * 100:.1f}{suffix}"


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

> **Rushing a shot with approximately 32 seconds remaining — so that your team
> gets a second possession before time expires while the opponent only gets one —
> increases expected win probability.**

---

## Motivation

The "2-for-1" is one of the most recognisable situational strategies in
basketball. With ~35 seconds on the game clock and possession, a team can shoot
quickly (say, with 28–32 s left), allow the opponent a *single* possession (~24 s
each), and then get the ball back with 4–8 s remaining for a final heave.
Net result: **two shots vs. one for the opponent**.

The open question is *when exactly the math tips in favour of rushing*. Shooting
too early (e.g. 45+ seconds remaining) gifts the opponent time for *two*
possessions themselves; shooting too late means the second possession is only a
Hail Mary.

---

## MDP Formulation

We fix possession at the home team ($\\pi = 1$) and a tied game ($\\Delta = 0$)
and evaluate two strategies:

**Strategy A – Rush:** Take a shot immediately at $\\tau$ seconds remaining.

$$\\text{{EV}}_{{\\text{{rush}}}}(\\tau) = \\sum_{{s'}} P(s' \\mid s_\\tau, \\texttt{{shoot}}) \\cdot V^*(s')$$

**Strategy B – Normal:** Hold the ball for a full-length possession (~24 s),
then shoot with $\\tau - 24$ seconds remaining.

$$\\text{{EV}}_{{\\text{{normal}}}}(\\tau) = \\sum_{{s_h}} P(s_h \\mid s_\\tau, \\texttt{{hold}}) \\cdot V^*(s_h)$$

The **EV gain** from rushing is:

$$\\Delta\\text{{EV}}(\\tau) = \\text{{EV}}_{{\\text{{rush}}}}(\\tau) - \\text{{EV}}_{{\\text{{normal}}}}(\\tau)$$

When $\\Delta\\text{{EV}}(\\tau) > 0$ rushing is optimal; when it is negative, a
normal possession is preferred.

---

## Results

![2-for-1 EV Curve](assets/images/two_for_one_ev_curve.png)

### Key Findings

The MDP analysis reveals several important thresholds:

{key_findings}

### Numerical Summary

Using league-average parameters:

| Scenario | $\\text{{EV}}_{{\\text{{rush}}}}$ | $\\text{{EV}}_{{\\text{{normal}}}}$ | $\\Delta\\text{{EV}}$ | Optimal |
|----------|------|--------|------|---------|
| 32 s, tied | {ev_rush_32} | {ev_normal_32} | {ev_gain_32} | {optimal_32} |
| 40 s, tied | {ev_rush_40} | {ev_normal_40} | {ev_gain_40} | {optimal_40} |
| 20 s, tied | {ev_rush_20} | {ev_normal_20} | {ev_gain_20} | {optimal_20} |

> *Values are home win-probability estimates from backward-induction solver.*

---

## Methodology Notes

* The analytic transition model uses the following league-average statistics
  (2019–24 regular season):
  - 2PT FG%: **{fg2_pct:.0%}**   |   3PT FG%: **{fg3_pct:.0%}**   |   FT%: **{ft_pct:.0%}**
  - Turnover rate: **{to_rate:.0%}**   |   Foul-drawn rate: **{foul_rate:.0%}**
* Strategy B ("Normal") models a typical possession of 20–25 seconds. For very
  short time values (< 25 s), the hold duration is clipped to the available clock.
* The "foul" action is restricted to defensive states (away possession) where the
  home team intentionally fouls the ball-handler. From home possession the shoot
  action already accounts for foul-drawn probability, so no separate foul shortcut
  is available — this keeps the value function grounded in realistic outcomes.
* The model treats rebounds probabilistically rather than tracking individual
  players.

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
            "1. **No time window found where rushing is strictly better** "
            "in the current analysis range. Normal possession is preferred throughout."
        )

    main_window = max(windows, key=lambda w: w[1] - w[0])
    all_secs = [e["seconds_remaining"] for e in sorted_sweep]
    sweep_min, sweep_max = min(all_secs), max(all_secs)
    window_covers_full_range = (
        main_window[0] == sweep_min and main_window[1] == sweep_max
    )

    lines: List[str] = []

    if window_covers_full_range:
        # Rush is better across the entire analyzed range.
        max_gain_entry = max(sorted_sweep, key=lambda e: e["ev_gain"])
        min_gain_entry = min(sorted_sweep, key=lambda e: e["ev_gain"])
        lines.append(
            f"1. **Rush advantage holds across the full analyzed window "
            f"({sweep_min}–{sweep_max} s).**"
            " The EV gain from rushing is positive throughout.\n"
            f"   The advantage is largest around ~{max_gain_entry['seconds_remaining']} s "
            f"and smallest near ~{min_gain_entry['seconds_remaining']} s."
        )
        # Find the classic 2-for-1 window (28-34s) and describe it
        classic_entries = [e for e in sorted_sweep if 28 <= e["seconds_remaining"] <= 34]
        if classic_entries:
            avg_gain_classic = sum(e["ev_gain"] for e in classic_entries) / len(classic_entries)
            lines.append(
                f"2. **Classic 2-for-1 window (~28–34 s): modest but consistent advantage.**"
                f" Average EV gain ≈ +{avg_gain_classic:.2f} in this range — "
                "rushing ensures two possessions vs. the opponent's one."
            )
        lines.append(
            "3. **The gain is larger at shorter time values** because holding for a full\n"
            "   possession at < 25 s consumes most of the remaining clock, while rushing\n"
            "   still gives the team a realistic scoring opportunity before time expires."
        )
    else:
        # Finding 1: main rush window
        lines.append(
            f"1. **Critical window: ~{main_window[0]}–{main_window[1]} seconds remaining.**"
            " This is where the EV gain from\n"
            "   rushing is maximised. A team with possession in this window should consider\n"
            "   pushing the pace to ensure two possessions."
        )

        # Finding 2: below threshold (if negative region exists below the main window)
        below_negs = [s for s in negative_secs if s < main_window[0]]
        if below_negs:
            boundary = max(below_negs)
            lines.append(
                f"2. **Below ~{boundary} seconds: normal possession is preferred** — "
                "insufficient time\n"
                "   for the opponent to mount a meaningful second possession, so the\n"
                "   risk-return of rushing does not pay off."
            )
        else:
            # No negative region below the main window: rush is always better at low times too
            secondary_windows = [w for w in windows if w != main_window]
            if secondary_windows:
                sw = secondary_windows[0]
                lines.append(
                    f"2. **Below ~{sw[1]} seconds: rushing is also better** because "
                    "there is insufficient\n"
                    "   time for a normal possession *plus* a second shot."
                )

        # Finding 3: above threshold (if negative region above the main window)
        above_negs = [s for s in negative_secs if s > main_window[1]]
        if above_negs:
            boundary = min(above_negs)
            lines.append(
                f"3. **Above ~{boundary} seconds: normal possession is preferable.** "
                f"Rushing at {boundary}+ seconds\n"
                "   gives the opponent two possessions, negating the advantage."
            )
        else:
            lines.append(
                "3. **Within the analyzed range the rush advantage holds** across most"
                " clock values in\n"
                "   the main window — but the margin narrows significantly outside of it."
            )

    return "\n\n".join(lines)


def _generate_theorem1_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 1 sweep data and write the theorem1 Markdown file."""
    sweep_path = processed_dir / "theorem1_sweep.json"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Theorem 1 sweep data not found at {sweep_path}. "
            "Run `python -m src.collect_data` first."
        )

    with open(sweep_path) as f:
        sweep: List[Dict] = json.load(f)

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

    # Build conclusion paragraph based on whether there is a genuine caution
    # window within the analyzed range or whether rush is always preferred.
    above_window_neg = [
        e["seconds_remaining"]
        for e in sorted_sweep
        if e["seconds_remaining"] > window_high and e["ev_gain"] <= 0
    ]
    if above_window_neg:
        conclusion_caution_sec = above_window_neg[0]
        conclusion = (
            f"**The 2-for-1 is mathematically justified in the window "
            f"~{window_low}–{window_high} s.**\n"
            "Outside this window, the conventional wisdom breaks down. Coaches should be aware\n"
            f"of the exact clock time — taking a shot at {conclusion_caution_sec} seconds\n"
            "can actually *reduce* win probability relative to playing for a clean look."
        )
    elif window_covers_full_range:
        conclusion = (
            f"**The rush advantage holds across the full analyzed range "
            f"({sweep_min}–{sweep_max} s).**\n"
            "Shooting immediately is always at least as good as holding for a full possession\n"
            "within the analyzed window. The gain is most modest in the classic 2-for-1 zone\n"
            "(~28–34 s), where the difference between rush and normal is small but consistently\n"
            "positive — coaches should push the pace when the clock is in this range to ensure\n"
            "two possessions. Much earlier or later in the game, other strategic considerations\n"
            "dominate."
        )
    else:
        conclusion = (
            f"**The 2-for-1 is mathematically justified in the time window "
            f"~{window_low}–{window_high} s.**\n"
            "Outside this window the conventional wisdom breaks down. Coaches should be aware\n"
            "of the exact clock time before committing to a rush."
        )

    key_findings = _build_theorem1_key_findings(sweep)

    def _optimal_label(gain: float) -> str:
        return "Rush ✓" if gain > 0 else "Normal ✓"

    def _gain_label(gain: float) -> str:
        """Format gain as displayed in the table (no pp suffix for theorem 1)."""
        if gain > 0:
            return f"**+{gain:.2f}**"
        return f"{gain:.2f}"

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
        fg2_pct=FG2_PCT,
        fg3_pct=FG3_PCT,
        ft_pct=FT_PCT,
        to_rate=TURNOVER_PROB,
        foul_rate=FOUL_DRAWN_PROB,
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

> **When leading by 3 points with fewer than 10 seconds remaining, the defending
> team should intentionally foul the ball-handler to prevent a game-tying
> 3-pointer. This strategy increases the home team's win probability.**

---

## Motivation

Few decisions in basketball generate more debate than whether to foul when leading
by three. The logic is seductive: deny the opponent a three-point attempt by
sending them to the free-throw line for two shots instead.

The counter-argument: the opponent makes both free throws {ft_both_pct:.0%} of the time
(${ft_pct:.2f}^2 \\approx {ft_both_pct:.2f}$), cutting the lead to one and granting them a free
throw to tie, while also creating a live-ball rebound opportunity.
Meanwhile, normal defense might hold the opponent to a
low-percentage heave.

We use the MDP framework to quantify *exactly when* fouling is beneficial and how
the opponent's 3PT shooting percentage affects the decision.

---

## MDP Formulation

The relevant game state is:

$$s = (\\Delta = +3,\\ \\tau \\in \\{{2, 4, 6, 8, 10\\}},\\ \\pi = 0,\\ f = 1)$$

(away team has the ball, home leads by 3.)

**Strategy A – Foul:** Force two free throws.

$$\\text{{WP}}_{{\\text{{foul}}}} = \\sum_{{s'}} P_{{\\text{{FT}}}}(s' \\mid s) \\cdot V^*(s')$$

where the free-throw transition probabilities are:

$$P(\\text{{make both}}) = \\text{{FT\\%}}^2, \\quad
P(\\text{{make one}}) = 2 \\cdot \\text{{FT\\%}} \\cdot (1 - \\text{{FT\\%}}), \\quad
P(\\text{{miss both}}) = (1 - \\text{{FT\\%}})^2$$

**Strategy B – Normal Defence:** Allow the opponent to shoot a 3-pointer.

$$\\text{{WP}}_{{\\text{{no\\_foul}}}} = \\sum_{{s'}} P_{{\\text{{3PT}}}}(s' \\mid s) \\cdot V^*(s')$$

The **win-probability gain** from fouling is:

$$\\Delta\\text{{WP}} = \\text{{WP}}_{{\\text{{foul}}}} - \\text{{WP}}_{{\\text{{no\\_foul}}}}$$

$\\Delta\\text{{WP}} > 0$ implies fouling is optimal.

---

## Results

![Foul Up 3 Heatmap](assets/images/foul_up_3_heatmap.png)

The heatmap displays $\\Delta\\text{{WP}}$ (in percentage points) across
combinations of seconds remaining (rows) and opponent 3PT% (columns).

- **Green cells** → Fouling increases win probability (positive $\\Delta\\text{{WP}}$)
- **Red cells** → Normal defense is preferred (negative $\\Delta\\text{{WP}}$)

### Key Findings

{key_findings}

### Numerical Summary

| Seconds | Opp 3PT% | $\\text{{WP}}_{{\\text{{foul}}}}$ | $\\text{{WP}}_{{\\text{{no\\_foul}}}}$ | $\\Delta\\text{{WP}}$ |
|---------|----------|-----------|------------|---------|
| 8 s | 28 % | {wp_foul_8_28} | {wp_no_foul_8_28} | {wp_gain_8_28} |
| 8 s | 36 % | {wp_foul_8_36} | {wp_no_foul_8_36} | {wp_gain_8_36} |
| 8 s | 44 % | {wp_foul_8_44} | {wp_no_foul_8_44} | {wp_gain_8_44} |
| 4 s | 36 % | {wp_foul_4_36} | {wp_no_foul_4_36} | {wp_gain_4_36} |

> *Values are approximate; exact figures depend on the solved MDP grid.*

---

## Sensitivity Analysis

The key driver of the decision is the **opponent's 3PT%**. As the opponent's
three-point shooting ability increases, the expected cost of allowing a 3PT attempt
grows, making the foul decision more valuable.

With league-average FT% = {ft_pct:.0%}, the free-throw outcomes impose an expected
win-probability cost of roughly:

$$\\text{{Expected cost of foul}} = {ft_both_pct:.2f} \\times (-2\\text{{ pp}}) \\approx {foul_cost:.1f}\\text{{ pp}}$$

In practice the MDP results show that fouling is beneficial across the **full
range** of analyzed 3PT percentages ({fg3_min:.0%}–{fg3_max:.0%}): even against
a relatively poor 3PT team the WP gain is materially positive (+{min_gain_pp:.1f} pp),
and it grows to +{max_gain_pp:.1f} pp against elite 3PT shooters.

---

## Methodology Notes

* Free-throw model: independent Bernoulli trials with league-average FT% = {ft_pct:.0%}.
* A **missed second free throw** is modelled as the fouled team retaining
  possession (rebounds to the offense at a 70 % rate in late-game situations).
* The MDP solver uses a near-undiscounted discount factor ($\\gamma = 0.99$)
  so that the value function approximates true win probabilities while keeping
  value iteration well-conditioned.

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

    # Are there any cells where fouling is NOT better?
    has_negative = bool((gain_grid <= 0).any())

    if not has_negative:
        # Fouling is always better across the entire grid.
        # Find the fg3 column with the largest gain to describe benefit direction.
        mean_by_fg3 = gain_grid.mean(axis=0)
        best_fg3_idx = int(np.argmax(mean_by_fg3))
        worst_fg3_idx = int(np.argmin(mean_by_fg3))
        best_fg3 = fg3_values[best_fg3_idx]
        worst_fg3 = fg3_values[worst_fg3_idx]

        # Build a data-driven finding #3: compare min-time gains vs max-time gains.
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
                f"   average WP gain at {min_time} s is "
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
            f"The WP gain ranges from **+{min_gain_pp:.1f} pp** to "
            f"**+{max_gain_pp:.1f} pp**.\n\n"
            f"2. **The advantage of fouling grows with the opponent's 3PT%.** "
            f"Against a {worst_fg3:.0%} shooter the gain is smallest "
            f"(+{float(gain_grid[:, worst_fg3_idx].mean() * 100):.1f} pp on average), "
            f"while against a {best_fg3:.0%} shooter it is largest "
            f"(+{float(gain_grid[:, best_fg3_idx].mean() * 100):.1f} pp on average).\n\n"
            f"{finding3}"
        )

    # There are some cells where normal defense is better.
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
        "   strategies converge to similar win probabilities."
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
            "**Fouling up 3 is mathematically justified across all analyzed game situations** "
            f"({min(time_values)}–{max(time_values)} seconds remaining, "
            f"opponent 3PT% {min(fg3_values):.0%}–{max(fg3_values):.0%}). "
            "The strategy is especially powerful against elite shooters. "
            "The key insight is that the decision is *opponent-specific*: "
            "the greater the opponent's 3PT ability, the larger the benefit of fouling — "
            "coaches should adjust based on who has the ball."
        )
    return (
        f"**Fouling up 3 is mathematically justified for most practical game situations\n"
        f"(≥ 4 s remaining, opponent 3PT% ≥ {threshold_low:.0%}).** The strategy is especially powerful\n"
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
    grid_path      = processed_dir / "theorem2_grid.npy"
    foul_path      = processed_dir / "theorem2_wp_foul_grid.npy"
    no_foul_path   = processed_dir / "theorem2_wp_no_foul_grid.npy"
    meta_path      = processed_dir / "theorem2_metadata.json"

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

    gain_grid = np.load(grid_path)

    # Load individual WP grids if available; otherwise fall back to computing
    # the four table rows on the fly.
    if foul_path.exists() and no_foul_path.exists():
        wp_foul_grid    = np.load(foul_path)
        wp_no_foul_grid = np.load(no_foul_path)
    else:
        logger.warning(
            "Individual WP grids not found; computing table rows from MDP. "
            "Re-run `python -m src.collect_data` to cache them."
        )
        from src.mdp_engine import Theorem2FoulUp3
        wp_foul_grid    = np.zeros_like(gain_grid)
        wp_no_foul_grid = np.zeros_like(gain_grid)
        for i, sec in enumerate(time_values):
            for j, fg3 in enumerate(fg3_values):
                res = Theorem2FoulUp3(opp_fg3_pct=fg3).compute(seconds_remaining=sec)
                wp_foul_grid[i, j]    = res["wp_foul"]
                wp_no_foul_grid[i, j] = res["wp_no_foul"]

    def _cell(grids: tuple, sec: int, fg3: float) -> tuple[float, float, float]:
        """Return (wp_foul, wp_no_foul, wp_gain) for the given (sec, fg3) cell."""
        wp_foul_g, wp_no_foul_g, gain_g = grids
        i = time_values.index(sec)
        # Find closest fg3 value
        j = min(range(len(fg3_values)), key=lambda k: abs(fg3_values[k] - fg3))
        return float(wp_foul_g[i, j]), float(wp_no_foul_g[i, j]), float(gain_g[i, j])

    grids = (wp_foul_grid, wp_no_foul_grid, gain_grid)

    wf_8_28, wn_8_28, wg_8_28 = _cell(grids, 8, 0.28)
    wf_8_36, wn_8_36, wg_8_36 = _cell(grids, 8, 0.36)
    wf_8_44, wn_8_44, wg_8_44 = _cell(grids, 8, 0.44)
    wf_4_36, wn_4_36, wg_4_36 = _cell(grids, 4, 0.36)

    ft_both = FT_PCT ** 2
    # Expected cost of fouling (in percentage points) from the analytic model:
    # P(both FTs made) × 2pp reduction in win probability.
    foul_cost_pp = ft_both * (-2.0)   # negative → cost to home team's WP

    # Analytic break-even 3PT% (from simplified model): FT%^2 * 2 / 3
    threshold = ft_both * 2 / 3
    threshold_low  = max(0.0, threshold - 0.01)
    threshold_high = threshold + 0.01

    key_findings = _build_theorem2_key_findings(gain_grid, time_values, fg3_values)
    conclusion    = _build_theorem2_conclusion(gain_grid, time_values, fg3_values, threshold_low)

    content = _THEOREM2_TEMPLATE.format(
        ft_pct=FT_PCT,
        ft_both_pct=ft_both,
        foul_cost=foul_cost_pp,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
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
# Registry
# ---------------------------------------------------------------------------

_DOC_GENERATORS: Dict = {
    "theorem1": _generate_theorem1_doc,
    "theorem2": _generate_theorem2_doc,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_docs(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> List[Path]:
    """
    Regenerate all theorem Markdown files from analysis results.

    Reads pre-computed sweep data from *processed_dir* and writes updated
    Markdown files to *docs_dir*.  Call after ``collect_all()`` so that the
    processed data is up-to-date.

    Returns
    -------
    List of Path objects for all generated Markdown files.
    """
    paths: List[Path] = []
    for key, generator in _DOC_GENERATORS.items():
        logger.info("Generating doc for %s…", key)
        paths.append(generator(processed_dir=processed_dir, docs_dir=docs_dir))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generated = generate_all_docs()
    for p in generated:
        print(f"Written: {p}")
