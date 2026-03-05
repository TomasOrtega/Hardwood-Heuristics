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

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Minimum number of consecutive positive-gain time buckets required to treat
# a run as a *dominant* timeout window worth highlighting in Theorem 3.
_MIN_SIGNIFICANT_WINDOW_SIZE = 4


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
    """Format a raw win-probability gain (0--1 scale) with bold for positive values."""
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

> **Based on NBA play-by-play data from 2019--2024, rushing a shot in tied
> games shows a positive win-rate signal around 18--22 seconds remaining.
> The effect is noisy and no single threshold reliably separates when
> rushing helps from when it hurts.**

---

## How We Measure It

We filter the historical play-by-play log for tied games **where the home team has possession** and group each possession by strategy:

- **Rush (shoot):** The possessing team takes a shot attempt.
- **Normal (hold):** The possessing team holds the ball (any non-shooting action).

We calculate the **historical win percentage** for each group — the fraction
of games where the home team went on to win given that choice.

---

## Results

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.svg)

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
        classic_entries = [
            e for e in sorted_sweep if 28 <= e["seconds_remaining"] <= 34
        ]
        if classic_entries:
            avg_gain_classic = sum(e["ev_gain"] for e in classic_entries) / len(
                classic_entries
            )
            lines.append(
                f"1. **Classic 2-for-1 window (~28--34 s): modest positive signal.**"
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
            f"1. **Rushing shows a positive signal around the ~{main_window[0]}--{main_window[1]} s window**, "
            "but results are noisy — individual time buckets often flip sign."
        )

    lines.append(
        f"{len(lines) + 1}. **Sample sizes are small per bucket** — treat these as "
        "directional signals, not precise thresholds."
    )

    return "\n\n".join(lines)


def _generate_theorem1_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 1 sweep data and write the theorem1 Markdown file."""
    from src.theorems.utils import load_sweep_csv

    sweep: List[Dict] = load_sweep_csv(processed_dir / "theorem1_sweep.csv")

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
            "**No consistent rushing advantage found** in the historical data. "
            "Normal possession performs at least as well across all analyzed time "
            "buckets. Do not sacrifice shot quality based on the clock alone."
        )
    elif window_covers_full_range:
        conclusion = (
            f"**Rushing shows a modest advantage across the full analyzed range "
            f"({sweep_min}--{sweep_max} s)**, but the gain is small and the results "
            "are noisy. Shot quality matters more than the exact clock value."
        )
    else:
        conclusion = (
            f"**The 2-for-1 shows a positive signal around the {window_low}--"
            f"{window_high} s range**, but results are noisy across individual "
            "time buckets. Favour rushing when a good shot is available in this "
            "window, but do not sacrifice shot quality for a specific clock value."
        )

    content = _THEOREM1_TEMPLATE.format(
        conclusion=conclusion,
    )

    out_path = docs_dir / "theorem1_two_for_one.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Written Theorem 1 doc to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Theorem 2 doc generator
# ---------------------------------------------------------------------------


def _generate_theorem2_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Write the static Theorem 2 Markdown file."""
    from src.theorems.theorem2 import generate_doc

    return generate_doc(processed_dir=processed_dir, docs_dir=docs_dir)


# ---------------------------------------------------------------------------
# Theorem 3 doc generator
# ---------------------------------------------------------------------------

_THEOREM3_TEMPLATE = """\
# Theorem 3: The Late-Game Timeout

## Claim

> **Based on NBA play-by-play data from 2019--2024, calling a timeout when
> trailing by 1--3 (or tied) with 36--50 seconds remaining shows a consistent
> win-rate advantage. Results are mixed below 36 seconds.**

---

## How We Measure It

We filter the historical play-by-play log for situations where:

- The home team has possession
- The score differential is between -3 and 0 (trailing by up to 3, or tied)
- Between 20 and 50 seconds remain

We group possessions by:

- **Timeout:** The team stops play with a timeout call.
- **Play On:** The team continues without calling a timeout.

We then calculate the **historical win percentage** for the home team in
each group across a sweep of time-remaining values.

---

## Results

![Late-Game Timeout Win Percentage Curve](assets/images/timeout_ev_curve.svg)

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
            "1. **Calling a timeout does not improve win rate** across any of "
            "the analyzed time windows — playing on is at least as good in all "
            "cases studied.\n\n"
            "2. **The gap is small**: the timeout decision has limited measurable "
            "impact on historical outcomes."
        )

    if positive_count == total:
        return (
            f"1. **A timeout is beneficial across all analyzed windows** "
            f"({sorted_sweep[0]['seconds_remaining']}--"
            f"{sorted_sweep[-1]['seconds_remaining']} s), "
            f"with an average win % gain of **+{mean_gain * 100:.1f} pp**.\n\n"
            f"2. **The advantage peaks around {max_entry['seconds_remaining']} s** "
            f"(+{max_entry['ev_gain'] * 100:.1f} pp) and is smallest near "
            f"{min_entry['seconds_remaining']} s "
            f"(+{min_entry['ev_gain'] * 100:.1f} pp)."
        )

    # Check for a dominant consecutive positive window
    windows = _consecutive_positive_windows(sweep)
    if windows:
        best_window = max(windows, key=lambda w: w[1] - w[0])
        window_entries = [
            e for e in sorted_sweep
            if best_window[0] <= e["seconds_remaining"] <= best_window[1]
        ]
        window_count = len(window_entries)
        if window_count >= _MIN_SIGNIFICANT_WINDOW_SIZE:
            window_mean_gain = sum(e["ev_gain"] for e in window_entries) / window_count
            outside = [
                e for e in sorted_sweep
                if not (best_window[0] <= e["seconds_remaining"] <= best_window[1])
            ]
            outside_positive = [e for e in outside if e["ev_gain"] > 0]
            outside_negative = [e for e in outside if e["ev_gain"] <= 0]
            return (
                f"1. **Consistent advantage from {best_window[0]}--{best_window[1]} s**: "
                f"a timeout improves win rate at all {window_count} time buckets in this "
                f"window (average gain: +{window_mean_gain * 100:.1f} pp; peak: "
                f"+{max_entry['ev_gain'] * 100:.1f} pp at ~{max_entry['seconds_remaining']} s).\n\n"
                f"2. **Mixed results below {best_window[0]} s**: "
                f"a timeout helps at {len(outside_positive)} bucket(s) but hurts at "
                f"{len(outside_negative)} other(s) in the {min(e['seconds_remaining'] for e in outside)}--"
                f"{max(e['seconds_remaining'] for e in outside)} s range.\n\n"
                f"3. **Largest disadvantage**: ~{min_entry['seconds_remaining']} s "
                f"({min_entry['ev_gain'] * 100:.1f} pp) — calling a timeout close to the "
                f"{min_entry['seconds_remaining']}-second mark carries meaningful risk."
            )

    positive_secs = [e["seconds_remaining"] for e in sorted_sweep if e["ev_gain"] > 0]
    negative_secs = [e["seconds_remaining"] for e in sorted_sweep if e["ev_gain"] <= 0]

    return (
        f"1. **Results are mixed**: a timeout helps at "
        f"{len(positive_secs)} of {total} time buckets "
        f"({min(positive_secs)}--{max(positive_secs)} s) "
        f"but hurts at {len(negative_secs)} others "
        f"({min(negative_secs)}--{max(negative_secs)} s).\n\n"
        f"2. **Largest advantage**: ~{max_entry['seconds_remaining']} s "
        f"(+{max_entry['ev_gain'] * 100:.1f} pp). "
        f"Largest disadvantage: ~{min_entry['seconds_remaining']} s "
        f"({min_entry['ev_gain'] * 100:.1f} pp).\n\n"
        "3. **No clean pattern**: the data does not identify a time window where "
        "calling a timeout is consistently better or worse."
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
            "**The historical data does not support a timeout advantage** when "
            "trailing in the final 20--50 seconds — playing on is at least as good "
            "in all cases studied. The timeout decision has limited measurable impact; "
            "focus on execution and matchups instead."
        )
    if positive_count == total:
        return (
            f"**Calling a timeout is historically beneficial** when trailing with "
            f"20--50 seconds remaining (average gain: +{mean_gain_pp:.1f} pp). "
            "Use available timeouts to set up the final possession."
        )
    # Mixed case — check for a dominant consecutive positive window
    windows = _consecutive_positive_windows(sweep)
    if windows:
        best_window = max(windows, key=lambda w: w[1] - w[0])
        window_entries = [
            e for e in sorted_sweep
            if best_window[0] <= e["seconds_remaining"] <= best_window[1]
        ]
        if len(window_entries) >= _MIN_SIGNIFICANT_WINDOW_SIZE:
            window_mean_gain_pp = (
                sum(e["ev_gain"] for e in window_entries) / len(window_entries)
            ) * 100
            return (
                f"**With {best_window[0]}--{best_window[1]} seconds remaining, calling "
                f"a timeout is historically beneficial** (average gain: "
                f"+{window_mean_gain_pp:.1f} pp across all {len(window_entries)} buckets "
                f"in this window). Below {best_window[0]} s the data is mixed — results "
                "are close to even. Rely on matchup specifics rather than a universal "
                "rule in the final 30 seconds."
            )
    return (
        f"**The data is inconclusive**: a timeout does not consistently help or "
        f"hurt in the 20--50 second window. On average the gain is "
        f"{mean_gain_pp:+.1f} pp. Base the decision on matchup specifics."
    )


def _generate_theorem3_doc(
    processed_dir: Path = PROCESSED_DIR,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Load Theorem 3 sweep data and write the theorem3 Markdown file."""
    from src.theorems.utils import load_sweep_csv

    sweep: List[Dict] = load_sweep_csv(processed_dir / "theorem3_sweep.csv")

    conclusion = _build_theorem3_conclusion(sweep)

    content = _THEOREM3_TEMPLATE.format(
        conclusion=conclusion,
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
