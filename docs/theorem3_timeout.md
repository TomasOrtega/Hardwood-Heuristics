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

### Key Findings

1. **Consistent advantage from 36--50 s**: a timeout improves win rate at all 8 time buckets in this window (average gain: +10.4 pp; peak: +17.3 pp at ~40 s).

2. **Mixed results below 36 s**: a timeout helps at 4 bucket(s) but hurts at 4 other(s) in the 20--34 s range.

3. **Largest disadvantage**: ~34 s (-8.2 pp) — calling a timeout close to the 34-second mark carries meaningful risk.

### Historical Data Summary

Data from 5 NBA seasons (2019--2024):

| Seconds | Timeout Win % | Play-On Win % | Win % Gain | Better Strategy |
|---------|--------------|--------------|------------|----------------|
| 40 s | 0.67 | 0.49 | **+0.17** | Timeout ✓ |
| 30 s | 0.46 | 0.45 | **+0.02** | Timeout ✓ |
| 20 s | 0.46 | 0.48 | -0.02 | Play On ✓ |

> *Values are historical win percentages from NBA play-by-play data, 2019--2024.*

---

## Conclusion

**With 36--50 seconds remaining, calling a timeout is historically beneficial** (average gain: +10.4 pp across all 8 buckets in this window). Below 36 s the data is mixed — results are close to even. Rely on matchup specifics rather than a universal rule in the final 30 seconds.
