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

1. **Calling a timeout is historically beneficial across all analyzed windows** (20–50 s). The average win % gain is **+50.0 pp**.

2. **The advantage peaks around 20 s** (gain ≈ +50.0 pp) and is smallest near 20 s (gain ≈ +50.0 pp).

3. **The result is consistent**: calling a timeout to set up the final possession appears to help regardless of how many seconds remain in the analyzed window.

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Seconds | Timeout Win % | Play-On Win % | Win % Gain | Better Strategy |
|---------|--------------|--------------|------------|----------------|
| 40 s | 0.50 | 0.00 | **+0.50** | Timeout ✓ |
| 30 s | 0.50 | 0.00 | **+0.50** | Timeout ✓ |
| 20 s | 0.50 | 0.00 | **+0.50** | Timeout ✓ |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Conclusion

**The historical data favours calling a timeout** when trailing with 20–50 seconds remaining (average win % gain: +50.0 pp). Setting up the final possession appears to be beneficial — coaches should use available timeouts to organise their offence in this window.
