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

![Late-Game Timeout Win Percentage Curve](assets/images/timeout_ev_curve.png)

### Key Findings

1. **Results are mixed**: a timeout historically helps at 8 of 16 time buckets (24–50 s) but hurts at 8 others (20–48 s).

2. **The largest timeout advantage** occurs around ~46 s (+12.4 pp). The largest disadvantage is near ~40 s (-15.0 pp).

3. **Overall the data is inconclusive**: there is no clean time window where timeouts are uniformly better or worse. The decision likely depends more on matchup and opponent tendencies than on the clock.

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Seconds | Timeout Win % | Play-On Win % | Win % Gain | Better Strategy |
|---------|--------------|--------------|------------|----------------|
| 40 s | 0.00 | 0.15 | -0.15 | Play On ✓ |
| 30 s | 0.14 | 0.07 | **+0.08** | Timeout ✓ |
| 20 s | 0.09 | 0.11 | -0.02 | Play On ✓ |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Conclusion

**The historical data is inconclusive**: a timeout does not consistently help or hurt across the analyzed 20–50 second window. On average the gain is +0.6 pp — essentially noise. Coaches should base the timeout decision on matchup specifics and player fatigue rather than treating it as a universal rule.
