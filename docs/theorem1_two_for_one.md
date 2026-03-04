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

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.png)

### Key Findings

1. **Rushing appears beneficial roughly in the ~18–22 s window** based on historical data, but the boundary is not sharp — adjacent time buckets often flip sign due to sample noise.

2. **Below ~16 s normal possession is preferred** — too little time remains for the opponent to mount a meaningful second possession, so the risk-return of rushing does not pay off historically.

3. **Above ~24 s normal possession is preferable** — rushing this early hands the opponent two possessions, negating the advantage.

4. **Sample sizes are small per bucket** — conclusions should be treated as directional signals rather than precise thresholds.

### Historical Data Summary

Data from 5 NBA seasons (2019–2024):

| Scenario | Rush Win % | Hold Win % | Win % Gain | Better Strategy |
|----------|-----------|-----------|------------|----------------|
| 32 s, tied | 0.20 | 0.25 | -0.05 | Normal ✓ |
| 40 s, tied | 0.13 | 0.29 | -0.16 | Normal ✓ |
| 20 s, tied | 0.30 | 0.18 | **+0.12** | Rush ✓ |

> *Values are historical win percentages from NBA play-by-play data, 2019–2024.*

---

## Conclusion

**The 2-for-1 shows a positive signal in roughly the 18–22 s range**, but there is no sharp, reliable threshold — individual second-by-second results are noisy. Rushing at 24+ seconds can reduce win probability. Use this as a directional guide: favour rushing when a good shot is available in this window, but do not sacrifice shot quality for a specific clock value.
