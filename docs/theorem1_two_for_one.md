# Theorem 1: The 2-for-1

## Claim

> **Based on NBA play-by-play data from 2019–2024, teams that rush a shot with
> approximately 32 seconds remaining — gaining a second possession before time
> expires — win at a higher historical rate than teams that hold for a full
> possession.**

---

## How We Measure It

We filter the historical play-by-play log for tied games and group each
possession by strategy:

- **Rush (shoot):** The possessing team takes a shot attempt.
- **Normal (hold):** The possessing team holds the ball (any non-shooting action).

We then calculate the **historical win percentage** for each group — the
fraction of games where the home team went on to win given that choice.

---

## Results

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.png)

### Key Findings

The historical data reveals several patterns:

1. **Critical window: ~18–22 seconds remaining.** This is where the historical
   win gain from rushing is largest. A team with possession in this window
   should consider pushing the pace to ensure two possessions.

2. **Below ~16 seconds: normal possession is preferred** — insufficient time
   for the opponent to mount a meaningful second possession, so the
   risk-return of rushing does not pay off historically.

3. **Above ~24 seconds: normal possession is preferable.** Rushing at 24+ seconds
   gives the opponent two possessions, negating the advantage.

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

**The 2-for-1 is historically justified in the window ~18–22 s.**
Outside this window, the historical data suggests normal possession
is better. Coaches should be aware of the exact clock time — taking
a shot at 24+ seconds can actually reduce win
probability relative to playing for a clean look.
