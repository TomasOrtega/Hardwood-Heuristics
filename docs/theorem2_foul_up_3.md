# Theorem 2: Foul Up 3

## Claim

> **Based on NBA play-by-play data from 2019--2024, intentionally fouling
> when leading by 3 with fewer than 12 seconds remaining shows mixed
> results — outcomes depend on time remaining and are not consistently
> better in this historical sample.**

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

1. **Results vary by time remaining**: fouling is better at 2 s, 8 s but worse at 4 s, 6 s, 10 s.

2. **Opponent 3PT% does not change outcomes** in this sample — gains are identical across all analyzed shooting percentages (28%--44%).

3. **No consistent pattern**: neither always-foul nor never-foul is optimal at all time values.

### Historical Data Summary

Data from 5 NBA seasons (2019--2024):

| Seconds | Opp 3PT% | Foul Win % | No-Foul Win % | Win % Gain |
|---------|----------|-----------|---------------|------------|
| 8 s | 28 % | 0.71 | 0.67 | **+3.7 pp** |
| 8 s | 36 % | 0.71 | 0.67 | **+3.7 pp** |
| 8 s | 44 % | 0.71 | 0.67 | **+3.7 pp** |
| 4 s | 36 % | 0.81 | 0.82 | -0.8 pp |

> *Values are historical win percentages from NBA play-by-play data, 2019--2024.*

---

## Sensitivity Analysis

Results vary by **time remaining** — the opponent's 3PT% does not explain the
variation in this historical sample.

Analyzed range (28%--44% opponent 3PT%):
win % gain from fouling ranges from -17.5 pp to +6.3 pp
(driven by time remaining, not shooting %).

---

## Conclusion

**The historical data does not consistently support intentional fouling** when leading by 3 with fewer than 12 seconds left. Outcomes vary by time remaining — fouling helps at some clock values and hurts at others. Opponent 3PT% does not explain the variation in this sample.
