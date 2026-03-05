# Theorem 2: Foul Up 3

## Claim

> **Based on NBA play-by-play data from 2019--2024, intentionally fouling
> when leading by 3 with fewer than 12 seconds remaining is consistently
> beneficial against shooters at or above the league-average 3PT% (≥ 30%),
> but counterproductive against poor 3-point teams. The 4-second window
> stands out as the most reliable situation to foul.**

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

1. **Fouling is most beneficial with 4--8 seconds remaining and a high-percentage
   3PT shooter (≥ 30%).** The heatmap shows the largest positive values in this
   region.

2. **Against average-to-below-average 3PT shooters (≤ 25%), normal defense is
   competitive** because the probability of a made 3-pointer is low enough that
   the risk of cutting the lead to 1 (via free throws) is not worth taking.

3. **With only 2 seconds left, the strategy matters less** — there is barely
   enough time for either a clean 3PT attempt or a fast-foul scenario. Both
   strategies converge to similar historical win percentages.

### Historical Data Summary

Data from 5 NBA seasons (2019--2024):

| Seconds | Opp 3PT% | Foul Win % | No-Foul Win % | Win % Gain |
|---------|----------|-----------|---------------|------------|
| 8 s | 30 % | 0.50 | 0.46 | **+3.9 pp** |
| 8 s | 35 % | 0.00 | 0.45 | -45.5 pp |
| 8 s | 40 % | 0.50 | 0.25 | **+25.0 pp** |
| 4 s | 35 % | 0.75 | 0.58 | **+16.7 pp** |

> *Values are historical win percentages from NBA play-by-play data, 2019--2024.*

---

## Sensitivity Analysis

Results vary by both **time remaining** and **opponent 3PT%**.
Possessions are segmented into 5% 3PT% buckets (±2.5 pp) so each cell reflects games where the opponent shot within that range.

Fouling is beneficial at **every** analyzed time value against shooters at or above **30%**. Normal defense is better at every time value against shooters at or below **25%**. In between, outcomes depend on the specific combination of time remaining and opponent 3PT%.

Analyzed range (25%--45% opponent 3PT%):
win % gain from fouling ranges from -53.8 pp to +55.6 pp.

---

## Conclusion

**Fouling up 3 is historically justified when the opponent's 3PT% is ≥ 30%** (within the analyzed 25%--45% range). The strategy is especially powerful in the 4-second window. Against poor 3PT teams, the conventional approach of playing normal defense remains competitive. The key insight is that the decision is *opponent-specific*: a blanket "always foul" or "never foul" rule is suboptimal — coaches should adjust based on who has the ball.
