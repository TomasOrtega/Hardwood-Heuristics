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
