# Theorem 1: The 2-for-1

## Claim

> **Rushing a shot in tied games is beneficial when there is more than one possession remaining.**

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

The 2-for-1 shows a positive signal for most of the analyzed time range.
