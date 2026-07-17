# Theorem 1: The 2-for-1

## Claim

> **Does taking a quick shot in a tied game improve the chance of winning when
> 30--40 seconds remain?**

---

## How We Measure It

At each clock value from 30 to 40 seconds, we filter for tied games and include
both home and away possessions whose next logged event is a shot by the team
with the ball. We group those possessions by timing:

- **Rush:** The shot occurs within five seconds of the target clock.
- **Normal:** The shot occurs more than five seconds later.

The saved sweep reports the possessing team's historical win percentage and
the number of observations in each group. It is a descriptive association,
not a causal estimate; team quality and game context are not adjusted for.

---

## Results

![2-for-1 Win Percentage Curve](assets/images/two_for_one_ev_curve.svg)

---

## Conclusion

Rushing had the higher observed win rate at 4 of 6 comparable clock points. The result is mixed and should not be interpreted as a causal effect.
