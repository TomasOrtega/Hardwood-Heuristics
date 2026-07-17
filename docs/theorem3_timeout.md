# Theorem 3: The Late-Game Timeout

## Claim

> **When trailing by up to three points (or tied), is calling a timeout better
> than playing on with 20--50 seconds remaining?**

---

## How We Measure It

We filter the historical play-by-play log for situations where:

- Either the home or away team has possession
- That team is trailing by up to 3 points, or tied
- Between 20 and 50 seconds remain

We group possessions by:

- **Timeout:** The team stops play with a timeout call.
- **Play On:** The team's first action is a shot or turnover.

Other first actions are excluded. We calculate the possessing team's historical
win percentage and save observation counts and pointwise 95% uncertainty
intervals for each group. Each interval uses the wider limits from a
game-cluster bootstrap and Wilson/Newcombe finite-sample bounds. Games are
resampled as blocks so repeated clock values and overtime periods from one game
remain together. This is descriptive: timeout availability, team quality, and
why a coach stopped play are not controlled for.

---

## Results

![Late-Game Timeout Win Percentage Curve](assets/images/timeout_ev_curve.svg)

---

## Conclusion

Calling timeout had the higher observed win rate at 11 of 16 comparable clock points. The 95% interval excluded zero in favor of timeout at 0 points. This association does not show that the timeout itself caused the difference.
