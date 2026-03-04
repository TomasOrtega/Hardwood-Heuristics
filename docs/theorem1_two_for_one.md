# Theorem 1: The 2-for-1

## Claim

> **Rushing a shot with approximately 32 seconds remaining — so that your team
> gets a second possession before time expires while the opponent only gets one —
> increases expected win probability.**

---

## Motivation

The "2-for-1" is one of the most recognisable situational strategies in
basketball. With ~35 seconds on the game clock and possession, a team can shoot
quickly (say, with 28–32 s left), allow the opponent a *single* possession (~24 s
each), and then get the ball back with 4–8 s remaining for a final heave.
Net result: **two shots vs. one for the opponent**.

The open question is *when exactly the math tips in favour of rushing*. Shooting
too early (e.g. 45+ seconds remaining) gifts the opponent time for *two*
possessions themselves; shooting too late means the second possession is only a
Hail Mary.

---

## MDP Formulation

We fix possession at the home team ($\pi = 1$) and a tied game ($\Delta = 0$)
and evaluate two strategies:

**Strategy A – Rush:** Take a shot immediately at $\tau$ seconds remaining.

$$\text{EV}_{\text{rush}}(\tau) = \sum_{s'} P(s' \mid s_\tau, \texttt{shoot}) \cdot V^*(s')$$

**Strategy B – Normal:** Hold the ball for a full-length possession (~24 s),
then shoot with $\tau - 24$ seconds remaining.

$$\text{EV}_{\text{normal}}(\tau) = \sum_{s_h} P(s_h \mid s_\tau, \texttt{hold}) \cdot V^*(s_h)$$

The **EV gain** from rushing is:

$$\Delta\text{EV}(\tau) = \text{EV}_{\text{rush}}(\tau) - \text{EV}_{\text{normal}}(\tau)$$

When $\Delta\text{EV}(\tau) > 0$ rushing is optimal; when it is negative, a
normal possession is preferred.

---

## Results

![2-for-1 EV Curve](assets/images/two_for_one_ev_curve.png)

### Key Findings

The MDP analysis reveals several important thresholds:

1. **Rush advantage holds across the full analyzed window (10–64 s).** The EV gain from rushing is positive throughout.
   The advantage is largest around ~10 s and smallest near ~28 s.

2. **Classic 2-for-1 window (~28–34 s): modest but consistent advantage.** Average EV gain ≈ +0.03 in this range — rushing ensures two possessions vs. the opponent's one.

3. **The gain is larger at shorter time values** because holding for a full
   possession at < 25 s consumes most of the remaining clock, while rushing
   still gives the team a realistic scoring opportunity before time expires.

### Numerical Summary

Using league-average parameters:

| Scenario | $\text{EV}_{\text{rush}}$ | $\text{EV}_{\text{normal}}$ | $\Delta\text{EV}$ | Optimal |
|----------|------|--------|------|---------|
| 32 s, tied | 0.47 | 0.44 | **+0.02** | Rush ✓ |
| 40 s, tied | 0.48 | 0.44 | **+0.04** | Rush ✓ |
| 20 s, tied | 0.47 | 0.22 | **+0.25** | Rush ✓ |

> *Values are home win-probability estimates from backward-induction solver.*

---

## Methodology Notes

* The analytic transition model uses the following league-average statistics
  (2019–24 regular season):
  - 2PT FG%: **52%**   |   3PT FG%: **36%**   |   FT%: **77%**
  - Turnover rate: **12%**   |   Foul-drawn rate: **15%**
* Strategy B ("Normal") models a typical possession of 20–25 seconds. For very
  short time values (< 25 s), the hold duration is clipped to the available clock.
* The "foul" action is restricted to defensive states (away possession) where the
  home team intentionally fouls the ball-handler. From home possession the shoot
  action already accounts for foul-drawn probability, so no separate foul shortcut
  is available — this keeps the value function grounded in realistic outcomes.
* The model treats rebounds probabilistically rather than tracking individual
  players.

---

## Conclusion

**The rush advantage holds across the full analyzed range (10–64 s).**
Shooting immediately is always at least as good as holding for a full possession
within the analyzed window. The gain is most modest in the classic 2-for-1 zone
(~28–34 s), where the difference between rush and normal is small but consistently
positive — coaches should push the pace when the clock is in this range to ensure
two possessions. Much earlier or later in the game, other strategic considerations
dominate.
