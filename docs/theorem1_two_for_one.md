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

1. **Critical window: ~28–35 seconds remaining.** This is where the EV gain from
   rushing is maximised. A team with possession in this window should consider
   pushing the pace to ensure two possessions.

2. **Below ~18 seconds: rushing is always better** because there is insufficient
   time for a normal possession *plus* a second shot.

3. **Above ~40 seconds: normal possession is preferable.** Rushing at 45+ seconds
   gives the opponent two possessions, negating the advantage.

### Numerical Summary

Using league-average parameters:

| Scenario | $\text{EV}_{\text{rush}}$ | $\text{EV}_{\text{normal}}$ | $\Delta\text{EV}$ | Optimal |
|----------|------|--------|------|---------|
| 32 s, tied | ~0.53 | ~0.50 | **+0.03** | Rush ✓ |
| 40 s, tied | ~0.51 | ~0.53 | −0.02 | Normal ✓ |
| 20 s, tied | ~0.54 | ~0.50 | **+0.04** | Rush ✓ |

> *Values are home win-probability estimates from backward-induction solver.*

---

## Methodology Notes

* The analytic transition model uses the following league-average statistics
  (2019–24 regular season):
  - 2PT FG%: **52 %**   |   3PT FG%: **36 %**   |   FT%: **77 %**
  - Turnover rate: **12 %**   |   Foul-drawn rate: **15 %**
* "Hold" action distributes time costs uniformly over 20–25 seconds to reflect
  real possession variability.
* The model treats rebounds probabilistically rather than tracking individual
  players.

---

## Conclusion

**The 2-for-1 is mathematically justified in a narrow time window (~18–35 s).**
Outside this window, the conventional wisdom breaks down. Coaches should be aware
of the exact clock time before committing to a rush — taking a shot at 42 seconds
can actually *reduce* win probability relative to playing for a clean look.
