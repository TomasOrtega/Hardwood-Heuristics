# Theorem 2: Foul Up 3

## Claim

> **When leading by 3 points with fewer than 10 seconds remaining, the defending
> team should intentionally foul the ball-handler to prevent a game-tying
> 3-pointer. This strategy increases the home team's win probability.**

---

## Motivation

Few decisions in basketball generate more debate than whether to foul when leading
by three. The logic is seductive: deny the opponent a three-point attempt by
sending them to the free-throw line for two shots instead.

The counter-argument: the opponent makes both free throws 59% of the time
($0.77^2 \approx 0.59$), cutting the lead to one and granting them a free
throw to tie, while also creating a live-ball rebound opportunity.
Meanwhile, normal defense might hold the opponent to a
low-percentage heave.

We use the MDP framework to quantify *exactly when* fouling is beneficial and how
the opponent's 3PT shooting percentage affects the decision.

---

## MDP Formulation

The relevant game state is:

$$s = (\Delta = +3,\ \tau \in \{2, 4, 6, 8, 10\},\ \pi = 0,\ f = 1)$$

(away team has the ball, home leads by 3.)

**Strategy A – Foul:** Force two free throws.

$$\text{WP}_{\text{foul}} = \sum_{s'} P_{\text{FT}}(s' \mid s) \cdot V^*(s')$$

where the free-throw transition probabilities are:

$$P(\text{make both}) = \text{FT\%}^2, \quad
P(\text{make one}) = 2 \cdot \text{FT\%} \cdot (1 - \text{FT\%}), \quad
P(\text{miss both}) = (1 - \text{FT\%})^2$$

**Strategy B – Normal Defence:** Allow the opponent to shoot a 3-pointer.

$$\text{WP}_{\text{no\_foul}} = \sum_{s'} P_{\text{3PT}}(s' \mid s) \cdot V^*(s')$$

The **win-probability gain** from fouling is:

$$\Delta\text{WP} = \text{WP}_{\text{foul}} - \text{WP}_{\text{no\_foul}}$$

$\Delta\text{WP} > 0$ implies fouling is optimal.

---

## Results

![Foul Up 3 Heatmap](assets/images/foul_up_3_heatmap.png)

The heatmap displays $\Delta\text{WP}$ (in percentage points) across
combinations of seconds remaining (rows) and opponent 3PT% (columns).

- **Green cells** → Fouling increases win probability (positive $\Delta\text{WP}$)
- **Red cells** → Normal defense is preferred (negative $\Delta\text{WP}$)

### Key Findings

1. **Fouling is beneficial across all analyzed scenarios** (2–10 seconds remaining, opponent 3PT% 28%–44%). The WP gain ranges from **+7.3 pp** to **+11.6 pp**.

2. **The advantage of fouling grows with the opponent's 3PT%.** Against a 28% shooter the gain is smallest (+7.4 pp on average), while against a 44% shooter it is largest (+11.6 pp on average).

3. **The benefit is remarkably stable across all time values** (2–10 s):
   average WP gain at 2 s is +9.4 pp vs. +9.5 pp at 10 s —
   fouling is advisable regardless of exactly how many seconds remain.

### Numerical Summary

| Seconds | Opp 3PT% | $\text{WP}_{\text{foul}}$ | $\text{WP}_{\text{no\_foul}}$ | $\Delta\text{WP}$ |
|---------|----------|-----------|------------|---------|
| 8 s | 28 % | 1.00 | 0.93 | **+7.5 pp** |
| 8 s | 36 % | 1.00 | 0.90 | **+9.5 pp** |
| 8 s | 44 % | 1.00 | 0.88 | **+11.6 pp** |
| 4 s | 36 % | 1.00 | 0.91 | **+9.4 pp** |

> *Values are approximate; exact figures depend on the solved MDP grid.*

---

## Sensitivity Analysis

The key driver of the decision is the **opponent's 3PT%**. As the opponent's
three-point shooting ability increases, the expected cost of allowing a 3PT attempt
grows, making the foul decision more valuable.

With league-average FT% = 77%, the free-throw outcomes impose an expected
win-probability cost of roughly:

$$\text{Expected cost of foul} = 0.59 \times (-2\text{ pp}) \approx -1.2\text{ pp}$$

In practice the MDP results show that fouling is beneficial across the **full
range** of analyzed 3PT percentages (28%–44%): even against
a relatively poor 3PT team the WP gain is materially positive (+7.3 pp),
and it grows to +11.6 pp against elite 3PT shooters.

---

## Methodology Notes

* Free-throw model: independent Bernoulli trials with league-average FT% = 77%.
* A **missed second free throw** is modelled as turning over to the defense,
  aligning with the ~85% defensive rebounding rate on free throws.
* The MDP solver uses a near-undiscounted discount factor ($\gamma = 0.99$)
  so that the value function approximates true win probabilities while keeping
  value iteration well-conditioned.

---

## Conclusion

**Fouling up 3 is mathematically justified across all analyzed game situations** (2–10 seconds remaining, opponent 3PT% 28%–44%). The strategy is especially powerful against elite shooters. The key insight is that the decision is *opponent-specific*: the greater the opponent's 3PT ability, the larger the benefit of fouling — coaches should adjust based on who has the ball.
