"""
mdp_engine.py
=============
Implements a finite-horizon Markov Decision Process solver using backward
induction (value iteration) and provides simulation harnesses for two
canonical NBA late-game theorems.

Bellman equation (finite horizon)
----------------------------------
For time step t and terminal horizon T:

    V_t(s) = max_{a ∈ A}  Σ_{s'} P(s' | s, a) · [R(s, a, s') + V_{t+1}(s')]
    V_T(s) = R_terminal(s)

where R_terminal(s) = +1 if home team wins, −1 if away team wins, 0 for tie
(used for OT in extensions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
StateKey = Tuple[int, int, int, int]   # (score_diff, seconds, possession, ftg)
Policy   = Dict[StateKey, str]         # state → optimal action label
Values   = Dict[StateKey, float]       # state → expected value (win probability)

# ---------------------------------------------------------------------------
# MDP state / action space definition
# ---------------------------------------------------------------------------
SCORE_RANGE = list(range(-10, 11))        # home − away, 21 values
TIME_BINS   = list(range(0, 185, 5))      # 5-second buckets, 37 values
POSSESSIONS = [0, 1]                      # 0 = away, 1 = home
FOULS_RANGE = [0, 1, 2]                   # fouls to give

ACTIONS = ["shoot", "foul", "hold"]       # strategic action labels

# NBA shooting percentages (empirical league averages 2019-24)
FG2_PCT   = 0.52   # 2-point field goal percentage
FG3_PCT   = 0.36   # 3-point field goal percentage
FT_PCT    = 0.77   # free-throw percentage
FG2_RATE  = 0.65   # proportion of shots that are 2-pointers (late game)
FG3_RATE  = 0.35

# Expected points per possession by action
EPP_SHOOT = FG2_RATE * FG2_PCT * 2 + FG3_RATE * FG3_PCT * 3  # ≈ 1.07
TURNOVER_PROB = 0.12   # probability of a turnover before a shot
FOUL_DRAWN_PROB = 0.15  # probability of drawing a foul on a shot attempt


# ---------------------------------------------------------------------------
# Terminal reward
# ---------------------------------------------------------------------------
def terminal_reward(score_diff: int) -> float:
    """
    Return the terminal game value from the *home* team's perspective.

    +1.0 = home win, -1.0 = away win, 0.0 = overtime (treated as neutral).
    """
    if score_diff > 0:
        return 1.0
    if score_diff < 0:
        return -1.0
    return 0.0  # tie → OT, simplified as 0.5 EV for home


# ---------------------------------------------------------------------------
# Analytic transition model
# ---------------------------------------------------------------------------
@dataclass
class TransitionModel:
    """
    Analytic (closed-form) transition model for late-game NBA possessions.

    Each action produces a probability distribution over next states.
    The model is intentionally simplified yet calibrated to league averages.
    """

    fg2_pct:          float = FG2_PCT
    fg3_pct:          float = FG3_PCT
    ft_pct:           float = FT_PCT
    fg2_rate:         float = FG2_RATE
    fg3_rate:         float = FG3_RATE
    turnover_prob:    float = TURNOVER_PROB
    foul_drawn_prob:  float = FOUL_DRAWN_PROB
    opponent_fg3_pct: float = FG3_PCT      # used for foul-up-3 analysis

    @property
    def p_2pt_make(self) -> float:
        return (1 - self.turnover_prob) * (1 - self.foul_drawn_prob) * self.fg2_rate * self.fg2_pct

    @property
    def p_3pt_make(self) -> float:
        return (1 - self.turnover_prob) * (1 - self.foul_drawn_prob) * self.fg3_rate * self.fg3_pct

    @property
    def p_miss(self) -> float:
        return (
            (1 - self.turnover_prob)
            * (1 - self.foul_drawn_prob)
            * (self.fg2_rate * (1 - self.fg2_pct) + self.fg3_rate * (1 - self.fg3_pct))
        )

    @property
    def p_turnover(self) -> float:
        return self.turnover_prob

    @property
    def p_foul_drawn(self) -> float:
        return (1 - self.turnover_prob) * self.foul_drawn_prob

    def shoot_transitions(
        self,
        state: StateKey,
        time_cost_2pt: int = 10,
        time_cost_3pt: int = 10,
        time_cost_miss: int = 8,
        time_cost_to: int = 6,
    ) -> List[Tuple[StateKey, float, float]]:
        """
        Return [(next_state, probability, reward)] for the "shoot" action.
        The home team (possession=1) takes the shot.
        """
        sd, sec, pos, ftg = state
        outcomes: List[Tuple[StateKey, float, float]] = []

        sign = 1 if pos == 1 else -1   # +1 for home shooting, −1 for away

        # 2-point make
        new_sd = np.clip(sd + sign * 2, -10, 10)
        next_sec = max(0, sec - time_cost_2pt)
        outcomes.append(((int(new_sd), next_sec, 1 - pos, ftg), self.p_2pt_make, 0.0))

        # 3-point make
        new_sd = np.clip(sd + sign * 3, -10, 10)
        next_sec = max(0, sec - time_cost_3pt)
        outcomes.append(((int(new_sd), next_sec, 1 - pos, ftg), self.p_3pt_make, 0.0))

        # Miss → opponent rebound (simplified: possession always flips on miss)
        next_sec = max(0, sec - time_cost_miss)
        outcomes.append(((sd, next_sec, 1 - pos, ftg), self.p_miss, 0.0))

        # Turnover
        next_sec = max(0, sec - time_cost_to)
        outcomes.append(((sd, next_sec, 1 - pos, ftg), self.p_turnover, 0.0))

        # Foul drawn → two free throws
        p_ft = self.p_foul_drawn
        ft_make_2 = self.ft_pct ** 2
        ft_make_1 = 2 * self.ft_pct * (1 - self.ft_pct)
        ft_miss_2 = (1 - self.ft_pct) ** 2
        next_sec_ft = max(0, sec - 12)
        # Make both
        new_sd = np.clip(sd + sign * 2, -10, 10)
        outcomes.append(((int(new_sd), next_sec_ft, 1 - pos, ftg), p_ft * ft_make_2, 0.0))
        # Make one
        new_sd = np.clip(sd + sign * 1, -10, 10)
        outcomes.append(((int(new_sd), next_sec_ft, pos, ftg), p_ft * ft_make_1, 0.0))
        # Miss both
        outcomes.append(((sd, next_sec_ft, 1 - pos, ftg), p_ft * ft_miss_2, 0.0))

        return self._normalize(outcomes)

    def foul_transitions(
        self,
        state: StateKey,
        time_cost: int = 8,
    ) -> List[Tuple[StateKey, float, float]]:
        """
        Return [(next_state, probability, reward)] for intentional foul.
        The *defending* team fouls the opponent who shoots two free throws.
        Possession *does not change* after free throws (they retain the ball
        only if a miss is rebounded, simplified here).
        """
        sd, sec, pos, ftg = state
        sign = 1 if pos == 1 else -1   # fouled team's perspective
        next_sec = max(0, sec - time_cost)
        new_ftg = max(0, ftg - 1)

        outcomes: List[Tuple[StateKey, float, float]] = []

        ft_make_2 = self.ft_pct ** 2
        ft_make_1 = 2 * self.ft_pct * (1 - self.ft_pct)
        ft_miss_2 = (1 - self.ft_pct) ** 2

        # Make both → possession flips
        new_sd = np.clip(sd + sign * 2, -10, 10)
        outcomes.append(((int(new_sd), next_sec, 1 - pos, new_ftg), ft_make_2, 0.0))
        # Make one → miss rebound retained by fouled team (simplified)
        new_sd = np.clip(sd + sign * 1, -10, 10)
        outcomes.append(((int(new_sd), next_sec, pos, new_ftg), ft_make_1, 0.0))
        # Miss both → fouled team rebound
        outcomes.append(((sd, next_sec, pos, new_ftg), ft_miss_2, 0.0))

        return self._normalize(outcomes)

    def hold_transitions(
        self,
        state: StateKey,
        time_cost_min: int = 20,
        time_cost_max: int = 30,
    ) -> List[Tuple[StateKey, float, float]]:
        """
        Dribble-out / stall possession. Distributed across time costs
        to reflect variation in how quickly the clock runs down.
        """
        sd, sec, pos, ftg = state
        outcomes: List[Tuple[StateKey, float, float]] = []

        steps = range(time_cost_min, time_cost_max + 1, 5)
        p_each = 1.0 / len(list(steps))
        for dt in steps:
            next_sec = max(0, sec - dt)
            # Small probability of turnover during dribble-out
            p_to = 0.05
            outcomes.append(((sd, next_sec, pos, ftg), p_each * (1 - p_to), 0.0))
            outcomes.append(((sd, next_sec, 1 - pos, ftg), p_each * p_to, 0.0))

        return self._normalize(outcomes)

    @staticmethod
    def _normalize(
        outcomes: List[Tuple[StateKey, float, float]]
    ) -> List[Tuple[StateKey, float, float]]:
        """Ensure probabilities sum to 1 (correct for floating point drift)."""
        total_p = sum(p for _, p, _ in outcomes)
        if total_p == 0:
            return outcomes
        return [(s, p / total_p, r) for s, p, r in outcomes]


# ---------------------------------------------------------------------------
# MDP Solver (backward induction / value iteration)
# ---------------------------------------------------------------------------
@dataclass
class MDPSolver:
    """
    Finite-horizon MDP solver using backward induction.

    Parameters
    ----------
    model : TransitionModel
        Analytic transition model for NBA possessions.
    gamma : float
        Discount factor (default 1.0 for win-probability optimisation).
    """

    model: TransitionModel = field(default_factory=TransitionModel)
    gamma: float = 1.0

    def _all_states(self) -> List[StateKey]:
        states: List[StateKey] = []
        for sd in SCORE_RANGE:
            for sec in TIME_BINS:
                for pos in POSSESSIONS:
                    for ftg in FOULS_RANGE:
                        states.append((sd, sec, pos, ftg))
        return states

    def _get_transitions(
        self, state: StateKey, action: str
    ) -> List[Tuple[StateKey, float, float]]:
        if action == "shoot":
            return self.model.shoot_transitions(state)
        if action == "foul":
            return self.model.foul_transitions(state)
        if action == "hold":
            return self.model.hold_transitions(state)
        raise ValueError(f"Unknown action: {action}")

    def solve(self) -> Tuple[Values, Policy]:
        """
        Run backward induction over the finite time horizon.

        Returns
        -------
        values : dict mapping state → optimal win probability (home perspective)
        policy : dict mapping state → optimal action label
        """
        states = self._all_states()

        # Initialise terminal values
        values: Values = {}
        for s in states:
            sd, sec, pos, ftg = s
            if sec == 0:
                values[s] = terminal_reward(sd)
            else:
                values[s] = terminal_reward(sd)  # fallback until iterated

        # Backward induction: iterate from smallest time-remaining upward
        sorted_time_bins = sorted(TIME_BINS)
        for sec in sorted_time_bins:
            for sd in SCORE_RANGE:
                for pos in POSSESSIONS:
                    for ftg in FOULS_RANGE:
                        s = (sd, sec, pos, ftg)
                        if sec == 0:
                            values[s] = terminal_reward(sd)
                            continue
                        # Evaluate each action
                        action_values: Dict[str, float] = {}
                        for action in ACTIONS:
                            trans = self._get_transitions(s, action)
                            q = 0.0
                            for sp, prob, reward in trans:
                                # Snap sp to valid grid
                                sp_snapped = self._snap_state(sp)
                                v_sp = values.get(sp_snapped, terminal_reward(sp_snapped[0]))
                                q += prob * (reward + self.gamma * v_sp)
                            action_values[action] = q
                        # Home team maximises; away team minimises
                        if pos == 1:  # home possession
                            values[s] = max(action_values.values())
                        else:         # away possession
                            values[s] = min(action_values.values())

        policy: Policy = {}
        for s in states:
            sd, sec, pos, ftg = s
            if sec == 0:
                policy[s] = "terminal"
                continue
            action_values = {}
            for action in ACTIONS:
                trans = self._get_transitions(s, action)
                q = sum(
                    prob * (reward + self.gamma * values.get(self._snap_state(sp), terminal_reward(sp[0])))
                    for sp, prob, reward in trans
                )
                action_values[action] = q
            if pos == 1:
                policy[s] = max(action_values, key=action_values.__getitem__)
            else:
                policy[s] = min(action_values, key=action_values.__getitem__)

        return values, policy

    @staticmethod
    def _snap_state(s: StateKey) -> StateKey:
        """Snap a state to the nearest valid grid point."""
        sd, sec, pos, ftg = s
        sd  = max(-10, min(10, sd))
        sec = max(0, min(180, int(round(sec / 5) * 5)))
        pos = pos % 2
        ftg = max(0, min(2, ftg))
        return (sd, sec, pos, ftg)


# ---------------------------------------------------------------------------
# Theorem simulators
# ---------------------------------------------------------------------------
@dataclass
class Theorem1TwoForOne:
    """
    Theorem 1: The 2-for-1 strategy.

    Tests whether it is optimal to rush a shot with ~32 seconds remaining
    so that your team gets a second possession before the half (or end of
    game) while the opponent only gets one.

    Compares:
      - Strategy A (rush): shoot immediately at ~32 seconds left
      - Strategy B (normal): take one full-length possession, leaving ~8 s

    Returns EV for each strategy and the EV *gain* from rushing.
    """

    model: TransitionModel = field(default_factory=TransitionModel)

    def compute(
        self,
        score_differential: int = 0,
        possession: int = 1,
        fouls_to_give: int = 1,
    ) -> Dict[str, float]:
        """
        Compute expected values for rush vs. normal at 32 seconds remaining.

        Returns
        -------
        dict with keys: ev_rush, ev_normal, ev_gain, rush_is_optimal
        """
        solver = MDPSolver(model=self.model)
        values, policy = solver.solve()

        # Strategy A: Rush shot at 32 seconds
        rush_state = MDPSolver._snap_state(
            (score_differential, 32, possession, fouls_to_give)
        )
        rush_transitions = self.model.shoot_transitions(rush_state, time_cost_2pt=10, time_cost_3pt=10)
        ev_rush = sum(
            prob * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
            for sp, prob, _ in rush_transitions
        )

        # Strategy B: Normal possession at 32 seconds (hold for ~24s, then shoot)
        normal_state = MDPSolver._snap_state(
            (score_differential, 32, possession, fouls_to_give)
        )
        hold_transitions = self.model.hold_transitions(normal_state, time_cost_min=20, time_cost_max=25)
        ev_after_hold = sum(
            prob * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
            for sp, prob, _ in hold_transitions
        )
        # Then shoot from whatever state we held into
        # The EV of "hold then shoot" is approximated as the value of the held state
        ev_normal = ev_after_hold

        ev_gain = ev_rush - ev_normal

        return {
            "ev_rush":       float(ev_rush),
            "ev_normal":     float(ev_normal),
            "ev_gain":       float(ev_gain),
            "rush_is_optimal": bool(ev_gain > 0),
        }

    def sweep_time(
        self,
        time_range: Optional[List[int]] = None,
        score_differential: int = 0,
        possession: int = 1,
    ) -> List[Dict]:
        """
        Compute EV gain for rushing across a range of seconds-remaining values.
        Returns a list of dicts suitable for plotting.
        """
        if time_range is None:
            time_range = list(range(10, 65, 2))

        solver = MDPSolver(model=self.model)
        values, _ = solver.solve()

        results = []
        for sec in time_range:
            rush_state = MDPSolver._snap_state((score_differential, sec, possession, 1))
            rush_trans = self.model.shoot_transitions(rush_state)
            ev_rush = sum(
                p * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
                for sp, p, _ in rush_trans
            )
            hold_trans = self.model.hold_transitions(
                rush_state, time_cost_min=max(5, sec - 10), time_cost_max=sec
            )
            ev_normal = sum(
                p * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
                for sp, p, _ in hold_trans
            )
            results.append(
                {
                    "seconds_remaining": sec,
                    "ev_rush":   ev_rush,
                    "ev_normal": ev_normal,
                    "ev_gain":   ev_rush - ev_normal,
                }
            )
        return results


@dataclass
class Theorem2FoulUp3:
    """
    Theorem 2: Foul when up by 3.

    Tests whether intentionally fouling (forcing two free throws) is better
    than playing normal defence when leading by 3 points with < 10 seconds left.

    The key risk of NOT fouling: the opponent hits a quick 3-pointer to tie.
    The risk of fouling: the opponent makes both FTs (−2, cuts lead to 1).

    Parameters
    ----------
    opp_ft_pct : float
        Opponent's free-throw percentage.
    opp_fg3_pct : float
        Opponent's 3-point field goal percentage (used for normal-defence risk).
    """

    opp_ft_pct:  float = FT_PCT
    opp_fg3_pct: float = FG3_PCT

    def _build_model(self, opp_fg3_pct: float) -> TransitionModel:
        return TransitionModel(
            fg3_pct=opp_fg3_pct,
            fg2_pct=FG2_PCT,
            ft_pct=self.opp_ft_pct,
            opponent_fg3_pct=opp_fg3_pct,
        )

    def compute(
        self,
        seconds_remaining: int = 8,
        score_differential: int = 3,   # home leads by 3
        possession: int = 0,           # away team has the ball
    ) -> Dict[str, float]:
        """
        Compare win probability under "foul" vs. "normal defence" strategies.

        Returns
        -------
        dict with:
            wp_foul      : home win probability if we foul
            wp_no_foul   : home win probability if we play normal defence
            wp_gain      : wp_foul − wp_no_foul (positive → foul is better)
            foul_is_optimal : bool
        """
        model = self._build_model(self.opp_fg3_pct)
        solver = MDPSolver(model=model)
        values, _ = solver.solve()

        state = MDPSolver._snap_state((score_differential, seconds_remaining, possession, 1))

        # Strategy A: Foul (home team fouls away ball-handler)
        foul_trans = model.foul_transitions(state)
        wp_foul = sum(
            p * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
            for sp, p, _ in foul_trans
        )

        # Strategy B: Normal defence — opponent shoots a 3
        shoot_trans = model.shoot_transitions(state)
        wp_no_foul = sum(
            p * values.get(MDPSolver._snap_state(sp), terminal_reward(sp[0]))
            for sp, p, _ in shoot_trans
        )

        return {
            "wp_foul":         float(wp_foul),
            "wp_no_foul":      float(wp_no_foul),
            "wp_gain":         float(wp_foul - wp_no_foul),
            "foul_is_optimal": bool(wp_foul > wp_no_foul),
        }

    def sweep(
        self,
        time_values: Optional[List[int]] = None,
        fg3_pct_values: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compute a 2-D grid of win-probability *gain* (foul − no-foul).

        Parameters
        ----------
        time_values    : list of seconds_remaining values (rows)
        fg3_pct_values : list of opponent 3PT% values (columns)

        Returns
        -------
        np.ndarray of shape (len(time_values), len(fg3_pct_values))
        containing the WP gain from fouling at each combination.
        """
        if time_values is None:
            time_values = list(range(2, 12, 2))
        if fg3_pct_values is None:
            fg3_pct_values = [round(x, 2) for x in np.arange(0.28, 0.46, 0.02)]

        grid = np.zeros((len(time_values), len(fg3_pct_values)))
        for i, sec in enumerate(time_values):
            for j, fg3 in enumerate(fg3_pct_values):
                t2 = Theorem2FoulUp3(opp_ft_pct=self.opp_ft_pct, opp_fg3_pct=fg3)
                result = t2.compute(seconds_remaining=sec)
                grid[i, j] = result["wp_gain"]
        return grid


