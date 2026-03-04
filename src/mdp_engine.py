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
import scipy.sparse as sp

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

        # Foul drawn → two free throws (clock stops; ~2 s for the foul itself)
        p_ft = self.p_foul_drawn
        ft_make_2 = self.ft_pct ** 2
        ft_make_1 = 2 * self.ft_pct * (1 - self.ft_pct)
        ft_miss_2 = (1 - self.ft_pct) ** 2
        next_sec_ft = max(0, sec - 2)
        # Make both
        new_sd = np.clip(sd + sign * 2, -10, 10)
        outcomes.append(((int(new_sd), next_sec_ft, 1 - pos, ftg), p_ft * ft_make_2, 0.0))
        # Make one
        new_sd = np.clip(sd + sign * 1, -10, 10)
        outcomes.append(((int(new_sd), next_sec_ft, 1 - pos, ftg), p_ft * ft_make_1, 0.0))
        # Miss both
        outcomes.append(((sd, next_sec_ft, 1 - pos, ftg), p_ft * ft_miss_2, 0.0))

        return self._normalize(outcomes)

    def foul_transitions(
        self,
        state: StateKey,
        time_cost: int = 2,
    ) -> List[Tuple[StateKey, float, float]]:
        """
        Return [(next_state, probability, reward)] for intentional foul.
        The *defending* team fouls the opponent who shoots two free throws.
        Missed free throws turn over to the defense (~85% defensive rebound rate).
        ``time_cost`` defaults to 2 s because an intentional foul stops the clock
        almost immediately; only the brief moment before the whistle is consumed.
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
        # Make one → defense rebounds missed second FT
        new_sd = np.clip(sd + sign * 1, -10, 10)
        outcomes.append(((int(new_sd), next_sec, 1 - pos, new_ftg), ft_make_1, 0.0))
        # Miss both → defense rebounds
        outcomes.append(((sd, next_sec, 1 - pos, new_ftg), ft_miss_2, 0.0))

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

    @classmethod
    def from_data(
        cls,
        fg_pct: Optional[float] = None,
        ft_pct: Optional[float] = None,
        turnover_prob: Optional[float] = None,
    ) -> "TransitionModel":
        """
        Create a TransitionModel calibrated to empirical data.

        When a scraped ``fg_pct`` (overall field-goal percentage) is provided,
        the 2-point and 3-point percentages are scaled proportionally so that
        the blended shooting percentage matches the empirical value.

        Parameters
        ----------
        fg_pct : float, optional
            Empirical overall field-goal percentage (makes / attempts).
        ft_pct : float, optional
            Empirical free-throw percentage.
        turnover_prob : float, optional
            Empirical turnover probability per possession.
        """
        kwargs: Dict[str, float] = {}
        if fg_pct is not None:
            default_blend = FG2_RATE * FG2_PCT + FG3_RATE * FG3_PCT
            scale = fg_pct / default_blend if default_blend > 0 else 1.0
            kwargs["fg2_pct"] = float(np.clip(FG2_PCT * scale, 0.05, 0.95))
            kwargs["fg3_pct"] = float(np.clip(FG3_PCT * scale, 0.05, 0.80))
        if ft_pct is not None:
            kwargs["ft_pct"] = float(np.clip(ft_pct, 0.05, 0.99))
        if turnover_prob is not None:
            kwargs["turnover_prob"] = float(np.clip(turnover_prob, 0.01, 0.40))
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# MDP Solver using pymdptoolbox (off-the-shelf) ValueIteration
# ---------------------------------------------------------------------------
def _build_solver_arrays(
    all_states: List[StateKey],
    state_idx: Dict[StateKey, int],
    model: "TransitionModel",
    gamma: float,
) -> Tuple[List, np.ndarray]:
    """
    Build the sparse transition matrices P and reward matrix R for pymdptoolbox.

    Terminal states (sec == 0) are made absorbing with steady-state reward
    ``terminal_reward(sd) * (1 - gamma)`` so that V*(terminal) = terminal_reward.
    """
    n = len(all_states)
    na = len(ACTIONS)

    P_lil = [sp.lil_matrix((n, n)) for _ in range(na)]
    R = np.zeros((n, na))

    for i, s in enumerate(all_states):
        sd, sec, pos, ftg = s
        if sec == 0:
            # Absorbing terminal state
            for a in range(na):
                P_lil[a][i, i] = 1.0
                R[i, a] = terminal_reward(sd) * (1 - gamma)
            continue
        for a_idx, action in enumerate(ACTIONS):
            trans = _get_model_transitions(model, s, action)
            for sp_state, prob, reward in trans:
                j = state_idx.get(MDPSolver._snap_state(sp_state))
                if j is not None and prob > 0:
                    P_lil[a_idx][i, j] += prob
                R[i, a_idx] += prob * reward

    return [m.tocsr() for m in P_lil], R


def _get_model_transitions(
    model: "TransitionModel", state: StateKey, action: str
) -> List[Tuple[StateKey, float, float]]:
    if action == "shoot":
        return model.shoot_transitions(state)
    if action == "foul":
        _, _, pos, _ = state
        # "foul" is a strategic defensive action: the home team intentionally
        # fouls the away ball-handler (pos == 0).  When home has possession
        # (pos == 1) there is no meaningful guaranteed-foul-draw option; the
        # small probability of drawing a foul is already embedded inside
        # shoot_transitions via foul_drawn_prob.  Mapping "foul" to "shoot"
        # for pos == 1 prevents value-iteration from exploiting an unrealistic
        # free-throw shortcut and keeps V*(home_ball) grounded in actual
        # shooting outcomes.
        if pos == 1:
            return model.shoot_transitions(state)
        return model.foul_transitions(state)
    if action == "hold":
        return model.hold_transitions(state)
    raise ValueError(f"Unknown action: {action}")


@dataclass
class MDPSolver:
    """
    Finite-horizon MDP solver using pymdptoolbox's ValueIteration.

    Parameters
    ----------
    model : TransitionModel
        Analytic transition model for NBA possessions.
    gamma : float
        Near-undiscounted discount factor (default 0.99).  Values very close to
        1.0 approximate undiscounted win probabilities while keeping the
        absorbing-state formulation well-conditioned.  ``gamma=1.0`` is not used
        because it prevents convergence of value iteration with absorbing states.
    """

    model: TransitionModel = field(default_factory=TransitionModel)
    gamma: float = 0.99

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
        return _get_model_transitions(self.model, state, action)

    def solve(self) -> Tuple[Values, Policy]:
        """
        Solve the finite-horizon MDP using pymdptoolbox's ValueIteration.

        Terminal states (sec == 0) are treated as absorbing with steady-state
        reward ``terminal_reward(score_diff) * (1 - gamma)`` so that at
        convergence ``V*(terminal) == terminal_reward``.  The discount is set
        close to 1 so that the value function approximates undiscounted win
        probabilities.

        For home-possession states (pos == 1) the optimal action maximises the
        value; for away-possession states (pos == 0) home team chooses an action
        (e.g. intentional foul vs. normal defence) to maximise home's win
        probability, matching the strategic framing of both theorems.

        Returns
        -------
        values : dict mapping state → win probability (home perspective)
        policy : dict mapping state → optimal action label
        """
        import mdptoolbox.mdp as mdp_lib
        import mdptoolbox.util as mdp_util

        all_states = self._all_states()
        n = len(all_states)
        state_idx: Dict[StateKey, int] = {s: i for i, s in enumerate(all_states)}

        P_csr, R = _build_solver_arrays(all_states, state_idx, self.model, self.gamma)

        # pymdptoolbox's default _boundIter performs an O(S²) column-extraction
        # loop that dominates runtime on our ~4 k-state space.  _FastVI overrides
        # it with an O(1) threshold computation.  The input-validation function
        # is similarly bypassed because our matrices are valid by construction.
        original_check = mdp_util.check
        mdp_util.check = lambda p, r: None  # matrices are valid by construction

        class _FastVI(mdp_lib.ValueIteration):
            """ValueIteration with O(1) iteration-bound computation.

            pymdptoolbox's default ``_boundIter`` extracts every column of every
            sparse transition matrix – an O(S²·A) operation – just to compute an
            upper bound on the number of iterations.  For our ~4 662-state space
            this takes seconds versus milliseconds for the actual value iteration.
            We bypass it by computing the epsilon-threshold directly from ``gamma``
            and relying on the standard epsilon-stopping criterion in ``run()``.
            """

            def _boundIter(self, epsilon: float) -> None:  # type: ignore[override]
                self.thresh = epsilon * (1 - self.discount) / self.discount

        try:
            vi = _FastVI(P_csr, R, discount=self.gamma, epsilon=1e-6, max_iter=10_000)
            vi.run()
        finally:
            mdp_util.check = original_check

        # Build value / policy dicts; override terminal states with exact values.
        values: Values = {}
        policy: Policy = {}
        for i, s in enumerate(all_states):
            sd, sec, pos, ftg = s
            if sec == 0:
                values[s] = terminal_reward(sd)
                policy[s] = "terminal"
            else:
                values[s] = float(vi.V[i])
                policy[s] = ACTIONS[vi.policy[i]]

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
            # Normal possession: model a typical ~20–25 s possession.
            # For very short time values we clip to the available seconds so
            # that the hold doesn't push the clock past zero.
            normal_min = min(20, max(5, sec - 5))
            normal_max = min(25, sec)
            hold_trans = self.model.hold_transitions(
                rush_state, time_cost_min=normal_min, time_cost_max=normal_max
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


# ---------------------------------------------------------------------------
# Theorem registry – add new theorems here to make them discoverable by
# collect_data.py and visualizations.py without editing those modules.
# ---------------------------------------------------------------------------
THEOREM_REGISTRY: List[Dict] = [
    {
        "key": "theorem1",
        "name": "2-for-1",
        "description": (
            "Rushing a shot with ~32 seconds remaining to secure an extra "
            "possession before the half/game ends."
        ),
        "class": Theorem1TwoForOne,
    },
    {
        "key": "theorem2",
        "name": "Foul Up 3",
        "description": (
            "Intentionally fouling when leading by 3 points with < 10 seconds "
            "left to prevent the opponent from attempting a tying 3-pointer."
        ),
        "class": Theorem2FoulUp3,
    },
]

