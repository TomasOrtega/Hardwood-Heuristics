"""
Unit tests for mdp_engine.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mdp_engine import (
    ACTIONS,
    MDPSolver,
    Theorem1TwoForOne,
    Theorem2FoulUp3,
    TransitionModel,
    terminal_reward,
    SCORE_RANGE,
    TIME_BINS,
    POSSESSIONS,
    FOULS_RANGE,
)


# ---------------------------------------------------------------------------
# terminal_reward
# ---------------------------------------------------------------------------
class TestTerminalReward:
    def test_home_win(self):
        assert terminal_reward(5) == 1.0

    def test_away_win(self):
        assert terminal_reward(-3) == -1.0

    def test_tie(self):
        assert terminal_reward(0) == 0.0

    def test_single_point(self):
        assert terminal_reward(1) == 1.0
        assert terminal_reward(-1) == -1.0


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------
class TestTransitionModel:
    def setup_method(self):
        self.model = TransitionModel()

    def test_shoot_probabilities_sum_to_one(self):
        state = (0, 30, 1, 1)
        outcomes = self.model.shoot_transitions(state)
        total = sum(p for _, p, _ in outcomes)
        assert abs(total - 1.0) < 1e-9, f"shoot probs sum to {total}"

    def test_foul_probabilities_sum_to_one(self):
        state = (3, 8, 0, 1)
        outcomes = self.model.foul_transitions(state)
        total = sum(p for _, p, _ in outcomes)
        assert abs(total - 1.0) < 1e-9, f"foul probs sum to {total}"

    def test_hold_probabilities_sum_to_one(self):
        state = (0, 60, 1, 1)
        outcomes = self.model.hold_transitions(state)
        total = sum(p for _, p, _ in outcomes)
        assert abs(total - 1.0) < 1e-9, f"hold probs sum to {total}"

    def test_shoot_outcome_states_are_valid_tuples(self):
        state = (0, 30, 1, 1)
        outcomes = self.model.shoot_transitions(state)
        for sp, p, r in outcomes:
            assert len(sp) == 4
            assert 0.0 <= p <= 1.0

    def test_foul_reduces_fouls_to_give(self):
        state = (3, 8, 0, 2)
        outcomes = self.model.foul_transitions(state)
        for sp, _, _ in outcomes:
            assert sp[3] <= state[3]  # ftg can only decrease or stay same

    def test_custom_fg3_pct(self):
        model_hot = TransitionModel(fg3_pct=0.50)
        model_cold = TransitionModel(fg3_pct=0.20)
        state = (0, 30, 1, 1)
        hot_outcomes = {sp: p for sp, p, _ in model_hot.shoot_transitions(state)}
        cold_outcomes = {sp: p for sp, p, _ in model_cold.shoot_transitions(state)}
        # Better 3PT% → higher cumulative probability of 3-point makes
        hot_3pt_prob = sum(
            p for sp, p in hot_outcomes.items()
            if sp[0] > state[0] + 2  # +3 implies a 3-pt make
        )
        cold_3pt_prob = sum(
            p for sp, p in cold_outcomes.items()
            if sp[0] > state[0] + 2
        )
        assert hot_3pt_prob > cold_3pt_prob

    def test_time_decreases_after_all_actions(self):
        state = (0, 60, 1, 1)
        for method in [
            self.model.shoot_transitions,
            self.model.foul_transitions,
            self.model.hold_transitions,
        ]:
            outcomes = method(state)
            for sp, _, _ in outcomes:
                assert sp[1] <= state[1], f"Time did not decrease: {sp[1]} > {state[1]}"


# ---------------------------------------------------------------------------
# MDPSolver
# ---------------------------------------------------------------------------
class TestMDPSolver:
    def test_solve_returns_values_and_policy(self):
        solver = MDPSolver()
        values, policy = solver.solve()
        assert isinstance(values, dict)
        assert isinstance(policy, dict)

    def test_values_bounded(self):
        solver = MDPSolver()
        values, _ = solver.solve()
        for s, v in values.items():
            assert -1.0 <= v <= 1.0, f"Value out of bounds at {s}: {v}"

    def test_terminal_states_match_reward(self):
        solver = MDPSolver()
        values, _ = solver.solve()
        for sd in SCORE_RANGE:
            for pos in POSSESSIONS:
                for ftg in FOULS_RANGE:
                    s = (sd, 0, pos, ftg)
                    expected = terminal_reward(sd)
                    assert values.get(s, expected) == expected, (
                        f"Terminal value mismatch at {s}: got {values.get(s)}"
                    )

    def test_home_leading_has_positive_value(self):
        """With 30 seconds left and home up by 5, EV should be strongly positive."""
        solver = MDPSolver()
        values, _ = solver.solve()
        s = MDPSolver._snap_state((5, 30, 0, 1))
        assert values.get(s, 0) > 0.5

    def test_away_leading_has_negative_value(self):
        """With 30 seconds left and away up by 5, EV should be strongly negative."""
        solver = MDPSolver()
        values, _ = solver.solve()
        s = MDPSolver._snap_state((-5, 30, 1, 1))
        assert values.get(s, 0) < -0.5

    def test_policy_contains_valid_actions(self):
        solver = MDPSolver()
        _, policy = solver.solve()
        valid = set(ACTIONS) | {"terminal"}
        for s, a in policy.items():
            assert a in valid, f"Invalid action '{a}' for state {s}"

    def test_snap_state(self):
        snapped = MDPSolver._snap_state((15, 47, 3, 5))
        sd, sec, pos, ftg = snapped
        assert -10 <= sd <= 10
        assert sec % 5 == 0
        assert pos in (0, 1)
        assert 0 <= ftg <= 2

    def test_coverage_of_all_states(self):
        """solve() should populate values for every state in the grid."""
        solver = MDPSolver()
        values, policy = solver.solve()
        n_expected = (
            len(SCORE_RANGE) * len(TIME_BINS) * len(POSSESSIONS) * len(FOULS_RANGE)
        )
        assert len(values) == n_expected
        assert len(policy) == n_expected


# ---------------------------------------------------------------------------
# Theorem1TwoForOne
# ---------------------------------------------------------------------------
class TestTheorem1TwoForOne:
    def test_compute_returns_expected_keys(self):
        t1 = Theorem1TwoForOne()
        result = t1.compute()
        assert "ev_rush" in result
        assert "ev_normal" in result
        assert "ev_gain" in result
        assert "rush_is_optimal" in result

    def test_ev_values_are_float(self):
        t1 = Theorem1TwoForOne()
        result = t1.compute()
        assert isinstance(result["ev_rush"], float)
        assert isinstance(result["ev_normal"], float)

    def test_ev_values_bounded(self):
        t1 = Theorem1TwoForOne()
        result = t1.compute()
        assert -1.0 <= result["ev_rush"]   <= 1.0
        assert -1.0 <= result["ev_normal"] <= 1.0

    def test_ev_gain_consistency(self):
        t1 = Theorem1TwoForOne()
        result = t1.compute()
        assert abs(result["ev_gain"] - (result["ev_rush"] - result["ev_normal"])) < 1e-9

    def test_rush_optimal_flag_matches_gain(self):
        t1 = Theorem1TwoForOne()
        result = t1.compute()
        assert result["rush_is_optimal"] == (result["ev_gain"] > 0)

    def test_sweep_time_returns_list(self):
        t1 = Theorem1TwoForOne()
        results = t1.sweep_time(time_range=list(range(10, 40, 5)))
        assert isinstance(results, list)
        assert len(results) == 6

    def test_sweep_time_keys(self):
        t1 = Theorem1TwoForOne()
        results = t1.sweep_time(time_range=[20, 30])
        for r in results:
            assert "seconds_remaining" in r
            assert "ev_rush" in r
            assert "ev_normal" in r
            assert "ev_gain" in r

    def test_sweep_time_monotonically_varies(self):
        """EV gain should have some variation across different time values."""
        t1 = Theorem1TwoForOne()
        results = t1.sweep_time(time_range=list(range(10, 60, 5)))
        gains = [r["ev_gain"] for r in results]
        # The sweep shouldn't be constant (MDP should differentiate seconds)
        assert max(gains) != min(gains)


# ---------------------------------------------------------------------------
# Theorem2FoulUp3
# ---------------------------------------------------------------------------
class TestTheorem2FoulUp3:
    def test_compute_returns_expected_keys(self):
        t2 = Theorem2FoulUp3()
        result = t2.compute()
        assert "wp_foul" in result
        assert "wp_no_foul" in result
        assert "wp_gain" in result
        assert "foul_is_optimal" in result

    def test_wp_values_bounded(self):
        t2 = Theorem2FoulUp3()
        result = t2.compute()
        assert -1.0 <= result["wp_foul"]    <= 1.0
        assert -1.0 <= result["wp_no_foul"] <= 1.0

    def test_wp_gain_consistency(self):
        t2 = Theorem2FoulUp3()
        result = t2.compute()
        assert abs(result["wp_gain"] - (result["wp_foul"] - result["wp_no_foul"])) < 1e-9

    def test_foul_optimal_flag(self):
        t2 = Theorem2FoulUp3()
        result = t2.compute()
        assert result["foul_is_optimal"] == (result["wp_gain"] > 0)

    def test_higher_opp_3pt_pct_favors_fouling(self):
        """Against a hotter 3PT shooter, fouling should be relatively more attractive."""
        t2_cold = Theorem2FoulUp3(opp_fg3_pct=0.28)
        t2_hot  = Theorem2FoulUp3(opp_fg3_pct=0.45)
        cold_result = t2_cold.compute(seconds_remaining=5)
        hot_result  = t2_hot.compute(seconds_remaining=5)
        # Fouling gain should be higher (or less negative) vs. hot shooter
        assert hot_result["wp_gain"] >= cold_result["wp_gain"] - 0.01

    def test_sweep_returns_correct_shape(self):
        t2 = Theorem2FoulUp3()
        time_vals = [2, 4, 6]
        fg3_vals  = [0.30, 0.36, 0.42]
        grid = t2.sweep(time_values=time_vals, fg3_pct_values=fg3_vals)
        assert grid.shape == (3, 3)

    def test_sweep_values_bounded(self):
        t2 = Theorem2FoulUp3()
        grid = t2.sweep(
            time_values=[4, 8],
            fg3_pct_values=[0.30, 0.36, 0.42],
        )
        assert np.all(grid >= -2.0)
        assert np.all(grid <= 2.0)

    def test_home_wins_when_up_large(self):
        """Up by 10 with 5 seconds left: home should almost certainly win."""
        t2 = Theorem2FoulUp3()
        result = t2.compute(seconds_remaining=5, score_differential=10)
        # Both strategies should yield high win probability
        assert result["wp_foul"]    > 0.8
        assert result["wp_no_foul"] > 0.8
