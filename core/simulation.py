"""
OracleXI v2.0 – Monte Carlo Simulator
==========================================
Simulates thousands of match outcomes based on Elo ratings,
recent form, and statistical distributions.
"""

from typing import Dict

import numpy as np

from core.data_engine import DataEngine
from utils.constants import DEFAULT_SIMULATION_COUNT
from utils.helper import setup_logging

logger = setup_logging("MonteCarloSimulator")


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for match outcome probabilities.
    Uses Poisson distributions for football and normally distributed
    run margins for cricket, centered around Elo expected outcomes.
    """

    def __init__(self, data_engine: DataEngine) -> None:
        """Initialize the simulator."""
        self.data_engine = data_engine
        np.random.seed()
        logger.info("MonteCarloSimulator initialized")

    def simulate_football(
        self,
        team1: str,
        team2: str,
        n_simulations: int = DEFAULT_SIMULATION_COUNT,
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation for a football match.
        """
        s1 = self.data_engine.get_football_team_stats(team1)
        s2 = self.data_engine.get_football_team_stats(team2)

        # Baseline expected goals
        base_goals_1 = max(s1["avg_goals_scored"] * 0.7 + s2["avg_goals_conceded"] * 0.3, 0.5)
        base_goals_2 = max(s2["avg_goals_scored"] * 0.7 + s1["avg_goals_conceded"] * 0.3, 0.5)

        # Elo scaling
        elo_diff = s1["elo_rating"] - s2["elo_rating"]
        # 100 elo diff = ~0.2 goals advantage
        elo_factor_1 = max(0, elo_diff / 500.0)
        elo_factor_2 = max(0, -elo_diff / 500.0)

        # Form factor
        form_diff = s1["form_5"] - s2["form_5"]
        form_factor_1 = max(0, form_diff * 0.5)
        form_factor_2 = max(0, -form_diff * 0.5)

        lam1 = base_goals_1 + elo_factor_1 + form_factor_1
        lam2 = base_goals_2 + elo_factor_2 + form_factor_2

        # Simulate scores
        sim_goals_1 = np.random.poisson(lam1, n_simulations)
        sim_goals_2 = np.random.poisson(lam2, n_simulations)

        wins1 = np.sum(sim_goals_1 > sim_goals_2)
        wins2 = np.sum(sim_goals_2 > sim_goals_1)
        draws = n_simulations - wins1 - wins2

        return {
            "team_a_win": wins1 / n_simulations,
            "draw": draws / n_simulations,
            "team_b_win": wins2 / n_simulations,
            "confidence": min(abs(wins1 - wins2) / n_simulations * 1.5, 1.0),
        }

    def simulate_cricket(
        self,
        team1: str,
        team2: str,
        n_simulations: int = DEFAULT_SIMULATION_COUNT,
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation for a cricket match.
        """
        s1 = self.data_engine.get_cricket_team_stats(team1)
        s2 = self.data_engine.get_cricket_team_stats(team2)

        # Combine win rate and form
        base_str_1 = (s1["win_rate"] * 0.5) + (s1["form_5"] * 0.5)
        base_str_2 = (s2["win_rate"] * 0.5) + (s2["form_5"] * 0.5)

        # Elo scaling
        elo_diff = s1["elo_rating"] - s2["elo_rating"]
        elo_prob_1 = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        elo_prob_2 = 1.0 - elo_prob_1

        # Final probabilities
        prob1 = (base_str_1 * 0.3) + (elo_prob_1 * 0.7)
        prob2 = (base_str_2 * 0.3) + (elo_prob_2 * 0.7)

        # Normalize
        total = prob1 + prob2
        if total == 0:
            prob1, prob2 = 0.5, 0.5
        else:
            prob1 /= total
            prob2 /= total

        # Run simulations (simple binomial for win/loss)
        simulations = np.random.random(n_simulations)
        wins1 = np.sum(simulations < prob1)
        wins2 = n_simulations - wins1

        return {
            "team_a_win": wins1 / n_simulations,
            "team_b_win": wins2 / n_simulations,
            "confidence": min(abs(wins1 - wins2) / n_simulations * 1.5, 1.0),
        }
