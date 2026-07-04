"""
OracleXI v2.0 – Data Preprocessing
=======================================
Prepares data for machine learning models (TensorFlow).
Extracts 20 features per team (40 total) using the DataEngine.

Features extracted include Elo rating, multi-window form,
momentum, experience, consistency, and sport-specific stats.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data_engine import DataEngine
from utils.constants import NUM_FEATURES_PER_TEAM
from utils.helper import setup_logging

logger = setup_logging("DataPreprocessor")


class DataPreprocessor:
    """
    Transforms raw DataEngine stats into normalized ML feature vectors.
    """

    def __init__(self, data_engine: DataEngine) -> None:
        """Initialize with a reference to the DataEngine."""
        self.data_engine = data_engine
        logger.info("DataPreprocessor initialized (20 features per team)")

    def prepare_football_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare full training dataset for football from historical matches.

        Returns:
            Tuple of (X_train, y_train).
        """
        df = self.data_engine.load_football_data()
        if df.empty:
            logger.warning("No football data available for training.")
            return np.array([]), np.array([])

        X = []
        y = []

        # Iterate through matches to build features
        # Note: In a true time-series setting, we'd compute point-in-time stats.
        # For simplicity in this demo, we use overall team stats.
        for _, row in df.iterrows():
            features = self.extract_football_features(row["home_team"], row["away_team"])
            if features is not None:
                X.append(features)
                if row["result"] == "home_win":
                    y.append(0)
                elif row["result"] == "draw":
                    y.append(1)
                else:  # away_win
                    y.append(2)

        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.int32)
        logger.info(f"Prepared football training data: X={X_train.shape}, y={y_train.shape}")
        return X_train, y_train

    def prepare_cricket_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare full training dataset for cricket from historical matches.

        Returns:
            Tuple of (X_train, y_train).
        """
        match_df, _ = self.data_engine.load_cricket_data()
        if match_df.empty:
            logger.warning("No cricket data available for training.")
            return np.array([]), np.array([])

        X = []
        y = []

        for _, row in match_df.iterrows():
            features = self.extract_cricket_features(row["team1"], row["team2"])
            if features is not None:
                X.append(features)
                if row["winner"] == row["team1"]:
                    y.append(0)
                else:
                    y.append(1)

        X_train = np.array(X, dtype=np.float32)
        y_train = np.array(y, dtype=np.int32)
        logger.info(f"Prepared cricket training data: X={X_train.shape}, y={y_train.shape}")
        return X_train, y_train

    def extract_football_features(self, team1: str, team2: str) -> Optional[np.ndarray]:
        """
        Extract 40-dimensional feature vector for a football matchup.
        (20 features for team1 + 20 features for team2)
        """
        s1 = self.data_engine.get_football_team_stats(team1)
        s2 = self.data_engine.get_football_team_stats(team2)
        h2h = self.data_engine.get_football_head_to_head(team1, team2)

        if s1["total_matches"] == 0 or s2["total_matches"] == 0:
            return None

        # Build 20 features for Team 1
        t1_features = [
            s1["elo_rating"] / 2500.0,                   # 1. Normalized Elo
            s1["win_rate"],                              # 2. Overall Win Rate
            s1["form_5"],                                # 3. Short-term form
            s1["form_10"],                               # 4. Momentum
            s1["form_20"],                               # 5. Long-term form
            min(s1["avg_goals_scored"] / 4.0, 1.0),      # 6. Attack power
            min(s1["avg_goals_conceded"] / 4.0, 1.0),    # 7. Defense weakness
            min(abs(s1["goal_diff"]) / 2.0, 1.0) * np.sign(s1["goal_diff"]), # 8. Goal diff impact
            s1["consistency"],                           # 9. Consistency index
            min(s1["win_streak"] / 10.0, 1.0),           # 10. Winning streak
            min(s1["loss_streak"] / 10.0, 1.0),          # 11. Losing streak
            min(s1["total_matches"] / 200.0, 1.0),       # 12. Experience factor
            h2h["t1_win_pct"],                           # 13. H2H dominance
            (s1["draw_rate"] * 2.0),                     # 14. Draw tendency
            s1["win_rate"] * s1["form_5"],               # 15. Form * Win Rate synergy
            (s1["elo_rating"] - 1500) / 1000.0,          # 16. Elo diff from mean
            max(0.0, s1["avg_goals_scored"] - s1["avg_goals_conceded"]) / 4.0, # 17. Net attack
            s1["form_10"] - s1["form_20"],               # 18. Momentum shift
            h2h["matches"] / 50.0,                       # 19. H2H familiarity
            1.0 if s1["win_streak"] > 3 else 0.0,        # 20. Hot streak indicator
        ]

        # Build 20 features for Team 2
        t2_features = [
            s2["elo_rating"] / 2500.0,
            s2["win_rate"],
            s2["form_5"],
            s2["form_10"],
            s2["form_20"],
            min(s2["avg_goals_scored"] / 4.0, 1.0),
            min(s2["avg_goals_conceded"] / 4.0, 1.0),
            min(abs(s2["goal_diff"]) / 2.0, 1.0) * np.sign(s2["goal_diff"]),
            s2["consistency"],
            min(s2["win_streak"] / 10.0, 1.0),
            min(s2["loss_streak"] / 10.0, 1.0),
            min(s2["total_matches"] / 200.0, 1.0),
            h2h["t2_win_pct"],
            (s2["draw_rate"] * 2.0),
            s2["win_rate"] * s2["form_5"],
            (s2["elo_rating"] - 1500) / 1000.0,
            max(0.0, s2["avg_goals_scored"] - s2["avg_goals_conceded"]) / 4.0,
            s2["form_10"] - s2["form_20"],
            h2h["matches"] / 50.0,
            1.0 if s2["win_streak"] > 3 else 0.0,
        ]

        combined = np.array(t1_features + t2_features, dtype=np.float32)
        assert len(combined) == NUM_FEATURES_PER_TEAM * 2
        return combined

    def extract_cricket_features(self, team1: str, team2: str) -> Optional[np.ndarray]:
        """
        Extract 40-dimensional feature vector for a cricket matchup.
        (20 features for team1 + 20 features for team2)
        """
        s1 = self.data_engine.get_cricket_team_stats(team1)
        s2 = self.data_engine.get_cricket_team_stats(team2)
        h2h = self.data_engine.get_cricket_head_to_head(team1, team2)

        if s1["total_matches"] == 0 or s2["total_matches"] == 0:
            return None

        # Build 20 features for Team 1
        t1_features = [
            s1["elo_rating"] / 2500.0,                   # 1. Normalized Elo
            s1["win_rate"],                              # 2. Overall Win Rate
            s1["form_5"],                                # 3. Short-term form
            s1["form_10"],                               # 4. Momentum
            s1["form_20"],                               # 5. Long-term form
            min(s1["avg_runs"] / 250.0, 1.0),            # 6. Batting strength
            min(s1["economy"] / 15.0, 1.0),              # 7. Bowling economy
            min(s1["strike_rate"] / 200.0, 1.0),         # 8. Batting aggressiveness
            min(s1["avg_wickets"] / 10.0, 1.0),          # 9. Bowling penetration
            s1["consistency"],                           # 10. Consistency index
            min(s1["win_streak"] / 10.0, 1.0),           # 11. Winning streak
            min(s1["loss_streak"] / 10.0, 1.0),          # 12. Losing streak
            min(s1["total_matches"] / 200.0, 1.0),       # 13. Experience factor
            h2h["t1_win_pct"],                           # 14. H2H dominance
            s1["toss_impact"],                           # 15. Toss importance
            min(s1["powerplay_rr"] / 12.0, 1.0),         # 16. Powerplay strength
            min(s1["death_economy"] / 15.0, 1.0),        # 17. Death bowling
            (s1["elo_rating"] - 1500) / 1000.0,          # 18. Elo diff from mean
            h2h["matches"] / 50.0,                       # 19. H2H familiarity
            1.0 if s1["win_streak"] > 3 else 0.0,        # 20. Hot streak indicator
        ]

        # Build 20 features for Team 2
        t2_features = [
            s2["elo_rating"] / 2500.0,
            s2["win_rate"],
            s2["form_5"],
            s2["form_10"],
            s2["form_20"],
            min(s2["avg_runs"] / 250.0, 1.0),
            min(s2["economy"] / 15.0, 1.0),
            min(s2["strike_rate"] / 200.0, 1.0),
            min(s2["avg_wickets"] / 10.0, 1.0),
            s2["consistency"],
            min(s2["win_streak"] / 10.0, 1.0),
            min(s2["loss_streak"] / 10.0, 1.0),
            min(s2["total_matches"] / 200.0, 1.0),
            h2h["t2_win_pct"],
            s2["toss_impact"],
            min(s2["powerplay_rr"] / 12.0, 1.0),
            min(s2["death_economy"] / 15.0, 1.0),
            (s2["elo_rating"] - 1500) / 1000.0,
            h2h["matches"] / 50.0,
            1.0 if s2["win_streak"] > 3 else 0.0,
        ]

        combined = np.array(t1_features + t2_features, dtype=np.float32)
        assert len(combined) == NUM_FEATURES_PER_TEAM * 2
        return combined
