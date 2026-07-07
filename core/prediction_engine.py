"""
OracleXI v2.0 – Prediction Engine
======================================
The central controller that coordinates data loading, preprocessing,
statistical analysis, simulation, and machine learning to produce
a final ensemble prediction and a rich comparison payload for the GUI.
"""

from typing import Dict, Any, Optional

import numpy as np

from core.data_engine import DataEngine
from core.preprocessing import DataPreprocessor
from core.simulation import MonteCarloSimulator
from core.statistics import StatisticalAnalyzer
from core.tensorflow_model import TensorFlowPredictor
from utils.constants import ENSEMBLE_WEIGHTS, DEFAULT_SIMULATION_COUNT
from utils.helper import setup_logging, validate_team_selection, clamp

logger = setup_logging("PredictionEngine")


class PredictionEngine:
    """
    Central orchestration engine for sports predictions.
    Computes ensemble predictions and returns rich GUI comparison data.
    """

    def __init__(self) -> None:
        """Initialize the PredictionEngine and all sub-modules."""
        logger.info("Initializing PredictionEngine...")

        # Data Layer
        self.data_engine = DataEngine()

        # Analytics & ML Layers
        self.preprocessor = DataPreprocessor(self.data_engine)
        self.stats_analyzer = StatisticalAnalyzer(self.data_engine)
        self.simulator = MonteCarloSimulator(self.data_engine)
        self.tf_predictor = TensorFlowPredictor()

        # Load datasets on startup
        try:
            self.data_engine.load_football_data()
            self.data_engine.load_cricket_data()
        except Exception as e:
            logger.error(f"Error during initial data load: {e}")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def get_football_teams(self) -> list[str]:
        return self.data_engine.get_football_teams()

    def get_cricket_teams(self) -> list[str]:
        return self.data_engine.get_cricket_teams()

    def get_dataset_summary(self) -> Dict[str, Any]:
        return self.data_engine.get_dataset_summary()

    def get_prediction_history(self) -> list[Dict[str, Any]]:
        from utils.helper import load_prediction_history
        return load_prediction_history()

    def add_dataset(self, filepath: str) -> Dict[str, Any]:
        """Add a new dataset via the DatasetManager."""
        res = self.data_engine.dataset_manager.add_dataset(filepath)
        if res.get("success"):
            # Reload cached engine data
            self.data_engine._football_df = None
            self.data_engine._cricket_match_df = None
            self.data_engine._cricket_ball_df = None
            self.data_engine._stats_cache = {}
            self.data_engine.load_football_data()
            self.data_engine.load_cricket_data()
        return res

    def predict_football(
        self,
        team_1: str,
        team_2: str,
        category: str = None,
        n_simulations: int = DEFAULT_SIMULATION_COUNT,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive ensemble prediction for a football match.
        """
        valid, msg = validate_team_selection(team_1, team_2, self.get_football_teams())
        if not valid:
            return {"error": msg}

        logger.info(f"Predicting Football: {team_1} vs {team_2}")

        # 1. Statistical Analysis
        stats_pred = self.stats_analyzer.analyze_football(team_1, team_2)

        # 2. Monte Carlo Simulation
        sim_pred = self.simulator.simulate_football(team_1, team_2, n_simulations)

        # 3. TensorFlow ML
        tf_pred = {"team_a_win": 0.4, "draw": 0.2, "team_b_win": 0.4, "confidence": 0}
        if self.tf_predictor.is_model_trained("football"):
            features = self.preprocessor.extract_football_features(team_1, team_2)
            if features is not None:
                tf_pred = self.tf_predictor.predict(features, "football")

        # 4. Ensemble Voting
        w_stats = ENSEMBLE_WEIGHTS["statsmodels"]
        w_sim = ENSEMBLE_WEIGHTS["monte_carlo"]
        w_tf = ENSEMBLE_WEIGHTS["tensorflow"] if self.tf_predictor.is_model_trained("football") else 0.0

        total_weight = w_stats + w_sim + w_tf
        if total_weight == 0:
            total_weight = 1.0

        prob_1 = (stats_pred["team_a_win"] * w_stats + sim_pred["team_a_win"] * w_sim + tf_pred["team_a_win"] * w_tf) / total_weight
        prob_d = (stats_pred["draw"] * w_stats + sim_pred["draw"] * w_sim + tf_pred["draw"] * w_tf) / total_weight
        prob_2 = (stats_pred["team_b_win"] * w_stats + sim_pred["team_b_win"] * w_sim + tf_pred["team_b_win"] * w_tf) / total_weight

        # Normalize
        total = prob_1 + prob_d + prob_2
        prob_1 /= total
        prob_d /= total
        prob_2 /= total

        # Confidence
        confidence = (stats_pred["confidence"] * w_stats + sim_pred["confidence"] * w_sim + tf_pred["confidence"] * w_tf) / total_weight

        # Determine winner
        if prob_1 > prob_2 and prob_1 > prob_d:
            winner = team_1
        elif prob_2 > prob_1 and prob_2 > prob_d:
            winner = team_2
        else:
            winner = "Draw"

        # 5. Build rich comparison payload for GUI
        comparison = self.data_engine.get_team_comparison("Football", team_1, team_2, category)

        # 6. TF Player Spotlight
        t1_players = self.data_engine.get_top_football_scorers(team_1, category)
        t2_players = self.data_engine.get_top_football_scorers(team_2, category)
        top_players = self.tf_predictor.predict_top_players("Football", t1_players, t2_players)

        return {
            "sport": "Football",
            "team_1": team_1,
            "team_2": team_2,
            "prediction": {
                "winner": winner,
                "team_1_win": clamp(prob_1),
                "draw": clamp(prob_d),
                "team_2_win": clamp(prob_2),
                "confidence": clamp(confidence),
            },
            "comparison": comparison,
            "top_players": top_players,
            "models": {
                "statsmodels": stats_pred,
                "monte_carlo": sim_pred,
                "tensorflow": tf_pred,
            },
        }

    def predict_cricket(
        self,
        team_1: str,
        team_2: str,
        category: str = None,
        n_simulations: int = DEFAULT_SIMULATION_COUNT,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive ensemble prediction for a cricket match.
        """
        valid, msg = validate_team_selection(team_1, team_2, self.get_cricket_teams())
        if not valid:
            return {"error": msg}

        logger.info(f"Predicting Cricket: {team_1} vs {team_2}")

        # 1. Statistical Analysis
        stats_pred = self.stats_analyzer.analyze_cricket(team_1, team_2)

        # 2. Monte Carlo Simulation
        sim_pred = self.simulator.simulate_cricket(team_1, team_2, n_simulations)

        # 3. TensorFlow ML
        tf_pred = {"team_a_win": 0.5, "team_b_win": 0.5, "confidence": 0}
        if self.tf_predictor.is_model_trained("cricket"):
            features = self.preprocessor.extract_cricket_features(team_1, team_2)
            if features is not None:
                tf_pred = self.tf_predictor.predict(features, "cricket")

        # 4. Ensemble Voting
        w_stats = ENSEMBLE_WEIGHTS["statsmodels"]
        w_sim = ENSEMBLE_WEIGHTS["monte_carlo"]
        w_tf = ENSEMBLE_WEIGHTS["tensorflow"] if self.tf_predictor.is_model_trained("cricket") else 0.0

        total_weight = w_stats + w_sim + w_tf
        if total_weight == 0:
            total_weight = 1.0

        prob_1 = (stats_pred["team_a_win"] * w_stats + sim_pred["team_a_win"] * w_sim + tf_pred["team_a_win"] * w_tf) / total_weight
        prob_2 = (stats_pred["team_b_win"] * w_stats + sim_pred["team_b_win"] * w_sim + tf_pred["team_b_win"] * w_tf) / total_weight

        total = prob_1 + prob_2
        prob_1 /= total
        prob_2 /= total

        confidence = (stats_pred["confidence"] * w_stats + sim_pred["confidence"] * w_sim + tf_pred["confidence"] * w_tf) / total_weight

        winner = team_1 if prob_1 > prob_2 else team_2

        # 5. Build rich comparison payload for GUI
        comparison = self.data_engine.get_team_comparison("Cricket", team_1, team_2, category)

        # 6. TF Player Spotlight
        t1_players = self.data_engine.get_top_cricket_players(team_1, category)
        t2_players = self.data_engine.get_top_cricket_players(team_2, category)
        top_players = self.tf_predictor.predict_top_players("Cricket", t1_players, t2_players)

        return {
            "sport": "Cricket",
            "team_1": team_1,
            "team_2": team_2,
            "prediction": {
                "winner": winner,
                "team_1_win": clamp(prob_1),
                "team_2_win": clamp(prob_2),
                "confidence": clamp(confidence),
            },
            "comparison": comparison,
            "top_players": top_players,
            "models": {
                "statsmodels": stats_pred,
                "monte_carlo": sim_pred,
                "tensorflow": tf_pred,
            },
        }

    # ─────────────────────────────────────────
    # ML Training Trigger
    # ─────────────────────────────────────────

    def train_models(self) -> Dict[str, Any]:
        """Trigger training of TensorFlow models for both sports."""
        results = {"football": False, "cricket": False}
        
        logger.info("Starting model training for Football...")
        X_fb, y_fb = self.preprocessor.prepare_football_training_data()
        if len(X_fb) > 0:
            res_fb = self.tf_predictor.train(X_fb, y_fb, sport="football")
            results["football"] = res_fb.get("trained", False)
            
        logger.info("Starting model training for Cricket...")
        X_cr, y_cr = self.preprocessor.prepare_cricket_training_data()
        if len(X_cr) > 0:
            res_cr = self.tf_predictor.train(X_cr, y_cr, sport="cricket")
            results["cricket"] = res_cr.get("trained", False)
            
        return results
