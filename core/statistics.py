"""
OracleXI v2.0 – Statistical Analysis
=========================================
Time-series and statistical models using statsmodels.
Incorporates Elo ratings and ARIMA/OLS for trend prediction.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

from core.data_engine import DataEngine
from utils.helper import setup_logging, safe_div

logger = setup_logging("StatisticalAnalyzer")


class StatisticalAnalyzer:
    """
    Analyzes historical trends and momentum using statistical models.
    """

    def __init__(self, data_engine: DataEngine) -> None:
        """Initialize with a reference to the DataEngine."""
        self.data_engine = data_engine
        logger.info("StatisticalAnalyzer initialized")

    def analyze_football(self, team1: str, team2: str) -> Dict[str, float]:
        """
        Perform statistical analysis for a football match.
        """
        # Get historical win rates over time
        df1 = self.data_engine.get_football_rolling_stats(team1, window=10)
        df2 = self.data_engine.get_football_rolling_stats(team2, window=10)

        # Base probabilities from overall stats and Elo
        s1 = self.data_engine.get_football_team_stats(team1)
        s2 = self.data_engine.get_football_team_stats(team2)
        elo_diff = s1["elo_rating"] - s2["elo_rating"]
        
        # Expected score from Elo
        e1 = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        e2 = 1.0 - e1

        # Momentum from ARIMA
        trend1 = self._predict_trend(df1, "rolling_win_rate") if not df1.empty else 0.0
        trend2 = self._predict_trend(df2, "rolling_win_rate") if not df2.empty else 0.0

        # OLS Goal difference trend
        goal_trend1 = self._predict_ols(df1, "goals_for") if not df1.empty else 0.0
        goal_trend2 = self._predict_ols(df2, "goals_for") if not df2.empty else 0.0

        # Combine Elo, momentum, and goal scoring trends
        score1 = e1 * 0.6 + max(trend1, 0) * 0.2 + max(goal_trend1, 0) * 0.2
        score2 = e2 * 0.6 + max(trend2, 0) * 0.2 + max(goal_trend2, 0) * 0.2

        total = score1 + score2 + 0.3  # 0.3 base draw probability factor
        prob_t1 = score1 / total
        prob_t2 = score2 / total
        prob_draw = 1.0 - (prob_t1 + prob_t2)

        return {
            "team_a_win": prob_t1,
            "draw": prob_draw,
            "team_b_win": prob_t2,
            "confidence": min(abs(prob_t1 - prob_t2) * 2, 1.0),
        }

    def analyze_cricket(self, team1: str, team2: str) -> Dict[str, float]:
        """
        Perform statistical analysis for a cricket match.
        """
        df1 = self.data_engine.get_cricket_rolling_stats(team1, window=10)
        df2 = self.data_engine.get_cricket_rolling_stats(team2, window=10)

        s1 = self.data_engine.get_cricket_team_stats(team1)
        s2 = self.data_engine.get_cricket_team_stats(team2)
        
        elo_diff = s1["elo_rating"] - s2["elo_rating"]
        e1 = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        e2 = 1.0 - e1

        trend1 = self._predict_trend(df1, "rolling_win_rate") if not df1.empty else 0.0
        trend2 = self._predict_trend(df2, "rolling_win_rate") if not df2.empty else 0.0

        run_trend1 = self._predict_ols(df1, "match_runs") if not df1.empty else 0.0
        run_trend2 = self._predict_ols(df2, "match_runs") if not df2.empty else 0.0

        # Normalize run trends (e.g. +10 runs = small bump)
        rt_norm1 = min(max(run_trend1 / 50.0, -0.2), 0.2)
        rt_norm2 = min(max(run_trend2 / 50.0, -0.2), 0.2)

        score1 = e1 * 0.7 + max(trend1, 0) * 0.15 + max(rt_norm1, 0) * 0.15
        score2 = e2 * 0.7 + max(trend2, 0) * 0.15 + max(rt_norm2, 0) * 0.15

        total = score1 + score2
        if total == 0:
            return {"team_a_win": 0.5, "team_b_win": 0.5, "confidence": 0.0}

        prob_t1 = score1 / total
        prob_t2 = score2 / total

        return {
            "team_a_win": prob_t1,
            "team_b_win": prob_t2,
            "confidence": min(abs(prob_t1 - prob_t2) * 2, 1.0),
        }

    def _predict_trend(self, df: pd.DataFrame, column: str) -> float:
        """Use ARIMA to predict the next value in a time series."""
        if df.empty or column not in df.columns or len(df.dropna()) < 10:
            return 0.0

        try:
            series = df[column].dropna().values
            model = ARIMA(series, order=(1, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=1)[0]
            # Return diff from current mean as the "momentum" indicator
            return float(forecast - np.mean(series))
        except Exception:
            return 0.0

    def _predict_ols(self, df: pd.DataFrame, column: str) -> float:
        """Use OLS linear regression to find the trend slope."""
        if df.empty or column not in df.columns or len(df.dropna()) < 5:
            return 0.0
            
        try:
            series = df[column].dropna().values
            X = np.arange(len(series))
            X = sm.add_constant(X)
            model = sm.OLS(series, X)
            results = model.fit()
            # Return the slope coefficient
            return float(results.params[1])
        except Exception:
            return 0.0
