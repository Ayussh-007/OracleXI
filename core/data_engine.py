"""
OracleXI v2.0 – Data Engine
================================
Central data loading, Elo rating, and team comparison module.

Key changes from v1:
    - Integrates with DatasetManager for multi-CSV loading
    - Computes Elo ratings from historical results
    - Removes home/away distinction for team stats
    - Provides get_team_comparison() for the new comparison dashboard
    - Multi-window rolling stats (5, 10, 20)
    - Winning/losing streak tracking
    - Tournament-specific performance
    - Consistency and strength-of-schedule metrics
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.dataset_manager import DatasetManager
from utils.constants import (
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_K_FACTOR_TOURNAMENT,
    CRICKET_TEAMS,
    FOOTBALL_TEAMS,
)
from utils.helper import setup_logging, safe_div

logger = setup_logging("DataEngine")


class DataEngine:
    """
    Central data engine with Elo ratings, team comparison,
    and comprehensive statistics computation.
    """

    def __init__(self) -> None:
        """Initialize data engine with DatasetManager."""
        self.dataset_manager = DatasetManager()
        self._football_df: Optional[pd.DataFrame] = None
        self._cricket_match_df: Optional[pd.DataFrame] = None
        self._cricket_ball_df: Optional[pd.DataFrame] = None
        self._football_elo: Dict[str, float] = {}
        self._cricket_elo: Dict[str, float] = {}
        self._stats_cache: Dict[str, dict] = {}
        logger.info("DataEngine initialized")

    # ─────────────────────────────────────────
    # Data Loading (via DatasetManager)
    # ─────────────────────────────────────────

    def load_football_data(self) -> pd.DataFrame:
        """Load and preprocess all football data via DatasetManager."""
        if self._football_df is not None:
            return self._football_df

        df = self.dataset_manager.get_football_data()
        if df.empty:
            logger.warning("No football data available")
            return pd.DataFrame()

        # Ensure date is parsed
        if not pd.api.types.is_datetime64_any_dtype(df.get("date", pd.Series())):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df["home_score"] = df["home_score"].fillna(0).astype(int)
        df["away_score"] = df["away_score"].fillna(0).astype(int)
        df["tournament"] = df["tournament"].fillna("Unknown")

        # Feature engineering
        df["result"] = df.apply(
            lambda r: "home_win" if r["home_score"] > r["away_score"]
            else "away_win" if r["away_score"] > r["home_score"]
            else "draw", axis=1)
        df["goal_diff"] = df["home_score"] - df["away_score"]
        df["total_goals"] = df["home_score"] + df["away_score"]
        df["year"] = df["date"].dt.year

        df = df.sort_values("date").reset_index(drop=True)
        self._football_df = df

        # Compute Elo ratings
        self._compute_football_elo(df)

        logger.info(f"Loaded {len(df)} football matches ({self._count_football_teams(df)} teams)")
        return df

    def load_cricket_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all cricket data via DatasetManager."""
        if self._cricket_match_df is not None and self._cricket_ball_df is not None:
            return self._cricket_match_df, self._cricket_ball_df

        match_df = self.dataset_manager.get_cricket_match_data()
        ball_df = self.dataset_manager.get_cricket_ball_data()

        if not match_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(match_df.get("date", pd.Series())):
                match_df["date"] = pd.to_datetime(match_df["date"], errors="coerce")
            match_df["winner"] = match_df["winner"].fillna("No Result")
            match_df["year"] = match_df["date"].dt.year
            match_df = match_df.sort_values("date").reset_index(drop=True)
            self._cricket_match_df = match_df
            self._compute_cricket_elo(match_df)
            logger.info(f"Loaded {len(match_df)} cricket matches")
        else:
            self._cricket_match_df = pd.DataFrame()

        if not ball_df.empty:
            ball_df["extra_run"] = ball_df["extra_run"].fillna(0).astype(int)
            ball_df["is_wicket"] = ball_df["is_wicket"].fillna(0).astype(int)
            ball_df["total_run"] = ball_df["total_run"].fillna(0).astype(int)
            self._cricket_ball_df = ball_df
            logger.info(f"Loaded {len(ball_df)} ball-by-ball records")
        else:
            self._cricket_ball_df = pd.DataFrame()

        return self._cricket_match_df, self._cricket_ball_df

    def _count_football_teams(self, df: pd.DataFrame) -> int:
        """Count unique football teams."""
        if df.empty:
            return 0
        return len(set(df["home_team"].unique()) | set(df["away_team"].unique()))

    # ─────────────────────────────────────────
    # Elo Rating System
    # ─────────────────────────────────────────

    def _compute_football_elo(self, df: pd.DataFrame) -> None:
        """Compute Elo ratings for all football teams from historical results."""
        elo = {}
        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            if home not in elo:
                elo[home] = ELO_INITIAL
            if away not in elo:
                elo[away] = ELO_INITIAL

            # Determine K factor (higher for tournaments)
            tournament = str(row.get("tournament", ""))
            k = ELO_K_FACTOR_TOURNAMENT if "World Cup" in tournament or "Euro" in tournament or "Champions" in tournament else ELO_K_FACTOR

            # Expected scores
            e_home = 1.0 / (1.0 + 10 ** ((elo[away] - elo[home]) / 400.0))
            e_away = 1.0 - e_home

            # Actual scores
            if row["result"] == "home_win":
                s_home, s_away = 1.0, 0.0
            elif row["result"] == "away_win":
                s_home, s_away = 0.0, 1.0
            else:
                s_home, s_away = 0.5, 0.5

            elo[home] += k * (s_home - e_home)
            elo[away] += k * (s_away - e_away)

        self._football_elo = elo

    def _compute_cricket_elo(self, df: pd.DataFrame) -> None:
        """Compute Elo ratings for all cricket teams."""
        elo = {}
        for _, row in df.iterrows():
            t1, t2 = row["team1"], row["team2"]
            winner = row["winner"]
            if t1 not in elo:
                elo[t1] = ELO_INITIAL
            if t2 not in elo:
                elo[t2] = ELO_INITIAL

            tournament = str(row.get("tournament", ""))
            k = ELO_K_FACTOR_TOURNAMENT if "World Cup" in tournament or "Champions" in tournament else ELO_K_FACTOR

            e1 = 1.0 / (1.0 + 10 ** ((elo[t2] - elo[t1]) / 400.0))
            e2 = 1.0 - e1

            if winner == t1:
                s1, s2 = 1.0, 0.0
            elif winner == t2:
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            elo[t1] += k * (s1 - e1)
            elo[t2] += k * (s2 - e2)

        self._cricket_elo = elo

    def get_football_elo(self, team: str) -> float:
        """Get current Elo rating for a football team."""
        self.load_football_data()
        return self._football_elo.get(team, ELO_INITIAL)

    def get_cricket_elo(self, team: str) -> float:
        """Get current Elo rating for a cricket team."""
        self.load_cricket_data()
        return self._cricket_elo.get(team, ELO_INITIAL)

    # ─────────────────────────────────────────
    # Team Lists
    # ─────────────────────────────────────────

    def get_football_teams(self) -> List[str]:
        """Get sorted list of all football teams from data."""
        df = self.load_football_data()
        if df.empty:
            return sorted(FOOTBALL_TEAMS)
        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
        return sorted(teams)

    def get_cricket_teams(self) -> List[str]:
        """Get sorted list of all cricket teams from data."""
        match_df, _ = self.load_cricket_data()
        if match_df.empty:
            return sorted(CRICKET_TEAMS)
        teams = set(match_df["team1"].unique()) | set(match_df["team2"].unique())
        return sorted(teams)

    # ─────────────────────────────────────────
    # Football Team Stats (Unified – no home/away)
    # ─────────────────────────────────────────

    def get_football_team_stats(self, team: str) -> Dict[str, float]:
        """
        Comprehensive unified stats for a football team.
        No home/away distinction – treats team symmetrically.
        """
        cache_key = f"fb_{team}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        df = self.load_football_data()
        if df.empty:
            return self._default_football_stats()

        # All matches where team played (either side)
        home = df[df["home_team"] == team].copy()
        away = df[df["away_team"] == team].copy()

        home["goals_for"] = home["home_score"]
        home["goals_against"] = home["away_score"]
        home["is_win"] = (home["result"] == "home_win").astype(int)
        home["is_draw"] = (home["result"] == "draw").astype(int)
        home["opponent"] = home["away_team"]

        away["goals_for"] = away["away_score"]
        away["goals_against"] = away["home_score"]
        away["is_win"] = (away["result"] == "away_win").astype(int)
        away["is_draw"] = (away["result"] == "draw").astype(int)
        away["opponent"] = away["home_team"]

        matches = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
        n = len(matches)
        if n == 0:
            return self._default_football_stats()

        wins = int(matches["is_win"].sum())
        draws = int(matches["is_draw"].sum())
        losses = n - wins - draws
        goals_scored = int(matches["goals_for"].sum())
        goals_conceded = int(matches["goals_against"].sum())

        # Streaks
        win_streak, loss_streak = self._compute_streaks(matches["is_win"].values)

        # Multi-window form
        recent5 = matches.tail(5)["is_win"].mean() if n >= 5 else matches["is_win"].mean()
        recent10 = matches.tail(10)["is_win"].mean() if n >= 10 else matches["is_win"].mean()
        recent20 = matches.tail(20)["is_win"].mean() if n >= 20 else matches["is_win"].mean()

        # Consistency (lower std = more consistent)
        consistency = 1.0 - min(matches["goals_for"].std() / max(matches["goals_for"].mean(), 1), 1.0)

        # Recent form string (last 5)
        last5 = matches.tail(5)
        form_str = ""
        for _, m in last5.iterrows():
            if m["is_win"]:
                form_str += "W"
            elif m["is_draw"]:
                form_str += "D"
            else:
                form_str += "L"

        elo = self.get_football_elo(team)

        stats = {
            "total_matches": n,
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": round(safe_div(wins, n), 4),
            "draw_rate": round(safe_div(draws, n), 4),
            "loss_rate": round(safe_div(losses, n), 4),
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "avg_goals_scored": round(safe_div(goals_scored, n), 2),
            "avg_goals_conceded": round(safe_div(goals_conceded, n), 2),
            "goal_diff": round(safe_div(goals_scored - goals_conceded, n), 2),
            "elo_rating": round(elo, 1),
            "form_5": round(recent5, 4),
            "form_10": round(recent10, 4),
            "form_20": round(recent20, 4),
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "consistency": round(consistency, 4),
            "form_string": form_str,
        }

        self._stats_cache[cache_key] = stats
        return stats

    # ─────────────────────────────────────────
    # Cricket Team Stats (Unified)
    # ─────────────────────────────────────────

    def get_cricket_team_stats(self, team: str) -> Dict[str, float]:
        """Comprehensive unified stats for a cricket team."""
        cache_key = f"cr_{team}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        match_df, ball_df = self.load_cricket_data()
        if match_df.empty:
            return self._default_cricket_stats()

        team_matches = match_df[
            (match_df["team1"] == team) | (match_df["team2"] == team)
        ].sort_values("date").copy()

        n = len(team_matches)
        if n == 0:
            return self._default_cricket_stats()

        team_matches["is_win"] = (team_matches["winner"] == team).astype(int)
        wins = int(team_matches["is_win"].sum())
        losses = n - wins

        # Streaks
        win_streak, loss_streak = self._compute_streaks(team_matches["is_win"].values)

        # Multi-window form
        recent5 = team_matches.tail(5)["is_win"].mean() if n >= 5 else team_matches["is_win"].mean()
        recent10 = team_matches.tail(10)["is_win"].mean() if n >= 10 else team_matches["is_win"].mean()
        recent20 = team_matches.tail(20)["is_win"].mean() if n >= 20 else team_matches["is_win"].mean()

        # Form string
        form_str = ""
        for _, m in team_matches.tail(5).iterrows():
            form_str += "W" if m["is_win"] else "L"

        # Toss impact
        toss_wins = len(team_matches[team_matches.get("toss_winner", pd.Series()) == team]) if "toss_winner" in team_matches.columns else 0
        toss_and_match_wins = 0
        if "toss_winner" in team_matches.columns:
            toss_and_match_wins = len(team_matches[
                (team_matches["toss_winner"] == team) & (team_matches["winner"] == team)
            ])
        toss_impact = safe_div(toss_and_match_wins, max(toss_wins, 1))

        # Ball-by-ball stats
        avg_runs = 155.0
        strike_rate = 130.0
        run_rate = 7.5
        economy = 8.0
        avg_wickets = 5.0
        powerplay_rr = 7.0
        death_economy = 10.0

        if not ball_df.empty:
            batting = ball_df[ball_df["batting_team"] == team]
            bowling = ball_df[ball_df["bowling_team"] == team]

            bat_by_match = batting.groupby("match_id").agg(
                runs=("total_run", "sum"), balls=("ball", "count"),
                wickets_lost=("is_wicket", "sum"),
            )
            if len(bat_by_match) > 0:
                avg_runs = bat_by_match["runs"].mean()
                avg_balls = bat_by_match["balls"].mean()
                strike_rate = safe_div(avg_runs, avg_balls, 130) * 100
                run_rate = safe_div(avg_runs, safe_div(avg_balls, 6, 20), 7.5)

            bowl_by_match = bowling.groupby("match_id").agg(
                conceded=("total_run", "sum"), bowled=("ball", "count"),
                wk=("is_wicket", "sum"),
            )
            if len(bowl_by_match) > 0:
                economy = safe_div(bowl_by_match["conceded"].mean(),
                                   safe_div(bowl_by_match["bowled"].mean(), 6, 20), 8.0)
                avg_wickets = bowl_by_match["wk"].mean()

            # Powerplay (overs 0-5)
            pp_batting = batting[batting["over"] < 6]
            if len(pp_batting) > 0:
                pp_by_match = pp_batting.groupby("match_id").agg(r=("total_run", "sum"), b=("ball", "count"))
                if len(pp_by_match) > 0:
                    powerplay_rr = safe_div(pp_by_match["r"].mean(), safe_div(pp_by_match["b"].mean(), 6, 6), 7.0)

            # Death overs (16-19 for T20)
            death_bowling = bowling[bowling["over"] >= 16]
            if len(death_bowling) > 0:
                death_by_match = death_bowling.groupby("match_id").agg(r=("total_run", "sum"), b=("ball", "count"))
                if len(death_by_match) > 0:
                    death_economy = safe_div(death_by_match["r"].mean(), safe_div(death_by_match["b"].mean(), 6, 4), 10.0)

        # Consistency
        consistency = 1.0
        if not ball_df.empty:
            bat_runs = ball_df[ball_df["batting_team"] == team].groupby("match_id")["total_run"].sum()
            if len(bat_runs) > 1:
                consistency = 1.0 - min(bat_runs.std() / max(bat_runs.mean(), 1), 1.0)

        elo = self.get_cricket_elo(team)

        stats = {
            "total_matches": n,
            "wins": wins, "losses": losses,
            "win_rate": round(safe_div(wins, n), 4),
            "elo_rating": round(elo, 1),
            "avg_runs": round(avg_runs, 2),
            "strike_rate": round(strike_rate, 2),
            "run_rate": round(run_rate, 2),
            "avg_wickets": round(avg_wickets, 2),
            "economy": round(economy, 2),
            "powerplay_rr": round(powerplay_rr, 2),
            "death_economy": round(death_economy, 2),
            "toss_impact": round(toss_impact, 4),
            "form_5": round(recent5, 4),
            "form_10": round(recent10, 4),
            "form_20": round(recent20, 4),
            "win_streak": win_streak,
            "loss_streak": loss_streak,
            "consistency": round(consistency, 4),
            "form_string": form_str,
        }

        self._stats_cache[cache_key] = stats
        return stats

    # ─────────────────────────────────────────
    # Head-to-Head
    # ─────────────────────────────────────────

    def get_football_head_to_head(self, t1: str, t2: str) -> Dict[str, any]:
        """Get head-to-head statistics between two football teams."""
        df = self.load_football_data()
        if df.empty:
            return {"matches": 0, "t1_wins": 0, "t2_wins": 0, "draws": 0}

        h2h = df[
            ((df["home_team"] == t1) & (df["away_team"] == t2))
            | ((df["home_team"] == t2) & (df["away_team"] == t1))
        ]

        if h2h.empty:
            return {"matches": 0, "t1_wins": 0, "t2_wins": 0, "draws": 0}

        t1_wins = 0
        t2_wins = 0
        draws = 0
        for _, row in h2h.iterrows():
            if row["result"] == "draw":
                draws += 1
            elif row["result"] == "home_win":
                if row["home_team"] == t1:
                    t1_wins += 1
                else:
                    t2_wins += 1
            else:
                if row["away_team"] == t1:
                    t1_wins += 1
                else:
                    t2_wins += 1

        return {
            "matches": len(h2h),
            "t1_wins": t1_wins,
            "t2_wins": t2_wins,
            "draws": draws,
            "t1_win_pct": round(safe_div(t1_wins, len(h2h)), 4),
            "t2_win_pct": round(safe_div(t2_wins, len(h2h)), 4),
        }

    def get_cricket_head_to_head(self, t1: str, t2: str) -> Dict[str, any]:
        """Get head-to-head statistics between two cricket teams."""
        match_df, _ = self.load_cricket_data()
        if match_df.empty:
            return {"matches": 0, "t1_wins": 0, "t2_wins": 0}

        h2h = match_df[
            ((match_df["team1"] == t1) & (match_df["team2"] == t2))
            | ((match_df["team1"] == t2) & (match_df["team2"] == t1))
        ]

        if h2h.empty:
            return {"matches": 0, "t1_wins": 0, "t2_wins": 0}

        t1_wins = len(h2h[h2h["winner"] == t1])
        t2_wins = len(h2h[h2h["winner"] == t2])

        return {
            "matches": len(h2h),
            "t1_wins": t1_wins,
            "t2_wins": t2_wins,
            "t1_win_pct": round(safe_div(t1_wins, len(h2h)), 4),
            "t2_win_pct": round(safe_div(t2_wins, len(h2h)), 4),
        }

    # ─────────────────────────────────────────
    # Rolling Stats
    # ─────────────────────────────────────────

    def get_football_rolling_stats(self, team: str, window: int = 10) -> pd.DataFrame:
        """Get rolling stats for a football team (unified, no home/away)."""
        df = self.load_football_data()
        if df.empty:
            return pd.DataFrame()

        home = df[df["home_team"] == team].copy()
        home["goals_for"] = home["home_score"]
        home["goals_against"] = home["away_score"]
        home["is_win"] = (home["result"] == "home_win").astype(int)

        away = df[df["away_team"] == team].copy()
        away["goals_for"] = away["away_score"]
        away["goals_against"] = away["home_score"]
        away["is_win"] = (away["result"] == "away_win").astype(int)

        matches = pd.concat([home, away]).sort_values("date").reset_index(drop=True)
        if matches.empty:
            return pd.DataFrame()

        matches["rolling_goals_for"] = matches["goals_for"].rolling(window=window, min_periods=1).mean()
        matches["rolling_goals_against"] = matches["goals_against"].rolling(window=window, min_periods=1).mean()
        matches["rolling_win_rate"] = matches["is_win"].rolling(window=window, min_periods=1).mean()

        return matches

    def get_cricket_rolling_stats(self, team: str, window: int = 10) -> pd.DataFrame:
        """Get rolling stats for a cricket team."""
        match_df, ball_df = self.load_cricket_data()
        if match_df.empty:
            return pd.DataFrame()

        team_matches = match_df[
            (match_df["team1"] == team) | (match_df["team2"] == team)
        ].sort_values("date").copy()

        if team_matches.empty:
            return pd.DataFrame()

        team_matches["is_win"] = (team_matches["winner"] == team).astype(int)

        if not ball_df.empty:
            runs_per_match = ball_df[ball_df["batting_team"] == team].groupby("match_id")["total_run"].sum()
            team_matches = team_matches.merge(
                runs_per_match.rename("match_runs"),
                left_on="match_id", right_index=True, how="left",
            )
            team_matches["match_runs"] = team_matches["match_runs"].fillna(150)
        else:
            team_matches["match_runs"] = 150

        team_matches["rolling_win_rate"] = team_matches["is_win"].rolling(window=window, min_periods=1).mean()
        team_matches["rolling_avg_runs"] = team_matches["match_runs"].rolling(window=window, min_periods=1).mean()

        return team_matches

    # ─────────────────────────────────────────
    # Recent Form
    # ─────────────────────────────────────────

    def get_football_recent_form(self, team: str, n: int = 10) -> pd.DataFrame:
        """Get most recent N matches for a football team."""
        df = self.load_football_data()
        if df.empty:
            return pd.DataFrame()
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        return pd.concat([home, away]).sort_values("date", ascending=False).head(n)

    def get_cricket_recent_form(self, team: str, n: int = 10) -> pd.DataFrame:
        """Get most recent N matches for a cricket team."""
        match_df, _ = self.load_cricket_data()
        if match_df.empty:
            return pd.DataFrame()
        return match_df[
            (match_df["team1"] == team) | (match_df["team2"] == team)
        ].sort_values("date", ascending=False).head(n)

    # ─────────────────────────────────────────
    # Full Team Comparison (for GUI)
    # ─────────────────────────────────────────

    def get_team_comparison(self, sport: str, t1: str, t2: str) -> Dict[str, any]:
        """
        Build a full comparison dictionary for the comparison dashboard.

        Returns all metrics needed for side-by-side display.
        """
        if sport == "Football":
            s1 = self.get_football_team_stats(t1)
            s2 = self.get_football_team_stats(t2)
            h2h = self.get_football_head_to_head(t1, t2)

            # Compute normalized ratings (0-100)
            max_elo = max(s1["elo_rating"], s2["elo_rating"], 1600)
            min_elo = min(s1["elo_rating"], s2["elo_rating"], 1400)
            elo_range = max(max_elo - min_elo, 100)

            return {
                "sport": "Football",
                "team_1": t1, "team_2": t2,
                "stats_1": s1, "stats_2": s2,
                "h2h": h2h,
                "ratings": {
                    "overall_1": min(99, max(40, int(50 + (s1["elo_rating"] - 1500) / 10))),
                    "overall_2": min(99, max(40, int(50 + (s2["elo_rating"] - 1500) / 10))),
                    "attack_1": min(99, int(s1["avg_goals_scored"] / 3.0 * 100)),
                    "attack_2": min(99, int(s2["avg_goals_scored"] / 3.0 * 100)),
                    "defense_1": min(99, max(20, int((1 - s1["avg_goals_conceded"] / 3.0) * 100))),
                    "defense_2": min(99, max(20, int((1 - s2["avg_goals_conceded"] / 3.0) * 100))),
                    "form_1": int(s1["form_5"] * 100),
                    "form_2": int(s2["form_5"] * 100),
                    "momentum_1": int(s1["form_10"] * 100),
                    "momentum_2": int(s2["form_10"] * 100),
                    "experience_1": min(99, int(s1["total_matches"] / 100 * 50 + 30)),
                    "experience_2": min(99, int(s2["total_matches"] / 100 * 50 + 30)),
                    "consistency_1": int(s1["consistency"] * 100),
                    "consistency_2": int(s2["consistency"] * 100),
                },
            }
        else:  # Cricket
            s1 = self.get_cricket_team_stats(t1)
            s2 = self.get_cricket_team_stats(t2)
            h2h = self.get_cricket_head_to_head(t1, t2)

            return {
                "sport": "Cricket",
                "team_1": t1, "team_2": t2,
                "stats_1": s1, "stats_2": s2,
                "h2h": h2h,
                "ratings": {
                    "overall_1": min(99, max(40, int(50 + (s1["elo_rating"] - 1500) / 10))),
                    "overall_2": min(99, max(40, int(50 + (s2["elo_rating"] - 1500) / 10))),
                    "attack_1": min(99, int(s1["avg_runs"] / 220 * 100)),
                    "attack_2": min(99, int(s2["avg_runs"] / 220 * 100)),
                    "defense_1": min(99, max(20, int((1 - s1["economy"] / 15) * 100))),
                    "defense_2": min(99, max(20, int((1 - s2["economy"] / 15) * 100))),
                    "form_1": int(s1["form_5"] * 100),
                    "form_2": int(s2["form_5"] * 100),
                    "momentum_1": int(s1["form_10"] * 100),
                    "momentum_2": int(s2["form_10"] * 100),
                    "experience_1": min(99, int(s1["total_matches"] / 200 * 50 + 30)),
                    "experience_2": min(99, int(s2["total_matches"] / 200 * 50 + 30)),
                    "consistency_1": int(s1["consistency"] * 100),
                    "consistency_2": int(s2["consistency"] * 100),
                },
            }

    # ─────────────────────────────────────────
    # Dataset Summary
    # ─────────────────────────────────────────

    def get_dataset_summary(self) -> Dict[str, any]:
        """Get summary statistics about loaded datasets."""
        fb = self.load_football_data()
        cr_m, cr_b = self.load_cricket_data()

        return {
            "football_matches": len(fb),
            "football_teams": self._count_football_teams(fb) if not fb.empty else 0,
            "cricket_matches": len(cr_m),
            "cricket_teams": len(set(cr_m["team1"].unique()) | set(cr_m["team2"].unique())) if not cr_m.empty else 0,
            "cricket_deliveries": len(cr_b),
        }

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    @staticmethod
    def _compute_streaks(win_array) -> Tuple[int, int]:
        """Compute current winning and losing streak from a win indicator array."""
        if len(win_array) == 0:
            return 0, 0

        # Current winning streak (from end)
        win_streak = 0
        for v in reversed(win_array):
            if v == 1:
                win_streak += 1
            else:
                break

        # Current losing streak (from end)
        loss_streak = 0
        for v in reversed(win_array):
            if v == 0:
                loss_streak += 1
            else:
                break

        return win_streak, loss_streak

    @staticmethod
    def _default_football_stats() -> Dict[str, float]:
        return {
            "total_matches": 0, "wins": 0, "draws": 0, "losses": 0,
            "win_rate": 0.40, "draw_rate": 0.25, "loss_rate": 0.35,
            "goals_scored": 0, "goals_conceded": 0,
            "avg_goals_scored": 1.5, "avg_goals_conceded": 1.2, "goal_diff": 0.3,
            "elo_rating": ELO_INITIAL, "form_5": 0.5, "form_10": 0.5, "form_20": 0.5,
            "win_streak": 0, "loss_streak": 0, "consistency": 0.5, "form_string": "-----",
        }

    @staticmethod
    def _default_cricket_stats() -> Dict[str, float]:
        return {
            "total_matches": 0, "wins": 0, "losses": 0,
            "win_rate": 0.50, "elo_rating": ELO_INITIAL,
            "avg_runs": 155.0, "strike_rate": 130.0, "run_rate": 7.5,
            "avg_wickets": 5.0, "economy": 8.0,
            "powerplay_rr": 7.0, "death_economy": 10.0, "toss_impact": 0.5,
            "form_5": 0.5, "form_10": 0.5, "form_20": 0.5,
            "win_streak": 0, "loss_streak": 0, "consistency": 0.5, "form_string": "-----",
        }
