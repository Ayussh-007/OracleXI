"""
OracleXI v2.0 – Constants and Configuration
===============================================
Central configuration: colors, fonts, paths, model config,
team lists (club + international), dataset schemas, Elo config.
"""

import os
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# Application Info
# ─────────────────────────────────────────────
APP_NAME: str = "OracleXI"
APP_SUBTITLE: str = "AI Sports Forecasting System"
APP_VERSION: str = "2.0.0"

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
DATA_FOOTBALL_DIR: str = os.path.join(DATA_DIR, "football")
DATA_CRICKET_DIR: str = os.path.join(DATA_DIR, "cricket")
MODEL_DIR: str = os.path.join(BASE_DIR, "models")
HISTORY_DIR: str = os.path.join(BASE_DIR, "history")
EXPORT_DIR: str = os.path.join(BASE_DIR, "exports")

# Legacy single-file paths (kept for backward compat)
FOOTBALL_CSV: str = os.path.join(DATA_DIR, "football_results.csv")
CRICKET_MATCH_CSV: str = os.path.join(DATA_DIR, "cricket_match_info.csv")
CRICKET_BALL_CSV: str = os.path.join(DATA_DIR, "cricket_ball_by_ball.csv")

# Model files
FOOTBALL_MODEL_PATH: str = os.path.join(MODEL_DIR, "football_model.keras")
CRICKET_MODEL_PATH: str = os.path.join(MODEL_DIR, "cricket_model.keras")
PREDICTION_HISTORY_FILE: str = os.path.join(HISTORY_DIR, "predictions.json")

# ─────────────────────────────────────────────
# Color Palette (Dark Theme)
# ─────────────────────────────────────────────
COLORS: Dict[str, str] = {
    "bg_primary": "#0D1117",
    "bg_secondary": "#161B22",
    "bg_tertiary": "#1C2333",
    "bg_hover": "#21262D",
    "bg_input": "#0D1117",
    "accent_cyan": "#00D4FF",
    "accent_purple": "#7B2FBE",
    "accent_blue": "#58A6FF",
    "accent_green": "#3FB950",
    "accent_red": "#F85149",
    "accent_orange": "#D29922",
    "accent_pink": "#F778BA",
    "accent_gold": "#FFD700",
    "accent_gradient_start": "#00D4FF",
    "accent_gradient_end": "#7B2FBE",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted": "#484F58",
    "text_link": "#58A6FF",
    "border_default": "#30363D",
    "border_hover": "#484F58",
    "border_accent": "#00D4FF",
    "chart_1": "#00D4FF",
    "chart_2": "#7B2FBE",
    "chart_3": "#3FB950",
    "chart_4": "#F85149",
    "chart_5": "#D29922",
    "chart_6": "#F778BA",
    "chart_7": "#58A6FF",
    "chart_8": "#79C0FF",
    # Form badge colors
    "badge_win": "#3FB950",
    "badge_loss": "#F85149",
    "badge_draw": "#D29922",
}

# ─────────────────────────────────────────────
# Fonts
# ─────────────────────────────────────────────
FONTS: Dict[str, Tuple[str, int, str]] = {
    "title": ("Helvetica Neue", 28, "bold"),
    "heading": ("Helvetica Neue", 20, "bold"),
    "subheading": ("Helvetica Neue", 16, "bold"),
    "body": ("Helvetica Neue", 13, "normal"),
    "body_bold": ("Helvetica Neue", 13, "bold"),
    "small": ("Helvetica Neue", 11, "normal"),
    "small_bold": ("Helvetica Neue", 11, "bold"),
    "caption": ("Helvetica Neue", 10, "normal"),
    "mono": ("Menlo", 12, "normal"),
    "button": ("Helvetica Neue", 13, "bold"),
    "nav": ("Helvetica Neue", 12, "bold"),
    "stat_value": ("Helvetica Neue", 32, "bold"),
    "stat_label": ("Helvetica Neue", 11, "normal"),
    "vs_big": ("Helvetica Neue", 36, "bold"),
    "rating_big": ("Helvetica Neue", 48, "bold"),
    "form_badge": ("Helvetica Neue", 11, "bold"),
}

# ─────────────────────────────────────────────
# GUI Configuration
# ─────────────────────────────────────────────
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 800
MIN_WIDTH: int = 1024
MIN_HEIGHT: int = 680
SIDEBAR_WIDTH: int = 220
CARD_PADDING: int = 20
CARD_RADIUS: int = 12

PAGES: List[str] = ["Home", "Compare", "Analytics", "History"]
PAGE_ICONS: Dict[str, str] = {
    "Home": "🏠",
    "Compare": "⚡",
    "Analytics": "📈",
    "History": "🕒",
}

# ─────────────────────────────────────────────
# Supported Sports
# ─────────────────────────────────────────────
SPORTS: List[str] = ["Football", "Cricket"]

# ─────────────────────────────────────────────
# Football Teams – International
# ─────────────────────────────────────────────
FOOTBALL_INTERNATIONAL: List[str] = [
    "Argentina", "Australia", "Belgium", "Brazil", "Colombia",
    "Croatia", "Denmark", "Ecuador", "England", "France",
    "Germany", "Ghana", "Iran", "Italy", "Japan",
    "Mexico", "Morocco", "Netherlands", "Nigeria", "Poland",
    "Portugal", "Saudi Arabia", "Senegal", "Serbia", "South Korea",
    "Spain", "Switzerland", "Tunisia", "United States", "Uruguay",
    "Wales", "Cameroon", "Canada", "Costa Rica", "Qatar",
    "India", "China", "Egypt", "Algeria", "Chile",
    "Paraguay", "Peru", "Sweden", "Norway", "Czech Republic",
    "Turkey", "Russia", "Ukraine", "Scotland", "Ireland",
]

# ─────────────────────────────────────────────
# Football Teams – Clubs (Top Leagues)
# ─────────────────────────────────────────────
FOOTBALL_CLUBS: List[str] = [
    # Premier League
    "Manchester City", "Arsenal", "Liverpool", "Manchester United",
    "Chelsea", "Tottenham", "Newcastle United", "Aston Villa",
    "Brighton", "West Ham",
    # La Liga
    "Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad",
    "Villarreal", "Athletic Bilbao", "Sevilla", "Real Betis",
    # Bundesliga
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
    # Serie A
    "Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma", "Lazio",
    # Ligue 1
    "Paris Saint-Germain", "Marseille", "Monaco", "Lyon",
    # Other
    "Ajax", "Benfica", "Porto", "Celtic", "Rangers",
    "Galatasaray", "Fenerbahce", "Flamengo", "River Plate", "Boca Juniors",
]

FOOTBALL_TEAMS: List[str] = sorted(set(FOOTBALL_INTERNATIONAL + FOOTBALL_CLUBS))

# ─────────────────────────────────────────────
# Football Tournaments
# ─────────────────────────────────────────────
FOOTBALL_TOURNAMENTS: List[str] = [
    "FIFA World Cup", "FIFA World Cup Qualification",
    "UEFA Euro", "UEFA Euro Qualification",
    "Copa America", "African Cup of Nations",
    "AFC Asian Cup", "CONCACAF Gold Cup",
    "UEFA Nations League", "FIFA Confederations Cup",
    "UEFA Champions League", "UEFA Europa League",
    "Premier League", "La Liga", "Bundesliga",
    "Serie A", "Ligue 1", "Eredivisie",
    "Friendly",
]

# ─────────────────────────────────────────────
# Cricket Teams – IPL
# ─────────────────────────────────────────────
CRICKET_IPL_TEAMS: List[str] = [
    "Mumbai Indians", "Chennai Super Kings",
    "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Delhi Capitals", "Punjab Kings",
    "Rajasthan Royals", "Sunrisers Hyderabad",
    "Gujarat Titans", "Lucknow Super Giants",
]

# ─────────────────────────────────────────────
# Cricket Teams – International
# ─────────────────────────────────────────────
CRICKET_INTERNATIONAL: List[str] = [
    "India", "Australia", "England", "South Africa",
    "New Zealand", "Pakistan", "Sri Lanka", "West Indies",
    "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland",
    "Netherlands", "Scotland", "Nepal", "Namibia",
    "Oman", "UAE", "USA", "Canada",
]

CRICKET_TEAMS: List[str] = sorted(set(CRICKET_IPL_TEAMS + CRICKET_INTERNATIONAL))

# ─────────────────────────────────────────────
# Cricket Formats & Tournaments
# ─────────────────────────────────────────────
CRICKET_FORMATS: List[str] = ["T20", "ODI", "Test"]
CRICKET_TOURNAMENTS: List[str] = [
    "IPL", "ICC T20 World Cup", "ICC ODI World Cup",
    "ICC Champions Trophy", "Big Bash League",
    "Caribbean Premier League", "T20 Bilateral",
    "ODI Bilateral", "Asia Cup", "ICC World Test Championship",
]

# ─────────────────────────────────────────────
# Team Name Aliases (for standardization)
# ─────────────────────────────────────────────
TEAM_NAME_ALIASES: Dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "United States of America": "United States",
    "USA": "United States",
    "US": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Eire": "Ireland",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham",
    "PSG": "Paris Saint-Germain",
    "RCB": "Royal Challengers Bangalore",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "Delhi Daredevils": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "Kings XI Punjab": "Punjab Kings",
    "KXIP": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Chennai Super Kings",
    "Rising Pune Supergiants": "Chennai Super Kings",
    "Pune Warriors India": "Delhi Capitals",
    "Gujarat Lions": "Gujarat Titans",
    "Kochi Tuskers Kerala": "Lucknow Super Giants",
}

# ─────────────────────────────────────────────
# Elo Rating Configuration
# ─────────────────────────────────────────────
ELO_INITIAL: float = 1500.0
ELO_K_FACTOR: float = 32.0
ELO_K_FACTOR_TOURNAMENT: float = 48.0  # Higher for tournament matches
ELO_HOME_ADVANTAGE: float = 100.0

# ─────────────────────────────────────────────
# Model Hyperparameters (v2 – 20 features/team)
# ─────────────────────────────────────────────
NUM_FEATURES_PER_TEAM: int = 20

MODEL_CONFIG: Dict[str, dict] = {
    "football": {
        "input_features": NUM_FEATURES_PER_TEAM,
        "hidden_layers": [128, 64, 32, 16],
        "output_classes": 3,         # Win, Draw, Loss
        "learning_rate": 0.001,
        "epochs": 80,
        "batch_size": 32,
        "dropout_rate": 0.25,
    },
    "cricket": {
        "input_features": NUM_FEATURES_PER_TEAM,
        "hidden_layers": [128, 64, 32, 16],
        "output_classes": 2,         # Win, Loss
        "learning_rate": 0.001,
        "epochs": 80,
        "batch_size": 32,
        "dropout_rate": 0.25,
    },
}

# ─────────────────────────────────────────────
# Simulation Defaults
# ─────────────────────────────────────────────
SIMULATION_COUNTS: List[int] = [1000, 5000, 10000, 50000]
DEFAULT_SIMULATION_COUNT: int = 10000

# ─────────────────────────────────────────────
# Prediction Weights (Ensemble)
# ─────────────────────────────────────────────
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "statsmodels": 0.30,
    "tensorflow": 0.40,
    "monte_carlo": 0.30,
}

# ─────────────────────────────────────────────
# Dataset Schemas (for validation)
# ─────────────────────────────────────────────
DATASET_SCHEMAS: Dict[str, List[str]] = {
    "football_results": [
        "date", "home_team", "away_team", "home_score", "away_score",
    ],
    "football_rankings": [
        "rank", "team", "points",
    ],
    "football_goalscorers": [
        "date", "home_team", "away_team", "scorer",
    ],
    "cricket_match_info": [
        "date", "team1", "team2", "winner",
    ],
    "cricket_ball_by_ball": [
        "match_id", "batting_team", "bowling_team", "total_run",
    ],
}

# ─────────────────────────────────────────────
# Comparison Metrics (for GUI)
# ─────────────────────────────────────────────
COMPARISON_METRICS: List[str] = [
    "Elo Rating", "Win Rate", "Recent Form",
    "Momentum", "Attack Rating", "Defense Rating",
    "Consistency", "Experience", "Head-to-Head",
    "Tournament Performance", "Scoring Average",
    "Goals/Runs Conceded", "Winning Streak",
    "Trend", "Strength of Schedule",
]

# Radar chart axes
RADAR_AXES: List[str] = [
    "Attack", "Defense", "Form", "Momentum", "Experience", "Consistency",
]

# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"
