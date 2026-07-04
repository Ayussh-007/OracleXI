"""
OracleXI v2.0 – Helper Utilities
===================================
Logging, formatting, I/O, validation, team name normalization.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.constants import (
    EXPORT_DIR,
    HISTORY_DIR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_DIR,
    DATA_FOOTBALL_DIR,
    DATA_CRICKET_DIR,
    PREDICTION_HISTORY_FILE,
    TEAM_NAME_ALIASES,
)


def setup_logging(name: str = "OracleXI") -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    directories = [
        MODEL_DIR, HISTORY_DIR, EXPORT_DIR,
        DATA_FOOTBALL_DIR, DATA_CRICKET_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# ─────────────────────────────────────────────
# Team Name Normalization
# ─────────────────────────────────────────────


def normalize_team_name(name: str) -> str:
    """
    Normalize a team name using the alias mapping.

    Handles abbreviations, alternate spellings, and defunct teams.

    Args:
        name: Raw team name from dataset.

    Returns:
        Standardized team name.
    """
    if not name or not isinstance(name, str):
        return str(name) if name else ""
    name = name.strip()
    return TEAM_NAME_ALIASES.get(name, name)


def normalize_team_names_in_df(df, columns: List[str]):
    """
    Normalize team names in specified DataFrame columns in-place.

    Args:
        df: Pandas DataFrame.
        columns: List of column names containing team names.

    Returns:
        Modified DataFrame.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)
    return df


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────


def format_date(date_obj: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime object to string."""
    return date_obj.strftime(fmt)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    if value <= 1.0:
        value *= 100
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a float with specified decimal places."""
    return f"{value:.{decimals}f}"


def format_elo(elo: float) -> str:
    """Format an Elo rating as a clean integer string."""
    return str(int(round(elo)))


# ─────────────────────────────────────────────
# Prediction History I/O
# ─────────────────────────────────────────────


def load_prediction_history() -> List[Dict[str, Any]]:
    """Load prediction history from JSON file."""
    ensure_directories()
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return []
    try:
        with open(PREDICTION_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def save_prediction_history(history: List[Dict[str, Any]]) -> bool:
    """Save prediction history to JSON file."""
    ensure_directories()
    try:
        with open(PREDICTION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        return True
    except IOError:
        return False


def add_prediction_to_history(prediction: Dict[str, Any]) -> bool:
    """Append a single prediction to the history file."""
    history = load_prediction_history()
    prediction["timestamp"] = format_date(datetime.now())
    prediction["id"] = len(history) + 1
    history.append(prediction)
    return save_prediction_history(history)


# ─────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────


def export_predictions_to_csv(
    predictions: List[Dict[str, Any]],
    filename: Optional[str] = None,
) -> str:
    """Export prediction results to a CSV file."""
    ensure_directories()
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.csv"

    filepath = os.path.join(EXPORT_DIR, filename)
    if not predictions:
        return filepath

    flat_predictions = []
    for pred in predictions:
        flat = {}
        for key, value in pred.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}_{sub_key}"] = sub_value
            else:
                flat[key] = value
        flat_predictions.append(flat)

    all_keys = []
    for pred in flat_predictions:
        for key in pred.keys():
            if key not in all_keys:
                all_keys.append(key)

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(flat_predictions)
        return filepath
    except IOError:
        return ""


def export_single_prediction_to_csv(prediction: Dict[str, Any]) -> str:
    """Export a single prediction result to CSV."""
    return export_predictions_to_csv([prediction])


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────


def validate_team_selection(
    team_1: str,
    team_2: str,
    available_teams: List[str],
) -> Tuple[bool, str]:
    """
    Validate team selection for prediction.

    Args:
        team_1: First team name.
        team_2: Second team name.
        available_teams: List of valid team names.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not team_1 or team_1 == "Select Team":
        return False, "Please select Team 1."
    if not team_2 or team_2 == "Select Team":
        return False, "Please select Team 2."
    if team_1 == team_2:
        return False, "Team 1 and Team 2 must be different."
    if team_1 not in available_teams:
        return False, f"'{team_1}' is not a valid team."
    if team_2 not in available_teams:
        return False, f"'{team_2}' is not a valid team."
    return True, ""


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max bounds."""
    return max(min_val, min(max_val, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    return numerator / denominator if denominator != 0 else default
