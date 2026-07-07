"""
OracleXI v2.0 – Dataset Manager
====================================
Handles multi-CSV discovery, schema validation, deduplication,
team name normalization, and unified DataFrame construction.

Features:
    - Scans data/football/ and data/cricket/ for all CSVs
    - Auto-detects schema type (results, rankings, ball-by-ball, etc.)
    - Validates required columns
    - Normalizes team names via alias mapping
    - Removes duplicate records
    - Merges overlapping datasets
    - Provides add_dataset() for future additions
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from utils.constants import (
    DATA_DIR,
    DATA_FOOTBALL_DIR,
    DATA_CRICKET_DIR,
    FOOTBALL_CSV,
    CRICKET_MATCH_CSV,
    CRICKET_BALL_CSV,
    DATASET_SCHEMAS,
    TEAM_NAME_ALIASES,
)
from utils.helper import (
    normalize_team_names_in_df,
    ensure_directories,
    setup_logging,
)

logger = setup_logging("DatasetManager")


class DatasetManager:
    """
    Manages multi-CSV dataset discovery, validation, and merging.

    Scans subdirectories for all compatible CSVs, validates them
    against known schemas, normalizes team names, removes
    duplicates, and produces unified master DataFrames.
    """

    def __init__(self) -> None:
        """Initialize the dataset manager."""
        ensure_directories()
        self._football_master: Optional[pd.DataFrame] = None
        self._cricket_match_master: Optional[pd.DataFrame] = None
        self._cricket_ball_master: Optional[pd.DataFrame] = None
        self._rankings_df: Optional[pd.DataFrame] = None
        self._goalscorers_df: Optional[pd.DataFrame] = None
        self._cricket_player_stats: Optional[pd.DataFrame] = None
        self._ingested_files: List[Dict[str, str]] = []
        logger.info("DatasetManager initialized")

    # ─────────────────────────────────────────
    # CSV Discovery
    # ─────────────────────────────────────────

    def discover_csvs(self, directory: str) -> List[str]:
        """
        Discover all CSV files in a directory (non-recursive).

        Args:
            directory: Path to scan.

        Returns:
            List of absolute CSV file paths.
        """
        if not os.path.exists(directory):
            return []
        pattern = os.path.join(directory, "*.csv")
        return sorted(glob.glob(pattern))

    # ─────────────────────────────────────────
    # Schema Detection & Validation
    # ─────────────────────────────────────────

    def detect_schema(self, filepath: str) -> Optional[str]:
        """
        Detect the schema type of a CSV file by inspecting its columns.

        Args:
            filepath: Path to CSV file.

        Returns:
            Schema type string or None if unrecognized.
        """
        try:
            # Read just the header
            df_head = pd.read_csv(filepath, nrows=2)
            cols = set(df_head.columns.str.lower())

            # Check against known schemas
            for schema_name, required_cols in DATASET_SCHEMAS.items():
                if set(c.lower() for c in required_cols).issubset(cols):
                    return schema_name

            # Heuristic detection
            if "home_team" in cols and "away_team" in cols:
                return "football_results"
            if "batting_team" in cols and "bowling_team" in cols:
                return "cricket_ball_by_ball"
            if {"team1", "team2", "winner"}.issubset(cols):
                return "cricket_match_info"
            if "rank" in cols and "team" in cols:
                return "football_rankings"
            if "scorer" in cols:
                return "football_goalscorers"
            if {"player", "runs", "wickets"}.issubset(cols):
                return "cricket_player_stats"

            logger.warning(f"Unrecognized schema in {filepath}: {cols}")
            return None
        except Exception as e:
            logger.error(f"Failed to detect schema for {filepath}: {e}")
            return None

    def validate_csv(self, filepath: str, schema: str) -> bool:
        """
        Validate a CSV against its expected schema.

        Args:
            filepath: Path to CSV.
            schema: Schema type name.

        Returns:
            True if valid.
        """
        required = DATASET_SCHEMAS.get(schema, [])
        if not required:
            return True

        try:
            df_head = pd.read_csv(filepath, nrows=1)
            cols = set(df_head.columns)
            missing = set(required) - cols
            if missing:
                logger.warning(
                    f"CSV {filepath} missing columns for {schema}: {missing}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Validation failed for {filepath}: {e}")
            return False

    # ─────────────────────────────────────────
    # Loading & Merging
    # ─────────────────────────────────────────

    def scan_and_load(self) -> Dict[str, int]:
        """
        Scan all data directories, load and merge all compatible CSVs.

        Returns:
            Dictionary with counts of loaded records.
        """
        logger.info("Scanning for datasets...")
        counts = {
            "football_matches": 0,
            "cricket_matches": 0,
            "cricket_deliveries": 0,
            "cricket_players": 0,
            "rankings": 0,
            "goalscorers": 0,
        }

        # ── Load Football ──
        fb_dfs = []

        # Check subdirectory
        for csv_path in self.discover_csvs(DATA_FOOTBALL_DIR):
            schema = self.detect_schema(csv_path)
            if schema == "football_results":
                df = self._load_football_csv(csv_path)
                if df is not None and not df.empty:
                    fb_dfs.append(df)
            elif schema == "football_rankings":
                self._load_rankings_csv(csv_path)
            elif schema == "football_goalscorers":
                self._load_goalscorers_csv(csv_path)

        # Check legacy root-level CSV
        if os.path.exists(FOOTBALL_CSV):
            schema = self.detect_schema(FOOTBALL_CSV)
            if schema == "football_results":
                df = self._load_football_csv(FOOTBALL_CSV)
                if df is not None and not df.empty:
                    fb_dfs.append(df)

        if fb_dfs:
            self._football_master = pd.concat(fb_dfs, ignore_index=True)
            self._football_master = self._deduplicate_football(self._football_master)
            counts["football_matches"] = len(self._football_master)
            logger.info(f"Loaded {counts['football_matches']} football matches total")

        # ── Load Cricket ──
        cr_match_dfs = []
        cr_ball_dfs = []

        for csv_path in self.discover_csvs(DATA_CRICKET_DIR):
            schema = self.detect_schema(csv_path)
            if schema == "cricket_match_info":
                df = self._load_cricket_match_csv(csv_path)
                if df is not None and not df.empty:
                    cr_match_dfs.append(df)
            elif schema == "cricket_ball_by_ball":
                df = self._load_cricket_ball_csv(csv_path)
                if df is not None and not df.empty:
                    cr_ball_dfs.append(df)
            elif schema == "cricket_player_stats":
                self._load_cricket_player_stats_csv(csv_path)

        # Check legacy root-level CSVs
        if os.path.exists(CRICKET_MATCH_CSV):
            df = self._load_cricket_match_csv(CRICKET_MATCH_CSV)
            if df is not None and not df.empty:
                cr_match_dfs.append(df)

        if os.path.exists(CRICKET_BALL_CSV):
            df = self._load_cricket_ball_csv(CRICKET_BALL_CSV)
            if df is not None and not df.empty:
                cr_ball_dfs.append(df)

        if cr_match_dfs:
            self._cricket_match_master = pd.concat(cr_match_dfs, ignore_index=True)
            self._cricket_match_master = self._deduplicate_cricket_matches(
                self._cricket_match_master
            )
            counts["cricket_matches"] = len(self._cricket_match_master)
            logger.info(f"Loaded {counts['cricket_matches']} cricket matches total")

        if cr_ball_dfs:
            self._cricket_ball_master = pd.concat(cr_ball_dfs, ignore_index=True)
            self._cricket_ball_master = self._deduplicate_cricket_balls(
                self._cricket_ball_master
            )
            counts["cricket_deliveries"] = len(self._cricket_ball_master)

        if self._rankings_df is not None:
            counts["rankings"] = len(self._rankings_df)
        if self._goalscorers_df is not None:
            counts["goalscorers"] = len(self._goalscorers_df)
        if self._cricket_player_stats is not None:
            counts["cricket_players"] = len(self._cricket_player_stats)

        return counts

    def _load_football_csv(self, path: str) -> Optional[pd.DataFrame]:
        """Load and normalize a football results CSV."""
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = normalize_team_names_in_df(df, ["home_team", "away_team"])
            df["home_score"] = df["home_score"].fillna(0).astype(int)
            df["away_score"] = df["away_score"].fillna(0).astype(int)
            df["tournament"] = df["tournament"].fillna("Unknown")
            self._ingested_files.append({"file": path, "type": "football_results", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _load_cricket_match_csv(self, path: str) -> Optional[pd.DataFrame]:
        """Load and normalize a cricket match info CSV."""
        try:
            df = pd.read_csv(path)
            # Handle date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "match_id" in df.columns:
                df["match_id"] = df["match_id"].astype(str)
            df = normalize_team_names_in_df(df, ["team1", "team2", "winner", "toss_winner"])
            df["winner"] = df["winner"].fillna("No Result")
            df["result_margin"] = df["result_margin"].fillna(0).astype(int)
            df["result_type"] = df["result_type"].fillna("N/A")
            # Add format column if missing
            if "format" not in df.columns:
                df["format"] = "T20"
            if "tournament" not in df.columns:
                df["tournament"] = "Unknown"
            self._ingested_files.append({"file": path, "type": "cricket_match_info", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _load_cricket_ball_csv(self, path: str) -> Optional[pd.DataFrame]:
        """Load and normalize a cricket ball-by-ball CSV."""
        try:
            df = pd.read_csv(path)
            if "match_id" in df.columns:
                df["match_id"] = df["match_id"].astype(str)
            df = normalize_team_names_in_df(df, ["batting_team", "bowling_team"])
            df["extra_run"] = df["extra_run"].fillna(0).astype(int)
            df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)
            df["total_run"] = df["total_run"].fillna(0).astype(int)
            self._ingested_files.append({"file": path, "type": "cricket_ball_by_ball", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def _load_rankings_csv(self, path: str) -> None:
        """Load FIFA rankings CSV."""
        try:
            df = pd.read_csv(path)
            if "team" in df.columns:
                df = normalize_team_names_in_df(df, ["team"])
            self._rankings_df = df
            self._ingested_files.append({"file": path, "type": "rankings", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} ranking records")
        except Exception as e:
            logger.error(f"Failed to load rankings: {e}")

    def _load_goalscorers_csv(self, path: str) -> None:
        """Load goalscorers CSV."""
        try:
            df = pd.read_csv(path)
            df = normalize_team_names_in_df(df, ["home_team", "away_team", "team"])
            self._goalscorers_df = df
            self._ingested_files.append({"file": path, "type": "goalscorers", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} goalscorer records")
        except Exception as e:
            logger.error(f"Failed to load goalscorers: {e}")

    def _load_cricket_player_stats_csv(self, path: str) -> None:
        """Load cricket player stats CSV."""
        try:
            df = pd.read_csv(path)
            df = normalize_team_names_in_df(df, ["team"])
            self._cricket_player_stats = df
            self._ingested_files.append({"file": path, "type": "cricket_player_stats", "rows": len(df)})
            logger.info(f"  Loaded {len(df)} cricket player records")
        except Exception as e:
            logger.error(f"Failed to load cricket player stats: {e}")

    # ─────────────────────────────────────────
    # Deduplication
    # ─────────────────────────────────────────

    def _deduplicate_football(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate football matches."""
        before = len(df)
        df = df.drop_duplicates(
            subset=["date", "home_team", "away_team"],
            keep="first",
        )
        df = df.sort_values("date").reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate football records")
        return df

    def _deduplicate_cricket_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate cricket matches."""
        before = len(df)
        if "match_id" in df.columns:
            # Preserve original match IDs instead of re-assigning sequential ints
            df = df.drop_duplicates(
                subset=["date", "team1", "team2"],
                keep="first",
            )
        df = df.sort_values("date").reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate cricket match records")
        return df

    def _deduplicate_cricket_balls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate ball-by-ball records."""
        before = len(df)
        df = df.drop_duplicates(
            subset=["match_id", "inning", "over", "ball"],
            keep="first",
        )
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed} duplicate ball-by-ball records")
        return df

    # ─────────────────────────────────────────
    # Public Accessors
    # ─────────────────────────────────────────

    def get_football_data(self) -> pd.DataFrame:
        """Get the unified football DataFrame."""
        if self._football_master is None:
            self.scan_and_load()
        return self._football_master if self._football_master is not None else pd.DataFrame()

    def get_cricket_match_data(self) -> pd.DataFrame:
        """Get the unified cricket match DataFrame."""
        if self._cricket_match_master is None:
            self.scan_and_load()
        return self._cricket_match_master if self._cricket_match_master is not None else pd.DataFrame()

    def get_cricket_ball_data(self) -> pd.DataFrame:
        """Get the unified cricket ball-by-ball DataFrame."""
        if self._cricket_ball_master is None:
            self.scan_and_load()
        return self._cricket_ball_master if self._cricket_ball_master is not None else pd.DataFrame()

    def get_rankings_data(self) -> Optional[pd.DataFrame]:
        """Get FIFA rankings DataFrame."""
        return self._rankings_df

    def get_goalscorers_data(self) -> Optional[pd.DataFrame]:
        """Get football goalscorers DataFrame."""
        return self._goalscorers_df

    def get_cricket_player_stats_data(self) -> Optional[pd.DataFrame]:
        """Get cricket player stats DataFrame."""
        return self._cricket_player_stats

    def get_summary(self) -> Dict[str, any]:
        """Get summary of all loaded data."""
        fb = self.get_football_data()
        cr = self.get_cricket_match_data()
        bb = self.get_cricket_ball_data()

        fb_teams = set()
        if not fb.empty:
            fb_teams = set(fb["home_team"].unique()) | set(fb["away_team"].unique())

        cr_teams = set()
        if not cr.empty:
            cr_teams = set(cr["team1"].unique()) | set(cr["team2"].unique())

        return {
            "football_matches": len(fb),
            "football_teams": len(fb_teams),
            "cricket_matches": len(cr),
            "cricket_teams": len(cr_teams),
            "cricket_deliveries": len(bb),
            "rankings_records": len(self._rankings_df) if self._rankings_df is not None else 0,
            "goalscorer_records": len(self._goalscorers_df) if self._goalscorers_df is not None else 0,
            "ingested_files": len(self._ingested_files),
            "files": self._ingested_files,
        }

    # ─────────────────────────────────────────
    # Dynamic Dataset Addition
    # ─────────────────────────────────────────

    def add_dataset(self, filepath: str) -> Dict[str, any]:
        """
        Add a new CSV dataset at runtime.

        Auto-detects schema, validates, normalizes, and merges.

        Args:
            filepath: Path to new CSV file.

        Returns:
            Dictionary with result status and row count.
        """
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}

        schema = self.detect_schema(filepath)
        if schema is None:
            return {"success": False, "error": "Unrecognized CSV schema"}

        if schema == "football_results":
            df = self._load_football_csv(filepath)
            if df is not None and not df.empty:
                if self._football_master is not None:
                    self._football_master = pd.concat(
                        [self._football_master, df], ignore_index=True
                    )
                    self._football_master = self._deduplicate_football(self._football_master)
                else:
                    self._football_master = df
                return {"success": True, "rows": len(df), "schema": schema}

        elif schema == "cricket_match_info":
            df = self._load_cricket_match_csv(filepath)
            if df is not None and not df.empty:
                if self._cricket_match_master is not None:
                    self._cricket_match_master = pd.concat(
                        [self._cricket_match_master, df], ignore_index=True
                    )
                    self._cricket_match_master = self._deduplicate_cricket_matches(
                        self._cricket_match_master
                    )
                else:
                    self._cricket_match_master = df
                return {"success": True, "rows": len(df), "schema": schema}

        elif schema == "cricket_ball_by_ball":
            df = self._load_cricket_ball_csv(filepath)
            if df is not None and not df.empty:
                if self._cricket_ball_master is not None:
                    self._cricket_ball_master = pd.concat(
                        [self._cricket_ball_master, df], ignore_index=True
                    )
                    self._cricket_ball_master = self._deduplicate_cricket_balls(
                        self._cricket_ball_master
                    )
                else:
                    self._cricket_ball_master = df
                return {"success": True, "rows": len(df), "schema": schema}

        return {"success": False, "error": f"Unsupported schema: {schema}"}
