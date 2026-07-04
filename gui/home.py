"""
OracleXI v2.0 – Home Dashboard
=================================
Landing page displaying global system status, loaded datasets,
and quick links to prediction models.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any

from core.prediction_engine import PredictionEngine
from utils.constants import COLORS, FONTS
from gui.widgets import (
    StatsCard,
    RoundedButton,
    create_section_header,
    create_card,
)


class HomePage(tk.Frame):
    """
    The main dashboard landing page.
    """

    def __init__(self, parent: tk.Widget, engine: PredictionEngine, navigate_callback) -> None:
        """Initialize the home page."""
        super().__init__(parent, bg=COLORS["bg_primary"])
        self.engine = engine
        self.navigate = navigate_callback
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the dashboard layout."""
        # Header
        header_frame = tk.Frame(self, bg=COLORS["bg_primary"])
        header_frame.pack(fill="x", padx=30, pady=(30, 10))

        tk.Label(
            header_frame,
            text="Welcome to OracleXI v2.0",
            font=FONTS["title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w")

        tk.Label(
            header_frame,
            text="Advanced Sports Forecasting & Analytics Dashboard",
            font=FONTS["subheading"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w", pady=(5, 0))

        # Stats Cards Row
        stats_frame = tk.Frame(self, bg=COLORS["bg_primary"])
        stats_frame.pack(fill="x", padx=30, pady=20)

        summary = self.engine.get_dataset_summary()
        f_matches = summary.get("football_matches", 0)
        c_matches = summary.get("cricket_matches", 0)
        total_teams = summary.get("football_teams", 0) + summary.get("cricket_teams", 0)

        self.card_fb = StatsCard(
            stats_frame,
            label="Football Matches",
            value=f"{f_matches:,}",
            icon="⚽",
            accent_color=COLORS["accent_cyan"],
        )
        self.card_fb.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.card_cr = StatsCard(
            stats_frame,
            label="Cricket Matches",
            value=f"{c_matches:,}",
            icon="🏏",
            accent_color=COLORS["accent_purple"],
        )
        self.card_cr.pack(side="left", fill="x", expand=True, padx=10)

        self.card_tm = StatsCard(
            stats_frame,
            label="Tracked Teams",
            value=f"{total_teams:,}",
            icon="🛡",
            accent_color=COLORS["accent_green"],
        )
        self.card_tm.pack(side="left", fill="x", expand=True, padx=(10, 0))

        # Quick Actions
        actions_container = tk.Frame(self, bg=COLORS["bg_primary"])
        actions_container.pack(fill="both", expand=True, padx=30, pady=10)

        create_section_header(actions_container, "Prediction Modules").pack(fill="x", pady=(0, 20))

        modules_frame = tk.Frame(actions_container, bg=COLORS["bg_primary"])
        modules_frame.pack(fill="both", expand=True)

        # Football Module
        fb_module = create_card(modules_frame)
        fb_module.pack(side="left", fill="both", expand=True, padx=(0, 15))

        tk.Label(
            fb_module.inner,
            text="⚽ Football AI",
            font=FONTS["heading"],
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(
            fb_module.inner,
            text="Predict international and club football matches using Elo ratings, neural networks, and recent form analysis.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            wraplength=350,
            justify="left",
        ).pack(anchor="w", pady=(0, 20))

        RoundedButton(
            fb_module.inner,
            text="Launch Comparison",
            command=lambda: self.navigate("Compare"),
            bg_color=COLORS["accent_cyan"],
            fg_color="#000000",
            width=160,
        ).pack(anchor="w")

        # Cricket Module
        cr_module = create_card(modules_frame)
        cr_module.pack(side="left", fill="both", expand=True, padx=(15, 0))

        tk.Label(
            cr_module.inner,
            text="🏏 Cricket AI",
            font=FONTS["heading"],
            fg=COLORS["accent_purple"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(
            cr_module.inner,
            text="Predict IPL, T20, and ODI matches analyzing ball-by-ball performance, toss impact, and multi-format statistics.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            wraplength=350,
            justify="left",
        ).pack(anchor="w", pady=(0, 20))

        RoundedButton(
            cr_module.inner,
            text="Launch Comparison",
            command=lambda: self.navigate("Compare"),
            bg_color=COLORS["accent_purple"],
            fg_color="#FFFFFF",
            width=160,
        ).pack(anchor="w")

    def refresh(self) -> None:
        """Update dashboard stats when switching to this page."""
        summary = self.engine.get_dataset_summary()
        self.card_fb.update_value(f"{summary.get('football_matches', 0):,}")
        self.card_cr.update_value(f"{summary.get('cricket_matches', 0):,}")
        total = summary.get("football_teams", 0) + summary.get("cricket_teams", 0)
        self.card_tm.update_value(f"{total:,}")
