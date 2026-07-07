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
from utils.constants import COLORS, FONTS, APP_VERSION
from gui.widgets import (
    StatsCard,
    RoundedButton,
    GradientFrame,
    create_section_header,
    create_card,
    ScrollableFrame,
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
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # ── Hero Header ──
        hero = tk.Frame(container, bg=COLORS["bg_primary"])
        hero.pack(fill="x", padx=30, pady=(30, 0))

        tk.Label(
            hero,
            text="⚡ OracleXI",
            font=("Helvetica Neue", 32, "bold"),
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w")

        tk.Label(
            hero,
            text="AI Sports Forecasting System",
            font=("Helvetica Neue", 14, "normal"),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w", pady=(2, 0))

        # Version + Status bar
        status_bar = tk.Frame(hero, bg=COLORS["bg_primary"])
        status_bar.pack(anchor="w", pady=(8, 0))

        tk.Label(
            status_bar,
            text=f"v{APP_VERSION}",
            font=FONTS["small_bold"],
            fg=COLORS["bg_primary"],
            bg=COLORS["accent_cyan"],
            padx=8,
            pady=2,
        ).pack(side="left")

        tk.Label(
            status_bar,
            text="  READY",
            font=FONTS["small_bold"],
            fg=COLORS["accent_green"],
            bg=COLORS["bg_primary"],
        ).pack(side="left", padx=(8, 0))

        GradientFrame(hero, height=2).pack(fill="x", pady=(15, 0))

        # ── Stats Cards Row ──
        stats_frame = tk.Frame(container, bg=COLORS["bg_primary"])
        stats_frame.pack(fill="x", padx=30, pady=20)

        summary = self.engine.get_dataset_summary()
        f_matches = summary.get("football_matches", 0)
        c_matches = summary.get("cricket_matches", 0)
        total_teams = summary.get("football_teams", 0) + summary.get("cricket_teams", 0)
        total_files = summary.get("ingested_files", 0)

        self.card_fb = StatsCard(
            stats_frame,
            label="Football Matches",
            value=f"{f_matches:,}",
            icon="⚽",
            accent_color=COLORS["accent_cyan"],
        )
        self.card_fb.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.card_cr = StatsCard(
            stats_frame,
            label="Cricket Matches",
            value=f"{c_matches:,}",
            icon="🏏",
            accent_color=COLORS["accent_purple"],
        )
        self.card_cr.pack(side="left", fill="x", expand=True, padx=8)

        self.card_tm = StatsCard(
            stats_frame,
            label="Tracked Teams",
            value=f"{total_teams:,}",
            icon="🛡",
            accent_color=COLORS["accent_green"],
        )
        self.card_tm.pack(side="left", fill="x", expand=True, padx=8)

        self.card_ds = StatsCard(
            stats_frame,
            label="Datasets Loaded",
            value=f"{total_files}",
            icon="📦",
            accent_color=COLORS["accent_orange"],
        )
        self.card_ds.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # ── Quick Actions ──
        create_section_header(container, "Prediction Modules", "Select a sport to start comparing teams").pack(fill="x", padx=30, pady=(0, 15))

        modules_frame = tk.Frame(container, bg=COLORS["bg_primary"])
        modules_frame.pack(fill="both", expand=True, padx=30)

        # Football Module
        fb_module = create_card(modules_frame)
        fb_module.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(
            fb_module.inner,
            text="⚽",
            font=("Helvetica Neue", 36, "normal"),
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        tk.Label(
            fb_module.inner,
            text="Football AI",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w", pady=(5, 5))

        tk.Label(
            fb_module.inner,
            text="52,000+ international matches • Elo ratings\nNeural network predictions • Monte Carlo simulation",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            justify="left",
        ).pack(anchor="w", pady=(0, 15))

        RoundedButton(
            fb_module.inner,
            text="⚡ Launch Comparison",
            command=lambda: self.navigate("Compare", sport="Football"),
            bg_color=COLORS["accent_cyan"],
            fg_color="#000000",
            width=200,
            height=38,
        ).pack(anchor="w")

        # Cricket Module
        cr_module = create_card(modules_frame)
        cr_module.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(
            cr_module.inner,
            text="🏏",
            font=("Helvetica Neue", 36, "normal"),
            fg=COLORS["accent_purple"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        tk.Label(
            cr_module.inner,
            text="Cricket AI",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w", pady=(5, 5))

        tk.Label(
            cr_module.inner,
            text="IPL, T20, ODI & World Cup matches\nBall-by-ball analysis • Toss impact • Multi-format stats",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            justify="left",
        ).pack(anchor="w", pady=(0, 15))

        RoundedButton(
            cr_module.inner,
            text="⚡ Launch Comparison",
            command=lambda: self.navigate("Compare", sport="Cricket"),
            bg_color=COLORS["accent_purple"],
            fg_color="#FFFFFF",
            width=200,
            height=38,
        ).pack(anchor="w")

        # ── Tech Stack Footer ──
        footer = tk.Frame(container, bg=COLORS["bg_primary"])
        footer.pack(fill="x", padx=30, pady=(25, 20))

        tk.Label(
            footer,
            text="POWERED BY",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w")

        tech_frame = tk.Frame(footer, bg=COLORS["bg_primary"])
        tech_frame.pack(anchor="w", pady=(5, 0))

        techs = [
            ("TensorFlow", COLORS["accent_orange"]),
            ("NumPy", COLORS["accent_blue"]),
            ("Pandas", COLORS["accent_green"]),
            ("Statsmodels", COLORS["accent_purple"]),
            ("Tkinter", COLORS["accent_cyan"]),
        ]
        for name, color in techs:
            tk.Label(
                tech_frame,
                text=f" {name} ",
                font=FONTS["caption"],
                fg=color,
                bg=COLORS["bg_tertiary"],
                padx=8,
                pady=3,
            ).pack(side="left", padx=(0, 6))

    def refresh(self) -> None:
        """Update dashboard stats when switching to this page."""
        summary = self.engine.get_dataset_summary()
        self.card_fb.update_value(f"{summary.get('football_matches', 0):,}")
        self.card_cr.update_value(f"{summary.get('cricket_matches', 0):,}")
        total = summary.get("football_teams", 0) + summary.get("cricket_teams", 0)
        self.card_tm.update_value(f"{total:,}")
        self.card_ds.update_value(f"{summary.get('ingested_files', 0)}")
