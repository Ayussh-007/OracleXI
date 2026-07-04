"""
OracleXI – Analytics Dashboard
==================================
Dashboard with embedded Matplotlib charts for data visualization.

Charts displayed:
    - Win/Loss/Draw distribution
    - Average Goals/Runs per team
    - Momentum Trend
    - Team Performance comparison
    - Prediction Confidence distribution
    - Forecast Trend (time series)
"""

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from gui.widgets import (
    COLORS,
    FONTS,
    GradientFrame,
    RoundedButton,
    ScrollableFrame,
    create_card,
    create_section_header,
)

if TYPE_CHECKING:
    from core.prediction_engine import PredictionEngine


# Chart colors matching the app theme
CHART_COLORS = [
    "#00D4FF", "#7B2FBE", "#3FB950", "#F85149",
    "#D29922", "#F778BA", "#58A6FF", "#79C0FF",
    "#A5D6FF", "#FF7B72",
]

CHART_BG = "#161B22"
CHART_TEXT = "#E6EDF3"
CHART_GRID = "#30363D"


def configure_chart_style() -> None:
    """Apply dark theme to matplotlib charts."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.facecolor": CHART_BG,
        "axes.facecolor": CHART_BG,
        "axes.edgecolor": CHART_GRID,
        "axes.labelcolor": CHART_TEXT,
        "text.color": CHART_TEXT,
        "xtick.color": CHART_TEXT,
        "ytick.color": CHART_TEXT,
        "grid.color": CHART_GRID,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


class AnalyticsPage(tk.Frame):
    """
    Analytics dashboard with embedded Matplotlib charts.

    Displays comprehensive visualizations for team performance,
    prediction analysis, and statistical trends.
    """

    def __init__(
        self,
        parent: tk.Widget,
        engine: "PredictionEngine",
        navigate_callback: Callable,
        **kwargs,
    ) -> None:
        """
        Initialize the analytics page.

        Args:
            parent: Parent widget.
            engine: Prediction engine instance.
            navigate_callback: Navigation callback.
        """
        super().__init__(parent, bg=COLORS["bg_primary"], **kwargs)
        self.engine = engine
        self.navigate = navigate_callback
        self.current_sport = "Football"
        self.canvas_widgets: List[FigureCanvasTkAgg] = []

        configure_chart_style()
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the complete analytics UI."""
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header_section = tk.Frame(container, bg=COLORS["bg_primary"])
        header_section.pack(fill="x", padx=30, pady=(25, 5))
        header = create_section_header(
            header_section,
            "Analytics Dashboard",
            "Interactive charts and statistical visualizations",
        )
        header.pack(fill="x")

        # Controls bar
        controls = tk.Frame(container, bg=COLORS["bg_primary"])
        controls.pack(fill="x", padx=30, pady=(10, 5))

        # Sport selector
        tk.Label(
            controls,
            text="Sport:",
            font=FONTS["small_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side="left", padx=(0, 8))

        self.sport_var = tk.StringVar(value="Football")
        sport_combo = ttk.Combobox(
            controls,
            textvariable=self.sport_var,
            state="readonly",
            values=["Football", "Cricket"],
            style="Dark.TCombobox",
            width=15,
        )
        sport_combo.pack(side="left", padx=(0, 15))
        sport_combo.bind("<<ComboboxSelected>>", self._on_sport_change)

        # Team selector
        tk.Label(
            controls,
            text="Team:",
            font=FONTS["small_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side="left", padx=(0, 8))

        self.team_var = tk.StringVar(value="All Teams")
        self.team_combo = ttk.Combobox(
            controls,
            textvariable=self.team_var,
            state="readonly",
            style="Dark.TCombobox",
            width=25,
        )
        self.team_combo.pack(side="left", padx=(0, 15))

        # Refresh button
        RoundedButton(
            controls,
            text="🔄 Refresh Charts",
            command=self._refresh_charts,
            bg_color=COLORS["accent_cyan"],
            width=150,
            height=34,
        ).pack(side="left")

        self._update_team_list()

        # Charts container
        self.charts_frame = tk.Frame(container, bg=COLORS["bg_primary"])
        self.charts_frame.pack(fill="both", expand=True, padx=30, pady=(15, 30))

        self._render_charts()

    def _on_sport_change(self, event: tk.Event = None) -> None:
        """Handle sport selection change."""
        self.current_sport = self.sport_var.get()
        self._update_team_list()
        self._refresh_charts()

    def _update_team_list(self) -> None:
        """Update team dropdown for selected sport."""
        if self.current_sport == "Football":
            teams = self.engine.data_engine.get_football_teams()
        else:
            teams = self.engine.data_engine.get_cricket_teams()

        self.team_combo["values"] = ["All Teams"] + teams
        self.team_var.set("All Teams")

    def _refresh_charts(self) -> None:
        """Refresh all charts with current data."""
        # Close existing figures
        for canvas in self.canvas_widgets:
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
        self.canvas_widgets.clear()

        for widget in self.charts_frame.winfo_children():
            widget.destroy()

        self._render_charts()

    def _render_charts(self) -> None:
        """Render all analytics charts."""
        if not MATPLOTLIB_AVAILABLE:
            tk.Label(
                self.charts_frame,
                text="Matplotlib is required for charts. Install with: pip install matplotlib",
                font=FONTS["body"],
                fg=COLORS["accent_red"],
                bg=COLORS["bg_primary"],
            ).pack(pady=40)
            return

        # Charts in 2-column grid
        row1 = tk.Frame(self.charts_frame, bg=COLORS["bg_primary"])
        row1.pack(fill="x", pady=(0, 12))

        row2 = tk.Frame(self.charts_frame, bg=COLORS["bg_primary"])
        row2.pack(fill="x", pady=(0, 12))

        row3 = tk.Frame(self.charts_frame, bg=COLORS["bg_primary"])
        row3.pack(fill="x", pady=(0, 12))

        if self.current_sport == "Football":
            self._chart_win_draw_loss(row1, "left")
            self._chart_avg_goals(row1, "right")
            self._chart_team_performance(row2, "left")
            self._chart_momentum_trend(row2, "right")
            self._chart_prediction_confidence(row3, "left")
            self._chart_forecast_trend(row3, "right")
        else:
            self._chart_cricket_win_loss(row1, "left")
            self._chart_avg_runs(row1, "right")
            self._chart_cricket_performance(row2, "left")
            self._chart_cricket_momentum(row2, "right")
            self._chart_prediction_confidence(row3, "left")
            self._chart_cricket_forecast(row3, "right")

    # ─────────────────────────────────────────
    # Football Charts
    # ─────────────────────────────────────────

    def _chart_win_draw_loss(self, parent: tk.Widget, side: str) -> None:
        """Win/Draw/Loss distribution pie chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(0, 6 if side == "left" else 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_football_teams()[:10]
        wins, draws, losses = [], [], []

        for team in teams:
            stats = self.engine.data_engine.get_football_team_stats(team)
            wins.append(stats.get("win_rate", 0))
            draws.append(stats.get("draw_rate", 0))
            losses.append(stats.get("loss_rate", 0))

        avg_win = np.mean(wins) if wins else 0.33
        avg_draw = np.mean(draws) if draws else 0.33
        avg_loss = np.mean(losses) if losses else 0.33

        sizes = [avg_win, avg_draw, avg_loss]
        labels = [f"Win\n{avg_win:.0%}", f"Draw\n{avg_draw:.0%}", f"Loss\n{avg_loss:.0%}"]
        colors = [CHART_COLORS[2], CHART_COLORS[4], CHART_COLORS[3]]

        wedges, texts = ax.pie(
            sizes, labels=labels, colors=colors,
            startangle=90, textprops={"color": CHART_TEXT, "fontsize": 9},
        )
        ax.set_title("Win / Draw / Loss Distribution", fontsize=11, color=CHART_TEXT, pad=10)

        self._embed_chart(fig, card.inner)

    def _chart_avg_goals(self, parent: tk.Widget, side: str) -> None:
        """Average goals per team bar chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_football_teams()[:8]
        short_names = [t[:3].upper() for t in teams]
        goals_scored = []
        goals_conceded = []

        for team in teams:
            stats = self.engine.data_engine.get_football_team_stats(team)
            goals_scored.append(stats.get("avg_goals_scored", 0))
            goals_conceded.append(stats.get("avg_goals_conceded", 0))

        x = np.arange(len(short_names))
        width = 0.35

        ax.bar(x - width / 2, goals_scored, width, label="Scored", color=CHART_COLORS[0], alpha=0.85)
        ax.bar(x + width / 2, goals_conceded, width, label="Conceded", color=CHART_COLORS[3], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_ylabel("Goals/Match", fontsize=9)
        ax.set_title("Average Goals Scored vs Conceded", fontsize=11, color=CHART_TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=CHART_BG, edgecolor=CHART_GRID)
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_team_performance(self, parent: tk.Widget, side: str) -> None:
        """Team win rate comparison bar chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(0, 6 if side == "left" else 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_football_teams()[:10]
        win_rates = []
        for team in teams:
            stats = self.engine.data_engine.get_football_team_stats(team)
            win_rates.append(stats.get("win_rate", 0))

        # Sort by win rate
        sorted_pairs = sorted(zip(teams, win_rates), key=lambda x: x[1], reverse=True)
        sorted_teams = [t[:3].upper() for t, _ in sorted_pairs]
        sorted_rates = [r for _, r in sorted_pairs]

        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(sorted_teams))]
        bars = ax.barh(sorted_teams, sorted_rates, color=colors, alpha=0.85)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Win Rate", fontsize=9)
        ax.set_title("Team Win Rate Ranking", fontsize=11, color=CHART_TEXT, pad=10)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_momentum_trend(self, parent: tk.Widget, side: str) -> None:
        """Momentum trend line chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        # Show rolling win rate for top 3 teams
        teams = self.engine.data_engine.get_football_teams()[:3]

        for i, team in enumerate(teams):
            rolling = self.engine.data_engine.get_football_rolling_stats(team)
            if not rolling.empty and "rolling_win_rate" in rolling.columns:
                values = rolling["rolling_win_rate"].values[-30:]
                ax.plot(
                    range(len(values)), values,
                    label=team[:3].upper(),
                    color=CHART_COLORS[i],
                    linewidth=2,
                    alpha=0.85,
                )

        ax.set_xlabel("Recent Matches", fontsize=9)
        ax.set_ylabel("Rolling Win Rate", fontsize=9)
        ax.set_title("Momentum Trend", fontsize=11, color=CHART_TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=CHART_BG, edgecolor=CHART_GRID)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_prediction_confidence(self, parent: tk.Widget, side: str) -> None:
        """Prediction confidence distribution from history."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(0, 6 if side == "left" else 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        history = self.engine.get_prediction_history()
        if history:
            confidences = [p.get("confidence", 0) for p in history]
            ax.hist(
                confidences, bins=10, color=CHART_COLORS[0],
                alpha=0.85, edgecolor=CHART_GRID,
            )
            ax.set_xlabel("Confidence Score", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
        else:
            ax.text(
                0.5, 0.5, "No prediction data yet",
                ha="center", va="center",
                color=CHART_TEXT, fontsize=11,
                transform=ax.transAxes,
            )

        ax.set_title("Prediction Confidence Distribution", fontsize=11, color=CHART_TEXT, pad=10)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_forecast_trend(self, parent: tk.Widget, side: str) -> None:
        """Forecast trend from statsmodels analysis."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        # Get rolling goals for a sample team
        team = self.engine.data_engine.get_football_teams()[0] if self.engine.data_engine.get_football_teams() else "Brazil"
        rolling = self.engine.data_engine.get_football_rolling_stats(team)

        if not rolling.empty and "rolling_goals_for" in rolling.columns:
            values = rolling["rolling_goals_for"].values[-40:]
            ax.plot(
                range(len(values)), values,
                color=CHART_COLORS[0], linewidth=2, label="Rolling Avg Goals",
            )

            if "moving_avg_goals" in rolling.columns:
                ma_values = rolling["moving_avg_goals"].values[-40:]
                ax.plot(
                    range(len(ma_values)), ma_values,
                    color=CHART_COLORS[1], linewidth=2, linestyle="--",
                    label="Moving Average",
                )
        else:
            ax.text(
                0.5, 0.5, "No trend data available",
                ha="center", va="center",
                color=CHART_TEXT, fontsize=11,
                transform=ax.transAxes,
            )

        ax.set_xlabel("Matches", fontsize=9)
        ax.set_ylabel("Goals", fontsize=9)
        ax.set_title(f"Forecast Trend ({team})", fontsize=11, color=CHART_TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=CHART_BG, edgecolor=CHART_GRID)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    # ─────────────────────────────────────────
    # Cricket Charts
    # ─────────────────────────────────────────

    def _chart_cricket_win_loss(self, parent: tk.Widget, side: str) -> None:
        """Cricket win/loss pie chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(0, 6 if side == "left" else 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_cricket_teams()[:10]
        total_wins, total_losses = 0, 0

        for team in teams:
            stats = self.engine.data_engine.get_cricket_team_stats(team)
            total_wins += stats.get("wins", 0)
            total_losses += stats.get("losses", 0)

        total = total_wins + total_losses
        if total > 0:
            sizes = [total_wins / total, total_losses / total]
            labels = [f"Wins\n{total_wins / total:.0%}", f"Losses\n{total_losses / total:.0%}"]
            colors = [CHART_COLORS[2], CHART_COLORS[3]]
        else:
            sizes = [0.5, 0.5]
            labels = ["Wins\n50%", "Losses\n50%"]
            colors = [CHART_COLORS[2], CHART_COLORS[3]]

        ax.pie(sizes, labels=labels, colors=colors, startangle=90,
               textprops={"color": CHART_TEXT, "fontsize": 9})
        ax.set_title("Win / Loss Distribution", fontsize=11, color=CHART_TEXT, pad=10)

        self._embed_chart(fig, card.inner)

    def _chart_avg_runs(self, parent: tk.Widget, side: str) -> None:
        """Average runs per team bar chart."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_cricket_teams()
        short_names = [t.split()[-1][:4].upper() for t in teams]
        avg_runs = []

        for team in teams:
            stats = self.engine.data_engine.get_cricket_team_stats(team)
            avg_runs.append(stats.get("avg_runs", 0))

        bars = ax.bar(short_names, avg_runs, color=CHART_COLORS[:len(teams)], alpha=0.85)

        ax.set_ylabel("Average Runs", fontsize=9)
        ax.set_title("Average Runs per Team", fontsize=11, color=CHART_TEXT, pad=10)
        ax.grid(axis="y", alpha=0.2)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_cricket_performance(self, parent: tk.Widget, side: str) -> None:
        """Cricket team performance comparison."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(0, 6 if side == "left" else 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_cricket_teams()
        win_rates = []
        for team in teams:
            stats = self.engine.data_engine.get_cricket_team_stats(team)
            win_rates.append(stats.get("win_rate", 0))

        sorted_pairs = sorted(zip(teams, win_rates), key=lambda x: x[1], reverse=True)
        sorted_teams = [t.split()[-1][:4].upper() for t, _ in sorted_pairs]
        sorted_rates = [r for _, r in sorted_pairs]

        colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(sorted_teams))]
        ax.barh(sorted_teams, sorted_rates, color=colors, alpha=0.85)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Win Rate", fontsize=9)
        ax.set_title("Cricket Team Rankings", fontsize=11, color=CHART_TEXT, pad=10)
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_cricket_momentum(self, parent: tk.Widget, side: str) -> None:
        """Cricket momentum trend."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_cricket_teams()[:3]

        for i, team in enumerate(teams):
            rolling = self.engine.data_engine.get_cricket_rolling_stats(team)
            if not rolling.empty and "rolling_win_rate" in rolling.columns:
                values = rolling["rolling_win_rate"].values[-30:]
                short_name = team.split()[-1][:4].upper()
                ax.plot(
                    range(len(values)), values,
                    label=short_name,
                    color=CHART_COLORS[i],
                    linewidth=2, alpha=0.85,
                )

        ax.set_xlabel("Recent Matches", fontsize=9)
        ax.set_ylabel("Rolling Win Rate", fontsize=9)
        ax.set_title("Cricket Momentum Trend", fontsize=11, color=CHART_TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=CHART_BG, edgecolor=CHART_GRID)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    def _chart_cricket_forecast(self, parent: tk.Widget, side: str) -> None:
        """Cricket runs forecast trend."""
        card = create_card(parent, padding=10)
        card.pack(side=side, fill="both", expand=True, padx=(6 if side == "right" else 0, 0))

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        teams = self.engine.data_engine.get_cricket_teams()
        team = teams[0] if teams else "Mumbai Indians"

        rolling = self.engine.data_engine.get_cricket_rolling_stats(team)
        if not rolling.empty and "rolling_avg_runs" in rolling.columns:
            values = rolling["rolling_avg_runs"].values[-40:]
            ax.plot(
                range(len(values)), values,
                color=CHART_COLORS[0], linewidth=2, label="Rolling Avg Runs",
            )
        else:
            ax.text(
                0.5, 0.5, "No trend data available",
                ha="center", va="center",
                color=CHART_TEXT, fontsize=11,
                transform=ax.transAxes,
            )

        short_name = team.split()[-1]
        ax.set_xlabel("Matches", fontsize=9)
        ax.set_ylabel("Runs", fontsize=9)
        ax.set_title(f"Runs Forecast ({short_name})", fontsize=11, color=CHART_TEXT, pad=10)
        ax.legend(fontsize=8, facecolor=CHART_BG, edgecolor=CHART_GRID)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        self._embed_chart(fig, card.inner)

    # ─────────────────────────────────────────
    # Chart Embedding
    # ─────────────────────────────────────────

    def _embed_chart(self, fig: Figure, parent: tk.Widget) -> None:
        """
        Embed a Matplotlib figure in a Tkinter widget.

        Uses FigureCanvasTkAgg from matplotlib.backends.

        Args:
            fig: Matplotlib Figure object.
            parent: Parent Tkinter widget.
        """
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widgets.append(canvas)

    def refresh(self) -> None:
        """Refresh the analytics page."""
        self._refresh_charts()

