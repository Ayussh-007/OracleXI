"""
OracleXI v2.0 – Comparison Dashboard
========================================
Replaces the old prediction page.
Provides a rich side-by-side comparison of two teams with:
- Radar charts for overall stats
- Form badges
- Comparison grid (20+ metrics)
- Head-to-Head history
- Final ensemble prediction probabilities
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from core.prediction_engine import PredictionEngine
from utils.constants import COLORS, FONTS, SPORTS, RADAR_AXES
from utils.helper import setup_logging, format_percentage, format_elo
from gui.widgets import (
    create_section_header,
    create_card,
    RoundedButton,
    LoadingSpinner,
    ScrollableFrame,
    ComparisonBar,
    FormBadges,
    VSBadge,
)

logger = setup_logging("ComparePage")


class ComparePage(tk.Frame):
    """
    Main Comparison and Prediction Dashboard.
    """

    def __init__(self, parent: tk.Widget, engine: PredictionEngine) -> None:
        """Initialize the compare page."""
        super().__init__(parent, bg=COLORS["bg_primary"])
        self.engine = engine

        self.current_sport = tk.StringVar(value=SPORTS[0])
        self.t1_var = tk.StringVar(value="Select Team")
        self.t2_var = tk.StringVar(value="Select Team")

        # Layout
        self._build_top_bar()

        # Content area (scrollable)
        self.content_container = tk.Frame(self, bg=COLORS["bg_primary"])
        self.content_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.scroll_frame = ScrollableFrame(self.content_container)
        self.scroll_frame.pack(fill="both", expand=True)
        self.inner = self.scroll_frame.scrollable_frame

        self.results_frame = tk.Frame(self.inner, bg=COLORS["bg_primary"])
        self.results_frame.pack(fill="both", expand=True, pady=10)

        # Initial teams update
        self._on_sport_change()

    def _build_top_bar(self) -> None:
        """Build the top selection bar."""
        top_bar = tk.Frame(self, bg=COLORS["bg_secondary"], height=80)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        inner = tk.Frame(top_bar, bg=COLORS["bg_secondary"])
        inner.pack(expand=True, fill="both", padx=30, pady=15)

        # Sport Selector
        sport_frame = tk.Frame(inner, bg=COLORS["bg_secondary"])
        sport_frame.pack(side="left")

        for sport in SPORTS:
            rb = tk.Radiobutton(
                sport_frame,
                text=sport.upper(),
                variable=self.current_sport,
                value=sport,
                command=self._on_sport_change,
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_tertiary"],
                activebackground=COLORS["bg_secondary"],
                activeforeground=COLORS["accent_cyan"],
                font=FONTS["nav"],
                indicatoron=False,
                bd=0,
                padx=15,
                pady=5,
            )
            rb.pack(side="left", padx=5)

        # Team Selection
        sel_frame = tk.Frame(inner, bg=COLORS["bg_secondary"])
        sel_frame.pack(side="left", padx=40)

        self.t1_cb = ttk.Combobox(
            sel_frame,
            textvariable=self.t1_var,
            state="readonly",
            width=25,
            font=FONTS["body"],
            style="Dark.TCombobox",
        )
        self.t1_cb.pack(side="left")

        VSBadge(sel_frame, size=40, bg_color=COLORS["bg_secondary"]).pack(side="left", padx=15)

        self.t2_cb = ttk.Combobox(
            sel_frame,
            textvariable=self.t2_var,
            state="readonly",
            width=25,
            font=FONTS["body"],
            style="Dark.TCombobox",
        )
        self.t2_cb.pack(side="left")

        # Predict Button
        self.btn_predict = RoundedButton(
            inner,
            text="COMPARE & PREDICT",
            command=self._run_prediction,
            bg_color=COLORS["accent_cyan"],
            fg_color="#000000",
            width=180,
            height=40,
        )
        self.btn_predict.pack(side="right")

        # Spinner
        self.spinner = LoadingSpinner(inner, size=30, color=COLORS["accent_cyan"])
        self.spinner.pack(side="right", padx=15)

    def _on_sport_change(self) -> None:
        """Update comboboxes when sport changes."""
        sport = self.current_sport.get()
        if sport == "Football":
            teams = self.engine.get_football_teams()
        else:
            teams = self.engine.get_cricket_teams()

        self.t1_cb["values"] = ["Select Team"] + teams
        self.t2_cb["values"] = ["Select Team"] + teams
        self.t1_var.set("Select Team")
        self.t2_var.set("Select Team")

        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def _run_prediction(self) -> None:
        """Run prediction in a background thread."""
        sport = self.current_sport.get()
        t1 = self.t1_var.get()
        t2 = self.t2_var.get()

        if t1 == "Select Team" or t2 == "Select Team" or t1 == t2:
            messagebox.showwarning("Invalid Selection", "Please select two distinct teams.")
            return

        self.btn_predict.configure(state="disabled")
        self.spinner.start()

        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        def worker():
            try:
                if sport == "Football":
                    res = self.engine.predict_football(t1, t2)
                else:
                    res = self.engine.predict_cricket(t1, t2)
                self.after(0, self._render_results, res)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, self._prediction_done)

        threading.Thread(target=worker, daemon=True).start()

    def _prediction_done(self) -> None:
        """Restore UI state after prediction."""
        self.btn_predict.configure(state="normal")
        self.spinner.stop()

    def _render_results(self, data: Dict[str, Any]) -> None:
        """Render the rich comparison dashboard."""
        if "error" in data:
            messagebox.showerror("Error", data["error"])
            return

        t1 = data["team_1"]
        t2 = data["team_2"]
        sport = data["sport"]
        pred = data["prediction"]
        comp = data["comparison"]
        r = comp["ratings"]

        # Layout: Top Row (Winner + Probs + Radar)
        top_row = tk.Frame(self.results_frame, bg=COLORS["bg_primary"])
        top_row.pack(fill="x", pady=10)

        # 1. Winner Card (Left)
        win_card = create_card(top_row, padding=20)
        win_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(win_card.inner, text="PREDICTED WINNER", font=FONTS["small_bold"], fg=COLORS["accent_cyan"], bg=COLORS["bg_secondary"]).pack(anchor="w")
        tk.Label(win_card.inner, text=pred["winner"].upper(), font=("Helvetica Neue", 32, "bold"), fg=COLORS["text_primary"], bg=COLORS["bg_secondary"]).pack(anchor="w", pady=(10, 0))
        tk.Label(win_card.inner, text=f"Confidence: {format_percentage(pred['confidence'])}", font=FONTS["body"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(anchor="w")

        # 2. Probability Bar (Center)
        prob_card = create_card(top_row, padding=20)
        prob_card.pack(side="left", fill="both", expand=True, padx=10)
        
        tk.Label(prob_card.inner, text="WIN PROBABILITY", font=FONTS["small_bold"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(anchor="center")
        
        cbar = ComparisonBar(
            prob_card.inner, 
            val1=pred["team_1_win"], 
            val2=pred["team_2_win"], 
            val_draw=pred.get("draw", 0.0),
            width=350, height=30
        )
        cbar.pack(pady=20)

        team_names = tk.Frame(prob_card.inner, bg=COLORS["bg_secondary"])
        team_names.pack(fill="x")
        tk.Label(team_names, text=t1, font=FONTS["body_bold"], fg=COLORS["accent_cyan"], bg=COLORS["bg_secondary"]).pack(side="left")
        if "draw" in pred:
            tk.Label(team_names, text="Draw", font=FONTS["small"], fg=COLORS["accent_orange"], bg=COLORS["bg_secondary"]).pack(side="left", expand=True)
        tk.Label(team_names, text=t2, font=FONTS["body_bold"], fg=COLORS["accent_purple"], bg=COLORS["bg_secondary"]).pack(side="right")

        # 3. Radar Chart (Right)
        radar_card = create_card(top_row, padding=10)
        radar_card.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._embed_radar_chart(radar_card.inner, t1, t2, r)

        # Bottom Row: Head-to-Head & Stats Grid
        bot_row = tk.Frame(self.results_frame, bg=COLORS["bg_primary"])
        bot_row.pack(fill="both", expand=True, pady=10)

        # 1. H2H Card
        h2h_card = create_card(bot_row, padding=20)
        h2h_card.pack(side="left", fill="y", padx=(0, 10))
        self._build_h2h(h2h_card.inner, comp["h2h"], t1, t2, sport)

        # 2. Stats Grid
        grid_card = create_card(bot_row, padding=20)
        grid_card.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._build_stats_grid(grid_card.inner, comp, sport)

    def _embed_radar_chart(self, parent: tk.Widget, t1: str, t2: str, r: dict) -> None:
        """Embed a matplotlib radar chart comparing the two teams."""
        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(COLORS["bg_secondary"])
        ax.set_facecolor(COLORS["bg_secondary"])

        angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False).tolist()
        angles += angles[:1]

        v1 = [r["attack_1"], r["defense_1"], r["form_1"], r["momentum_1"], r["experience_1"], r["consistency_1"]]
        v2 = [r["attack_2"], r["defense_2"], r["form_2"], r["momentum_2"], r["experience_2"], r["consistency_2"]]
        v1 += v1[:1]
        v2 += v2[:1]

        ax.plot(angles, v1, color=COLORS["accent_cyan"], linewidth=2, label=t1)
        ax.fill(angles, v1, color=COLORS["accent_cyan"], alpha=0.25)
        ax.plot(angles, v2, color=COLORS["accent_purple"], linewidth=2, label=t2)
        ax.fill(angles, v2, color=COLORS["accent_purple"], alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), RADAR_AXES, color=COLORS["text_secondary"], fontsize=9)
        
        ax.set_ylim(0, 100)
        ax.set_yticklabels([])
        ax.grid(color=COLORS["border_default"], linestyle="--", alpha=0.7)
        ax.spines['polar'].set_color(COLORS["border_default"])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_h2h(self, parent: tk.Widget, h2h: dict, t1: str, t2: str, sport: str) -> None:
        tk.Label(parent, text="HEAD TO HEAD", font=FONTS["subheading"], fg=COLORS["text_primary"], bg=COLORS["bg_secondary"]).pack(anchor="w")
        tk.Label(parent, text=f"{h2h['matches']} Matches Played", font=FONTS["small"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(anchor="w", pady=(0, 20))

        if h2h["matches"] == 0:
            tk.Label(parent, text="No Historical Data", font=FONTS["body"], fg=COLORS["text_muted"], bg=COLORS["bg_secondary"]).pack(pady=20)
            return

        # Circle stats
        row = tk.Frame(parent, bg=COLORS["bg_secondary"])
        row.pack(fill="x")

        # T1 Wins
        tk.Label(row, text=str(h2h["t1_wins"]), font=("Helvetica Neue", 28, "bold"), fg=COLORS["accent_cyan"], bg=COLORS["bg_secondary"]).pack(side="left", padx=10)
        
        # Draws (football only)
        if sport == "Football":
            tk.Label(row, text=str(h2h.get("draws", 0)), font=("Helvetica Neue", 28, "bold"), fg=COLORS["accent_orange"], bg=COLORS["bg_secondary"]).pack(side="left", expand=True)
            
        # T2 Wins
        tk.Label(row, text=str(h2h["t2_wins"]), font=("Helvetica Neue", 28, "bold"), fg=COLORS["accent_purple"], bg=COLORS["bg_secondary"]).pack(side="right", padx=10)

        # Labels
        lrow = tk.Frame(parent, bg=COLORS["bg_secondary"])
        lrow.pack(fill="x", pady=(0, 20))
        tk.Label(lrow, text=t1[:10], font=FONTS["small_bold"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(side="left", padx=10)
        if sport == "Football":
            tk.Label(lrow, text="DRAWS", font=FONTS["small_bold"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(side="left", expand=True)
        tk.Label(lrow, text=t2[:10], font=FONTS["small_bold"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(side="right", padx=10)

    def _build_stats_grid(self, parent: tk.Widget, comp: dict, sport: str) -> None:
        create_section_header(parent, "STATISTICAL COMPARISON").pack(fill="x", pady=(0, 15))
        
        s1 = comp["stats_1"]
        s2 = comp["stats_2"]

        grid = tk.Frame(parent, bg=COLORS["bg_secondary"])
        grid.pack(fill="both", expand=True)

        def add_row(label: str, val1: Any, val2: Any, highlight_higher: bool = True):
            r = tk.Frame(grid, bg=COLORS["bg_secondary"])
            r.pack(fill="x", pady=4)
            
            c1, c2 = COLORS["text_primary"], COLORS["text_primary"]
            
            # Simple numeric highlight
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 != val2:
                    if (val1 > val2 and highlight_higher) or (val1 < val2 and not highlight_higher):
                        c1 = COLORS["accent_cyan"]
                    else:
                        c2 = COLORS["accent_purple"]
                
                # Format floats
                if isinstance(val1, float): val1 = f"{val1:.2f}"
                if isinstance(val2, float): val2 = f"{val2:.2f}"

            tk.Label(r, text=str(val1), font=FONTS["body_bold"], fg=c1, bg=COLORS["bg_secondary"], width=15, anchor="e").pack(side="left")
            tk.Label(r, text=label, font=FONTS["small"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(side="left", expand=True)
            tk.Label(r, text=str(val2), font=FONTS["body_bold"], fg=c2, bg=COLORS["bg_secondary"], width=15, anchor="w").pack(side="right")

        # Custom Recent Form row with Badges
        r_form = tk.Frame(grid, bg=COLORS["bg_secondary"])
        r_form.pack(fill="x", pady=4)
        left = tk.Frame(r_form, bg=COLORS["bg_secondary"], width=100)
        left.pack(side="left")
        left.pack_propagate(False)
        FormBadges(left, s1["form_string"]).pack(anchor="e")
        
        tk.Label(r_form, text="Recent Form (L5)", font=FONTS["small"], fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"]).pack(side="left", expand=True)
        
        right = tk.Frame(r_form, bg=COLORS["bg_secondary"], width=100)
        right.pack(side="right")
        right.pack_propagate(False)
        FormBadges(right, s2["form_string"]).pack(anchor="w")

        # Common Stats
        add_row("Elo Rating", format_elo(s1["elo_rating"]), format_elo(s2["elo_rating"]))
        add_row("Win Rate", f"{s1['win_rate']*100:.1f}%", f"{s2['win_rate']*100:.1f}%")
        add_row("Matches Played", s1["total_matches"], s2["total_matches"])
        
        if sport == "Football":
            add_row("Goals Scored (Avg)", s1["avg_goals_scored"], s2["avg_goals_scored"])
            add_row("Goals Conceded (Avg)", s1["avg_goals_conceded"], s2["avg_goals_conceded"], False)
            add_row("Goal Difference (Avg)", s1["goal_diff"], s2["goal_diff"])
        else:
            add_row("Avg Runs", s1["avg_runs"], s2["avg_runs"])
            add_row("Strike Rate", s1["strike_rate"], s2["strike_rate"])
            add_row("Run Rate", s1["run_rate"], s2["run_rate"])
            add_row("Bowling Economy", s1["economy"], s2["economy"], False)
            add_row("Avg Wickets", s1["avg_wickets"], s2["avg_wickets"])
            
        add_row("Current Win Streak", s1["win_streak"], s2["win_streak"])
        add_row("Momentum Index", f"{s1['form_10']*100:.0f}", f"{s2['form_10']*100:.0f}")
