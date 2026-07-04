"""
OracleXI – Prediction Page
=============================
Sport-specific prediction interface with team selection,
simulation controls, loading animation, and results display.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Callable, Dict, Optional

from gui.widgets import (
    COLORS,
    FONTS,
    GradientFrame,
    LoadingSpinner,
    ModernProgressBar,
    RoundedButton,
    ScrollableFrame,
    create_card,
    create_section_header,
)
from utils.constants import CRICKET_TEAMS, FOOTBALL_TEAMS, SIMULATION_COUNTS
from utils.helper import (
    export_single_prediction_to_csv,
    format_percentage,
    validate_team_selection,
)

if TYPE_CHECKING:
    from core.prediction_engine import PredictionEngine


class PredictionPage(tk.Frame):
    """
    Prediction page for both Football and Cricket.

    Features:
        - Team A / Team B dropdowns
        - Simulation count selector
        - Predict, Compare, Reset, Export buttons
        - Loading animation during prediction
        - Results display with progress bars
        - Statistical summary card
    """

    def __init__(
        self,
        parent: tk.Widget,
        engine: "PredictionEngine",
        navigate_callback: Callable,
        **kwargs,
    ) -> None:
        """
        Initialize the prediction page.

        Args:
            parent: Parent widget.
            engine: Prediction engine instance.
            navigate_callback: Navigation callback function.
        """
        super().__init__(parent, bg=COLORS["bg_primary"], **kwargs)
        self.engine = engine
        self.navigate = navigate_callback
        self.current_sport: str = "Football"
        self.current_result: Optional[Dict] = None
        self._prediction_running = False

        self._build_ui()

    def set_sport(self, sport: str) -> None:
        """
        Set the active sport and update team dropdowns.

        Args:
            sport: 'Football' or 'Cricket'.
        """
        self.current_sport = sport
        self._update_sport_tabs()
        self._update_team_lists()

    def _build_ui(self) -> None:
        """Build the complete prediction page UI."""
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # ── Page Header ──
        header_section = tk.Frame(container, bg=COLORS["bg_primary"])
        header_section.pack(fill="x", padx=30, pady=(25, 5))

        header = create_section_header(
            header_section,
            "Match Prediction",
            "Configure teams and run AI-powered prediction",
        )
        header.pack(fill="x")

        # ── Sport Tabs ──
        self._build_sport_tabs(container)

        # ── Main Content: Controls + Results ──
        main = tk.Frame(container, bg=COLORS["bg_primary"])
        main.pack(fill="both", expand=True, padx=30, pady=(10, 30))

        # Left panel: Controls
        controls_frame = tk.Frame(main, bg=COLORS["bg_primary"])
        controls_frame.pack(side="left", fill="y", padx=(0, 15))
        self._build_controls(controls_frame)

        # Right panel: Results
        self.results_frame = tk.Frame(main, bg=COLORS["bg_primary"])
        self.results_frame.pack(side="left", fill="both", expand=True)
        self._build_empty_results()

    # ─────────────────────────────────────────
    # Sport Tabs
    # ─────────────────────────────────────────

    def _build_sport_tabs(self, parent: tk.Widget) -> None:
        """Build sport selection tabs."""
        tabs = tk.Frame(parent, bg=COLORS["bg_primary"])
        tabs.pack(fill="x", padx=30, pady=(10, 5))

        self.tab_buttons: Dict[str, tk.Label] = {}

        for sport in ["Football", "Cricket"]:
            icon = "⚽" if sport == "Football" else "🏏"
            tab = tk.Label(
                tabs,
                text=f"  {icon}  {sport}  ",
                font=FONTS["nav"],
                cursor="hand2",
                padx=20,
                pady=8,
            )
            tab.pack(side="left", padx=(0, 4))
            tab.bind("<Button-1>", lambda e, s=sport: self.set_sport(s))
            self.tab_buttons[sport] = tab

        self._update_sport_tabs()

    def _update_sport_tabs(self) -> None:
        """Update tab styling based on current sport."""
        for sport, tab in self.tab_buttons.items():
            if sport == self.current_sport:
                tab.configure(
                    bg=COLORS["accent_cyan"],
                    fg=COLORS["bg_primary"],
                )
            else:
                tab.configure(
                    bg=COLORS["bg_tertiary"],
                    fg=COLORS["text_secondary"],
                )

    # ─────────────────────────────────────────
    # Controls Panel
    # ─────────────────────────────────────────

    def _build_controls(self, parent: tk.Widget) -> None:
        """Build the team selection and control buttons."""
        card = create_card(parent, padding=0)
        card.pack(fill="x")

        inner = card.inner

        # Card title
        title_bar = tk.Frame(inner, bg=COLORS["bg_tertiary"])
        title_bar.pack(fill="x")
        tk.Label(
            title_bar,
            text="  ⚙  Match Configuration",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            pady=10,
        ).pack(anchor="w", padx=12)

        content = tk.Frame(inner, bg=COLORS["bg_secondary"])
        content.pack(fill="x", padx=20, pady=20)

        # ── Team A ──
        tk.Label(
            content,
            text="Team A (Home)",
            font=FONTS["small_bold"],
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        self.team_a_var = tk.StringVar(value="Select Team")
        self.team_a_combo = ttk.Combobox(
            content,
            textvariable=self.team_a_var,
            state="readonly",
            style="Dark.TCombobox",
            width=30,
        )
        self.team_a_combo.pack(fill="x", pady=(4, 12))

        # ── Team B ──
        tk.Label(
            content,
            text="Team B (Away)",
            font=FONTS["small_bold"],
            fg=COLORS["accent_purple"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        self.team_b_var = tk.StringVar(value="Select Team")
        self.team_b_combo = ttk.Combobox(
            content,
            textvariable=self.team_b_var,
            state="readonly",
            style="Dark.TCombobox",
            width=30,
        )
        self.team_b_combo.pack(fill="x", pady=(4, 12))

        # ── Simulation Count ──
        tk.Label(
            content,
            text="Simulation Count",
            font=FONTS["small_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        self.sim_var = tk.StringVar(value="10000")
        sim_combo = ttk.Combobox(
            content,
            textvariable=self.sim_var,
            state="readonly",
            values=[str(s) for s in SIMULATION_COUNTS],
            style="Dark.TCombobox",
            width=30,
        )
        sim_combo.pack(fill="x", pady=(4, 20))

        # ── Buttons ──
        btn_frame = tk.Frame(content, bg=COLORS["bg_secondary"])
        btn_frame.pack(fill="x")

        # Predict button
        RoundedButton(
            btn_frame,
            text="⚡ Predict",
            command=self._run_prediction,
            bg_color=COLORS["accent_cyan"],
            width=260,
            height=42,
        ).pack(fill="x", pady=(0, 8))

        # Button row
        btn_row = tk.Frame(btn_frame, bg=COLORS["bg_secondary"])
        btn_row.pack(fill="x")

        RoundedButton(
            btn_row,
            text="🔄 Reset",
            command=self._reset,
            bg_color=COLORS["bg_tertiary"],
            fg_color=COLORS["text_primary"],
            width=125,
            height=36,
        ).pack(side="left", padx=(0, 8))

        RoundedButton(
            btn_row,
            text="📥 Export",
            command=self._export_result,
            bg_color=COLORS["bg_tertiary"],
            fg_color=COLORS["text_primary"],
            width=125,
            height=36,
        ).pack(side="left")

        # ── Loading Indicator ──
        self.loading_frame = tk.Frame(content, bg=COLORS["bg_secondary"])
        self.loading_frame.pack(fill="x", pady=(15, 0))

        self.spinner = LoadingSpinner(self.loading_frame, size=30)
        self.spinner.pack(side="left")

        self.loading_label = tk.Label(
            self.loading_frame,
            text="",
            font=FONTS["small"],
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        )
        self.loading_label.pack(side="left", padx=(8, 0))

        # Initialize team lists
        self._update_team_lists()

    def _update_team_lists(self) -> None:
        """Update team dropdown options based on current sport."""
        if self.current_sport == "Football":
            teams = self.engine.data_engine.get_football_teams()
        else:
            teams = self.engine.data_engine.get_cricket_teams()

        self.team_a_combo["values"] = teams
        self.team_b_combo["values"] = teams
        self.team_a_var.set("Select Team")
        self.team_b_var.set("Select Team")

    # ─────────────────────────────────────────
    # Prediction Execution
    # ─────────────────────────────────────────

    def _run_prediction(self) -> None:
        """Run the prediction in a background thread."""
        if self._prediction_running:
            return

        team_a = self.team_a_var.get()
        team_b = self.team_b_var.get()

        if self.current_sport == "Football":
            teams = self.engine.data_engine.get_football_teams()
        else:
            teams = self.engine.data_engine.get_cricket_teams()

        is_valid, error_msg = validate_team_selection(team_a, team_b, teams)
        if not is_valid:
            messagebox.showwarning("Invalid Selection", error_msg)
            return

        n_sims = int(self.sim_var.get())

        # Show loading
        self._prediction_running = True
        self.loading_label.config(
            text=f"Running {n_sims:,} simulations..."
        )
        self.spinner.start()

        # Run in background thread
        thread = threading.Thread(
            target=self._prediction_worker,
            args=(team_a, team_b, n_sims),
            daemon=True,
        )
        thread.start()

    def _prediction_worker(
        self, team_a: str, team_b: str, n_sims: int
    ) -> None:
        """Worker function for prediction (runs in background thread)."""
        try:
            if self.current_sport == "Football":
                result = self.engine.predict_football(team_a, team_b, n_sims)
            else:
                result = self.engine.predict_cricket(team_a, team_b, n_sims)

            self.current_result = result

            # Update UI from main thread
            self.after(0, lambda: self._display_results(result))
        except Exception as e:
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Prediction Error", str(e)
                ),
            )
        finally:
            self.after(0, self._stop_loading)

    def _stop_loading(self) -> None:
        """Stop the loading indicator."""
        self._prediction_running = False
        self.spinner.stop()
        self.loading_label.config(text="")

    # ─────────────────────────────────────────
    # Results Display
    # ─────────────────────────────────────────

    def _build_empty_results(self) -> None:
        """Show placeholder when no prediction has been made."""
        card = create_card(self.results_frame)
        card.pack(fill="both", expand=True)

        tk.Label(
            card.inner,
            text="⚡",
            font=("Helvetica Neue", 48, "normal"),
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack(pady=(40, 10))

        tk.Label(
            card.inner,
            text="Select teams and click Predict",
            font=FONTS["heading"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack()

        tk.Label(
            card.inner,
            text="Results will appear here with detailed analysis",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack(pady=(5, 40))

    def _display_results(self, result: Dict) -> None:
        """
        Display prediction results with progress bars and stats.

        Args:
            result: Prediction result dictionary.
        """
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        sport = result.get("sport", "Football")
        team_a = result.get("team_a", "Team A")
        team_b = result.get("team_b", "Team B")
        prediction = result.get("prediction", {})
        confidence = result.get("confidence", 0)

        # ── Winner Card ──
        winner_card = create_card(self.results_frame, padding=0)
        winner_card.pack(fill="x", pady=(0, 12))

        inner = winner_card.inner
        accent_bar = tk.Frame(inner, bg=COLORS["accent_cyan"], height=4)
        accent_bar.pack(fill="x")

        winner_content = tk.Frame(inner, bg=COLORS["bg_secondary"])
        winner_content.pack(fill="x", padx=20, pady=16)

        # Match title
        tk.Label(
            winner_content,
            text=f"{team_a}  vs  {team_b}",
            font=FONTS["heading"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        tk.Label(
            winner_content,
            text=f"{sport}  •  {result.get('simulations', 0):,} simulations  •  {result.get('elapsed_seconds', 0):.1f}s",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w", pady=(2, 12))

        # ── Winning Probability Bars ──
        a_win = prediction.get("team_a_win", 0)
        b_win = prediction.get("team_b_win", 0)
        draw = prediction.get("draw", 0)

        # Team A bar
        self._create_result_bar(
            winner_content, team_a, a_win,
            COLORS["accent_cyan"],
        )

        if sport == "Football":
            self._create_result_bar(
                winner_content, "Draw", draw,
                COLORS["accent_orange"],
            )

        # Team B bar
        self._create_result_bar(
            winner_content, team_b, b_win,
            COLORS["accent_purple"],
        )

        # ── Detailed Stats Grid ──
        stats_card = create_card(self.results_frame, padding=0)
        stats_card.pack(fill="x", pady=(0, 12))

        stats_inner = stats_card.inner

        # Stats header
        stats_header = tk.Frame(stats_inner, bg=COLORS["bg_tertiary"])
        stats_header.pack(fill="x")
        tk.Label(
            stats_header,
            text="  📊  Detailed Analysis",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            pady=10,
        ).pack(anchor="w", padx=12)

        stats_content = tk.Frame(stats_inner, bg=COLORS["bg_secondary"])
        stats_content.pack(fill="x", padx=20, pady=16)

        # Stats grid
        if sport == "Football":
            expected = result.get("expected_goals", {})
            stats_items = [
                ("Expected Goals (A)", f"{expected.get('team_a', 0):.2f}", COLORS["accent_cyan"]),
                ("Expected Goals (B)", f"{expected.get('team_b', 0):.2f}", COLORS["accent_purple"]),
                ("Most Likely Score", result.get("most_likely_score", "N/A"), COLORS["accent_green"]),
                ("Confidence", format_percentage(confidence), COLORS["accent_orange"]),
            ]
        else:
            expected = result.get("expected_runs", {})
            stats_items = [
                ("Expected Runs (A)", f"{expected.get('team_a', 0):.1f}", COLORS["accent_cyan"]),
                ("Expected Runs (B)", f"{expected.get('team_b', 0):.1f}", COLORS["accent_purple"]),
                ("Confidence", format_percentage(confidence), COLORS["accent_orange"]),
            ]

        stats_grid = tk.Frame(stats_content, bg=COLORS["bg_secondary"])
        stats_grid.pack(fill="x")

        for i, (label, value, color) in enumerate(stats_items):
            cell = tk.Frame(stats_grid, bg=COLORS["bg_secondary"])
            cell.pack(side="left", fill="x", expand=True, padx=(0, 10))

            tk.Label(
                cell,
                text=label,
                font=FONTS["caption"],
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_secondary"],
            ).pack(anchor="w")
            tk.Label(
                cell,
                text=value,
                font=FONTS["subheading"],
                fg=color,
                bg=COLORS["bg_secondary"],
            ).pack(anchor="w")

        # ── Momentum & Form ──
        form_card = create_card(self.results_frame, padding=0)
        form_card.pack(fill="x", pady=(0, 12))

        form_inner = form_card.inner

        form_header = tk.Frame(form_inner, bg=COLORS["bg_tertiary"])
        form_header.pack(fill="x")
        tk.Label(
            form_header,
            text="  🔥  Form & Momentum",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_tertiary"],
            pady=10,
        ).pack(anchor="w", padx=12)

        form_content = tk.Frame(form_inner, bg=COLORS["bg_secondary"])
        form_content.pack(fill="x", padx=20, pady=16)

        momentum = result.get("momentum", {})
        recent_form = result.get("recent_form", {})
        trend = result.get("trend", {})

        form_items = [
            (f"Momentum ({team_a})", momentum.get("team_a", 0.5)),
            (f"Momentum ({team_b})", momentum.get("team_b", 0.5)),
            (f"Recent Form ({team_a})", recent_form.get("team_a", 0.5)),
            (f"Recent Form ({team_b})", recent_form.get("team_b", 0.5)),
        ]

        for label, value in form_items:
            row = tk.Frame(form_content, bg=COLORS["bg_secondary"])
            row.pack(fill="x", pady=(0, 8))

            tk.Label(
                row,
                text=label,
                font=FONTS["small"],
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_secondary"],
                width=25,
                anchor="w",
            ).pack(side="left")

            bar = ModernProgressBar(
                row,
                value=value,
                max_value=1.0,
                width=250,
                height=20,
                bar_color=COLORS["accent_cyan"] if "A" in label or team_a in label else COLORS["accent_purple"],
                show_label=True,
            )
            bar.pack(side="left", padx=(8, 0))

        # Trend indicators
        trend_frame = tk.Frame(form_content, bg=COLORS["bg_secondary"])
        trend_frame.pack(fill="x", pady=(10, 0))

        trend_a = trend.get("team_a", "stable")
        trend_b = trend.get("team_b", "stable")
        trend_icons = {"improving": "📈", "declining": "📉", "stable": "➡️"}

        tk.Label(
            trend_frame,
            text=f"{trend_icons.get(trend_a, '➡️')} {team_a}: {trend_a.title()}",
            font=FONTS["small_bold"],
            fg=COLORS["accent_green"] if trend_a == "improving" else COLORS["accent_red"] if trend_a == "declining" else COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left", padx=(0, 30))

        tk.Label(
            trend_frame,
            text=f"{trend_icons.get(trend_b, '➡️')} {team_b}: {trend_b.title()}",
            font=FONTS["small_bold"],
            fg=COLORS["accent_green"] if trend_b == "improving" else COLORS["accent_red"] if trend_b == "declining" else COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left")

        # ── H2H Summary ──
        h2h = result.get("head_to_head", {})
        if h2h.get("matches", 0) > 0:
            h2h_card = create_card(self.results_frame, padding=0)
            h2h_card.pack(fill="x")

            h2h_inner = h2h_card.inner

            h2h_header = tk.Frame(h2h_inner, bg=COLORS["bg_tertiary"])
            h2h_header.pack(fill="x")
            tk.Label(
                h2h_header,
                text="  🤝  Head-to-Head Record",
                font=FONTS["body_bold"],
                fg=COLORS["text_primary"],
                bg=COLORS["bg_tertiary"],
                pady=10,
            ).pack(anchor="w", padx=12)

            h2h_content = tk.Frame(h2h_inner, bg=COLORS["bg_secondary"])
            h2h_content.pack(fill="x", padx=20, pady=16)

            h2h_data = [
                ("Total Matches", str(h2h.get("matches", 0)), COLORS["text_primary"]),
                (f"{team_a} Wins", str(h2h.get("team_a_wins", 0)), COLORS["accent_cyan"]),
                (f"{team_b} Wins", str(h2h.get("team_b_wins", 0)), COLORS["accent_purple"]),
            ]
            if "draws" in h2h:
                h2h_data.append(("Draws", str(h2h.get("draws", 0)), COLORS["accent_orange"]))

            for label, value, color in h2h_data:
                cell = tk.Frame(h2h_content, bg=COLORS["bg_secondary"])
                cell.pack(side="left", fill="x", expand=True)

                tk.Label(
                    cell, text=label, font=FONTS["caption"],
                    fg=COLORS["text_secondary"], bg=COLORS["bg_secondary"],
                ).pack()
                tk.Label(
                    cell, text=value, font=FONTS["stat_value"],
                    fg=color, bg=COLORS["bg_secondary"],
                ).pack()

    def _create_result_bar(
        self,
        parent: tk.Widget,
        label: str,
        value: float,
        color: str,
    ) -> None:
        """Create a labeled result progress bar."""
        row = tk.Frame(parent, bg=COLORS["bg_secondary"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(
            row,
            text=label,
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_secondary"],
            width=20,
            anchor="w",
        ).pack(side="left")

        bar = ModernProgressBar(
            row,
            value=value,
            max_value=1.0,
            width=300,
            height=26,
            bar_color=color,
            show_label=True,
        )
        bar.pack(side="left", fill="x", expand=True, padx=(8, 0))

    # ─────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────

    def _reset(self) -> None:
        """Reset all inputs and results."""
        self.team_a_var.set("Select Team")
        self.team_b_var.set("Select Team")
        self.sim_var.set("10000")
        self.current_result = None

        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._build_empty_results()

    def _export_result(self) -> None:
        """Export current prediction result to CSV."""
        if not self.current_result:
            messagebox.showinfo("No Data", "Run a prediction first before exporting.")
            return

        filepath = export_single_prediction_to_csv(self.current_result)
        if filepath:
            messagebox.showinfo(
                "Export Complete",
                f"Prediction exported to:\n{filepath}",
            )

    def refresh(self) -> None:
        """Refresh the prediction page."""
        self._update_team_lists()
