"""
OracleXI – Prediction History Page
=====================================
Displays prediction history with search, filter, export,
and detail view functionality.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from gui.widgets import (
    COLORS,
    FONTS,
    GradientFrame,
    RoundedButton,
    ScrollableFrame,
    create_card,
    create_section_header,
)
from utils.helper import (
    export_predictions_to_csv,
    format_percentage,
    load_prediction_history,
)

if TYPE_CHECKING:
    from core.prediction_engine import PredictionEngine


class HistoryPage(tk.Frame):
    """
    Prediction history page with searchable, filterable table.

    Features:
        - Full prediction history table
        - Search by team name
        - Filter by sport
        - Export all to CSV
        - Click to expand prediction details
    """

    def __init__(
        self,
        parent: tk.Widget,
        engine: "PredictionEngine",
        navigate_callback: Callable,
        **kwargs,
    ) -> None:
        """
        Initialize the history page.

        Args:
            parent: Parent widget.
            engine: Prediction engine instance.
            navigate_callback: Navigation callback.
        """
        super().__init__(parent, bg=COLORS["bg_primary"], **kwargs)
        self.engine = engine
        self.navigate = navigate_callback
        self.all_history: List[Dict] = []

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the complete history page UI."""
        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header_section = tk.Frame(container, bg=COLORS["bg_primary"])
        header_section.pack(fill="x", padx=30, pady=(25, 5))
        header = create_section_header(
            header_section,
            "Prediction History",
            "Browse, search, and export all past predictions",
        )
        header.pack(fill="x")

        # Controls bar
        self._build_controls(container)

        # Table
        self.table_frame = tk.Frame(container, bg=COLORS["bg_primary"])
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(10, 30))

        self._load_and_display()

    def _build_controls(self, parent: tk.Widget) -> None:
        """Build the search and filter controls."""
        controls = tk.Frame(parent, bg=COLORS["bg_primary"])
        controls.pack(fill="x", padx=30, pady=(10, 5))

        # Search
        tk.Label(
            controls,
            text="🔍",
            font=("Helvetica Neue", 14, "normal"),
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side="left")

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search)

        search_entry = tk.Entry(
            controls,
            textvariable=self.search_var,
            font=FONTS["body"],
            bg=COLORS["bg_input"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["accent_cyan"],
            relief="flat",
            highlightbackground=COLORS["border_default"],
            highlightthickness=1,
            width=25,
        )
        search_entry.pack(side="left", padx=(6, 15), ipady=4)

        # Sport filter
        tk.Label(
            controls,
            text="Sport:",
            font=FONTS["small_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(side="left", padx=(0, 6))

        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(
            controls,
            textvariable=self.filter_var,
            state="readonly",
            values=["All", "Football", "Cricket"],
            style="Dark.TCombobox",
            width=12,
        )
        filter_combo.pack(side="left", padx=(0, 15))
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filters())

        # Export button
        RoundedButton(
            controls,
            text="📥 Export All",
            command=self._export_all,
            bg_color=COLORS["accent_cyan"],
            width=130,
            height=34,
        ).pack(side="left", padx=(0, 8))

        # Refresh button
        RoundedButton(
            controls,
            text="🔄 Refresh",
            command=self.refresh,
            bg_color=COLORS["bg_tertiary"],
            fg_color=COLORS["text_primary"],
            width=100,
            height=34,
        ).pack(side="left")

        # Count label
        self.count_label = tk.Label(
            controls,
            text="",
            font=FONTS["small"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_primary"],
        )
        self.count_label.pack(side="right")

    def _load_and_display(self) -> None:
        """Load history and display in table."""
        self.all_history = load_prediction_history()
        self._apply_filters()

    def _on_search(self, *args) -> None:
        """Handle search input changes."""
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Apply search and sport filters to history."""
        search_text = self.search_var.get().lower().strip()
        sport_filter = self.filter_var.get()

        filtered = self.all_history.copy()

        # Sport filter
        if sport_filter != "All":
            filtered = [
                p for p in filtered
                if p.get("sport", "") == sport_filter
            ]

        # Search filter
        if search_text:
            filtered = [
                p for p in filtered
                if search_text in p.get("team_a", "").lower()
                or search_text in p.get("team_b", "").lower()
            ]

        self._render_table(filtered)
        self.count_label.config(text=f"{len(filtered)} of {len(self.all_history)} predictions")

    def _render_table(self, predictions: List[Dict]) -> None:
        """
        Render the predictions table.

        Args:
            predictions: List of prediction dictionaries to display.
        """
        # Clear existing table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        if not predictions:
            empty_card = create_card(self.table_frame)
            empty_card.pack(fill="x")

            tk.Label(
                empty_card.inner,
                text="📋  No predictions found",
                font=FONTS["heading"],
                fg=COLORS["text_muted"],
                bg=COLORS["bg_secondary"],
            ).pack(pady=20)

            tk.Label(
                empty_card.inner,
                text="Make some predictions to see them here!",
                font=FONTS["small"],
                fg=COLORS["text_muted"],
                bg=COLORS["bg_secondary"],
            ).pack(pady=(0, 20))
            return

        # Table card
        table_card = create_card(self.table_frame, padding=0)
        table_card.pack(fill="x")

        # Header row
        header = tk.Frame(table_card.inner, bg=COLORS["bg_tertiary"])
        header.pack(fill="x")

        columns = [
            ("#", 40), ("Time", 140), ("Sport", 80),
            ("Team A", 150), ("Team B", 150),
            ("Winner", 160), ("Confidence", 90), ("Sims", 70),
        ]

        for col_name, col_width in columns:
            tk.Label(
                header,
                text=col_name,
                font=FONTS["small_bold"],
                fg=COLORS["text_secondary"],
                bg=COLORS["bg_tertiary"],
                width=col_width // 8,
                anchor="w",
            ).pack(side="left", padx=6, pady=8)

        # Data rows (most recent first)
        for i, pred in enumerate(reversed(predictions)):
            row_bg = COLORS["bg_secondary"] if i % 2 == 0 else COLORS["bg_tertiary"]
            row = tk.Frame(table_card.inner, bg=row_bg, cursor="hand2")
            row.pack(fill="x")

            # Parse prediction data
            sport = pred.get("sport", "N/A")
            team_a = pred.get("team_a", "N/A")
            team_b = pred.get("team_b", "N/A")
            timestamp = pred.get("timestamp", "N/A")
            prediction = pred.get("prediction", {})
            confidence = pred.get("confidence", 0)
            sims = pred.get("simulations", 0)
            pred_id = pred.get("id", i + 1)

            # Determine winner
            a_win = prediction.get("team_a_win", 0)
            b_win = prediction.get("team_b_win", 0)
            draw = prediction.get("draw", 0)

            if a_win > b_win and a_win > draw:
                winner = f"{team_a} ({a_win:.0%})"
                winner_color = COLORS["accent_green"]
            elif b_win > a_win and b_win > draw:
                winner = f"{team_b} ({b_win:.0%})"
                winner_color = COLORS["accent_green"]
            else:
                winner = f"Draw ({draw:.0%})"
                winner_color = COLORS["accent_orange"]

            values = [
                (str(pred_id), COLORS["text_muted"], 40),
                (str(timestamp)[:16], COLORS["text_secondary"], 140),
                (sport, COLORS["accent_blue"], 80),
                (team_a, COLORS["text_primary"], 150),
                (team_b, COLORS["text_primary"], 150),
                (winner, winner_color, 160),
                (format_percentage(confidence), COLORS["accent_cyan"], 90),
                (f"{sims:,}" if sims else "N/A", COLORS["text_muted"], 70),
            ]

            for val, color, width in values:
                tk.Label(
                    row,
                    text=val,
                    font=FONTS["small"],
                    fg=color,
                    bg=row_bg,
                    width=width // 8,
                    anchor="w",
                ).pack(side="left", padx=6, pady=5)

            # Click to show details
            row.bind(
                "<Button-1>",
                lambda e, p=pred: self._show_detail(p),
            )
            for child in row.winfo_children():
                child.bind(
                    "<Button-1>",
                    lambda e, p=pred: self._show_detail(p),
                )

    def _show_detail(self, prediction: Dict) -> None:
        """
        Show detailed prediction information in a popup.

        Args:
            prediction: Prediction dictionary to display.
        """
        detail_window = tk.Toplevel(self)
        detail_window.title("Prediction Details")
        detail_window.geometry("500x600")
        detail_window.configure(bg=COLORS["bg_primary"])
        detail_window.transient(self)
        detail_window.grab_set()

        scroll = ScrollableFrame(detail_window)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Title
        sport = prediction.get("sport", "N/A")
        team_a = prediction.get("team_a", "N/A")
        team_b = prediction.get("team_b", "N/A")

        tk.Label(
            container,
            text=f"⚡ {team_a} vs {team_b}",
            font=FONTS["heading"],
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w", padx=20, pady=(20, 5))

        tk.Label(
            container,
            text=f"{sport} • {prediction.get('timestamp', 'N/A')}",
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w", padx=20)

        GradientFrame(container, height=2).pack(fill="x", padx=20, pady=10)

        # Results
        pred_data = prediction.get("prediction", {})
        self._detail_row(container, "Team A Win", format_percentage(pred_data.get("team_a_win", 0)))
        self._detail_row(container, "Team B Win", format_percentage(pred_data.get("team_b_win", 0)))
        if "draw" in pred_data:
            self._detail_row(container, "Draw", format_percentage(pred_data.get("draw", 0)))

        self._detail_row(container, "Confidence", format_percentage(prediction.get("confidence", 0)))
        self._detail_row(container, "Simulations", f"{prediction.get('simulations', 0):,}")

        # Expected scores
        if sport == "Football":
            expected = prediction.get("expected_goals", {})
            self._detail_row(container, f"Expected Goals ({team_a})", f"{expected.get('team_a', 0):.2f}")
            self._detail_row(container, f"Expected Goals ({team_b})", f"{expected.get('team_b', 0):.2f}")
            self._detail_row(container, "Most Likely Score", prediction.get("most_likely_score", "N/A"))
        else:
            expected = prediction.get("expected_runs", {})
            self._detail_row(container, f"Expected Runs ({team_a})", f"{expected.get('team_a', 0):.1f}")
            self._detail_row(container, f"Expected Runs ({team_b})", f"{expected.get('team_b', 0):.1f}")

        # Momentum
        GradientFrame(container, height=2).pack(fill="x", padx=20, pady=10)
        momentum = prediction.get("momentum", {})
        self._detail_row(container, f"Momentum ({team_a})", f"{momentum.get('team_a', 0):.2%}")
        self._detail_row(container, f"Momentum ({team_b})", f"{momentum.get('team_b', 0):.2%}")

        form = prediction.get("recent_form", {})
        self._detail_row(container, f"Form ({team_a})", f"{form.get('team_a', 0):.2%}")
        self._detail_row(container, f"Form ({team_b})", f"{form.get('team_b', 0):.2%}")

        trend = prediction.get("trend", {})
        self._detail_row(container, f"Trend ({team_a})", trend.get("team_a", "stable").title())
        self._detail_row(container, f"Trend ({team_b})", trend.get("team_b", "stable").title())

        # Close button
        tk.Frame(container, bg=COLORS["bg_primary"], height=15).pack()
        RoundedButton(
            container,
            text="Close",
            command=detail_window.destroy,
            bg_color=COLORS["bg_tertiary"],
            fg_color=COLORS["text_primary"],
            width=100,
            height=34,
        ).pack(pady=(5, 20))

    def _detail_row(self, parent: tk.Widget, label: str, value: str) -> None:
        """Create a detail row with label and value."""
        row = tk.Frame(parent, bg=COLORS["bg_primary"])
        row.pack(fill="x", padx=20, pady=2)

        tk.Label(
            row,
            text=label,
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
            width=25,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            row,
            text=value,
            font=FONTS["small_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_primary"],
            anchor="w",
        ).pack(side="left")

    def _export_all(self) -> None:
        """Export all predictions to CSV."""
        if not self.all_history:
            messagebox.showinfo("No Data", "No predictions to export.")
            return

        filepath = export_predictions_to_csv(self.all_history)
        if filepath:
            messagebox.showinfo(
                "Export Complete",
                f"Exported {len(self.all_history)} predictions to:\n{filepath}",
            )

    def refresh(self) -> None:
        """Refresh the history page data."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self._load_and_display()
