"""
OracleXI – AI Sports Forecasting System
==========================================
Main application entry point.

Initializes the Tkinter root window with dark theme,
sets up sidebar navigation, loads all pages, and manages
page switching. Also initializes the core prediction engine
and passes it to GUI controllers.

Usage:
    python main.py

Requirements:
    pip install numpy pandas tensorflow statsmodels matplotlib
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.analytics import AnalyticsPage
from gui.history import HistoryPage
from gui.home import HomePage
from gui.compare import ComparePage
from gui.widgets import (
    COLORS,
    FONTS,
    GradientFrame,
    apply_dark_theme,
)
from utils.constants import (
    APP_NAME,
    APP_SUBTITLE,
    MIN_HEIGHT,
    MIN_WIDTH,
    PAGE_ICONS,
    PAGES,
    SIDEBAR_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from utils.helper import ensure_directories, setup_logging

logger = setup_logging("Main")


class OracleXIApp:
    """
    Main application class for OracleXI.

    Manages the root window, navigation sidebar, page switching,
    and initialization of the prediction engine.
    """

    def __init__(self) -> None:
        """Initialize the OracleXI application."""
        logger.info("=" * 50)
        logger.info(f"  Starting {APP_NAME} – {APP_SUBTITLE}")
        logger.info("=" * 50)

        # Ensure required directories exist
        ensure_directories()

        # Initialize Tkinter root
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} – {APP_SUBTITLE}")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(MIN_WIDTH, MIN_HEIGHT)

        # Apply dark theme
        apply_dark_theme(self.root)

        # Center window on screen
        self._center_window()

        # Initialize prediction engine (lazy – loads data on first use)
        from core.prediction_engine import PredictionEngine
        self.engine = PredictionEngine()

        # Build UI
        self._build_layout()

        # Start with Home page
        self.current_page: str = "Home"
        self._show_page("Home")

        logger.info("Application ready")

    def _center_window(self) -> None:
        """Center the application window on screen."""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - WINDOW_WIDTH) // 2
        y = (screen_height - WINDOW_HEIGHT) // 2
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

    def _build_layout(self) -> None:
        """Build the main application layout with sidebar and content area."""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        self.main_frame.pack(fill="both", expand=True)

        # ── Sidebar ──
        self._build_sidebar()

        # ── Content Area ──
        self.content_frame = tk.Frame(
            self.main_frame,
            bg=COLORS["bg_primary"],
        )
        self.content_frame.pack(side="left", fill="both", expand=True)

        # ── Initialize Pages ──
        self.pages: Dict[str, tk.Frame] = {}

        self.pages["Home"] = HomePage(
            self.content_frame,
            engine=self.engine,
            navigate_callback=self._navigate,
        )

        self.pages["Compare"] = ComparePage(
            self.content_frame,
            engine=self.engine,
        )

        self.pages["Analytics"] = AnalyticsPage(
            self.content_frame,
            engine=self.engine,
            navigate_callback=self._navigate,
        )

        self.pages["History"] = HistoryPage(
            self.content_frame,
            engine=self.engine,
            navigate_callback=self._navigate,
        )

    def _build_sidebar(self) -> None:
        """Build the navigation sidebar."""
        sidebar = tk.Frame(
            self.main_frame,
            bg=COLORS["bg_secondary"],
            width=SIDEBAR_WIDTH,
        )
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # ── Logo Area ──
        logo_frame = tk.Frame(sidebar, bg=COLORS["bg_secondary"])
        logo_frame.pack(fill="x", padx=15, pady=(20, 5))

        tk.Label(
            logo_frame,
            text="⚡",
            font=("Helvetica Neue", 28, "normal"),
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left")

        title_frame = tk.Frame(logo_frame, bg=COLORS["bg_secondary"])
        title_frame.pack(side="left", padx=(8, 0))

        tk.Label(
            title_frame,
            text=APP_NAME,
            font=("Helvetica Neue", 18, "bold"),
            fg=COLORS["accent_cyan"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text="v1.0",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack(anchor="w")

        # Gradient separator
        GradientFrame(
            sidebar, height=2,
            color_start=COLORS["accent_cyan"],
            color_end=COLORS["accent_purple"],
        ).pack(fill="x", padx=15, pady=(15, 20))

        # ── Navigation Items ──
        self.nav_labels: Dict[str, tk.Frame] = {}

        for page_name in PAGES:
            icon = PAGE_ICONS.get(page_name, "•")
            nav_item = self._create_nav_item(sidebar, page_name, icon)
            self.nav_labels[page_name] = nav_item

        # ── Bottom Section ──
        bottom = tk.Frame(sidebar, bg=COLORS["bg_secondary"])
        bottom.pack(side="bottom", fill="x", padx=15, pady=(0, 20))

        # Train models button
        train_frame = tk.Frame(bottom, bg=COLORS["bg_secondary"])
        train_frame.pack(fill="x", pady=(0, 10))

        train_btn = tk.Label(
            train_frame,
            text="🧠 Train Models",
            font=FONTS["small_bold"],
            fg=COLORS["accent_orange"],
            bg=COLORS["bg_tertiary"],
            cursor="hand2",
            padx=15,
            pady=8,
        )
        train_btn.pack(fill="x")
        train_btn.bind("<Button-1>", lambda e: self._train_models())

        # Version info
        tk.Label(
            bottom,
            text="Powered by TensorFlow",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack()

        tk.Label(
            bottom,
            text="NumPy • Pandas • Statsmodels",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["bg_secondary"],
        ).pack()

    def _create_nav_item(
        self, parent: tk.Widget, page_name: str, icon: str
    ) -> tk.Frame:
        """
        Create a sidebar navigation item.

        Args:
            parent: Parent widget (sidebar).
            page_name: Page name for navigation.
            icon: Unicode icon character.

        Returns:
            Navigation item frame.
        """
        nav_frame = tk.Frame(parent, bg=COLORS["bg_secondary"], cursor="hand2")
        nav_frame.pack(fill="x", padx=10, pady=2)

        # Active indicator
        indicator = tk.Frame(nav_frame, bg=COLORS["bg_secondary"], width=3)
        indicator.pack(side="left", fill="y")

        # Content
        content = tk.Label(
            nav_frame,
            text=f"  {icon}  {page_name}",
            font=FONTS["nav"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
            anchor="w",
            padx=8,
            pady=10,
        )
        content.pack(fill="x")

        # Hover effects
        def on_enter(e):
            if self.current_page != page_name:
                nav_frame.configure(bg=COLORS["bg_hover"])
                content.configure(bg=COLORS["bg_hover"])
                indicator.configure(bg=COLORS["bg_hover"])

        def on_leave(e):
            if self.current_page != page_name:
                nav_frame.configure(bg=COLORS["bg_secondary"])
                content.configure(bg=COLORS["bg_secondary"])
                indicator.configure(bg=COLORS["bg_secondary"])

        nav_frame.bind("<Enter>", on_enter)
        nav_frame.bind("<Leave>", on_leave)
        content.bind("<Enter>", on_enter)
        content.bind("<Leave>", on_leave)

        # Click handler
        nav_frame.bind(
            "<Button-1>", lambda e: self._show_page(page_name)
        )
        content.bind(
            "<Button-1>", lambda e: self._show_page(page_name)
        )

        # Store references for styling
        nav_frame._indicator = indicator
        nav_frame._content = content

        return nav_frame

    def _update_nav_styles(self) -> None:
        """Update navigation item styles based on current page."""
        for page_name, nav_frame in self.nav_labels.items():
            indicator = nav_frame._indicator
            content = nav_frame._content

            if page_name == self.current_page:
                nav_frame.configure(bg=COLORS["bg_tertiary"])
                indicator.configure(bg=COLORS["accent_cyan"])
                content.configure(
                    bg=COLORS["bg_tertiary"],
                    fg=COLORS["accent_cyan"],
                )
            else:
                nav_frame.configure(bg=COLORS["bg_secondary"])
                indicator.configure(bg=COLORS["bg_secondary"])
                content.configure(
                    bg=COLORS["bg_secondary"],
                    fg=COLORS["text_secondary"],
                )

    def _show_page(self, page_name: str) -> None:
        """
        Show the specified page and hide all others.

        Args:
            page_name: Name of the page to show.
        """
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()

        # Show requested page
        if page_name in self.pages:
            self.pages[page_name].pack(fill="both", expand=True)
            self.current_page = page_name
            self._update_nav_styles()

            # Refresh page data
            if hasattr(self.pages[page_name], "refresh"):
                self.pages[page_name].refresh()

            logger.info(f"Navigated to: {page_name}")

    def _navigate(self, page_name: str, **kwargs) -> None:
        """
        Navigation callback for child pages.

        Args:
            page_name: Target page name.
            **kwargs: Additional parameters (e.g., sport selection).
        """
        # Handle sport parameter for compare page
        if page_name == "Compare" and "sport" in kwargs:
            compare_page = self.pages.get("Compare")
            if compare_page:
                compare_page.current_sport.set(kwargs["sport"])
                compare_page._on_sport_change()

        self._show_page(page_name)

    def _train_models(self) -> None:
        """Train both football and cricket models."""
        import threading

        logger.info("Training models...")

        def train():
            try:
                results = self.engine.train_models()
                logger.info(f"Training results: {results}")

                fb_str = "Success" if results.get("football") else "Failed/Skipped"
                cr_str = "Success" if results.get("cricket") else "Failed/Skipped"

                self.root.after(0, lambda: tk.messagebox.showinfo(
                    "Training Complete",
                    f"Football: {fb_str}\n"
                    f"Cricket: {cr_str}",
                ))
            except Exception as e:
                logger.error(f"Training failed: {e}")
                self.root.after(0, lambda: tk.messagebox.showerror(
                    "Training Error", str(e)
                ))

        thread = threading.Thread(target=train, daemon=True)
        thread.start()

    def run(self) -> None:
        """Start the application main loop."""
        logger.info("Starting main loop")
        self.root.mainloop()


def main() -> None:
    """Application entry point."""
    app = OracleXIApp()
    app.run()


if __name__ == "__main__":
    main()
