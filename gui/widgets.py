"""
OracleXI – Custom Tkinter Widgets
====================================
Premium custom widgets with dark theme, gradient effects,
rounded cards, progress bars, and chart embedding.

Provides the visual design system for the entire application.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

from utils.constants import COLORS, FONTS

# ─────────────────────────────────────────────
# Theme Setup
# ─────────────────────────────────────────────


def apply_dark_theme(root: tk.Tk) -> None:
    """
    Apply the dark premium theme to the root window.

    Args:
        root: The Tkinter root window.
    """
    root.configure(bg=COLORS["bg_primary"])

    # Configure ttk style
    style = ttk.Style()
    style.theme_use("clam")

    # General widget styles
    style.configure(
        "Dark.TFrame",
        background=COLORS["bg_primary"],
    )
    style.configure(
        "Card.TFrame",
        background=COLORS["bg_secondary"],
    )
    style.configure(
        "Dark.TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_primary"],
        font=FONTS["body"],
    )
    style.configure(
        "Title.TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_primary"],
        font=FONTS["title"],
    )
    style.configure(
        "Heading.TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_primary"],
        font=FONTS["heading"],
    )
    style.configure(
        "Subheading.TLabel",
        background=COLORS["bg_secondary"],
        foreground=COLORS["text_primary"],
        font=FONTS["subheading"],
    )
    style.configure(
        "Muted.TLabel",
        background=COLORS["bg_primary"],
        foreground=COLORS["text_secondary"],
        font=FONTS["small"],
    )

    # Combobox style
    style.configure(
        "Dark.TCombobox",
        fieldbackground=COLORS["bg_input"],
        background=COLORS["bg_tertiary"],
        foreground=COLORS["text_primary"],
        arrowcolor=COLORS["accent_cyan"],
        bordercolor=COLORS["border_default"],
        lightcolor=COLORS["bg_tertiary"],
        darkcolor=COLORS["bg_tertiary"],
    )
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", COLORS["bg_input"])],
        foreground=[("readonly", COLORS["text_primary"])],
        bordercolor=[("focus", COLORS["accent_cyan"])],
    )

    # Scrollbar style
    style.configure(
        "Dark.Vertical.TScrollbar",
        background=COLORS["bg_tertiary"],
        troughcolor=COLORS["bg_primary"],
        arrowcolor=COLORS["text_secondary"],
    )


# ─────────────────────────────────────────────
# Gradient Frame
# ─────────────────────────────────────────────


class GradientFrame(tk.Canvas):
    """
    A frame with a horizontal gradient background.

    Creates a smooth color transition between two colors
    using individual line drawing on a Canvas.
    """

    def __init__(
        self,
        parent: tk.Widget,
        color_start: str = COLORS["accent_cyan"],
        color_end: str = COLORS["accent_purple"],
        height: int = 4,
        **kwargs,
    ) -> None:
        """
        Initialize the gradient frame.

        Args:
            parent: Parent widget.
            color_start: Starting gradient color (hex).
            color_end: Ending gradient color (hex).
            height: Height of the gradient bar.
        """
        super().__init__(
            parent,
            height=height,
            highlightthickness=0,
            **kwargs,
        )
        self.color_start = color_start
        self.color_end = color_end
        self.bind("<Configure>", self._draw_gradient)

    def _draw_gradient(self, event: tk.Event = None) -> None:
        """Draw the gradient by rendering colored lines."""
        self.delete("gradient")
        width = self.winfo_width()
        height = self.winfo_height()

        if width <= 0:
            return

        r1, g1, b1 = self._hex_to_rgb(self.color_start)
        r2, g2, b2 = self._hex_to_rgb(self.color_end)

        for i in range(width):
            ratio = i / max(width - 1, 1)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.create_line(
                i, 0, i, height, fill=color, tags="gradient"
            )

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# ─────────────────────────────────────────────
# Rounded Button
# ─────────────────────────────────────────────


class RoundedButton(tk.Canvas):
    """
    A modern rounded button with hover effects.

    Uses Canvas for custom shape rendering with
    smooth hover transitions.
    """

    def __init__(
        self,
        parent: tk.Widget,
        text: str = "Button",
        command: Optional[Callable] = None,
        bg_color: str = COLORS["accent_cyan"],
        fg_color: str = COLORS["bg_primary"],
        hover_color: Optional[str] = None,
        width: int = 140,
        height: int = 40,
        radius: int = 8,
        font: Optional[Tuple] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a rounded button.

        Args:
            parent: Parent widget.
            text: Button label text.
            command: Callback function on click.
            bg_color: Background color.
            fg_color: Text color.
            hover_color: Color on hover.
            width: Button width.
            height: Button height.
            radius: Corner radius.
            font: Custom font tuple.
        """
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bg=parent.cget("bg") if hasattr(parent, "cget") else COLORS["bg_primary"],
            **kwargs,
        )

        self.text = text
        self.command = command
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.hover_color = hover_color or self._lighten_color(bg_color, 30)
        self.btn_width = width
        self.btn_height = height
        self.radius = radius
        self.font = font or FONTS["button"]
        self._is_hovered = False

        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw(self) -> None:
        """Draw the rounded rectangle button."""
        self.delete("all")
        color = self.hover_color if self._is_hovered else self.bg_color

        # Draw rounded rectangle
        r = self.radius
        w = self.btn_width
        h = self.btn_height

        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w - r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h - r, fill=color, outline=color)

        # Draw text
        self.create_text(
            w / 2,
            h / 2,
            text=self.text,
            fill=self.fg_color,
            font=self.font,
        )

    def _on_enter(self, event: tk.Event) -> None:
        """Handle mouse enter (hover effect)."""
        self._is_hovered = True
        self._draw()
        self.configure(cursor="hand2")

    def _on_leave(self, event: tk.Event) -> None:
        """Handle mouse leave."""
        self._is_hovered = False
        self._draw()

    def _on_click(self, event: tk.Event) -> None:
        """Handle button click."""
        if self.command:
            self.command()

    def set_text(self, text: str) -> None:
        """Update button text."""
        self.text = text
        self._draw()

    @staticmethod
    def _lighten_color(hex_color: str, amount: int = 30) -> str:
        """Lighten a hex color by a given amount."""
        hex_color = hex_color.lstrip("#")
        r = min(255, int(hex_color[0:2], 16) + amount)
        g = min(255, int(hex_color[2:4], 16) + amount)
        b = min(255, int(hex_color[4:6], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────
# Progress Bar
# ─────────────────────────────────────────────


class ModernProgressBar(tk.Canvas):
    """
    A modern styled progress bar with gradient fill.

    Displays a percentage value with smooth color transitions.
    """

    def __init__(
        self,
        parent: tk.Widget,
        value: float = 0.0,
        max_value: float = 1.0,
        width: int = 300,
        height: int = 24,
        bar_color: str = COLORS["accent_cyan"],
        bg_color: str = COLORS["bg_tertiary"],
        show_label: bool = True,
        label_text: str = "",
        **kwargs,
    ) -> None:
        """
        Initialize the progress bar.

        Args:
            parent: Parent widget.
            value: Current value.
            max_value: Maximum value.
            width: Bar width.
            height: Bar height.
            bar_color: Fill color.
            bg_color: Background color.
            show_label: Whether to show percentage label.
            label_text: Optional custom label.
        """
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bg=parent.cget("bg") if hasattr(parent, "cget") else COLORS["bg_secondary"],
            **kwargs,
        )
        self.value = value
        self.max_value = max_value
        self.bar_width = width
        self.bar_height = height
        self.bar_color = bar_color
        self.bar_bg_color = bg_color
        self.show_label = show_label
        self.label_text = label_text

        self._draw()

    def _draw(self) -> None:
        """Draw the progress bar."""
        self.delete("all")
        w = self.bar_width
        h = self.bar_height
        r = h // 2

        # Background track
        self.create_rounded_rect(0, 0, w, h, r, self.bar_bg_color)

        # Fill bar
        ratio = min(self.value / self.max_value, 1.0) if self.max_value > 0 else 0
        fill_width = max(r * 2, int(w * ratio))

        if ratio > 0.02:
            self.create_rounded_rect(0, 0, fill_width, h, r, self.bar_color)

        # Label
        if self.show_label:
            pct_text = f"{ratio * 100:.1f}%"
            if self.label_text:
                pct_text = f"{self.label_text}: {pct_text}"
            text_x = w / 2
            self.create_text(
                text_x, h / 2,
                text=pct_text,
                fill=COLORS["text_primary"],
                font=FONTS["small_bold"],
            )

    def create_rounded_rect(
        self, x1: int, y1: int, x2: int, y2: int, r: int, color: str
    ) -> None:
        """Draw a rounded rectangle on the canvas."""
        self.create_arc(x1, y1, x1 + r * 2, y2, start=90, extent=180, fill=color, outline=color)
        self.create_arc(x2 - r * 2, y1, x2, y2, start=270, extent=180, fill=color, outline=color)
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline=color)

    def set_value(self, value: float, color: Optional[str] = None) -> None:
        """Update the progress bar value."""
        self.value = value
        if color:
            self.bar_color = color
        self._draw()


# ─────────────────────────────────────────────
# Stats Card
# ─────────────────────────────────────────────


class StatsCard(tk.Frame):
    """
    A styled statistics card showing a label and value.

    Used for dashboard overview numbers.
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str = "Stat",
        value: str = "0",
        icon: str = "",
        accent_color: str = COLORS["accent_cyan"],
        width: int = 180,
        **kwargs,
    ) -> None:
        """
        Initialize a stats card.

        Args:
            parent: Parent widget.
            label: Stat description.
            value: Stat value.
            icon: Unicode icon character.
            accent_color: Card accent color.
            width: Card width.
        """
        super().__init__(
            parent,
            bg=COLORS["bg_secondary"],
            highlightbackground=COLORS["border_default"],
            highlightthickness=1,
            **kwargs,
        )

        self.accent_color = accent_color

        # Inner padding
        inner = tk.Frame(self, bg=COLORS["bg_secondary"])
        inner.pack(padx=16, pady=14, fill="both", expand=True)

        # Icon + Label row
        header = tk.Frame(inner, bg=COLORS["bg_secondary"])
        header.pack(fill="x")

        if icon:
            tk.Label(
                header,
                text=icon,
                font=("Helvetica Neue", 16, "normal"),
                fg=accent_color,
                bg=COLORS["bg_secondary"],
            ).pack(side="left")

        tk.Label(
            header,
            text=label,
            font=FONTS["stat_label"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_secondary"],
        ).pack(side="left", padx=(6, 0))

        # Value
        self.value_label = tk.Label(
            inner,
            text=value,
            font=FONTS["stat_value"],
            fg=accent_color,
            bg=COLORS["bg_secondary"],
        )
        self.value_label.pack(anchor="w", pady=(4, 0))

    def update_value(self, value: str) -> None:
        """Update the displayed value."""
        self.value_label.config(text=value)


# ─────────────────────────────────────────────
# Loading Spinner
# ─────────────────────────────────────────────


class LoadingSpinner(tk.Canvas):
    """
    Animated loading spinner using rotating arcs.

    Displays during prediction computation.
    """

    def __init__(
        self,
        parent: tk.Widget,
        size: int = 40,
        color: str = COLORS["accent_cyan"],
        **kwargs,
    ) -> None:
        """
        Initialize the loading spinner.

        Args:
            parent: Parent widget.
            size: Spinner diameter.
            color: Arc color.
        """
        bg = parent.cget("bg") if hasattr(parent, "cget") else COLORS["bg_primary"]
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bg=bg,
            **kwargs,
        )
        self.size = size
        self.color = color
        self.angle = 0
        self._running = False
        self._job_id = None

    def start(self) -> None:
        """Start the spinning animation."""
        self._running = True
        self._animate()

    def stop(self) -> None:
        """Stop the spinning animation."""
        self._running = False
        if self._job_id:
            self.after_cancel(self._job_id)
            self._job_id = None
            self.delete("all")

    def _animate(self) -> None:
        """Draw the next frame of the animation."""
        if not self._running:
            return

        self.delete("all")
        pad = 4
        self.create_arc(
            pad, pad,
            self.size - pad, self.size - pad,
            start=self.angle,
            extent=270,
            style="arc",
            outline=self.color,
            width=3,
        )
        self.angle = (self.angle + 15) % 360
        self._job_id = self.after(50, self._animate)


# ─────────────────────────────────────────────
# Scrollable Frame
# ─────────────────────────────────────────────


class ScrollableFrame(tk.Frame):
    """
    A scrollable frame container using Canvas + Scrollbar.

    Allows any content to be scrolled vertically.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        """Initialize the scrollable frame."""
        super().__init__(parent, bg=COLORS["bg_primary"], **kwargs)

        self.canvas = tk.Canvas(
            self,
            bg=COLORS["bg_primary"],
            highlightthickness=0,
        )

        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
            style="Dark.Vertical.TScrollbar",
        )

        self.scrollable_frame = tk.Frame(
            self.canvas,
            bg=COLORS["bg_primary"],
        )

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind canvas resize to update inner frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Update inner frame width to match canvas."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ─────────────────────────────────────────────
# Section Header
# ─────────────────────────────────────────────


def create_section_header(
    parent: tk.Widget,
    title: str,
    subtitle: str = "",
) -> tk.Frame:
    """
    Create a styled section header with title and optional subtitle.

    Args:
        parent: Parent widget.
        title: Section title text.
        subtitle: Optional subtitle text.

    Returns:
        Frame containing the header.
    """
    frame = tk.Frame(parent, bg=COLORS["bg_primary"])

    tk.Label(
        frame,
        text=title,
        font=FONTS["heading"],
        fg=COLORS["text_primary"],
        bg=COLORS["bg_primary"],
    ).pack(anchor="w")

    if subtitle:
        tk.Label(
            frame,
            text=subtitle,
            font=FONTS["small"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_primary"],
        ).pack(anchor="w", pady=(2, 0))

    # Gradient underline
    GradientFrame(frame, height=3).pack(fill="x", pady=(8, 0))

    return frame


# ─────────────────────────────────────────────
# Card Container
# ─────────────────────────────────────────────


def create_card(
    parent: tk.Widget,
    padding: int = 20,
    bg: str = COLORS["bg_secondary"],
) -> tk.Frame:
    """
    Create a styled card container with border.

    Args:
        parent: Parent widget.
        padding: Inner padding.
        bg: Background color.

    Returns:
        Frame representing the card.
    """
    card = tk.Frame(
        parent,
        bg=bg,
        highlightbackground=COLORS["border_default"],
        highlightthickness=1,
    )

    inner = tk.Frame(card, bg=bg)
    inner.pack(padx=padding, pady=padding, fill="both", expand=True)

    # Store inner frame reference
    card.inner = inner
    return card


# ─────────────────────────────────────────────
# NEW: Comparison Bar (Team 1 vs Team 2)
# ─────────────────────────────────────────────

class ComparisonBar(tk.Canvas):
    """
    A horizontal bar that fills outward from the center, showing 
    the relative strength/probability of Team 1 vs Team 2 vs Draw.
    """

    def __init__(
        self,
        parent: tk.Widget,
        val1: float = 0.0,
        val2: float = 0.0,
        val_draw: float = 0.0,
        width: int = 400,
        height: int = 24,
        color1: str = COLORS["accent_cyan"],
        color2: str = COLORS["accent_purple"],
        color_draw: str = COLORS["accent_orange"],
        bg_color: str = COLORS["bg_tertiary"],
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bg=parent.cget("bg") if hasattr(parent, "cget") else COLORS["bg_secondary"],
            **kwargs,
        )
        self.val1 = val1
        self.val2 = val2
        self.val_draw = val_draw
        self.bar_width = width
        self.bar_height = height
        self.color1 = color1
        self.color2 = color2
        self.color_draw = color_draw
        self.bg_color = bg_color

        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        w = self.bar_width
        h = self.bar_height
        r = h // 2

        # Background track
        self._create_rounded_rect(0, 0, w, h, r, self.bg_color)

        total = self.val1 + self.val2 + self.val_draw
        if total <= 0:
            return

        w1 = int((self.val1 / total) * w)
        wd = int((self.val_draw / total) * w)
        w2 = w - w1 - wd

        # Team 1 (Left)
        if w1 > 0:
            self.create_arc(0, 0, r * 2, h, start=90, extent=180, fill=self.color1, outline=self.color1)
            self.create_rectangle(r, 0, w1, h, fill=self.color1, outline=self.color1)
            self.create_text(w1 // 2, h // 2, text=f"{self.val1*100:.1f}%", fill="#FFFFFF", font=FONTS["small_bold"])

        # Draw (Middle)
        if wd > 0:
            self.create_rectangle(w1, 0, w1 + wd, h, fill=self.color_draw, outline=self.color_draw)
            if wd > 30:
                self.create_text(w1 + wd // 2, h // 2, text=f"{self.val_draw*100:.1f}%", fill="#FFFFFF", font=FONTS["small_bold"])

        # Team 2 (Right)
        if w2 > 0:
            self.create_rectangle(w1 + wd, 0, w - r, h, fill=self.color2, outline=self.color2)
            self.create_arc(w - r * 2, 0, w, h, start=270, extent=180, fill=self.color2, outline=self.color2)
            self.create_text(w - w2 // 2, h // 2, text=f"{self.val2*100:.1f}%", fill="#FFFFFF", font=FONTS["small_bold"])

    def _create_rounded_rect(self, x1: int, y1: int, x2: int, y2: int, r: int, color: str) -> None:
        self.create_arc(x1, y1, x1 + r * 2, y2, start=90, extent=180, fill=color, outline=color)
        self.create_arc(x2 - r * 2, y1, x2, y2, start=270, extent=180, fill=color, outline=color)
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=color, outline=color)

    def set_values(self, val1: float, val2: float, val_draw: float = 0.0) -> None:
        self.val1 = val1
        self.val2 = val2
        self.val_draw = val_draw
        self._draw()


# ─────────────────────────────────────────────
# NEW: Form Badges (W D L circles)
# ─────────────────────────────────────────────

class FormBadges(tk.Frame):
    """
    Displays a series of colored circles representing recent match outcomes (W, D, L).
    """

    def __init__(
        self,
        parent: tk.Widget,
        form_string: str = "",
        size: int = 24,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg") if hasattr(parent, "cget") else COLORS["bg_primary"], **kwargs)
        self.size = size
        self.set_form(form_string)

    def set_form(self, form_string: str) -> None:
        for widget in self.winfo_children():
            widget.destroy()

        if not form_string or form_string == "-----":
            tk.Label(self, text="No data", fg=COLORS["text_muted"], bg=self.cget("bg"), font=FONTS["small"]).pack()
            return

        for result in form_string:
            color = COLORS["bg_tertiary"]
            if result == "W":
                color = COLORS["badge_win"]
            elif result == "D":
                color = COLORS["badge_draw"]
            elif result == "L":
                color = COLORS["badge_loss"]

            c = tk.Canvas(self, width=self.size, height=self.size, bg=self.cget("bg"), highlightthickness=0)
            c.pack(side="left", padx=2)
            c.create_oval(2, 2, self.size - 2, self.size - 2, fill=color, outline=color)
            c.create_text(self.size // 2, self.size // 2, text=result, fill="#FFFFFF", font=FONTS["form_badge"])


# ─────────────────────────────────────────────
# NEW: VS Badge
# ─────────────────────────────────────────────

class VSBadge(tk.Canvas):
    """
    A stylized 'VS' badge for the comparison header.
    """

    def __init__(
        self,
        parent: tk.Widget,
        size: int = 48,
        bg_color: str = COLORS["bg_primary"],
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            width=size,
            height=size,
            highlightthickness=0,
            bg=bg_color,
            **kwargs,
        )
        # Hexagon / Circle background
        self.create_oval(2, 2, size - 2, size - 2, fill=COLORS["bg_tertiary"], outline=COLORS["accent_purple"], width=2)
        self.create_text(size // 2, size // 2, text="VS", fill=COLORS["accent_cyan"], font=FONTS["vs_big"])
