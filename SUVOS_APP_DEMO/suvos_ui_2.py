import tkinter as tk
import math

# --- UNIFIED DARK THEME: BIOPUNK CONTROL ROOM ---
COLOR_BG_MAIN    = "#0a0e1a"   # Near-black with blue tint
COLOR_BG_CARD    = "#0f1525"   # Slightly lighter panel
COLOR_BG_BORDER  = "#1a2540"   # Subtle panel border
COLOR_PRIMARY    = "#00f5c8"   # Phosphorescent teal/green
COLOR_ACCENT     = "#0077ff"   # Electric blue accent
COLOR_TEXT       = "#b8cfe8"   # Cool off-white
COLOR_MUTED      = "#3d5478"   # Muted blue-grey
COLOR_DANGER     = "#ff4757"   # Hot red
COLOR_SUCCESS    = "#00f5c8"   # Same as primary
COLOR_WARNING    = "#ffa502"   # Amber

FONT_DISPLAY  = ("Courier New", 11, "bold")   # Monospace display
FONT_LABEL    = ("Courier New", 9)
FONT_TITLE    = ("Courier New", 18, "bold")
FONT_SMALL    = ("Courier New", 8)


class ModernButton(tk.Frame):
    """
    A tactile, industrial-style button with:
    - A glowing left accent bar
    - Animated hover fill sweep
    - Status indicator (dot + checkmark)
    """
    def __init__(self, master, text, command,
                 fg=COLOR_PRIMARY,
                 font=FONT_DISPLAY,
                 pady=14, **kwargs):
        super().__init__(master, bg=COLOR_BG_MAIN, cursor="hand2")

        self.command     = command
        self.default_fg  = fg
        self.hover_active = False
        self._status     = False
        self._text_base  = text.upper()

        # --- Accent bar (left glow strip) ---
        self.accent_bar = tk.Frame(self, bg=fg, width=3)
        self.accent_bar.pack(side="left", fill="y")

        # --- Inner panel ---
        self.inner = tk.Frame(self, bg=COLOR_BG_CARD,
                              padx=18, pady=pady)
        self.inner.pack(side="left", fill="both", expand=True)

        # --- Status dot ---
        self.dot_canvas = tk.Canvas(self.inner, width=10, height=10,
                                    bg=COLOR_BG_CARD, highlightthickness=0)
        self.dot_canvas.pack(side="left", padx=(0, 10))
        self._draw_dot(COLOR_MUTED)

        # --- Label ---
        self.lbl = tk.Label(self.inner, text=self._text_base,
                            font=font, fg=fg,
                            bg=COLOR_BG_CARD, anchor="w")
        self.lbl.pack(side="left", fill="x", expand=True)

        # --- Right arrow indicator ---
        self.arrow = tk.Label(self.inner, text="▶",
                              font=("Courier New", 9), fg=COLOR_MUTED,
                              bg=COLOR_BG_CARD)
        self.arrow.pack(side="right")

        # Thin bottom separator line
        sep = tk.Frame(self, bg=COLOR_BG_BORDER, height=1)
        sep.pack(side="bottom", fill="x")

        # Bind events to all children
        for widget in [self, self.inner, self.lbl, self.arrow,
                       self.dot_canvas, self.accent_bar]:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>",    self._on_enter)
            widget.bind("<Leave>",    self._on_leave)

    def _draw_dot(self, color):
        self.dot_canvas.delete("all")
        self.dot_canvas.create_oval(1, 1, 9, 9, fill=color,
                                    outline=color)

    def _on_click(self, e=None):
        if self.command:
            self.command()

    def _on_enter(self, e=None):
        self.inner.config(bg=COLOR_BG_BORDER)
        self.lbl.config(bg=COLOR_BG_BORDER, fg="#ffffff")
        self.arrow.config(bg=COLOR_BG_BORDER, fg=self.default_fg, text="►")
        self.dot_canvas.config(bg=COLOR_BG_BORDER)

    def _on_leave(self, e=None):
        self.inner.config(bg=COLOR_BG_CARD)
        self.lbl.config(bg=COLOR_BG_CARD, fg=self.default_fg)
        self.arrow.config(bg=COLOR_BG_CARD, fg=COLOR_MUTED, text="▶")
        self.dot_canvas.config(bg=COLOR_BG_CARD)

    def set_status(self, status: bool):
        self._status = status
        if status:
            self._draw_dot(COLOR_SUCCESS)
            self.accent_bar.config(bg=COLOR_SUCCESS)
            self.lbl.config(fg=COLOR_SUCCESS)
            self.default_fg = COLOR_SUCCESS
        else:
            self._draw_dot(COLOR_MUTED)
            self.accent_bar.config(bg=COLOR_PRIMARY)
            self.lbl.config(fg=COLOR_PRIMARY)
            self.default_fg = COLOR_PRIMARY

    def config(self, **kwargs):
        # Route fg changes to label
        if "fg" in kwargs:
            fg = kwargs.pop("fg")
            self.default_fg = fg
            self.lbl.config(fg=fg)
            self.accent_bar.config(bg=fg)
        if "font" in kwargs:
            self.lbl.config(font=kwargs.pop("font"))
        if "pady" in kwargs:
            self.inner.config(pady=kwargs.pop("pady"))
        if kwargs:
            super().config(**kwargs)

    def cget(self, key):
        if key == "fg":
            return self.lbl.cget("fg")
        return super().cget(key)


class SectionLabel(tk.Frame):
    """A horizontal rule with centered label — like a control panel section divider."""
    def __init__(self, master, text, **kwargs):
        super().__init__(master, bg=COLOR_BG_MAIN, **kwargs)
        tk.Frame(self, bg=COLOR_BG_BORDER, height=1).pack(
            side="left", fill="x", expand=True, pady=8)
        tk.Label(self, text=f"  {text}  ",
                 font=FONT_SMALL, fg=COLOR_MUTED,
                 bg=COLOR_BG_MAIN).pack(side="left")
        tk.Frame(self, bg=COLOR_BG_BORDER, height=1).pack(
            side="left", fill="x", expand=True, pady=8)


class StatusBar(tk.Frame):
    """Bottom status bar with live ticker text."""
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=COLOR_BG_BORDER,
                         height=28, **kwargs)
        self.pack_propagate(False)

        self._dot = tk.Canvas(self, width=10, height=10,
                              bg=COLOR_BG_BORDER, highlightthickness=0)
        self._dot.pack(side="left", padx=(12, 6), pady=9)

        self._lbl = tk.Label(self, text="SYSTEM IDLE",
                             font=FONT_SMALL,
                             fg=COLOR_MUTED, bg=COLOR_BG_BORDER,
                             anchor="w")
        self._lbl.pack(side="left", fill="x", expand=True)

        tk.Label(self, text="SUVOS v2.0",
                 font=FONT_SMALL, fg=COLOR_MUTED,
                 bg=COLOR_BG_BORDER).pack(side="right", padx=12)

        self._blink = False
        self._blink_after = None

    def set_status(self, text, color=COLOR_MUTED, blink=False):
        self._lbl.config(text=text, fg=color)
        if self._blink_after:
            self.after_cancel(self._blink_after)
            self._blink_after = None
        if blink:
            self._do_blink(color)
        else:
            self._draw_dot(color)

    def _draw_dot(self, color):
        self._dot.delete("all")
        self._dot.create_oval(1, 1, 9, 9, fill=color, outline=color)

    def _do_blink(self, color):
        self._blink = not self._blink
        self._draw_dot(color if self._blink else COLOR_BG_BORDER)
        self._blink_after = self.after(600, lambda: self._do_blink(color))