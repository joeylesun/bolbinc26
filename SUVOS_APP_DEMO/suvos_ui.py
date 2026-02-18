import tkinter as tk

# --- UNIFIED DARK THEME ---
COLOR_BG_MAIN = "#1e1e2e"  # The main background color
COLOR_BG_CARD = "#1e1e2e"  # MATCHES MAIN (Invisible container)
COLOR_PRIMARY = "#00d2d3"  # Neon Cyan
COLOR_TEXT = "#c8d6e5"  # Off-White
COLOR_DANGER = "#ff6b6b"  # Neon Red
COLOR_SUCCESS = "#1dd1a1"  # Neon Green


# --- MODERN FLAT BUTTON ---
class ModernButton(tk.Label):
    def __init__(self, master, text, command, bg=COLOR_BG_MAIN, fg=COLOR_PRIMARY, font=("Segoe UI", 11, "bold"),
                 pady=12):
        # Note: Default bg is now COLOR_BG_MAIN to blend in
        super().__init__(master, text=text.upper(), bg=bg, fg=fg)

        self.command = command
        self.default_bg = bg
        self.default_fg = fg
        self.hover_bg = COLOR_PRIMARY
        self.hover_fg = "#ffffff"

        # Typography & Padding
        self.config(font=font, pady=pady, padx=20)

        # Events
        self.bind("<Button-1>", lambda e: self.on_click())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_click(self):
        if self.command: self.command()

    def on_enter(self, e):
        self.config(bg=self.hover_bg, fg=self.hover_fg)

    def on_leave(self, e):
        self.config(bg=self.default_bg, fg=self.default_fg)

    def set_status(self, status):
        """ Updates button look based on status """
        if status:
            self.config(text=self.cget("text") + "  ✔", fg=COLOR_SUCCESS)
            self.default_fg = COLOR_SUCCESS
        else:
            clean_text = self.cget("text").replace("  ✔", "")
            self.config(text=clean_text, fg=COLOR_PRIMARY)
            self.default_fg = COLOR_PRIMARY