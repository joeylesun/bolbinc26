import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import sys
import os
import math

# --- IMPORTS ---
from suvos_common import (
    resource_path, get_config_path, ZONES_FILE,
    get_calib_file, CAMERA_INDICES, FSM_TIMES, save_fsm_config, load_fsm_config
)
import suvos_calibration
import suvos_setup
import suvos_working
from suvos_ui import (
    ModernButton, SectionLabel, StatusBar,
    COLOR_BG_MAIN, COLOR_BG_CARD, COLOR_BG_BORDER,
    COLOR_PRIMARY, COLOR_ACCENT, COLOR_TEXT, COLOR_MUTED,
    COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    FONT_DISPLAY, FONT_LABEL, FONT_TITLE, FONT_SMALL
)

# --- GLOBALS ---
SYSTEM = suvos_working.SUVOS_System()
CAM_WINDOW = None


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def check_status():
    calib_path = get_config_path(get_calib_file(CAMERA_INDICES[0]))
    zones_path = get_config_path(ZONES_FILE)

    calib_ok = os.path.exists(calib_path)
    zones_ok  = os.path.exists(zones_path)

    btn1.set_status(calib_ok)
    btn2.set_status(zones_ok)

    if calib_ok and zones_ok:
        btn3.config(fg=COLOR_PRIMARY)
        status_bar.set_status("READY — All prerequisites met. System can be started.", COLOR_SUCCESS)
    elif calib_ok:
        btn3.config(fg=COLOR_MUTED)
        status_bar.set_status("WAITING — Zone setup required before starting.", COLOR_WARNING)
    else:
        btn3.config(fg=COLOR_MUTED)
        status_bar.set_status("WAITING — Calibration required.", COLOR_MUTED)


def step1():
    pts = filedialog.askopenfilename(filetypes=[("PTS", "*.pts")])
    if pts:
        status_bar.set_status("Running calibration…", COLOR_WARNING, blink=True)
        try:
            suvos_calibration.run_calibration(pts)
            check_status()
        except Exception as e:
            messagebox.showerror("Calibration Error", str(e))
            status_bar.set_status(f"Calibration failed: {e}", COLOR_DANGER)


def step2():
    if not os.path.exists(get_config_path(get_calib_file(CAMERA_INDICES[0]))):
        messagebox.showwarning("Prerequisite", "Complete Step 1 (Calibration) first.")
        return
    status_bar.set_status("Running zone setup…", COLOR_WARNING, blink=True)
    try:
        suvos_setup.run_setup()
        check_status()
    except Exception as e:
        messagebox.showerror("Setup Error", str(e))
        status_bar.set_status(f"Zone setup failed: {e}", COLOR_DANGER)


# ─────────────────────────────────────────────
#  CAMERA WINDOW
# ─────────────────────────────────────────────
def open_camera_window():
    global CAM_WINDOW
    CAM_WINDOW = tk.Toplevel(root)
    CAM_WINDOW.title("SUVOS — Live Monitoring")
    CAM_WINDOW.geometry("1280x720")
    CAM_WINDOW.configure(bg="black")

    # Header bar
    header = tk.Frame(CAM_WINDOW, bg=COLOR_BG_CARD, height=36)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(header, text="◉  LIVE FEED — UVC MONITORING ACTIVE",
             font=FONT_DISPLAY, fg=COLOR_PRIMARY, bg=COLOR_BG_CARD).pack(
             side="left", padx=16, pady=6)

    lbl_cam = tk.Label(CAM_WINDOW, bg="black")
    lbl_cam.pack(fill="both", expand=True)

    def update_cam():
        if SYSTEM.running and CAM_WINDOW:
            frame = SYSTEM.get_processed_frame()
            if frame is not None:
                import cv2
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_cam.imgtk = imgtk
                lbl_cam.configure(image=imgtk)
            CAM_WINDOW.after(30, update_cam)

    update_cam()
    CAM_WINDOW.protocol("WM_DELETE_WINDOW", stop_system)


# ─────────────────────────────────────────────
#  SYSTEM LIFECYCLE
# ─────────────────────────────────────────────
def step3_run():
    if str(btn3.cget("fg")) in (COLOR_MUTED, "#576574"):
        return
    main_frame.place_forget()
    running_frame.place(relx=0.5, rely=0.5, anchor="center")
    status_bar.set_status("SYSTEM ACTIVE — UVC disinfection monitoring running.", COLOR_SUCCESS, blink=True)
    SYSTEM.start()
    open_camera_window()


def stop_system():
    SYSTEM.stop()
    if CAM_WINDOW:
        CAM_WINDOW.destroy()
    running_frame.place_forget()
    main_frame.place(relx=0.5, rely=0.5, anchor="center")
    status_bar.set_status("System stopped.", COLOR_MUTED)
    check_status()


def shutdown():
    SYSTEM.stop()
    root.destroy()
    sys.exit(0)


# ─────────────────────────────────────────────
#  CLOCK / TIMER WIDGETS
# ─────────────────────────────────────────────
def change_fsm_time(fsm_key, canvas, time_label):
    current = FSM_TIMES.get(fsm_key, 0.0)
    new_val = simpledialog.askfloat(
        "Adjust Timer",
        f"Duration (seconds):",
        initialvalue=current,
        minvalue=0.0
    )
    if new_val is not None:
        save_fsm_config({fsm_key: new_val})
        draw_clock(canvas, new_val, time_label)


def draw_clock(canvas, time_val, time_label=None):
    """Draws a sleek arc-based timer face."""
    canvas.delete("all")
    W, H  = 100, 100
    cx, cy = W / 2, H / 2
    R_outer = 44
    R_inner = 34

    # Track ring
    canvas.create_oval(
        cx - R_outer, cy - R_outer,
        cx + R_outer, cy + R_outer,
        outline=COLOR_BG_BORDER, width=2
    )

    # Glowing arc (280° sweep)
    canvas.create_arc(
        cx - R_outer, cy - R_outer,
        cx + R_outer, cy + R_outer,
        start=90, extent=-280,
        style="arc", outline=COLOR_PRIMARY, width=3
    )

    # Inner accent ring
    canvas.create_oval(
        cx - R_inner, cy - R_inner,
        cx + R_inner, cy + R_inner,
        outline=COLOR_BG_BORDER, width=1
    )

    # Tick marks at each 60°
    for i in range(6):
        angle = math.radians(90 - i * 60)
        x1 = cx + (R_outer - 2) * math.cos(angle)
        y1 = cy - (R_outer - 2) * math.sin(angle)
        x2 = cx + (R_outer + 4) * math.cos(angle)
        y2 = cy - (R_outer + 4) * math.sin(angle)
        canvas.create_line(x1, y1, x2, y2, fill=COLOR_MUTED, width=1)

    # Time text
    canvas.create_text(
        cx, cy,
        text=f"{time_val:.0f}",
        font=("Courier New", 17, "bold"),
        fill="white"
    )
    # Unit
    canvas.create_text(
        cx, cy + 16,
        text="SEC",
        font=("Courier New", 7, "bold"),
        fill=COLOR_MUTED
    )

    # Update the external label if provided
    if time_label:
        time_label.config(text=f"{time_val:.0f}s")


def create_clock_widget(parent, fsm_key, label_text):
    """A bordered clock card with label and hover-to-edit."""
    outer = tk.Frame(parent, bg=COLOR_BG_BORDER, padx=1, pady=1)
    inner = tk.Frame(outer, bg=COLOR_BG_CARD, padx=14, pady=14)
    inner.pack(fill="both", expand=True)

    canvas = tk.Canvas(inner, width=100, height=100,
                       bg=COLOR_BG_CARD, highlightthickness=0,
                       cursor="hand2")
    canvas.pack()

    # Label row
    lbl_row = tk.Frame(inner, bg=COLOR_BG_CARD)
    lbl_row.pack(fill="x", pady=(8, 0))

    tk.Label(lbl_row, text=label_text.upper(),
             font=("Courier New", 8, "bold"),
             fg=COLOR_MUTED, bg=COLOR_BG_CARD).pack(side="left")

    time_lbl = tk.Label(lbl_row, text="",
                        font=("Courier New", 8),
                        fg=COLOR_PRIMARY, bg=COLOR_BG_CARD)
    time_lbl.pack(side="right")

    # Edit hint
    edit_lbl = tk.Label(inner, text="CLICK TO EDIT",
                        font=("Courier New", 7),
                        fg=COLOR_BG_BORDER, bg=COLOR_BG_CARD)
    edit_lbl.pack()

    draw_clock(canvas, FSM_TIMES.get(fsm_key, 0.0), time_lbl)

    def on_click(e=None):
        change_fsm_time(fsm_key, canvas, time_lbl)

    def on_enter(e=None):
        edit_lbl.config(fg=COLOR_ACCENT)
        canvas.config(bg=COLOR_BG_BORDER)

    def on_leave(e=None):
        edit_lbl.config(fg=COLOR_BG_BORDER)
        canvas.config(bg=COLOR_BG_CARD)

    for w in [canvas, inner, edit_lbl]:
        w.bind("<Button-1>", on_click)
        w.bind("<Enter>",    on_enter)
        w.bind("<Leave>",    on_leave)

    return outer


# ─────────────────────────────────────────────
#  MAIN GUI SETUP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.title("SUVOS Monitor")
    root.geometry("980x660")
    root.minsize(860, 580)
    root.configure(bg=COLOR_BG_MAIN)

    # ── Grid scanline texture (drawn on a canvas behind everything) ──
    bg_canvas = tk.Canvas(root, bg=COLOR_BG_MAIN, highlightthickness=0)
    bg_canvas.place(relwidth=1, relheight=1)

    def draw_grid(event=None):
        bg_canvas.delete("grid")
        w = bg_canvas.winfo_width()
        h = bg_canvas.winfo_height()
        step = 32
        for x in range(0, w, step):
            bg_canvas.create_line(x, 0, x, h, fill=COLOR_BG_BORDER,
                                  width=1, tags="grid")
        for y in range(0, h, step):
            bg_canvas.create_line(0, y, w, y, fill=COLOR_BG_BORDER,
                                  width=1, tags="grid")

    root.bind("<Configure>", draw_grid)

    # ── Status bar (bottom) ──
    status_bar = StatusBar(root)
    status_bar.pack(side="bottom", fill="x")

    # ── Top header bar ──
    top_bar = tk.Frame(root, bg=COLOR_BG_CARD, height=46)
    top_bar.pack(side="top", fill="x")
    top_bar.pack_propagate(False)

    tk.Label(top_bar, text="SUVOS",
             font=("Courier New", 16, "bold"),
             fg=COLOR_PRIMARY, bg=COLOR_BG_CARD).pack(side="left", padx=20, pady=8)

    tk.Label(top_bar, text="UV-C DISINFECTION MONITORING SYSTEM",
             font=FONT_SMALL, fg=COLOR_MUTED,
             bg=COLOR_BG_CARD).pack(side="left", pady=14)

    # Right side — live clock
    clock_lbl = tk.Label(top_bar, text="",
                         font=FONT_SMALL, fg=COLOR_MUTED,
                         bg=COLOR_BG_CARD)
    clock_lbl.pack(side="right", padx=20)

    def tick():
        import datetime
        clock_lbl.config(
            text=datetime.datetime.now().strftime("  %Y-%m-%d  %H:%M:%S  ")
        )
        root.after(1000, tick)
    tick()

    # Separator under header
    tk.Frame(root, bg=COLOR_PRIMARY, height=1).pack(fill="x")

    # ══════════════════════════════════════════
    #  MAIN FRAME
    # ══════════════════════════════════════════
    main_frame = tk.Frame(root, bg=COLOR_BG_MAIN)
    main_frame.place(relx=0.5, rely=0.5, anchor="center")

    # ── LEFT COLUMN ──
    left_panel = tk.Frame(main_frame, bg=COLOR_BG_MAIN, width=300)
    left_panel.grid(row=0, column=0, padx=(0, 60), sticky="ns")
    left_panel.grid_propagate(False)

    # Logo / Title
    logo_path = resource_path("assets/logo.png")
    if os.path.exists(logo_path):
        pil_img    = Image.open(logo_path)
        base_width = 160
        h_size     = int(pil_img.size[1] * (base_width / pil_img.size[0]))
        pil_img    = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        logo_photo = ImageTk.PhotoImage(pil_img)
        tk.Label(left_panel, image=logo_photo,
                 bg=COLOR_BG_MAIN).pack(pady=(0, 6))
        left_panel.logo_photo = logo_photo  # keep reference
    else:
        tk.Label(left_panel, text="BOLB INC.",
                 font=("Courier New", 22, "bold"),
                 bg=COLOR_BG_MAIN, fg="white").pack(pady=(0, 6))

    tk.Label(left_panel, text="OPERATIONS CONSOLE",
             font=FONT_SMALL, fg=COLOR_MUTED,
             bg=COLOR_BG_MAIN).pack(pady=(0, 28))

    # Setup section
    SectionLabel(left_panel, "SETUP").pack(fill="x", pady=(0, 6))

    btn1 = ModernButton(left_panel, text="01 — Scan & Calibrate", command=step1)
    btn1.pack(fill="x", pady=3)

    btn2 = ModernButton(left_panel, text="02 — Define Zones", command=step2)
    btn2.pack(fill="x", pady=3)

    # Operations section
    SectionLabel(left_panel, "OPERATION").pack(fill="x", pady=(20, 6))

    btn3 = ModernButton(left_panel, text="03 — Activate System",
                        command=step3_run,
                        font=("Courier New", 12, "bold"),
                        pady=18)
    btn3.pack(fill="x", pady=3)

    # Shutdown
    SectionLabel(left_panel, "SYSTEM").pack(fill="x", pady=(20, 6))

    exit_btn = ModernButton(left_panel, text="Shutdown",
                            command=shutdown, fg=COLOR_DANGER,
                            pady=10)
    exit_btn.pack(fill="x", pady=3)

    # ── RIGHT COLUMN ──
    right_panel = tk.Frame(main_frame, bg=COLOR_BG_MAIN)
    right_panel.grid(row=0, column=1, sticky="n")

    tk.Label(right_panel, text="OPERATIONAL TIMERS",
             font=("Courier New", 10, "bold"),
             fg=COLOR_MUTED, bg=COLOR_BG_MAIN).pack(pady=(0, 18))

    tk.Label(right_panel, text="Click any timer to adjust duration",
             font=("Courier New", 8),
             fg=COLOR_BG_BORDER, bg=COLOR_BG_MAIN).pack(pady=(0, 18))

    timers_frame = tk.Frame(right_panel, bg=COLOR_BG_MAIN)
    timers_frame.pack()

    create_clock_widget(timers_frame, "waitingTime",
                        "Safety Wait").grid(row=0, column=0, padx=10, pady=10)
    create_clock_widget(timers_frame, "disinfectionTime",
                        "UVC Disinfect").grid(row=0, column=1, padx=10, pady=10)
    create_clock_widget(timers_frame, "standbyTime",
                        "Standby").grid(row=1, column=0, columnspan=2,
                                        pady=10, sticky="ew")

    # System info panel
    info_frame = tk.Frame(right_panel, bg=COLOR_BG_CARD,
                          padx=14, pady=10)
    info_frame.pack(fill="x", pady=(14, 0))

    def info_row(parent, label, value, color=COLOR_TEXT):
        row = tk.Frame(parent, bg=COLOR_BG_CARD)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label,
                 font=("Courier New", 8), fg=COLOR_MUTED,
                 bg=COLOR_BG_CARD, width=14, anchor="w").pack(side="left")
        tk.Label(row, text=value,
                 font=("Courier New", 8), fg=color,
                 bg=COLOR_BG_CARD).pack(side="left")

    tk.Label(info_frame, text="SYSTEM INFO",
             font=("Courier New", 8, "bold"), fg=COLOR_MUTED,
             bg=COLOR_BG_CARD).pack(anchor="w", pady=(0, 6))
    info_row(info_frame, "VERSION", "SUVOS v2.0")
    info_row(info_frame, "CAMERAS",
             f"{len(CAMERA_INDICES)} active", COLOR_PRIMARY)

    # ══════════════════════════════════════════
    #  RUNNING FRAME (shown while system active)
    # ══════════════════════════════════════════
    running_frame = tk.Frame(root, bg=COLOR_BG_MAIN, padx=80, pady=60)

    # Pulsing indicator canvas
    pulse_canvas = tk.Canvas(running_frame, width=80, height=80,
                             bg=COLOR_BG_MAIN, highlightthickness=0)
    pulse_canvas.pack(pady=(0, 20))

    _pulse_radii = [0]
    def animate_pulse():
        pulse_canvas.delete("all")
        r = _pulse_radii[0]
        cx, cy = 40, 40
        for ring in range(3):
            rr = (r + ring * 10) % 36
            alpha_factor = 1 - (rr / 36)
            gray_val = int(alpha_factor * 0x55)
            color_hex = f"#{gray_val:02x}{0xff:02x}{0xc8:02x}"
            if rr > 0:
                pulse_canvas.create_oval(
                    cx - rr, cy - rr, cx + rr, cy + rr,
                    outline=color_hex, width=1
                )
        pulse_canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                                 fill=COLOR_PRIMARY, outline=COLOR_PRIMARY)
        _pulse_radii[0] = (r + 0.8) % 36
        running_frame.after(30, animate_pulse)

    animate_pulse()

    tk.Label(running_frame, text="SYSTEM ACTIVE",
             font=("Courier New", 24, "bold"),
             fg=COLOR_PRIMARY, bg=COLOR_BG_MAIN).pack(pady=(0, 6))

    tk.Label(running_frame,
             text="UVC DISINFECTION MONITORING IN PROGRESS",
             font=FONT_SMALL, fg=COLOR_MUTED,
             bg=COLOR_BG_MAIN).pack(pady=(0, 2))

    tk.Frame(running_frame, bg=COLOR_PRIMARY, height=1,
             width=320).pack(pady=20)

    stop_btn = ModernButton(running_frame, text="⬛  Stop Operation",
                            command=stop_system, fg=COLOR_DANGER,
                            font=("Courier New", 13, "bold"), pady=18)
    stop_btn.pack(fill="x")

    # ── Init ──
    load_fsm_config()
    check_status()
    root.mainloop()