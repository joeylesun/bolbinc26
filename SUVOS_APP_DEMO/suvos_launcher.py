import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import sys
import os

# --- IMPORTS ---
from suvos_common import (
    resource_path, get_config_path, ZONES_FILE,
    get_calib_file, CAMERA_INDICES, FSM_TIMES, save_fsm_config, load_fsm_config
)
import suvos_calibration
import suvos_setup
import suvos_working
from suvos_ui import ModernButton, COLOR_BG_MAIN, COLOR_PRIMARY, COLOR_DANGER

# --- GLOBALS ---
SYSTEM = suvos_working.SUVOS_System()
CAM_WINDOW = None


# --- HELPER FUNCTIONS ---
def check_status():
    """ Updates button status indicators """
    calib_path = get_config_path(get_calib_file(CAMERA_INDICES[0]))
    zones_path = get_config_path(ZONES_FILE)

    btn1.set_status(os.path.exists(calib_path))
    btn2.set_status(os.path.exists(zones_path))

    # Enable Start Button only if ready
    if os.path.exists(calib_path) and os.path.exists(zones_path):
        btn3.config(fg=COLOR_PRIMARY)
    else:
        btn3.config(fg="#576574")  # Dimmed out


def step1():
    pts = filedialog.askopenfilename(filetypes=[("PTS", "*.pts")])
    if pts:
        try:
            suvos_calibration.run_calibration(pts); check_status()
        except Exception as e:
            messagebox.showerror("Error", str(e))


def step2():
    if not os.path.exists(get_config_path(get_calib_file(CAMERA_INDICES[0]))):
        messagebox.showwarning("Wait", "Run Step 1 First.")
        return
    try:
        suvos_setup.run_setup(); check_status()
    except Exception as e:
        messagebox.showerror("Error", str(e))


# --- CAMERA WINDOW ---
def open_camera_window():
    global CAM_WINDOW
    CAM_WINDOW = tk.Toplevel(root)
    CAM_WINDOW.title("SUVOS Live Camera")
    CAM_WINDOW.geometry("1280x720")

    lbl_cam = tk.Label(CAM_WINDOW, bg="black")
    lbl_cam.pack(fill="both", expand=True)

    def update_cam():
        if SYSTEM.running and CAM_WINDOW:
            frame = SYSTEM.get_processed_frame()
            if frame is not None:
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_cam.imgtk = imgtk
                lbl_cam.configure(image=imgtk)
            CAM_WINDOW.after(30, update_cam)

    update_cam()
    CAM_WINDOW.protocol("WM_DELETE_WINDOW", stop_system)


def step3_run():
    # Only run if ready
    if str(btn3.cget("fg")) == "#576574": return

    main_frame.place_forget()
    running_frame.place(relx=0.5, rely=0.5, anchor="center")
    SYSTEM.start()
    open_camera_window()


def stop_system():
    SYSTEM.stop()
    if CAM_WINDOW: CAM_WINDOW.destroy()
    running_frame.place_forget()
    main_frame.place(relx=0.5, rely=0.5, anchor="center")
    check_status()


def shutdown():
    SYSTEM.stop()
    root.destroy()
    sys.exit(0)


# --- CLOCK UI LOGIC ---
def change_fsm_time(fsm_key, canvas):
    current = FSM_TIMES.get(fsm_key, 0.0)
    new_val = simpledialog.askfloat("Timer Setting", f"Enter duration (seconds):", initialvalue=current, minvalue=0.0)
    if new_val is not None:
        save_fsm_config({fsm_key: new_val})
        draw_clock(canvas, new_val)


def draw_clock(canvas, time_val):
    canvas.delete("all")
    w, h = 120, 120
    cx, cy = w / 2, h / 2
    radius = 45

    # 1. Background Ring (Darker Grey for contrast on dark BG)
    canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#333333", width=4)
    # 2. Neon Arc
    canvas.create_arc(cx - radius, cy - radius, cx + radius, cy + radius, start=90, extent=-280, style="arc",
                      outline=COLOR_PRIMARY, width=4)
    # 3. Text
    canvas.create_text(cx, cy, text=f"{time_val:.0f}s", font=("Segoe UI", 16, "bold"), fill="white")


def create_clock_widget(parent, fsm_key, label_text):
    # CRITICAL: Background matches Main Theme
    frame = tk.Frame(parent, bg=COLOR_BG_MAIN)

    canvas = tk.Canvas(frame, width=120, height=120, bg=COLOR_BG_MAIN, highlightthickness=0)
    canvas.pack(pady=(0, 5))
    draw_clock(canvas, FSM_TIMES.get(fsm_key, 0.0))

    lbl = tk.Label(frame, text=label_text.upper(), font=("Segoe UI", 9, "bold"), fg="#8395a7", bg=COLOR_BG_MAIN)
    lbl.pack()

    lbl.bind("<Button-1>", lambda e: change_fsm_time(fsm_key, canvas))
    canvas.bind("<Button-1>", lambda e: change_fsm_time(fsm_key, canvas))

    return frame


# --- GUI SETUP ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("SUVOS Monitor")
    root.geometry("1000x700")
    # 1. MAIN BACKGROUND
    root.configure(bg=COLOR_BG_MAIN)

    # 2. MAIN CONTAINER (Invisible Frame)
    # We removed padx/pady and bg colors that differed
    main_frame = tk.Frame(root, bg=COLOR_BG_MAIN)
    main_frame.place(relx=0.5, rely=0.5, anchor="center")

    # --- LEFT COLUMN: CONTROLS ---
    left_panel = tk.Frame(main_frame, bg=COLOR_BG_MAIN)
    left_panel.grid(row=0, column=0, padx=(0, 80), sticky="ns")

    # A. LOGO
    logo_path = resource_path("assets/logo.png")
    if os.path.exists(logo_path):
        pil_img = Image.open(logo_path)
        base_width = 200
        w_percent = (base_width / float(pil_img.size[0]))
        h_size = int((float(pil_img.size[1]) * float(w_percent)))
        pil_img = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)

        logo_photo = ImageTk.PhotoImage(pil_img)
        # Ensure logo label matches BG
        tk.Label(left_panel, image=logo_photo, bg=COLOR_BG_MAIN).pack(pady=(0, 40))
    else:
        tk.Label(left_panel, text="BOLB INC.", font=("Segoe UI", 28, "bold"), bg=COLOR_BG_MAIN, fg="white").pack(
            pady=(0, 40))

    # B. BUTTONS
    btn1 = ModernButton(left_panel, text="Scan & Calibrate", command=step1)
    btn1.pack(pady=10, fill="x")

    btn2 = ModernButton(left_panel, text="Setup Zones", command=step2)
    btn2.pack(pady=10, fill="x")

    # Spacer
    tk.Frame(left_panel, height=40, bg=COLOR_BG_MAIN).pack(fill="x")

    # C. START BUTTON
    btn3 = ModernButton(left_panel, text="Start System", command=step3_run,
                        font=("Segoe UI", 14, "bold"), pady=16)
    btn3.pack(pady=10, fill="x")

    # D. SHUTDOWN
    exit_btn = ModernButton(left_panel, text="Shutdown", command=shutdown, fg=COLOR_DANGER)
    exit_btn.config(pady=8, font=("Segoe UI", 10, "bold"))
    exit_btn.pack(pady=(20, 0), fill="x")

    # --- RIGHT COLUMN: TIMERS ---
    right_panel = tk.Frame(main_frame, bg=COLOR_BG_MAIN)
    right_panel.grid(row=0, column=1, sticky="n")

    # Header
    tk.Label(right_panel, text="OPERATIONAL CYCLES", font=("Segoe UI", 12, "bold"), bg=COLOR_BG_MAIN,
             fg="#576574").pack(pady=(10, 40))

    timers_frame = tk.Frame(right_panel, bg=COLOR_BG_MAIN)
    timers_frame.pack()

    create_clock_widget(timers_frame, "waitingTime", "Safety Wait").grid(row=0, column=0, padx=25, pady=25)
    create_clock_widget(timers_frame, "disinfectionTime", "UVC Disinfection").grid(row=0, column=1, padx=25, pady=25)
    create_clock_widget(timers_frame, "standbyTime", "Standby").grid(row=1, column=0, columnspan=2, pady=25)

    # 3. RUNNING PANEL
    running_frame = tk.Frame(root, bg=COLOR_BG_MAIN, padx=60, pady=60)
    tk.Label(running_frame, text="SYSTEM ACTIVE", font=("Segoe UI", 24, "bold"), fg=COLOR_PRIMARY,
             bg=COLOR_BG_MAIN).pack(pady=10)
    tk.Label(running_frame, text="UVC Disinfection Monitoring Active", font=("Segoe UI", 12), bg=COLOR_BG_MAIN,
             fg="white").pack(pady=5)

    ModernButton(running_frame, text="STOP OPERATION", command=stop_system, fg=COLOR_DANGER,
                 font=("Segoe UI", 14, "bold"), pady=16).pack(pady=40)

    load_fsm_config()
    check_status()
    root.mainloop()