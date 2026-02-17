import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import os
import cv2

# Shared Config
from suvos_common import (
    get_config_path, ZONES_FILE, get_calib_file, CAMERA_INDICES
)

# Modules
import suvos_calibration
import suvos_setup
import suvos_working
import suvos_config

SYSTEM = suvos_working.SUVOS_System()
CAM_WINDOW = None  # Reference to the separate camera window
VIDEO_LOOP_ID = None


def check_status():
    calib_path = get_config_path(get_calib_file(CAMERA_INDICES[0]))
    zones_path = get_config_path(ZONES_FILE)
    c_ok = os.path.exists(calib_path)
    z_ok = os.path.exists(zones_path)

    btn1.config(text="1. Load Scan & Calibrate" + (" (✅)" if c_ok else ""), bg="#e6ffe6" if c_ok else "white")
    btn2.config(text="2. Setup Zones" + (" (✅)" if z_ok else ""), bg="#e6ffe6" if z_ok else "white")

    if c_ok and z_ok:
        btn3.config(state="normal", bg="#4CAF50", text="3. START SYSTEM (Ready)")
    else:
        btn3.config(state="normal", text="3. START SYSTEM", bg="systemButtonFace")


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


# --- CAMERA WINDOW LOGIC ---
def update_cam_window(label):
    global VIDEO_LOOP_ID
    if SYSTEM.running:
        frame = SYSTEM.get_processed_frame()
        if frame is not None:
            # Resize for display if needed, or keep 1280x720
            # RGB Conversion
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            label.imgtk = imgtk
            label.configure(image=imgtk)

        # Keep updating
        VIDEO_LOOP_ID = root.after(30, lambda: update_cam_window(label))


def open_camera_window():
    global CAM_WINDOW
    # Create a separate floating window
    CAM_WINDOW = tk.Toplevel(root)
    CAM_WINDOW.title("SUVOS Live Camera")
    CAM_WINDOW.geometry("1280x720")

    # Label to hold the video
    lbl_cam = tk.Label(CAM_WINDOW, bg="black")
    lbl_cam.pack(fill="both", expand=True)

    # Start the update loop for this window
    update_cam_window(lbl_cam)

    # Handle window close button (X)
    CAM_WINDOW.protocol("WM_DELETE_WINDOW", stop_system)


def step3_run():
    # 1. Hide Menu
    frame_main.pack_forget()
    frame_tools.pack_forget()
    lbl_info.pack_forget()

    # 2. Show "Running" Control Panel
    frame_running.pack(fill="both", expand=True, pady=50)

    # 3. Start Backend
    SYSTEM.start()

    # 4. Open Separate Camera Window
    open_camera_window()


def stop_system():
    global CAM_WINDOW, VIDEO_LOOP_ID

    # Stop Backend
    SYSTEM.stop()
    if VIDEO_LOOP_ID: root.after_cancel(VIDEO_LOOP_ID)

    # Close Separate Window
    if CAM_WINDOW:
        CAM_WINDOW.destroy()
        CAM_WINDOW = None

    # Restore UI
    frame_running.pack_forget()
    frame_main.pack(pady=10)
    frame_tools.pack(pady=5)
    lbl_info.pack(side=tk.BOTTOM, pady=10)
    check_status()


def shutdown():
    SYSTEM.stop()
    root.destroy()
    sys.exit(0)


# --- GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("SUVOS Monitor")
    root.geometry("450x500")  # Small Control Window

    tk.Label(root, text="SUVOS MONITOR CONTROLLER", font=("Arial", 18, "bold")).pack(pady=(20, 5))

    # --- RUNNING UI ---
    frame_running = tk.Frame(root)
    tk.Label(frame_running, text="SYSTEM IS RUNNING", font=("Arial", 16, "bold"), fg="green").pack(pady=20)
    tk.Label(frame_running, text="Camera is in a separate window.", font=("Arial", 10)).pack(pady=5)
    tk.Button(frame_running, text="⏹ STOP SYSTEM", command=stop_system, bg="#ffcccc", fg="red",
              font=("Arial", 14, "bold"), height=2, width=20).pack(pady=30)

    # --- MAIN MENU UI ---
    frame_main = tk.Frame(root)
    frame_main.pack(pady=10)

    btn1 = tk.Button(frame_main, text="1. Load Scan & Calibrate", command=step1, height=2, width=35)
    btn1.pack(pady=8)
    btn2 = tk.Button(frame_main, text="2. Setup Zones", command=step2, height=2, width=35)
    btn2.pack(pady=8)

    tk.Frame(frame_main, height=15).pack()

    btn3 = tk.Button(frame_main, text="3. START SYSTEM", command=step3_run, height=3, width=35)
    btn3.pack(pady=10)

    frame_tools = tk.Frame(root)
    frame_tools.pack(pady=5)
    tk.Button(frame_tools, text="⚙️ Config", command=lambda: suvos_config.run_config_editor(root)).grid(row=0, column=0,
                                                                                                        padx=5)
    tk.Button(frame_tools, text="🛑 Exit", command=shutdown, bg="#ffcccc").grid(row=0, column=1, padx=5)

    lbl_info = tk.Label(root, text="v4.1 (Dual Window)", font=("Arial", 9), fg="gray")
    lbl_info.pack(side=tk.BOTTOM, pady=10)

    check_status()
    root.mainloop()