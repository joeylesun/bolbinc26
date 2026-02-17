import tkinter as tk
from tkinter import filedialog, messagebox
import json
import sys
import os

# Import shared config and path helpers
from suvos_common import (
    resource_path,
    get_config_path,
    SHAPE_FILE,
    ZONES_FILE,
    get_calib_file,
    CAMERA_INDICES
)

# IMPORT MODULES
import suvos_calibration
import suvos_setup
import suvos_working


def step1_load_and_calibrate():
    # 1. Select the .pts file
    pts_path = filedialog.askopenfilename(
        title="Select Dot3D Scan (.pts)",
        filetypes=[("PTS Files", "*.pts"), ("All Files", "*.*")]
    )
    if not pts_path: return

    # 2. Run Calibration
    # The calibration module now handles saving to get_config_path() internally
    print(f"Loading {pts_path}...")
    try:
        suvos_calibration.run_calibration(pts_path)
    except Exception as e:
        messagebox.showerror("Error", f"Calibration failed:\n{e}")


def step2_zones():
    # Check if calibration exists in the PERSISTENT storage
    # We check the first camera's calibration file
    calib_file = get_config_path(get_calib_file(CAMERA_INDICES[0]))

    if not os.path.exists(calib_file):
        messagebox.showwarning(
            "Warning",
            f"No calibration found at:\n{calib_file}\n\nPlease run Step 1 first."
        )
        return

    try:
        suvos_setup.run_setup()
    except Exception as e:
        messagebox.showerror("Error", f"Setup failed:\n{e}")


def step3_run():
    # Check for zones file in PERSISTENT storage
    zones_path = get_config_path(ZONES_FILE)

    if not os.path.exists(zones_path):
        messagebox.showerror("Error", "No Zones File found. Run Step 2.")
        return

    root.withdraw()  # Hide menu
    try:
        suvos_working.start_system()
    except Exception as e:
        messagebox.showerror("Error", f"System crashed:\n{e}")
        root.deiconify()  # Show menu again if it crashes immediately
    else:
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("SUVOS Setup & Control")
    root.geometry("450x420")

    # Title
    tk.Label(root, text="SUVOS MONITOR CONTROLLER", font=("Arial", 18, "bold")).pack(pady=(20, 5))
    tk.Label(root, text="v2.0 (Dual-Ready)", font=("Arial", 10, "italic"), fg="gray").pack(pady=(0, 20))

    # Buttons Frame
    frame = tk.Frame(root)
    frame.pack(pady=10)

    btn1 = tk.Button(frame, text="1. Load Scan & Calibrate (.pts)", command=step1_load_and_calibrate, height=2,
                     width=35)
    btn1.pack(pady=8)

    btn2 = tk.Button(frame, text="2. Setup Zones", command=step2_zones, height=2, width=35)
    btn2.pack(pady=8)

    tk.Frame(root, height=15).pack()

    btn3 = tk.Button(root, text="3. START SYSTEM", command=step3_run, height=3, width=35, bg="#4CAF50", fg="black")
    btn3.pack(pady=10)

    # Footer
    data_dir = os.path.expanduser("~/SUVOS_Data")
    tk.Label(root, text=f"Data Folder: {data_dir}", font=("Arial", 9), fg="gray").pack(side=tk.BOTTOM, pady=10)

    root.mainloop()