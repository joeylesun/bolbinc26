import tkinter as tk
from tkinter import filedialog, messagebox
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

# --- UI STATE CHECKER ---
def check_status():
    """Checks if persistent files exist and updates UI labels"""
    
    # 1. Check Calibration
    calib_path = get_config_path(get_calib_file(CAMERA_INDICES[0]))
    calib_exists = os.path.exists(calib_path)
    
    # 2. Check Zones
    zones_path = get_config_path(ZONES_FILE)
    zones_exists = os.path.exists(zones_path)

    # Update Button 1 Text
    if calib_exists:
        btn1.config(text="1. Load Scan & Calibrate (✅ Ready)", bg="#e6ffe6")
    else:
        btn1.config(text="1. Load Scan & Calibrate", bg="white")

    # Update Button 2 Text
    if zones_exists:
        btn2.config(text="2. Setup Zones (✅ Ready)", bg="#e6ffe6")
    else:
        btn2.config(text="2. Setup Zones", bg="white")

    # Update Start Button
    if calib_exists and zones_exists:
        btn3.config(state="normal", bg="#4CAF50", text="3. START SYSTEM (Ready to Run)")
    else:
        # We keep it enabled but standard color, or could disable it if strictly enforcing flow
        btn3.config(text="3. START SYSTEM", bg="systemButtonFace")

def step1_load_and_calibrate():
    pts_path = filedialog.askopenfilename(
        title="Select Dot3D Scan (.pts)", 
        filetypes=[("PTS Files", "*.pts"), ("All Files", "*.*")]
    )
    if not pts_path: return

    print(f"Loading {pts_path}...")
    try:
        suvos_calibration.run_calibration(pts_path)
        check_status() # Update UI after run
    except Exception as e:
        messagebox.showerror("Error", f"Calibration failed:\n{e}")

def step2_zones():
    calib_file = get_config_path(get_calib_file(CAMERA_INDICES[0]))
    if not os.path.exists(calib_file):
        messagebox.showwarning("Warning", f"No calibration found.\nPlease run Step 1 first.")
        return
        
    try:
        suvos_setup.run_setup()
        check_status() # Update UI after run
    except Exception as e:
        messagebox.showerror("Error", f"Setup failed:\n{e}")

def step3_run():
    zones_path = get_config_path(ZONES_FILE)
    if not os.path.exists(zones_path):
        messagebox.showerror("Error", "No Zones File found. Run Step 2.")
        return
        
    root.withdraw()
    try:
        suvos_working.start_system()
    except Exception as e:
        messagebox.showerror("Error", f"System crashed:\n{e}")
        root.deiconify()
    else:
        sys.exit(0)

# --- GUI SETUP ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("SUVOS Setup & Control")
    root.geometry("450x450")
    
    # Title
    tk.Label(root, text="SUVOS MONITOR CONTROLLER", font=("Arial", 18, "bold")).pack(pady=(20, 5))
    tk.Label(root, text="v2.1 (Smart Persistence)", font=("Arial", 10, "italic"), fg="gray").pack(pady=(0, 20))
    
    # Buttons Frame
    frame = tk.Frame(root)
    frame.pack(pady=10)
    
    btn1 = tk.Button(frame, text="1. Load Scan & Calibrate", command=step1_load_and_calibrate, height=2, width=35)
    btn1.pack(pady=8)
    
    btn2 = tk.Button(frame, text="2. Setup Zones", command=step2_zones, height=2, width=35)
    btn2.pack(pady=8)
    
    tk.Frame(root, height=15).pack()
    
    btn3 = tk.Button(root, text="3. START SYSTEM", command=step3_run, height=3, width=35)
    btn3.pack(pady=10)
    
    # Footer info
    data_dir = os.path.expanduser("~/SUVOS_Data")
    lbl_info = tk.Label(root, text=f"Data Folder: {data_dir}", font=("Arial", 9), fg="gray")
    lbl_info.pack(side=tk.BOTTOM, pady=10)
    
    # Initial Check
    check_status()
    
    root.mainloop()
