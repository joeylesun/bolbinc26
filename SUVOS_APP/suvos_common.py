import os
import sys

# --- HARDWARE CONFIGURATION ---
# List of cameras to use.
# Single Camera Mode: [0]
# Dual Camera Mode:   [0, 1]
CAMERA_INDICES = [0]

# Helper to get the single index for legacy single-cam scripts
CAMERA_INDEX = CAMERA_INDICES[0]

SERIAL_PORT = "/dev/cu.usbserial-120"
BAUD_RATE = 115200

# --- NETWORK CONFIGURATION ---
WS_HOST = "localhost"
WS_PORT = 8765

# --- FILE PATHS ---
ZONES_FILE = "led_zones.json"
SHAPE_FILE = "room_shape.json"
MODEL_FILE = "yolov8n.pt"


# --- HELPER FUNCTIONS ---
def get_calib_file(cam_idx):
    """Returns filename like 'calibration_data_cam0.npz'"""
    return f"calibration_data_cam{cam_idx}.npz"


# *** CRITICAL FIX: DEFINE CALIB_FILE HERE ***
# This is what suvos_calibration.py is looking for
CALIB_FILE = get_calib_file(CAMERA_INDEX)

# --- MAP & CALIBRATION SETTINGS ---
MAP_SIZE = 1000  # Resolution of the internal map
GRID_SPACING_M = 1.0  # Grid line spacing for visuals

# --- APPLICATION LOGIC ---
NUM_LEDS = 16
CONE_RADIUS_M = 3.0
CONF_THRES = 0.4
MERGE_DISTANCE_M = 1.5
HEARTBEAT_INTERVAL = 0.10

FSM_TIMES = {
    "waitingTime": 5.0,
    "disinfectionTime": 30.0,
    "standbyTime": 10.0
}


# --- SYSTEM HELPERS ---
def resource_path(relative_path):
    """ Used for READING static assets bundled with the app (Frontend, Model) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_config_path(filename):
    """
    Used for READING & WRITING dynamic config files.
    Saves to: /Users/<Username>/SUVOS_Data/
    """
    home = os.path.expanduser("~")
    data_dir = os.path.join(home, "SUVOS_Data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, filename)