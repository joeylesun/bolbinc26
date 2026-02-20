import os
import sys
import json

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
CONFIG_FILE = "suvos_config.json"

DEFAULT_FSM_TIMES = {
    "waitingTime": 5.0,       # Time before cleaning starts
    "disinfectionTime": 30.0, # Duration of cleaning
    "standbyTime": 10.0       # Cooldown after cleaning
}

CAM_WIDTH = 1920
CAM_HEIGHT = 1080

# --- HELPER FUNCTIONS ---
def get_calib_file(cam_idx):
    """Returns filename like 'calibration_data_cam0.npz'"""
    return f"calibration_data_cam{cam_idx}.npz"

def load_fsm_config():
    """ Loads timing config from persistent storage or returns defaults """
    path = get_config_path(CONFIG_FILE)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Ensure all keys exist (merge with defaults)
                config = DEFAULT_FSM_TIMES.copy()
                config.update(data)
                return config
        except Exception as e:
            print(f"[Config] Load error: {e}")
    return DEFAULT_FSM_TIMES.copy()

def save_fsm_config(config_dict):
    """ Saves timing config to persistent storage """
    path = get_config_path(CONFIG_FILE)
    try:
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"[Config] Save error: {e}")
        return False

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