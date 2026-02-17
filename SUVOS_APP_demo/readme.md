# SUVOS APP DEMO


| File | Description |
| :--- | :--- |
| **`suvos_launcher.py`** | **The Entry Point.** A Tkinter GUI that acts as the main controller. It launches the Setup, Calibration, and Working modes. It creates the floating "Camera Window" to avoid macOS thread freezing. |
| **`suvos_working.py`** | **The Backend.** Runs the YOLOv8 detection, FSM logic, Serial communication, and WebSocket server in a background thread. It processes video and creates the overlays. |
| **`suvos_setup.py`** | **Zone Configurator.** A tool to draw LED safety zones on the camera feed. It saves the zone coordinates to `led_zones.json`. |
| **`suvos_calibration.py`** | **Calibration Tool.** Calculates the Homography Matrix (`H`) to map 2D camera pixels to 3D real-world meters. |
| **`suvos_common.py`** | **Shared Config.** Stores global constants like `CAM_WIDTH` (1920), `CAM_HEIGHT` (1080), ports, and file paths. |
| **`suvos_config.py`** | **Settings Editor.** A GUI for editing the `suvos_common.py` variables (timers, thresholds) without touching code. |
| **`SUVOS_Monitor.spec`** | **Build Spec.** The configuration file for PyInstaller, defining included assets (models, HTML) and excluded libraries. |

---

## Environment Setup (Crucial)


### 1. Create Virtual Environment
```bash
# Create a fresh environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 2. Install Dependencies
```
# Upgrade pip
pip install --upgrade pip

# Install core libraries with version constraints
pip install "numpy<2.0" "opencv-python<4.10" ultralytics websockets pyserial Pillow pyinstaller
```

### 3. Packing Process (Building the APP via pyinstaller)
```
# Clean Previous Builds
rm -rf build dist

# Run the Build
python -m PyInstaller --noconfirm --clean SUVOS_Monitor.spec

# Sign the APP
xattr -cr dist/SUVOS_Monitor.app
codesign --force --deep --sign - "dist/SUVOS_Monitor.app"
