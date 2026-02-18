## SUVOS APP File Structure

| File | Description |
| :--- | :--- |
| **`suvos_launcher.py`** | **The Main Controller.** A Tkinter GUI that acts as the system entry point. It manages the high-tech "Dark Mode" dashboard, safety timers, and launches the other modules. |
| **`suvos_ui.py`** | **UI Theme Engine.** Defines the custom "ModernButton" class and the color palette (Deep Navy/Neon Cyan) to bypass standard macOS gray styling. |
| **`suvos_working.py`** | **The Backend.** Runs the YOLOv8 detection, FSM logic, Serial communication, and WebSocket server in a background thread. |
| **`suvos_setup.py`** | **Zone Configurator.** A tool to draw LED safety zones on the camera feed. It saves the zone coordinates to `led_zones.json`. |
| **`suvos_calibration.py`** | **Calibration Tool.** Calculates the Homography Matrix (`H`) to map 2D camera pixels to 3D real-world meters using Open3D and OpenCV. |
| **`suvos_common.py`** | **Shared Config.** Stores global constants (`CAM_WIDTH`, `CAM_HEIGHT`), file paths, and helper functions for resource management. |
| **`build.command`** | **Auto-Builder.** A shell script that automates the entire installation and packaging process for macOS. |
| **`SUVOS_Monitor.spec`** | **PyInstaller Spec.** Defines how the app is bundled, ensuring assets like the Logo and YOLO models are included. |

---

##  Build Process

We have automated the complex build process (handling Virtual Environments, Dependencies, and Signing) into a single script: **`build.command`**.

### How it Works
When you run the build script, it performs these steps automatically:
1.  **Clean:** Removes old build artifacts (`dist/`, `build/`).
2.  **Isolate:** Creates a fresh Python Virtual Environment (`venv`) to avoid conflicts with your system Python.
3.  **Install:** Installs specific versions of libraries (e.g., `numpy<2.0`, `open3d`) to ensure compatibility with macOS binaries.
4.  **Package:** Runs **PyInstaller** to bundle the Python interpreter, YOLO models, and your `assets/` folder into a single `.app` file.
5.  **Sign:** Applies an ad-hoc signature to the app so macOS Gatekeeper allows it to access the Camera and Microphone.

### How to Build
1.  Open Terminal in the project folder.
2.  Make the script executable (only needed once):
    ```bash
    chmod +x build.command
    ```
3.  Run the builder:
    ```bash
    ./build.command
    ```
4.  **Result:** The finished app will appear in the `dist/` folder.



