import cv2
import numpy as np
import json
import os
from tkinter import messagebox
from suvos_common import resource_path, ZONES_FILE, get_calib_file, CAMERA_INDICES, NUM_LEDS, CONE_RADIUS_M,get_config_path

# ================= STATE =================
CURRENT_CAM_IDX = 0
CAM_ZOOM = 1.0
CAM_PAN = [0, 0]
LED_CONES = []
H_MATRIX = None


# --- CALIBRATION HELPERS ---
def load_calibration(cam_idx):
    """ Loads H matrix from PERSISTENT storage """
    # CHANGED: resource_path -> get_config_path
    fname = get_config_path(get_calib_file(cam_idx))
    try:
        d = np.load(fname)
        return d['H']
    except Exception as e:
        print(f"[Error] Could not load {fname}: {e}")
        return None


def pixel_to_world(u, v, H):
    """ Converts Frame Pixel (u,v) -> Real World (x,y) """
    if H is None: return (0, 0)
    w_vec = H @ np.array([u, v, 1.0])
    if abs(w_vec[2]) < 1e-9: return (0, 0)
    return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])


def world_to_pixel(x, y, H_inv):
    """ Converts Real World (x,y) -> Frame Pixel (u,v) """
    if H_inv is None: return (-100, -100)
    w_vec = H_inv @ np.array([x, y, 1.0])
    if abs(w_vec[2]) < 1e-9: return (-100, -100)
    return int(round(w_vec[0] / w_vec[2])), int(round(w_vec[1] / w_vec[2]))


# --- ZOOM & PAN HELPERS ---
def get_cam_view(frame):
    """ Digital Zoom: Crops the frame based on ZOOM/PAN state """
    h, w = frame.shape[:2]
    view_w = int(w / CAM_ZOOM)
    view_h = int(h / CAM_ZOOM)

    # Calculate Center
    cx = int(w / 2 + CAM_PAN[0])
    cy = int(h / 2 + CAM_PAN[1])

    # Calculate Top-Left with Clamping
    tl_x = max(0, min(cx - view_w // 2, w - view_w))
    tl_y = max(0, min(cy - view_h // 2, h - view_h))

    crop = frame[tl_y:tl_y + view_h, tl_x:tl_x + view_w]
    if crop.size == 0: return frame

    # Resize back to original window size
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def cam_view_to_real(vx, vy, frame_w, frame_h):
    """ Converts Mouse Click on Zoomed View -> Real Frame Coordinates """
    view_w = int(frame_w / CAM_ZOOM)
    view_h = int(frame_h / CAM_ZOOM)

    cx = int(frame_w / 2 + CAM_PAN[0])
    cy = int(frame_h / 2 + CAM_PAN[1])

    tl_x = max(0, min(cx - view_w // 2, frame_w - view_w))
    tl_y = max(0, min(cy - view_h // 2, frame_h - view_h))

    real_x = tl_x + (vx / CAM_ZOOM)
    real_y = tl_y + (vy / CAM_ZOOM)
    return int(real_x), int(real_y)


# --- MOUSE HANDLER ---
def on_click(event, x, y, flags, param):
    global LED_CONES
    w, h = param

    # Place LED (Left Click)
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(LED_CONES) >= NUM_LEDS:
            print("[Limit] Max zones reached.")
            return
        if H_MATRIX is None:
            print("[Error] Calibration missing for this camera.")
            return

        # 1. Map Click (Zoomed) -> Frame Pixel (Real)
        real_x, real_y = cam_view_to_real(x, y, w, h)

        # 2. Frame Pixel -> World Coordinate (Meters)
        wx, wy = pixel_to_world(real_x, real_y, H_MATRIX)

        # 3. Save Zone
        idx = len(LED_CONES)
        LED_CONES.append({"pos": (wx, wy), "id": idx, "radius": CONE_RADIUS_M})
        print(f"[Setup] Placed S{idx} at ({wx:.2f}, {wy:.2f})")

    # Undo (Right Click)
    elif event == cv2.EVENT_RBUTTONDOWN and len(LED_CONES) > 0:
        removed = LED_CONES.pop()
        print(f"[Setup] Removed S{removed['id']}")


# --- MAIN SETUP LOOP ---
def run_setup():
    global CURRENT_CAM_IDX, H_MATRIX, LED_CONES, CAM_ZOOM, CAM_PAN

    # 1. Initialize Camera
    cam_list_idx = 0
    CURRENT_CAM_IDX = CAMERA_INDICES[0]
    H_MATRIX = load_calibration(CURRENT_CAM_IDX)

    cap = cv2.VideoCapture(CURRENT_CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Get actual frame dimensions
    ret, test = cap.read()
    if ret:
        W, H = test.shape[1], test.shape[0]
    else:
        W, H = 1920, 1080

    # 2. Load Existing Zones (Persistence)
    if os.path.exists(get_config_path(ZONES_FILE)):
        try:
            with open(ZONES_FILE, 'r') as f:
                data = json.load(f)
                LED_CONES = [{"pos": tuple(d["pos"]), "id": d["id"], "radius": d.get("radius", CONE_RADIUS_M)} for d in
                             data]
                print(f"[Info] Loaded {len(LED_CONES)} existing zones.")
        except:
            pass

    # 3. Window Setup
    win_name = "Setup Zones"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_click, param=(W, H))

    print("--- SETUP MODE ---")
    print("CONTROLS: [L-Click]=Place | [R-Click]=Undo | [TAB]=Switch Cam | [+/-]=Zoom | [WASD]=Pan | [S]=Save")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[Error] Camera {CURRENT_CAM_IDX} disconnected. Retrying...")
            cap.open(CURRENT_CAM_IDX)
            continue

        # --- DRAW ZONES ---
        # We draw them on the raw frame first, then zoom.
        if H_MATRIX is not None:
            H_INV = np.linalg.inv(H_MATRIX)

            for cone in LED_CONES:
                wx, wy = cone["pos"]
                # Convert World -> Pixel
                cx, cy = world_to_pixel(wx, wy, H_INV)

                # Only draw if visible
                if 0 <= cx < W and 0 <= cy < H:
                    # Draw ID
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                    cv2.putText(frame, f"S{cone['id']}", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Draw Safety Radius Circle
                    pts_px = []
                    for ang in np.linspace(0, 6.28, 20):
                        pwx = wx + CONE_RADIUS_M * np.cos(ang)
                        pwy = wy + CONE_RADIUS_M * np.sin(ang)
                        pcx, pcy = world_to_pixel(pwx, pwy, H_INV)
                        pts_px.append((pcx, pcy))
                    if len(pts_px) > 2:
                        cv2.polylines(frame, [np.array(pts_px)], True, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "NO CALIBRATION LOADED", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        # --- APPLY DIGITAL ZOOM ---
        disp_frame = get_cam_view(frame)

        # --- HUD ---
        cv2.putText(disp_frame, f"CAM {CURRENT_CAM_IDX} [{'ACTIVE'}]", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(disp_frame, f"Zoom: x{CAM_ZOOM:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disp_frame, f"Zones: {len(LED_CONES)}/{NUM_LEDS}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        cv2.imshow(win_name, disp_frame)

        # --- CONTROLS ---
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):  # Save
            save_data = [{"id": c["id"], "pos": c["pos"], "radius": c["radius"]} for c in LED_CONES]
            with open(get_config_path(ZONES_FILE), 'w') as f:
                json.dump(save_data, f, indent=4)
            messagebox.showinfo("Saved", f"Successfully saved {len(LED_CONES)} zones to {ZONES_FILE}")
            break

        # Camera Switch (TAB)
        elif k == 9:
            if len(CAMERA_INDICES) > 1:
                cam_list_idx = (cam_list_idx + 1) % len(CAMERA_INDICES)
                CURRENT_CAM_IDX = CAMERA_INDICES[cam_list_idx]
                print(f"[Info] Switching to Camera {CURRENT_CAM_IDX}...")
                cap.release()
                cap = cv2.VideoCapture(CURRENT_CAM_IDX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                # Reload H Matrix for new camera
                H_MATRIX = load_calibration(CURRENT_CAM_IDX)

        # Pan/Zoom
        elif k == ord('='):
            CAM_ZOOM = min(10.0, CAM_ZOOM + 0.5)
        elif k == ord('-'):
            CAM_ZOOM = max(1.0, CAM_ZOOM - 0.5)
        elif k == ord('w'):
            CAM_PAN[1] -= 50 / CAM_ZOOM
        elif k == ord('s'):
            CAM_PAN[1] += 50 / CAM_ZOOM
        elif k == ord('a'):
            CAM_PAN[0] -= 50 / CAM_ZOOM
        elif k == ord('d'):
            CAM_PAN[0] += 50 / CAM_ZOOM
        elif k == ord('z'):
            CAM_ZOOM = 1.0; CAM_PAN = [0, 0]  # Reset View

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_setup()