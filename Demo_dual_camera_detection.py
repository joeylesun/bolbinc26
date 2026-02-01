#!/usr/bin/env python3
"""
SUVOS Runtime Controller (Dual Camera) - Clustering Fix
- Update: Merges detections from multiple cameras into single "Centroid" dots.
- Logic: If points are within 1.5m of each other, they are the same person.
"""

import time
import cv2
import numpy as np
import serial
from ultralytics import YOLO
from threading import Thread

# ================= CONFIGURATION =================
CAMERA_INDICES = [0, 1]
MAP_IMG_FILE = "room_map.png"
SERIAL_PORT = "/dev/cu.usbserial-120"
BAUD_RATE = 115200

# App Logic
NUM_LEDS = 16
CONE_RADIUS_M = 2.0
CONF_THRES = 0.65
MERGE_DISTANCE_M = 1.5  # <--- NEW: Max distance to merge two dots
HEARTBEAT_INTERVAL = 0.10
FSM_TIMES = {"waitingTime": 10.0, "disinfectionTime": 30.0, "standbyTime": 10.0}

# =================================================

# --- DATA STRUCTURES ---
CAM_DATA = {}


# --- THREADED CAMERA ---
class CameraStream:
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True;
        self.stream.release()


# --- MATH HELPERS ---
MAP_SCALE = 1.0;
MAP_OFF_X = 0.0;
MAP_OFF_Y = 0.0;
MAP_ROTATION = 0;
MAP_SIZE = 800


def pixel_to_world(u, v, cam_idx):
    H = CAM_DATA[cam_idx]['H']
    w_vec = H @ np.array([u, v, 1.0])
    if abs(w_vec[2]) < 1e-9: return (0, 0)
    return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])


def world_to_pixel(x, y, cam_idx):
    H_INV = CAM_DATA[cam_idx]['H_INV']
    w_vec = H_INV @ np.array([x, y, 1.0])
    if abs(w_vec[2]) < 1e-9: return None
    return int(round(w_vec[0] / w_vec[2])), int(round(w_vec[1] / w_vec[2]))


def world_to_map_pixel(wx, wy):
    mx = (wx - MAP_OFF_X) * MAP_SCALE
    my = MAP_SIZE - ((wy - MAP_OFF_Y) * MAP_SCALE)
    if MAP_ROTATION == 0:
        rx, ry = mx, my
    elif MAP_ROTATION == 1:
        rx, ry = MAP_SIZE - 1 - my, mx
    elif MAP_ROTATION == 2:
        rx, ry = MAP_SIZE - 1 - mx, MAP_SIZE - 1 - my
    elif MAP_ROTATION == 3:
        rx, ry = my, MAP_SIZE - 1 - mx
    return int(rx), int(ry)


def draw_floor_circle_on_camera(img, world_x, world_y, radius_m, color, cam_idx):
    pts_px = []
    for ang in np.linspace(0, 6.28, 16):
        wx = world_x + radius_m * np.cos(ang)
        wy = world_y + radius_m * np.sin(ang)
        px = world_to_pixel(wx, wy, cam_idx)
        if px: pts_px.append(px)
    if len(pts_px) > 2:
        cv2.polylines(img, [np.array(pts_px)], True, color, 2, cv2.LINE_AA)


# --- CLUSTERING ALGORITHM (NEW) ---
def get_merged_centroids(raw_points, threshold=1.5):
    """
    Groups points that are close together and returns their averages.
    Simple greedy clustering.
    """
    merged = []
    used = [False] * len(raw_points)

    for i in range(len(raw_points)):
        if used[i]: continue

        current_cluster_x, current_cluster_y = raw_points[i]
        cluster_count = 1
        used[i] = True

        # Check against all other points
        for j in range(i + 1, len(raw_points)):
            if used[j]: continue

            px, py = raw_points[j]
            # Calculate distance to the SEED point of the cluster
            dist = np.sqrt((raw_points[i][0] - px) ** 2 + (raw_points[i][1] - py) ** 2)

            if dist < threshold:
                current_cluster_x += px
                current_cluster_y += py
                cluster_count += 1
                used[j] = True

        # Calculate Average
        avg_x = current_cluster_x / cluster_count
        avg_y = current_cluster_y / cluster_count
        merged.append((avg_x, avg_y))

    return merged


# --- FSM ---
class _State: WAIT = 0; CLEAN = 1; STBY = 2


class _LedFSM:
    def __init__(self):
        self.state = _State.WAIT; self.timer = 0.0

    def update(self, personDetected, dt):
        cfg = FSM_TIMES
        if self.state == _State.WAIT:
            if personDetected:
                self.timer = 0.0
            else:
                self.timer = min(self.timer + dt, cfg["waitingTime"])
                if self.timer >= cfg["waitingTime"]: self.state = _State.CLEAN; self.timer = 0
        elif self.state == _State.CLEAN:
            if personDetected:
                self.state = _State.WAIT; self.timer = 0
            else:
                self.timer = min(self.timer + dt, cfg["disinfectionTime"])
                if self.timer >= cfg["disinfectionTime"]: self.state = _State.STBY; self.timer = 0
        elif self.state == _State.STBY:
            if personDetected:
                self.state = _State.WAIT; self.timer = 0
            else:
                self.timer = min(self.timer + dt, cfg["standbyTime"])
                if self.timer >= cfg["standbyTime"]: self.state = _State.WAIT
        return self.state

    def get_status_info(self):
        cfg = FSM_TIMES
        if self.state == _State.WAIT: return "WAIT", f"{self.timer:.1f}/{cfg['waitingTime']:.0f}"
        if self.state == _State.CLEAN: return "CLEAN", f"{self.timer:.1f}/{cfg['disinfectionTime']:.0f}"
        return "STBY", f"{self.timer:.1f}/{cfg['standbyTime']:.0f}"


# --- HUD ---
def draw_dashboard(img, led_cones, led_fsms):
    h, w = img.shape[:2]
    panel_h = 40 + (len(led_cones) * 30)
    x_start = w - 260
    roi = img[0:panel_h, x_start:w];
    black = np.zeros_like(roi)
    res = cv2.addWeighted(roi, 0.7, black, 0.3, 0)
    img[0:panel_h, x_start:w] = res
    cv2.putText(img, "SUVOS STATUS", (x_start + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, cone in enumerate(led_cones):
        idx = cone["id"]
        fsm = led_fsms[idx]
        status, timer = fsm.get_status_info()
        col = (0, 255, 0) if fsm.state == _State.CLEAN else (
            (200, 200, 200) if fsm.state == _State.STBY else (0, 255, 255))
        cv2.putText(img, f"SUVOS{idx}: {status} [{timer}s]", (x_start + 10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1)


# --- MAIN ---
def main():
    global CAM_DATA, MAP_SCALE, MAP_OFF_X, MAP_OFF_Y, MAP_ROTATION, MAP_SIZE

    # 1. LOAD DATA
    try:
        map_bg_original = cv2.imread(MAP_IMG_FILE)
        if map_bg_original is None: raise Exception("Map Img Missing")
        for idx in CAMERA_INDICES:
            fname = f"calibration_data_cam{idx}.npz"
            d = np.load(fname)
            CAM_DATA[idx] = {'H': d['H'], 'H_INV': np.linalg.inv(d['H'])}
            if idx == CAMERA_INDICES[0]:
                MAP_SCALE = float(d['scale']);
                MAP_OFF_X = float(d['off_x']);
                MAP_OFF_Y = float(d['off_y']);
                MAP_ROTATION = int(d['rotation']);
                MAP_SIZE = int(d['map_size'])
        print("[Init] Calibration Loaded.")
    except Exception as e:
        print(f"[Error] {e}. Run calibration first."); return

    # 2. INIT
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    except:
        ser = None; print("[HW] No Serial")
    model = YOLO("yolov8n.pt")

    # 3. START CAMERAS
    cams = {}
    for idx in CAMERA_INDICES: cams[idx] = CameraStream(idx).start()
    time.sleep(1.0)

    LED_CONES = [];
    LED_FSMS = {};
    SYSTEM_ACTIVE = False

    def make_click_handler(cam_idx):
        def on_click(event, x, y, flags, param):
            if not SYSTEM_ACTIVE:
                if event == cv2.EVENT_LBUTTONDOWN and len(LED_CONES) < NUM_LEDS:
                    wx, wy = pixel_to_world(x, y, cam_idx)
                    idx = len(LED_CONES)
                    LED_CONES.append({"pos": (wx, wy), "id": idx})
                    LED_FSMS[idx] = _LedFSM()
                    print(f"[Setup] Placed SUVOS{idx} via Cam {cam_idx}")

        return on_click

    for idx in CAMERA_INDICES:
        win_name = f"Cam {idx}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, make_click_handler(idx))

    print("\n--- PHASE 1: SETUP ---")
    print(" - Click ANY Camera to place units.")
    print(" - Press 'c' to START SYSTEM.")

    last_time = time.time();
    last_bits = tuple();
    last_send = 0

    while True:
        frames = {}
        for idx in CAMERA_INDICES:
            _, f = cams[idx].read()
            frames[idx] = f if f is not None else np.zeros((1080, 1920, 3), np.uint8)

        dt = time.time() - last_time;
        last_time = time.time()

        if not SYSTEM_ACTIVE:
            for idx in CAMERA_INDICES:
                frm = frames[idx]
                for cone in LED_CONES:
                    wx, wy = cone["pos"]
                    draw_floor_circle_on_camera(frm, wx, wy, CONE_RADIUS_M, (0, 255, 255), idx)
                    c_px = world_to_pixel(wx, wy, idx)
                    if c_px: cv2.putText(frm, f"S{cone['id']}", c_px, 0, 0.6, (0, 255, 255), 2)
                cv2.putText(frm, "SETUP: Press 'c' to Start", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(f"Cam {idx}", frm)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('c'):
                SYSTEM_ACTIVE = True
                cv2.namedWindow("Digital Twin", cv2.WINDOW_NORMAL)
                print("\n--- PHASE 2: SYSTEM ACTIVE ---")

        else:
            disp_map = map_bg_original.copy()
            map_overlay = disp_map.copy()
            occupied = [False] * NUM_LEDS

            # RAW DETECTIONS LIST (Before Merging)
            # Format: (world_x, world_y)
            all_raw_detections = []

            for idx in CAMERA_INDICES:
                frm = frames[idx]
                results = model(frm, verbose=False, conf=CONF_THRES)

                for r in results:
                    boxes = r.boxes
                    if boxes is None: continue
                    for box in boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # 1. FEET LOGIC
                            feet_x = (x1 + x2) // 2
                            feet_y = y2
                            pwx, pwy = pixel_to_world(feet_x, feet_y, idx)

                            # Store raw detection for clustering
                            all_raw_detections.append((pwx, pwy))

                            # 2. SAFETY LOGIC (Conservative - Use Raw)
                            h = y2 - y1
                            y_start = int(y2 - 0.30 * h)
                            hit_circle = False
                            x_grid = np.linspace(x1, x2, 5)
                            y_grid = np.linspace(y_start, y2, 3)

                            for u in x_grid:
                                for v in y_grid:
                                    grid_wx, grid_wy = pixel_to_world(u, v, idx)
                                    for cone in LED_CONES:
                                        cwx, cwy = cone["pos"]
                                        if np.sqrt((grid_wx - cwx) ** 2 + (grid_wy - cwy) ** 2) < CONE_RADIUS_M:
                                            occupied[cone["id"]] = True
                                            hit_circle = True

                            col = (0, 0, 255) if hit_circle else (0, 255, 0)
                            cv2.rectangle(frm, (x1, y_start), (x2, y2), col, 2)
                            cv2.circle(frm, (feet_x, feet_y), 5, (255, 0, 255), -1)

            # --- SENSOR FUSION STEP ---
            # Merge dots that are close (same person seen by 2 cams)
            merged_detections = get_merged_centroids(all_raw_detections, MERGE_DISTANCE_M)

            # Update Hardware
            bits = [0] * NUM_LEDS
            for cone in LED_CONES:
                idx = cone["id"]
                state = LED_FSMS[idx].update(occupied[idx], dt)
                if state == 1: bits[idx] = 1
            if tuple(bits) != last_bits or (time.time() - last_send > HEARTBEAT_INTERVAL):
                if ser:
                    try:
                        ser.write(("B:" + ",".join(map(str, bits)) + "\n").encode())
                    except:
                        pass
                last_bits = tuple(bits);
                last_send = time.time()

            # --- DRAWING ZONES (Background) ---
            for cone in LED_CONES:
                cid = cone["id"];
                wx, wy = cone["pos"];
                fsm = LED_FSMS[cid]
                if fsm.state == 1:
                    col = (0, 255, 0)
                elif fsm.state == 2:
                    col = (200, 200, 200)
                else:
                    col = (0, 255, 255)
                if occupied[cid]: col = (0, 0, 255)

                mx, my = world_to_map_pixel(wx, wy)
                r_px = int(CONE_RADIUS_M * MAP_SCALE)
                cv2.circle(map_overlay, (mx, my), r_px, col, -1)
                cv2.circle(map_overlay, (mx, my), r_px, col, 2)
                cv2.putText(map_overlay, f"S{cid}", (mx - 10, my + 5), 0, 0.5, (0, 0, 0), 2)
                for cam_idx in CAMERA_INDICES:
                    draw_floor_circle_on_camera(frames[cam_idx], wx, wy, CONE_RADIUS_M, col, cam_idx)
                    c_px = world_to_pixel(wx, wy, cam_idx)
                    if c_px: cv2.putText(frames[cam_idx], f"S{cid}", c_px, 0, 0.5, col, 2)

            # --- DRAWING MERGED PEOPLE (Foreground) ---
            for (pwx, pwy) in merged_detections:
                mx, my = world_to_map_pixel(pwx, pwy)
                if 0 <= mx < MAP_SIZE:
                    # White Ring + Black Dot
                    cv2.circle(map_overlay, (mx, my), 9, (255, 255, 255), -1)
                    cv2.circle(map_overlay, (mx, my), 7, (0, 0, 0), -1)

            cv2.addWeighted(map_overlay, 0.4, disp_map, 0.6, 0, disp_map)

            for cam_idx in CAMERA_INDICES:
                draw_dashboard(frames[cam_idx], LED_CONES, LED_FSMS)
                cv2.imshow(f"Cam {cam_idx}", frames[cam_idx])
            draw_dashboard(disp_map, LED_CONES, LED_FSMS)
            cv2.imshow("Digital Twin", disp_map)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break

    for c in cams.values(): c.stop()
    cv2.destroyAllWindows()
    if ser: ser.close()


if __name__ == "__main__":
    main()