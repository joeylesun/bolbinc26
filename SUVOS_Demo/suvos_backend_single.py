#!/usr/bin/env python3
"""
SUVOS Runtime Controller (Final Stable Version)
- Logic: REVERTED to "Frame Intersection" (Your working code).
- Visualization: Camera Window + Frontend Auto-Launch.
- Packaging: Ready for PyInstaller.
"""

import time
import cv2
import numpy as np
import serial
from ultralytics import YOLO
from threading import Thread
import asyncio
import websockets
import json
import os
import sys
import webbrowser

# ================= CONFIGURATION =================
SETUP_MODE = False
CAMERA_INDEX = 0
SERIAL_PORT = "/dev/cu.usbserial-120"
BAUD_RATE = 115200

# Network
WS_HOST = "localhost"
WS_PORT = 8765
ZONES_FILE = "led_zones.json"
SHAPE_FILE = "room_shape.json"

# App Logic
NUM_LEDS = 16
CONE_RADIUS_M = 3.0
CONF_THRES = 0.4
MERGE_DISTANCE_M = 1.5
HEARTBEAT_INTERVAL = 0.10
FSM_TIMES = {"waitingTime": 5.0, "disinfectionTime": 30.0, "standbyTime": 10.0}
# =================================================

CAM_DATA = {}
CLIENTS = set()
GLOBAL_INIT_PACKET = {}


# --- PACKAGING HELPER ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


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


def pixel_to_world(u, v):
    if 'H' not in CAM_DATA: return (0, 0)
    w_vec = CAM_DATA['H'] @ np.array([u, v, 1.0])
    if abs(w_vec[2]) < 1e-9: return (0, 0)
    return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])


def world_to_pixel(x, y):
    if 'H_INV' not in CAM_DATA: return None
    w_vec = CAM_DATA['H_INV'] @ np.array([x, y, 1.0])
    if abs(w_vec[2]) < 1e-9: return None
    return int(round(w_vec[0] / w_vec[2])), int(round(w_vec[1] / w_vec[2]))


def draw_floor_circle_on_camera(img, world_x, world_y, radius_m, color):
    pts_px = []
    for ang in np.linspace(0, 6.28, 16):
        wx = world_x + radius_m * np.cos(ang)
        wy = world_y + radius_m * np.sin(ang)
        px = world_to_pixel(wx, wy)
        if px: pts_px.append(px)
    if len(pts_px) > 2:
        cv2.polylines(img, [np.array(pts_px)], True, color, 2, cv2.LINE_AA)


# --- CLUSTERING ---
def get_merged_centroids(raw_points, threshold=1.5):
    merged = []
    used = [False] * len(raw_points)
    for i in range(len(raw_points)):
        if used[i]: continue
        cx, cy = raw_points[i]
        count = 1;
        used[i] = True
        for j in range(i + 1, len(raw_points)):
            if used[j]: continue
            dist = np.sqrt((raw_points[i][0] - raw_points[j][0]) ** 2 + (raw_points[i][1] - raw_points[j][1]) ** 2)
            if dist < threshold:
                cx += raw_points[j][0];
                cy += raw_points[j][1];
                count += 1;
                used[j] = True
        merged.append((cx / count, cy / count))
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
        if idx in led_fsms:
            fsm = led_fsms[idx]
            status, timer = fsm.get_status_info()
            col = (0, 255, 0) if fsm.state == _State.CLEAN else (
                (200, 200, 200) if fsm.state == _State.STBY else (0, 255, 255))
            cv2.putText(img, f"SUVOS{idx}: {status} [{timer}s]", (x_start + 10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, 1)


# --- WEBSOCKET ---
async def register(websocket):
    CLIENTS.add(websocket)
    try:
        if GLOBAL_INIT_PACKET:
            await websocket.send(json.dumps(GLOBAL_INIT_PACKET))
        else:
            await websocket.send(json.dumps({"type": "init", "zones": []}))
        await websocket.wait_closed()
    finally:
        CLIENTS.remove(websocket)


async def broadcast_state(people, zones_status):
    if not CLIENTS: return
    # CRITICAL FIX: Format as list of dicts {"x":val, "y":val} for frontend
    formatted_people = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in people]
    msg = json.dumps({"type": "update", "people": formatted_people, "led_states": zones_status})
    await asyncio.gather(*[c.send(msg) for c in CLIENTS], return_exceptions=True)


# --- MAIN LOOP ---
LED_CONES_CACHE = []


async def main_loop():
    global CAM_DATA, MAP_SCALE, MAP_OFF_X, MAP_OFF_Y, MAP_ROTATION, MAP_SIZE, LED_CONES_CACHE, GLOBAL_INIT_PACKET

    # 1. LOAD CALIBRATION
    try:
        fname = resource_path(f"calibration_data_cam{CAMERA_INDEX}.npz")
        d = np.load(fname)
        CAM_DATA = {'H': d['H'], 'H_INV': np.linalg.inv(d['H'])}
        MAP_SCALE = float(d['scale']);
        MAP_OFF_X = float(d['off_x']);
        MAP_OFF_Y = float(d['off_y']);
        MAP_ROTATION = int(d['rotation']);
        MAP_SIZE = int(d['map_size'])
        print("[Init] Calibration Loaded.")
    except Exception as e:
        print(f"[Error] {e}. Check calibration file.");
        return

    # 2. INIT
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    except:
        ser = None; print("[HW] No Serial")

    model = YOLO(resource_path("yolov8n.pt"))
    cam = CameraStream(CAMERA_INDEX).start()
    await asyncio.sleep(1.0)

    LED_CONES = []
    LED_FSMS = {}

    print("\n--- MODE: WORK (Visual) ---")

    zfile = resource_path(ZONES_FILE)
    if os.path.exists(zfile):
        with open(zfile, 'r') as f:
            loaded = json.load(f)
            for l in loaded:
                LED_CONES.append({"id": l["id"], "pos": tuple(l["pos"]), "radius": l.get("radius", CONE_RADIUS_M)})
                LED_FSMS[l["id"]] = _LedFSM()
    else:
        print(f"[Error] {ZONES_FILE} missing."); return

    sfile = resource_path(SHAPE_FILE)
    ROOM_SHAPE = []
    if os.path.exists(sfile):
        with open(sfile, 'r') as f: ROOM_SHAPE = json.load(f)

    GLOBAL_INIT_PACKET = {
        "type": "init",
        "config": {
            "scale": MAP_SCALE, "off_x": MAP_OFF_X, "off_y": MAP_OFF_Y,
            "map_size": MAP_SIZE, "rotation": MAP_ROTATION
        },
        "room_shape": ROOM_SHAPE,
        "zones": [{"id": c["id"], "x": c["pos"][0], "y": c["pos"][1], "radius": c.get("radius", CONE_RADIUS_M)} for c in
                  LED_CONES],
        "fsm_config": FSM_TIMES
    }

    last_time = time.time();
    last_bits = tuple();
    last_send = 0

    # --- VISUALIZATION ENABLED ---
    cv2.namedWindow("Single Cam", cv2.WINDOW_NORMAL)
    browser_launched = False

    while True:
        _, frm = cam.read()
        if frm is None: frm = np.zeros((1080, 1920, 3), np.uint8)

        dt = time.time() - last_time;
        last_time = time.time()
        occupied = [False] * NUM_LEDS
        all_raw_detections = []

        # 1. YOLO
        results = model(frm, verbose=False, conf=CONF_THRES)
        img_h, img_w = frm.shape[:2]

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2

                    # --- REVERTED TO YOUR WORKING LOGIC ---
                    # Only check for frame bottom intersection
                    if y2 >= img_h - 15:
                        feet_y = img_h - 1
                        cv2.line(frm, (x1, img_h - 5), (x2, img_h - 5), (0, 0, 255), 2)
                    else:
                        feet_y = y2
                    # --------------------------------------

                    pwx, pwy = pixel_to_world(cx, feet_y)
                    all_raw_detections.append((pwx, pwy))

                    for cone in LED_CONES:
                        cwx, cwy = cone["pos"]
                        if np.sqrt((pwx - cwx) ** 2 + (pwy - cwy) ** 2) < cone.get("radius", CONE_RADIUS_M):
                            occupied[cone["id"]] = True

                    # VISUAL: Box and Dot
                    cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frm, (cx, feet_y), 5, (255, 0, 255), -1)

        # 2. MERGE & FSM
        merged_detections = get_merged_centroids(all_raw_detections, MERGE_DISTANCE_M)
        bits = [0] * NUM_LEDS
        led_states_for_ws = []

        for cone in LED_CONES:
            idx = cone["id"]
            fsm = LED_FSMS[idx]
            state = fsm.update(occupied[idx], dt)
            if state == 1: bits[idx] = 1

            status_str = "WAIT" if state == 0 else ("CLEAN" if state == 1 else "STBY")
            total_time = FSM_TIMES["waitingTime"] if state == 0 else (
                FSM_TIMES["disinfectionTime"] if state == 1 else FSM_TIMES["standbyTime"])

            led_states_for_ws.append({
                "id": idx, "state": status_str,
                "timer": round(max(0, total_time - fsm.timer), 1),
                "occupied": occupied[idx]
            })

        # 3. OUTPUT
        if tuple(bits) != last_bits or (time.time() - last_send > HEARTBEAT_INTERVAL):
            if ser:
                try:
                    ser.write(("B:" + ",".join(map(str, bits)) + "\n").encode())
                except:
                    pass
            last_bits = tuple(bits);
            last_send = time.time()

        await broadcast_state(merged_detections, led_states_for_ws)

        # 4. VISUALIZATION
        for cone in LED_CONES:
            wx, wy = cone["pos"];
            r = cone.get("radius", CONE_RADIUS_M)
            fsm = LED_FSMS[cone["id"]]
            col = (0, 255, 255) if fsm.state == 0 else ((0, 255, 0) if fsm.state == 1 else (200, 200, 200))
            if occupied[cone["id"]]: col = (0, 0, 255)
            draw_floor_circle_on_camera(frm, wx, wy, r, col)

        draw_dashboard(frm, LED_CONES, LED_FSMS)
        cv2.imshow("Single Cam", frm)

        # --- LAUNCH FRONTEND ---
        if not browser_launched:
            print("[System] Camera started. Launching frontend...")
            html_file = resource_path("interface/index.html")
            webbrowser.open("file://" + html_file)
            browser_launched = True
        # -----------------------

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        await asyncio.sleep(0.01)

    cam.stop()
    cv2.destroyAllWindows()
    if ser: ser.close()


if __name__ == "__main__":
    async def main():
        print(f"[System] WebSocket server on ws://{WS_HOST}:{WS_PORT}")
        async with websockets.serve(register, WS_HOST, WS_PORT):
            await main_loop()


    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("[System] Stopped.")