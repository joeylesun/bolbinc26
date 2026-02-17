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
import time
import webbrowser
from tkinter import messagebox

# Import Shared Config
from suvos_common import (
    resource_path, get_calib_file,
    ZONES_FILE, SHAPE_FILE, MODEL_FILE,
    CAMERA_INDICES, SERIAL_PORT, BAUD_RATE,
    WS_HOST, WS_PORT,
    NUM_LEDS, CONE_RADIUS_M, CONF_THRES, MERGE_DISTANCE_M,
    HEARTBEAT_INTERVAL, FSM_TIMES, get_config_path
)

# ================= STATE GLOBALS =================
CAM_DATA = {}
CLIENTS = set()
GLOBAL_INIT_PACKET = {}


# --- THREADED CAMERA (High Performance for Detection) ---
class CameraStream:
    def __init__(self, src=0):
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
def pixel_to_world(u, v):
    if 'H' not in CAM_DATA: return (0, 0)
    w_vec = CAM_DATA['H'] @ np.array([u, v, 1.0])
    if abs(w_vec[2]) < 1e-9: return (0, 0)
    return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])


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
            status = "WAIT" if fsm.state == 0 else ("CLEAN" if fsm.state == 1 else "STBY")
            rem_time = FSM_TIMES["waitingTime"] if fsm.state == 0 else (
                FSM_TIMES["disinfectionTime"] if fsm.state == 1 else FSM_TIMES["standbyTime"])
            timer = round(max(0, rem_time - fsm.timer), 1)
            col = (0, 255, 0) if fsm.state == 1 else ((200, 200, 200) if fsm.state == 2 else (0, 255, 255))
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
    formatted_people = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in people]
    msg = json.dumps({"type": "update", "people": formatted_people, "led_states": zones_status})
    await asyncio.gather(*[c.send(msg) for c in CLIENTS], return_exceptions=True)


# --- MAIN LOOP ---
async def run_detection_loop():
    global CAM_DATA, GLOBAL_INIT_PACKET

    # Use Primary Camera for Single-Cam Mode
    PRIMARY_CAM = CAMERA_INDICES[0]

    # 1. LOAD CONFIG
    try:
        # Load Calibration
        fname = get_config_path(get_calib_file(PRIMARY_CAM))
        d = np.load(fname)
        CAM_DATA = {'H': d['H'], 'H_INV': np.linalg.inv(d['H'])}

        # Load Zones
        with open(get_config_path(ZONES_FILE), 'r') as f:
            raw_zones = json.load(f)
            LED_CONES = [{"id": z["id"], "pos": tuple(z["pos"]), "radius": z.get("radius", CONE_RADIUS_M)} for z in
                         raw_zones]
            LED_FSMS = {z["id"]: _LedFSM() for z in raw_zones}

        # Load Shape
        shape_path = get_config_path(SHAPE_FILE)
        ROOM_SHAPE = []
        if os.path.exists(shape_path):
            with open(shape_path, 'r') as f: ROOM_SHAPE = json.load(f)

        GLOBAL_INIT_PACKET = {
            "type": "init",
            "config": {
                "scale": float(d['scale']), "off_x": float(d['off_x']),
                "off_y": float(d['off_y']), "map_size": int(d['map_size']), "rotation": int(d['rotation'])
            },
            "room_shape": ROOM_SHAPE,
            "zones": [{"id": c["id"], "x": c["pos"][0], "y": c["pos"][1], "radius": c["radius"]} for c in LED_CONES],
            "fsm_config": FSM_TIMES
        }
        print("[Init] Configuration Loaded.")
    except Exception as e:
        messagebox.showerror("Error", f"Files missing. Run Setup steps first!\n{e}")
        return

    # 2. HARDWARE INIT
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    except:
        ser = None; print("[HW] No Serial")

    model = YOLO(resource_path(MODEL_FILE))
    cam = CameraStream(PRIMARY_CAM).start()

    # 3. LAUNCH SERVER & FRONTEND
    print(f"[System] WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    server = await websockets.serve(register, WS_HOST, WS_PORT)

    time.sleep(1.0)
    html_file = resource_path("interface/index.html")
    webbrowser.open("file://" + html_file)
    print(f"[System] Launched Frontend.")

    cv2.namedWindow("Working Mode", cv2.WINDOW_NORMAL)
    last_time = time.time();
    last_bits = tuple();
    last_send = 0

    while True:
        ret, frm = cam.read()
        if not ret:
            print("Frame drop.");
            continue

        dt = time.time() - last_time;
        last_time = time.time()
        occupied = [False] * NUM_LEDS
        all_raw_detections = []
        img_h, img_w = frm.shape[:2]

        # YOLO
        results = model(frm, verbose=False, conf=CONF_THRES)
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2

                    # Logic: Frame Intersection
                    if y2 >= img_h - 15:
                        feet_y = img_h - 1
                        cv2.line(frm, (x1, img_h - 5), (x2, img_h - 5), (0, 0, 255), 2)
                    else:
                        feet_y = y2

                    pwx, pwy = pixel_to_world(cx, feet_y)
                    all_raw_detections.append((pwx, pwy))

                    for cone in LED_CONES:
                        cwx, cwy = cone["pos"]
                        if np.sqrt((pwx - cwx) ** 2 + (pwy - cwy) ** 2) < cone["radius"]:
                            occupied[cone["id"]] = True
                    cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frm, (cx, feet_y), 5, (255, 0, 255), -1)

        # Merge & FSM
        merged = get_merged_centroids(all_raw_detections, MERGE_DISTANCE_M)
        bits = [0] * NUM_LEDS
        led_states_ws = []

        for cone in LED_CONES:
            idx = cone["id"]
            fsm = LED_FSMS[idx]
            state = fsm.update(occupied[idx], dt)
            if state == 1: bits[idx] = 1

            status = "WAIT" if state == 0 else ("CLEAN" if state == 1 else "STBY")
            rem_time = FSM_TIMES["waitingTime"] if state == 0 else (
                FSM_TIMES["disinfectionTime"] if state == 1 else FSM_TIMES["standbyTime"])
            led_states_ws.append({
                "id": idx, "state": status, "timer": round(max(0, rem_time - fsm.timer), 1), "occupied": occupied[idx]
            })

        # Serial Output
        if tuple(bits) != last_bits or (time.time() - last_send > HEARTBEAT_INTERVAL):
            if ser:
                try:
                    ser.write(("B:" + ",".join(map(str, bits)) + "\n").encode())
                except:
                    pass
            last_bits = tuple(bits);
            last_send = time.time()

        # Visualization
        for cone in LED_CONES:
            wx, wy = cone["pos"];
            r = cone["radius"]
            # World -> Pixel
            w_vec = CAM_DATA['H_INV'] @ np.array([wx, wy, 1.0])
            cx, cy = int(w_vec[0] / w_vec[2]), int(w_vec[1] / w_vec[2])

            fsm = LED_FSMS[cone["id"]]
            col = (0, 255, 255) if fsm.state == 0 else ((0, 255, 0) if fsm.state == 1 else (200, 200, 200))
            if occupied[cone["id"]]: col = (0, 0, 255)

            # Draw Dot
            cv2.circle(frm, (cx, cy), 10, col, -1)
            cv2.putText(frm, f"S{cone['id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw Radius Ring
            pts_px = []
            for ang in np.linspace(0, 6.28, 16):
                pwx = wx + r * np.cos(ang);
                pwy = wy + r * np.sin(ang)
                w_vec = CAM_DATA['H_INV'] @ np.array([pwx, pwy, 1.0])
                pts_px.append((int(w_vec[0] / w_vec[2]), int(w_vec[1] / w_vec[2])))
            if len(pts_px) > 2: cv2.polylines(frm, [np.array(pts_px)], True, col, 2)

        draw_dashboard(frm, LED_CONES, LED_FSMS)
        cv2.imshow("Working Mode", frm)
        if cv2.waitKey(1) == ord('q'): break

        await broadcast_state(merged, led_states_ws)
        await asyncio.sleep(0.01)

    cam.stop()
    cv2.destroyAllWindows()
    if ser: ser.close()


# --- ENTRY POINT FOR LAUNCHER ---
def start_system():
    try:
        asyncio.run(run_detection_loop())
    except KeyboardInterrupt:
        pass
    finally:
        print("[System] Stopped.")