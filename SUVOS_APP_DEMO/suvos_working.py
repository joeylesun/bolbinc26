import cv2
import numpy as np
import serial
from ultralytics import YOLO
from threading import Thread, Lock
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
    WS_HOST, WS_PORT,CAM_WIDTH, CAM_HEIGHT,
    NUM_LEDS, CONE_RADIUS_M, CONF_THRES, MERGE_DISTANCE_M,
    HEARTBEAT_INTERVAL, load_fsm_config, FSM_TIMES, get_config_path
)


# --- FSM STATES ---
class _State: WAIT = 0; CLEAN = 1; STBY = 2


class _LedFSM:
    def __init__(self):
        self.state = _State.WAIT;
        self.timer = 0.0

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


# --- SYSTEM CONTROLLER ---
class SUVOS_System:
    def __init__(self):
        self.running = False
        self.thread = None
        self.clients = set()
        self.cam_data = {}
        self.led_cones = []
        self.led_fsms = {}
        self.global_init_packet = {}
        self.latest_frame = None
        self.frame_lock = Lock()

    def start(self):
        if self.running: return
        self.running = True
        self.thread = Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[System] Stopped.")

    def get_processed_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    # --- MATH ---
    def _pixel_to_world(self, u, v):
        if 'H' not in self.cam_data: return (0, 0)
        # Standard Homography Mapping
        w_vec = self.cam_data['H'] @ np.array([u, v, 1.0])
        if abs(w_vec[2]) < 1e-9: return (0, 0)
        return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])

    def _get_merged_centroids(self, raw_points, threshold=1.5):
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

    # --- WEBSOCKET ---
    async def _register(self, websocket):
        self.clients.add(websocket)
        try:
            if self.global_init_packet: await websocket.send(json.dumps(self.global_init_packet))
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def _broadcast_state(self, people, zones_status):
        if not self.clients: return
        # Format coords and ensure they are float, not numpy types
        formatted_people = [{"x": float(p[0]), "y": float(p[1])} for p in people]
        msg = json.dumps({"type": "update", "people": formatted_people, "led_states": zones_status})
        await asyncio.gather(*[c.send(msg) for c in self.clients], return_exceptions=True)

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._main_logic())

    async def _main_logic(self):
        PRIMARY_CAM = CAMERA_INDICES[0]
        new_times = load_fsm_config()
        FSM_TIMES.update(new_times)

        # 1. LOAD DATA
        try:
            # Calibration Data
            d = np.load(get_config_path(get_calib_file(PRIMARY_CAM)))
            self.cam_data = {'H': d['H'], 'H_INV': np.linalg.inv(d['H'])}

            # LED Zones
            with open(get_config_path(ZONES_FILE), 'r') as f:
                raw_zones = json.load(f)
                self.led_cones = [{"id": z["id"], "pos": tuple(z["pos"]), "radius": z.get("radius", CONE_RADIUS_M)} for
                                  z in raw_zones]
                self.led_fsms = {z["id"]: _LedFSM() for z in raw_zones}

            # Map Shape
            shape_path = get_config_path(SHAPE_FILE)
            ROOM_SHAPE = json.load(open(shape_path, 'r')) if os.path.exists(shape_path) else []

            self.global_init_packet = {
                "type": "init",
                "config": {"scale": float(d['scale']), "off_x": float(d['off_x']), "off_y": float(d['off_y']),
                           "map_size": int(d['map_size']), "rotation": int(d['rotation'])},
                "room_shape": ROOM_SHAPE,
                "zones": [{"id": c["id"], "x": c["pos"][0], "y": c["pos"][1], "radius": c["radius"]} for c in
                          self.led_cones],
                "fsm_config": FSM_TIMES
            }
        except Exception as e:
            print(f"[Error] Config missing: {e}")

        # 2. SETUP HARDWARE
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
        except:
            ser = None

        model = YOLO(resource_path(MODEL_FILE))

        # FORCE 1280x720 (Must match calibration!)
        cap = cv2.VideoCapture(PRIMARY_CAM)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        server = await websockets.serve(self._register, WS_HOST, WS_PORT)
        webbrowser.open("file://" + resource_path("interface/index.html"))

        last_time = time.time();
        last_send = 0;
        last_bits = tuple()

        # 3. LOOP
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue

            dt = time.time() - last_time;
            last_time = time.time()
            occupied = [False] * NUM_LEDS
            all_raw_detections = []
            img_h, img_w = frame.shape[:2]

            # AI Detection
            results = model(frame, verbose=False, conf=CONF_THRES)
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = (x1 + x2) // 2
                            feet_y = y2 if y2 < img_h - 15 else img_h - 1

                            pwx, pwy = self._pixel_to_world(cx, feet_y)
                            all_raw_detections.append((pwx, pwy))

                            for cone in self.led_cones:
                                cwx, cwy = cone["pos"]
                                if np.sqrt((pwx - cwx) ** 2 + (pwy - cwy) ** 2) < cone["radius"]:
                                    occupied[cone["id"]] = True

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, feet_y), 5, (255, 0, 255), -1)

            # Zone Logic
            merged = self._get_merged_centroids(all_raw_detections, MERGE_DISTANCE_M)
            bits = [0] * NUM_LEDS
            led_states_ws = []

            for cone in self.led_cones:
                idx = cone["id"]
                fsm = self.led_fsms[idx]
                state = fsm.update(occupied[idx], dt)
                if state == 1: bits[idx] = 1

                status = "WAIT" if state == 0 else ("CLEAN" if state == 1 else "STBY")
                rem = FSM_TIMES["waitingTime"] if state == 0 else (
                    FSM_TIMES["disinfectionTime"] if state == 1 else FSM_TIMES["standbyTime"])
                led_states_ws.append(
                    {"id": idx, "state": status, "timer": round(max(0, rem - fsm.timer), 1), "occupied": occupied[idx]})

                # Overlay Zones
                wx, wy = cone["pos"]
                w_vec = self.cam_data['H_INV'] @ np.array([wx, wy, 1.0])
                cx, cy = int(w_vec[0] / w_vec[2]), int(w_vec[1] / w_vec[2])
                col = (0, 0, 255) if occupied[idx] else ((0, 255, 0) if state == 1 else (255, 255, 0))
                cv2.circle(frame, (cx, cy), 10, col, -1)
                cv2.putText(frame, f"S{idx} {status}", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            # Update Frame for GUI
            with self.frame_lock:
                self.latest_frame = frame

            # Broadcast
            await self._broadcast_state(merged, led_states_ws)

            if tuple(bits) != last_bits or (time.time() - last_send > HEARTBEAT_INTERVAL):
                if ser:
                    try:
                        ser.write(("B:" + ",".join(map(str, bits)) + "\n").encode())
                    except:
                        pass
                last_bits = tuple(bits);
                last_send = time.time()

            await asyncio.sleep(0.001)

        cap.release()
        if ser: ser.close()