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

# >>> NEW: IMPORTS FOR LOGGING <<<
import sqlite3
from datetime import datetime

# Import Shared Config
from suvos_common import (
    resource_path, get_calib_file,
    ZONES_FILE, SHAPE_FILE, MODEL_FILE,
    CAMERA_INDICES, SERIAL_PORT, BAUD_RATE,
    WS_HOST, WS_PORT, CAM_WIDTH, CAM_HEIGHT,
    NUM_LEDS, CONE_RADIUS_M, CONF_THRES, MERGE_DISTANCE_M,
    HEARTBEAT_INTERVAL, load_fsm_config, FSM_TIMES, get_config_path
)


# --- THREADED CAMERA STREAM ---
class CameraStream:
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)

        # Force the USB UVC driver to compress the stream to MJPEG
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set dimensions AFTER setting the codec
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # >>> CHANGED: BUMPED TO 30 FPS <<<
        self.stream.set(cv2.CAP_PROP_FPS, 30)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True

# --- FSM STATES ---
class _State: WAIT = 0; CLEAN = 1; STBY = 2

class _LedFSM:
    def __init__(self):
        self.state = _State.WAIT
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
                self.state = _State.WAIT;
                self.timer = 0
            else:
                self.timer = min(self.timer + dt, cfg["disinfectionTime"])
                if self.timer >= cfg["disinfectionTime"]: self.state = _State.STBY; self.timer = 0
        elif self.state == _State.STBY:
            if personDetected:
                self.state = _State.WAIT;
                self.timer = 0
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
        self.cam_streams = {}
        self.led_cones = []
        self.led_fsms = {}
        self.global_init_packet = {}
        self.latest_frame = None
        self.frame_lock = Lock()

        # >>> NEW: DATABASE SETUP <<<
        self.db_path = os.path.expanduser('~/Documents/SUVOS_Logs.db')
        self._init_database()

    # >>> NEW: DATABASE METHODS <<<
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uvc_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                zone_id INTEGER,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _log_event(self, zone_id, status):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT INTO uvc_logs (timestamp, zone_id, status)
                VALUES (?, ?, ?)
            ''', (current_time, zone_id, status))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Database Error] Failed to log event: {e}")

    def start(self):
        if self.running: return
        self.running = True
        self.thread = Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        for stream in self.cam_streams.values():
            stream.stop()
        print("[System] Stopped.")

    def get_processed_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    # --- MATH ---
    def _pixel_to_world(self, u, v, cam_idx):
        if cam_idx not in self.cam_data: return (0, 0)
        H = self.cam_data[cam_idx]['H']
        w_vec = H @ np.array([u, v, 1.0])
        if abs(w_vec[2]) < 1e-9: return (0, 0)
        return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])

    def _get_merged_centroids(self, raw_points, threshold=MERGE_DISTANCE_M):
        merged = []
        used = [False] * len(raw_points)
        for i in range(len(raw_points)):
            if used[i]: continue
            cx, cy = raw_points[i]
            count = 1
            used[i] = True
            for j in range(i + 1, len(raw_points)):
                if used[j]: continue
                dist = np.sqrt((raw_points[i][0] - raw_points[j][0]) ** 2 + (raw_points[i][1] - raw_points[j][1]) ** 2)
                if dist < threshold:
                    cx += raw_points[j][0]
                    cy += raw_points[j][1]
                    count += 1
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

        try:
            for idx in CAMERA_INDICES:
                try:
                    d = np.load(get_config_path(get_calib_file(idx)))
                    self.cam_data[idx] = {'H': d['H'], 'H_INV': np.linalg.inv(d['H'])}
                    if idx == PRIMARY_CAM:
                        self.global_init_packet = {
                            "type": "init",
                            "config": {"scale": float(d['scale']), "off_x": float(d['off_x']),
                                       "off_y": float(d['off_y']),
                                       "map_size": int(d['map_size']), "rotation": int(d['rotation'])},
                            "fsm_config": FSM_TIMES
                        }
                except Exception as e:
                    print(f"[Warning] Calibration missing for cam {idx}: {e}")

            with open(get_config_path(ZONES_FILE), 'r') as f:
                raw_zones = json.load(f)
                self.led_cones = [{"id": z["id"], "pos": tuple(z["pos"]), "radius": z.get("radius", CONE_RADIUS_M)} for
                                  z in raw_zones]
                self.led_fsms = {z["id"]: _LedFSM() for z in raw_zones}

            shape_path = get_config_path(SHAPE_FILE)
            ROOM_SHAPE = json.load(open(shape_path, 'r')) if os.path.exists(shape_path) else []
            self.global_init_packet["room_shape"] = ROOM_SHAPE
            self.global_init_packet["zones"] = [
                {"id": c["id"], "x": c["pos"][0], "y": c["pos"][1], "radius": c["radius"]} for c in self.led_cones]

        except Exception as e:
            print(f"[Error] Config loading failed: {e}")

        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
        except:
            ser = None

        model = YOLO(resource_path(MODEL_FILE))

        for idx in CAMERA_INDICES:
            self.cam_streams[idx] = CameraStream(idx).start()

        server = await websockets.serve(self._register, WS_HOST, WS_PORT)
        webbrowser.open("file://" + resource_path("interface/index.html"))

        last_time = time.time()
        last_send = 0
        last_bits = tuple()

        # >>> CHANGED: BUMPED TO 30 FPS <<<
        target_fps = 30
        frame_interval = 1.0 / target_fps


        # 3. MAIN ASYNC LOOP
        while self.running:
            # >>> FIXED: INITIALIZE LOOP START TIME FOR THE LIMITER MATH <<<
            loop_start = time.time()

            frames = {}
            for idx in CAMERA_INDICES:
                ret, frm = self.cam_streams[idx].read()
                frames[idx] = frm if ret and frm is not None else np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)

            dt = time.time() - last_time
            last_time = time.time()

            occupied = {z["id"]: False for z in self.led_cones}
            all_raw_detections = []

            for cam_idx in CAMERA_INDICES:
                frame = frames[cam_idx]
                img_h, img_w = frame.shape[:2]

                results = model(frame, verbose=False, conf=CONF_THRES)
                for r in results:
                    if r.boxes:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx = (x1 + x2) // 2
                                feet_y = y2 if y2 < img_h - 15 else img_h - 1

                                pwx, pwy = self._pixel_to_world(cx, feet_y, cam_idx)
                                all_raw_detections.append((pwx, pwy))

                                for cone in self.led_cones:
                                    cwx, cwy = cone["pos"]
                                    if np.sqrt((pwx - cwx) ** 2 + (pwy - cwy) ** 2) < cone["radius"]:
                                        occupied[cone["id"]] = True

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(frame, (cx, feet_y), 5, (255, 0, 255), -1)

                for cone in self.led_cones:
                    if cam_idx in self.cam_data:
                        wx, wy = cone["pos"]
                        idx_id = cone["id"]
                        w_vec = self.cam_data[cam_idx]['H_INV'] @ np.array([wx, wy, 1.0])
                        if abs(w_vec[2]) > 1e-9:
                            px, py = int(w_vec[0] / w_vec[2]), int(w_vec[1] / w_vec[2])
                            col = (0, 0, 255) if occupied[idx_id] else (255, 255, 0)
                            cv2.circle(frame, (px, py), 10, col, -1)
                            cv2.putText(frame, f"S{idx_id}", (px, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            merged_centroids = self._get_merged_centroids(all_raw_detections, MERGE_DISTANCE_M)

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

            # >>> NEW: SQLITE DATABASE LOGGING LOGIC <<<
            if last_bits and len(last_bits) == NUM_LEDS:
                for i in range(NUM_LEDS):
                    if bits[i] != last_bits[i]:
                        if bits[i] == 1:
                            log_msg = "ON (CLEANING)"
                        else:
                            person_in_zone = occupied.get(i, False)
                            log_msg = "OFF (PERSON DETECTED)" if person_in_zone else "OFF (STANDBY TIMEOUT)"
                        self._log_event(i, log_msg)

            display_frames = []
            target_h = 400
            for idx in CAMERA_INDICES:
                frm = frames[idx]
                h, w = frm.shape[:2]
                scale = target_h / h
                resized = cv2.resize(frm, (int(w * scale), target_h))
                cv2.putText(resized, f"CAM {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                display_frames.append(resized)

            composite = np.hstack(display_frames)

            with self.frame_lock:
                self.latest_frame = composite

            await self._broadcast_state(merged_centroids, led_states_ws)

            if tuple(bits) != last_bits or (time.time() - last_send > HEARTBEAT_INTERVAL):
                if ser:
                    try:
                        ser.write(("B:" + ",".join(map(str, bits)) + "\n").encode())
                    except:
                        pass
                last_bits = tuple(bits)
                last_send = time.time()

            # >>> FIXED: REMOVED DUPLICATE BROADCAST/SERIAL BLOCK THAT WAS HERE <<<

            # --- THE FRAME LIMITER EDIT ---
            elapsed_time = time.time() - loop_start
            sleep_time = frame_interval - elapsed_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.001)

        if ser: ser.close()