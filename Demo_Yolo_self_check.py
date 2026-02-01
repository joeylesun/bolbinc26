#!/usr/bin/env python3
"""
YOLO + FSM + Homography + STABILIZED SAFETY
Update:
1. COOLDOWN: Added a 2-second delay after people leave before resuming safety checks.
   - Allows camera Auto-Exposure/White Balance to settle.
   - Prevents false alarms from lighting drift.
2. ROBUSTNESS: Reduced false positives significantly.
"""

import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import serial
from ultralytics import YOLO

# ========================= CONFIG =========================

HOMOGRAPHY_MATRIX = np.array([
    [-1.55517530e-02, 9.06540391e-03, 1.28146652e+01],
    [-2.15564579e-03, 2.44741022e-02, -2.30101425e+01],
    [-4.71868525e-04, -5.03582778e-03, 1.00000000e+00]
])
H_INV = np.linalg.inv(HOMOGRAPHY_MATRIX)

SERIAL_PORT = "/dev/cu.usbserial-120"
BAUD_RATE = 115200
CAM_INDEX = 0
CONF_THRES = 0.4
NUM_LEDS = 16
CONE_RADIUS_M = 2
HEARTBEAT_INTERVAL = 0.10
WINDOW_NAME = "Real-World Controller"
MODEL_PATH = "yolov8n.pt"

# TIMING CONFIG
SAFETY_CHECK_INTERVAL = 0.5
FAILURE_PERSISTENCE = 5.0
CAMERA_STABILIZE_TIME = 2.0  # Wait 2s after person leaves before checking

FSM_TIMES = {
    "waitingTime": 10.0,
    "disinfectionTime": 300.0,
    "standbyTime": 120.0,
}


# ========================= SAFETY MONITOR =========================

class SafetyMonitor:
    def __init__(self):
        self.ref_frame = None
        self.ref_gray = None
        self.ref_hist = None
        self.ref_kp = None
        self.ref_des = None

        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # --- TUNED THRESHOLDS ---
        self.DARK_THRESH = 30
        self.HIST_THRESH = 0.65
        self.STRUCT_MATCH_RATIO = 0.25
        self.STRUCT_SHIFT_LIMIT = 50.0

    def set_reference(self, frame):
        self.ref_frame = frame.copy()
        self.ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.ref_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.ref_hist, self.ref_hist, 0, 1, cv2.NORM_MINMAX)

        self.ref_kp, self.ref_des = self.orb.detectAndCompute(self.ref_frame, None)
        print(f"[Safety] Reference Updated. Features: {len(self.ref_kp)}")

    def check(self, frame):
        if self.ref_frame is None: return True, "NO REF"

        # 1. BRIGHTNESS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val < self.DARK_THRESH:
            return False, f"TOO DARK ({mean_val:.1f})"

        # 2. COLOR (Histogram)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

        correlation = cv2.compareHist(self.ref_hist, curr_hist, cv2.HISTCMP_CORREL)
        if correlation < self.HIST_THRESH:
            return False, f"COLOR MISMATCH ({correlation:.2f})"

        # 3. STRUCTURE
        kp, des = self.orb.detectAndCompute(frame, None)
        if des is None or len(kp) < 5:
            return False, "NO FEATURES"

        matches = self.bf.match(self.ref_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.5)]

        match_ratio = len(good_matches) / len(self.ref_kp)
        if match_ratio < self.STRUCT_MATCH_RATIO:
            return False, f"STRUCTURAL LOSS ({match_ratio:.0%})"

        shifts = [np.linalg.norm(np.array(self.ref_kp[m.queryIdx].pt) - np.array(kp[m.trainIdx].pt)) for m in
                  good_matches]
        if shifts and np.median(shifts) > self.STRUCT_SHIFT_LIMIT:
            return False, f"CAMERA MOVED ({np.median(shifts):.1f}px)"

        return True, "SAFE"


# ========================= MATH HELPERS =========================

def pixel_to_world(u, v):
    p = np.array([u, v, 1.0])
    w_vec = HOMOGRAPHY_MATRIX @ p
    if abs(w_vec[2]) < 1e-9: return (0, 0)
    return (w_vec[0] / w_vec[2], w_vec[1] / w_vec[2])


def world_to_pixel(x, y):
    w_vec = H_INV @ np.array([x, y, 1.0])
    if abs(w_vec[2]) < 1e-9: return None
    u = int(round(w_vec[0] / w_vec[2]))
    v = int(round(w_vec[1] / w_vec[2]))
    if abs(u) > 50000 or abs(v) > 50000: return None
    return (u, v)


def draw_floor_circle(img, world_x, world_y, radius_m, color):
    pts_px = []
    for ang in np.linspace(0, 2 * np.pi, 64):
        wx = world_x + radius_m * np.cos(ang)
        wy = world_y + radius_m * np.sin(ang)
        px = world_to_pixel(wx, wy)
        if px: pts_px.append(px)
    if len(pts_px) < 3: return
    pts_array = np.array([pts_px], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_array], color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, dst=img)
    cv2.polylines(img, [pts_array], True, color, 2, cv2.LINE_AA)
    center_px = world_to_pixel(world_x, world_y)
    if center_px:
        cv2.drawMarker(img, center_px, color, cv2.MARKER_CROSS, 15, 2)


# ========================= FSM =========================

class _State:
    WAITING_FOR_CLEAR = 0
    DISINFECTING = 1
    STANDBY_WAITING = 2


class _LedFSM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = 0;
        self.wait = 0.0;
        self.dis = 0.0;
        self.std = 0.0

    def reset(self):
        self.state = 0;
        self.wait = 0.0;
        self.dis = 0.0;
        self.std = 0.0

    def update(self, personDetected, dt):
        s = self.state
        if s == 0:
            if personDetected:
                self.wait = 0.0
            else:
                self.wait = min(self.wait + dt, self.cfg["waitingTime"])
                if self.wait >= self.cfg["waitingTime"]: self.state = 1
        elif s == 1:
            if personDetected:
                self.state = 0; self.wait = 0.0
            else:
                self.dis = min(self.dis + dt, self.cfg["disinfectionTime"])
                if self.dis >= self.cfg["disinfectionTime"]: self.state = 2
        else:
            if personDetected:
                self.state = 0
            else:
                self.std = min(self.std + dt, self.cfg["standbyTime"])
                if self.std >= self.cfg["standbyTime"]: self.state = 0
        return self.state

    def get_timer_str(self):
        if self.state == 0: return f"WAIT {self.wait:.1f}"
        if self.state == 1: return f"CLN {self.dis:.1f}"
        return f"STBY {self.std:.1f}"


# ========================= APP =========================

LED_CONES = []
LED_FSMS: Dict[int, _LedFSM] = {}


def on_mouse(event, x, y, flags, userdata):
    if userdata["state"] != "SETUP": return
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(LED_CONES) >= NUM_LEDS: return
        wx, wy = pixel_to_world(x, y)
        idx = len(LED_CONES)
        LED_CONES.append({"world_pos": (wx, wy), "led_index": idx})
        LED_FSMS[idx] = _LedFSM(FSM_TIMES)


def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
        time.sleep(2.0)
    except:
        ser = None

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAM_INDEX)

    app_context = {"state": "SETUP"}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, app_context)

    safety = SafetyMonitor()

    last_frame_time = time.time()
    last_check_time = time.time()
    last_bits = None
    last_send_time = 0.0

    failure_start_time = None
    halt_reason = ""

    # NEW: Track when the room was last occupied
    last_person_time = time.time()

    print("--- SETUP MODE ---")
    print("Point at empty room -> Press 'c' -> Place Zones -> Press 's'")

    while True:
        ok, frame = cap.read()
        if not ok: break

        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        h, w = frame.shape[:2]

        results = model(frame, conf=CONF_THRES, verbose=False)
        person_boxes = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_boxes.append((x1, y1, x2, y2))

        people_present = (len(person_boxes) > 0)

        # Update last person time
        if people_present:
            last_person_time = current_time

        # ================= STATE: SETUP =================
        if app_context["state"] == "SETUP":
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, "[SETUP] 'c': Ref | Click: Zone | 's': Start", (10, 40), 0, 0.6, (0, 255, 255), 2)

            if safety.ref_frame is not None:
                cv2.putText(frame, "REF: OK", (w - 100, 40), 0, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "REF: MISSING", (w - 150, 40), 0, 0.6, (0, 0, 255), 2)

            for cone in LED_CONES:
                wx, wy = cone["world_pos"]
                draw_floor_circle(frame, wx, wy, CONE_RADIUS_M, (255, 0, 0))

            cv2.imshow(WINDOW_NAME, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'):
                if people_present:
                    print("Err: Room must be empty")
                else:
                    safety.set_reference(frame)
            elif k == ord('s'):
                if safety.ref_frame is not None: app_context["state"] = "RUNNING"
            elif k == ord('q'):
                break
            continue

        # ================= STATE: RUNNING =================
        if app_context["state"] == "RUNNING":

            # --- Safety Check Logic ---
            if current_time - last_check_time > SAFETY_CHECK_INTERVAL:
                last_check_time = current_time

                # Check 1: Must be empty
                if not people_present:
                    # Check 2: Must be empty for > 2.0s (Stabilization)
                    time_since_empty = current_time - last_person_time

                    if time_since_empty > CAMERA_STABILIZE_TIME:
                        # SAFE TO CHECK
                        is_safe, msg = safety.check(frame)

                        if not is_safe:
                            if failure_start_time is None:
                                failure_start_time = current_time
                                print(f"[Warn] Check Failed: {msg}")

                            if (current_time - failure_start_time) > FAILURE_PERSISTENCE:
                                app_context["state"] = "HALTED"
                                halt_reason = msg
                                print(f"[ALARM] HALTED: {msg}")
                                continue
                        else:
                            failure_start_time = None
                    else:
                        # Waiting for camera to settle
                        failure_start_time = None
                else:
                    # Person present
                    failure_start_time = None

            # Timer Visualization
            if failure_start_time is not None:
                elapsed = current_time - failure_start_time
                rem = FAILURE_PERSISTENCE - elapsed
                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(frame, f"WARNING: VERIFYING... {rem:.1f}s | {msg}", (20, 40), 0, 0.6, (255, 255, 255), 2)
            elif people_present:
                cv2.putText(frame, "SAFETY CHECK: PAUSED (PEOPLE)", (10, 30), 0, 0.6, (0, 255, 255), 2)
            else:
                time_since = current_time - last_person_time
                if time_since < CAMERA_STABILIZE_TIME:
                    rem = CAMERA_STABILIZE_TIME - time_since
                    cv2.putText(frame, f"SAFETY CHECK: STABILIZING... ({rem:.1f}s)", (10, 30), 0, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "SAFETY CHECK: ACTIVE", (10, 30), 0, 0.6, (0, 255, 0), 2)

            # LED Logic
            led_occupied = [False] * NUM_LEDS
            for (x1, y1, x2, y2) in person_boxes:
                box_h = y2 - y1
                y_start = int(y2 - (0.30 * box_h))
                hit_circle = False
                for u in np.linspace(x1, x2, 10):
                    for v in np.linspace(y_start, y2, 3):
                        wx, wy = pixel_to_world(u, v)
                        for cone in LED_CONES:
                            cwx, cwy = cone["world_pos"]
                            if np.hypot(wx - cwx, wy - cwy) < CONE_RADIUS_M:
                                led_occupied[cone["led_index"]] = True
                                hit_circle = True
                c = (0, 0, 255) if hit_circle else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

            final_bits = [0] * NUM_LEDS
            for cone in LED_CONES:
                idx = cone["led_index"]
                fsm = LED_FSMS[idx]
                state = fsm.update(led_occupied[idx], dt)
                final_bits[idx] = 1 if state == 1 else 0

            if ser:
                tup = tuple(final_bits)
                if tup != last_bits or (current_time - last_send_time > HEARTBEAT_INTERVAL):
                    try:
                        ser.write(("B:" + ",".join(str(b) for b in tup + (0,) * (NUM_LEDS - len(tup))) + "\n").encode(
                            'ascii')); ser.flush()
                    except:
                        pass
                    last_bits = tup;
                    last_send_time = current_time

            for i, cone in enumerate(LED_CONES):
                idx = cone["led_index"]
                fsm = LED_FSMS[idx]
                c = (0, 255, 0) if fsm.state == 1 else ((200, 200, 200) if fsm.state == 2 else (0, 255, 255))
                draw_floor_circle(frame, cone["world_pos"][0], cone["world_pos"][1], CONE_RADIUS_M, c)
                cv2.putText(frame, f"{idx}:{fsm.get_timer_str()}", (w - 220, 50 + i * 30), 0, 0.6, c, 2)

            cv2.imshow(WINDOW_NAME, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('u'):
                if not people_present:
                    safety.set_reference(frame)
                    failure_start_time = None
                    print("[System] Reference Updated Manually.")

        # ================= STATE: HALTED =================
        elif app_context["state"] == "HALTED":
            failure_start_time = None
            if ser:
                ser.write(("B:" + ",".join(["0"] * NUM_LEDS) + "\n").encode('ascii'));
                ser.flush()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)
            cv2.putText(frame, "SYSTEM HALTED", (w // 2 - 200, h // 2 - 60), 0, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"Reason: {halt_reason}", (w // 2 - 200, h // 2), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to Recalibrate", (w // 2 - 200, h // 2 + 80), 0, 0.7, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                app_context["state"] = "RECOVERY"

        # ================= STATE: RECOVERY =================
        elif app_context["state"] == "RECOVERY":
            passed = 0
            for i in range(5):
                ret, r_frame = cap.read()
                if not ret: break

                r_res = model(r_frame, conf=CONF_THRES, verbose=False)
                if any(int(b.cls[0]) == 0 for r in r_res for b in r.boxes):
                    cv2.putText(frame, "WAITING FOR CLEAR VIEW...", (w // 2 - 200, h // 2), 0, 1.0, (0, 255, 255), 2)
                    cv2.imshow(WINDOW_NAME, frame);
                    cv2.waitKey(500)
                    i -= 1;
                    continue

                is_safe, msg = safety.check(r_frame)
                col = (0, 255, 0) if is_safe else (0, 0, 255)
                cv2.circle(frame, (w // 2 - 100 + i * 50, h // 2 + 40), 15, col, -1)
                cv2.putText(frame, "VERIFYING...", (w // 2 - 100, h // 2), 0, 1.0, (255, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, frame);
                cv2.waitKey(200)

                if is_safe:
                    passed += 1
                else:
                    halt_reason = msg; break

            if passed == 5:
                print("Resuming.");
                app_context["state"] = "RUNNING"
                for fsm in LED_FSMS.values(): fsm.reset()
            else:
                app_context["state"] = "HALTED"

    cap.release()
    cv2.destroyAllWindows()
    if ser: ser.close()


if __name__ == "__main__":
    main()