import cv2
import numpy as np
import open3d as o3d
import json
import os
from tkinter import messagebox
from suvos_common import resource_path, get_config_path, CALIB_FILE, SHAPE_FILE, CAMERA_INDICES

# ================= CONFIGURATION =================
MAP_SIZE = 1000
GRID_SPACING_M = 1.0

# Global State
STATE = "WAIT_MAP"
PAIRS = []
TEMP_WORLD = None

# View States
MAP_SCALE = 1.0
MAP_OFF_X = 0.0
MAP_OFF_Y = 0.0
MAP_ROTATION = 0
ACTIVE_MODE = "MAP"

# Zoom/Pan States
VIEW_ZOOM = 1.0;
VIEW_PAN = [0, 0]
CAM_ZOOM = 1.0;
CAM_PAN = [0, 0]


# --- IO & GENERATION ---
def read_and_level_scan(filepath):
    print(f"[IO] Reading {filepath}...")
    try:
        if not os.path.exists(filepath): return None, None
        data = np.loadtxt(filepath, skiprows=1)
        if data.size == 0: return None, None

        xyz = data[:, 0:3]
        rgb = data[:, 4:7] / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    except Exception as e:
        print(f"[Error] {e}")
        return None, None

    plane, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane
    normal = np.array([a, b, c])
    if np.dot(normal, [0, 0, 1]) < 0: normal = -normal
    rot_axis = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        pcd.rotate(R, center=(0, 0, 0))

    return np.asarray(pcd.points)[inliers], np.asarray(pcd.colors)[inliers]


def generate_click_map(points, colors, size):
    global MAP_SCALE, MAP_OFF_X, MAP_OFF_Y
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    span = max(max_x - min_x, max_y - min_y)
    padding = size * 0.1
    MAP_SCALE = (size - 2 * padding) / span
    MAP_OFF_X = min_x - (padding / MAP_SCALE)
    MAP_OFF_Y = min_y - (padding / MAP_SCALE)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    px_x = ((points[:, 0] - MAP_OFF_X) * MAP_SCALE).astype(int)
    px_y = size - ((points[:, 1] - MAP_OFF_Y) * MAP_SCALE).astype(int)
    valid = (px_x >= 0) & (px_x < size) & (px_y >= 0) & (px_y < size)
    bgr = (colors[:, [2, 1, 0]] * 255).astype(np.uint8)
    img[px_y[valid], px_x[valid]] = bgr[valid]
    return img


def generate_schematic_and_shape(points, size):
    px_x = ((points[:, 0] - MAP_OFF_X) * MAP_SCALE).astype(int)
    px_y = size - ((points[:, 1] - MAP_OFF_Y) * MAP_SCALE).astype(int)
    mask = np.zeros((size, size), dtype=np.uint8)
    valid = (px_x >= 0) & (px_x < size) & (px_y >= 0) & (px_y < size)
    mask[px_y[valid], px_x[valid]] = 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    room_polygon_meters = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.0005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        for pt in approx:
            px, py = pt[0]
            wx = (px / MAP_SCALE) + MAP_OFF_X
            wy = ((size - py) / MAP_SCALE) + MAP_OFF_Y
            room_polygon_meters.append({"x": round(wx, 3), "y": round(wy, 3)})

    visual = np.ones((size, size, 3), dtype=np.uint8) * 40
    visual[mask == 255] = (180, 180, 180)
    grid_col = (100, 100, 100)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    for x in range(int(np.floor(min_x)), int(np.ceil(max_x)) + 1, int(GRID_SPACING_M)):
        px = int((x - MAP_OFF_X) * MAP_SCALE)
        if 0 <= px < size: cv2.line(visual, (px, 0), (px, size), grid_col, 1)
    for y in range(int(np.floor(min_y)), int(np.ceil(max_y)) + 1, int(GRID_SPACING_M)):
        py = size - int((y - MAP_OFF_Y) * MAP_SCALE)
        if 0 <= py < size: cv2.line(visual, (0, py), (size, py), grid_col, 1)

    return visual, room_polygon_meters


def rotate_image(image, rotation):
    if rotation == 1: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 2: return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 3: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# --- VIEW TRANSFORMS (CORRECTED MATH) ---
def get_clamped_map_state():
    view_w = int(MAP_SIZE / VIEW_ZOOM)
    view_h = int(MAP_SIZE / VIEW_ZOOM)
    cx = int(MAP_SIZE / 2 + VIEW_PAN[0])
    cy = int(MAP_SIZE / 2 + VIEW_PAN[1])
    tl_x = cx - view_w // 2
    tl_y = cy - view_h // 2

    tl_x = max(0, min(tl_x, MAP_SIZE - view_w))
    tl_y = max(0, min(tl_y, MAP_SIZE - view_h))

    return tl_x, tl_y, view_w, view_h


def world_to_view_pixels(wx, wy):
    # 1. World -> Unrotated Full Map Pixel
    mx = (wx - MAP_OFF_X) * MAP_SCALE
    my = MAP_SIZE - ((wy - MAP_OFF_Y) * MAP_SCALE)

    # 2. Apply Rotation (Fixed to match OpenCV)
    # S = MAP_SIZE - 1
    S = MAP_SIZE - 1
    if MAP_ROTATION == 1:  # 90 Clockwise
        # New X is Inverted Old Y; New Y is Old X
        # (x, y) -> (S-y, x)
        mx, my = S - my, mx
    elif MAP_ROTATION == 2:  # 180
        # (x, y) -> (S-x, S-y)
        mx, my = S - mx, S - my
    elif MAP_ROTATION == 3:  # 90 Counter-Clockwise
        # New X is Old Y; New Y is Inverted Old X
        # (x, y) -> (y, S-x)
        mx, my = my, S - mx

    # 3. Full Map -> View Pixel (Apply Zoom/Pan)
    tl_x, tl_y, _, _ = get_clamped_map_state()
    vx = int((mx - tl_x) * VIEW_ZOOM)
    vy = int((my - tl_y) * VIEW_ZOOM)
    return vx, vy


def view_pixels_to_world(vx, vy):
    # 1. View Pixel -> Full Map Pixel (Un-Zoom/Pan)
    tl_x, tl_y, _, _ = get_clamped_map_state()
    rx = (vx / VIEW_ZOOM) + tl_x
    ry = (vy / VIEW_ZOOM) + tl_y

    # 2. Un-Rotate (Inverse of world_to_view)
    S = MAP_SIZE - 1
    if MAP_ROTATION == 0:
        mx, my = rx, ry
    elif MAP_ROTATION == 1:  # Inverse of 90 CW is 90 CCW
        # Input: (S-y, x).  We want (x,y).
        # Let u = S-y, v = x.
        # Then x = v, y = S-u.
        mx, my = ry, S - rx
    elif MAP_ROTATION == 2:  # Inverse of 180 is 180
        mx, my = S - rx, S - ry
    elif MAP_ROTATION == 3:  # Inverse of 90 CCW is 90 CW
        # Input: (y, S-x). We want (x,y).
        # Let u = y, v = S-x.
        # Then y = u, x = S-v.
        mx, my = S - ry, rx

    # 3. Full Map -> World
    wx = MAP_OFF_X + (mx / MAP_SCALE)
    wy = MAP_OFF_Y + ((MAP_SIZE - my) / MAP_SCALE)
    return wx, wy


# --- CAM TRANSFORMS ---
def get_cam_view(frame):
    h, w = frame.shape[:2]
    view_w = int(w / CAM_ZOOM)
    view_h = int(h / CAM_ZOOM)
    cx = int(w / 2 + CAM_PAN[0])
    cy = int(h / 2 + CAM_PAN[1])
    tl_x = max(0, min(cx - view_w // 2, w - view_w))
    tl_y = max(0, min(cy - view_h // 2, h - view_h))

    crop = frame[tl_y:tl_y + view_h, tl_x:tl_x + view_w]
    if crop.size == 0: return frame
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def cam_view_to_real(vx, vy, frame_w, frame_h):
    view_w = int(frame_w / CAM_ZOOM)
    view_h = int(frame_h / CAM_ZOOM)
    cx = int(frame_w / 2 + CAM_PAN[0])
    cy = int(frame_h / 2 + CAM_PAN[1])
    tl_x = max(0, min(cx - view_w // 2, frame_w - view_w))
    tl_y = max(0, min(cy - view_h // 2, frame_h - view_h))

    real_x = tl_x + (vx / CAM_ZOOM)
    real_y = tl_y + (vy / CAM_ZOOM)
    return int(real_x), int(real_y)


def real_to_cam_view(rx, ry, frame_w, frame_h):
    view_w = int(frame_w / CAM_ZOOM)
    view_h = int(frame_h / CAM_ZOOM)
    cx = int(frame_w / 2 + CAM_PAN[0])
    cy = int(frame_h / 2 + CAM_PAN[1])
    tl_x = max(0, min(cx - view_w // 2, frame_w - view_w))
    tl_y = max(0, min(cy - view_h // 2, frame_h - view_h))

    vx = (rx - tl_x) * CAM_ZOOM
    vy = (ry - tl_y) * CAM_ZOOM
    return int(vx), int(vy)


# --- MOUSE CALLBACKS ---
def mouse_map(event, x, y, flags, param):
    global STATE, TEMP_WORLD
    if event == cv2.EVENT_LBUTTONDOWN and STATE == "WAIT_MAP":
        wx, wy = view_pixels_to_world(x, y)
        TEMP_WORLD = (wx, wy)
        STATE = "WAIT_CAM"
        print(f" -> Map Click: ({wx:.2f}, {wy:.2f})")


def mouse_cam(event, x, y, flags, param):
    global STATE, PAIRS, TEMP_WORLD
    w, h = param
    if event == cv2.EVENT_LBUTTONDOWN and STATE == "WAIT_CAM":
        real_x, real_y = cam_view_to_real(x, y, w, h)
        PAIRS.append((TEMP_WORLD, (real_x, real_y)))
        STATE = "WAIT_MAP"
        TEMP_WORLD = None
        print(f" -> Cam Click: ({real_x}, {real_y}) | Pair #{len(PAIRS)} Saved")


# --- MAIN ENTRY POINT ---
def run_calibration(pts_path):
    global MAP_ROTATION, VIEW_ZOOM, VIEW_PAN, CAM_ZOOM, CAM_PAN, ACTIVE_MODE, PAIRS, STATE, TEMP_WORLD

    PAIRS = []
    STATE = "WAIT_MAP"
    TEMP_WORLD = None
    VIEW_ZOOM = 1.0;
    VIEW_PAN = [0, 0]
    CAM_ZOOM = 1.0;
    CAM_PAN = [0, 0]
    ACTIVE_MODE = "MAP"
    MAP_ROTATION = 0

    pts, cols = read_and_level_scan(pts_path)
    if pts is None:
        messagebox.showerror("Error", "Could not load .pts file")
        return

    click_map_img = generate_click_map(pts, cols, MAP_SIZE)
    schematic_img, raw_shape = generate_schematic_and_shape(pts, MAP_SIZE)

    cap = cv2.VideoCapture(CAMERA_INDICES[0])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, test_frm = cap.read()
    if ret:
        CAM_W, CAM_H = test_frm.shape[1], test_frm.shape[0]
    else:
        CAM_W, CAM_H = 1920, 1080

    cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Map", mouse_map)
    cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Cam", mouse_cam, param=(CAM_W, CAM_H))

    cv2.moveWindow("Map", 0, 0)
    cv2.moveWindow("Cam", 800, 0)

    print("--- ADVANCED CALIBRATION ---")
    print("CONTROLS: [TAB]=Switch View | [WASD]=Pan | [+/-]=Zoom | [z]=Undo | [c]=Save")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. RENDER MAP VIEW
        rot_map = rotate_image(click_map_img.copy(), MAP_ROTATION)
        tl_x, tl_y, view_w, view_h = get_clamped_map_state()
        disp_map = rot_map[tl_y:tl_y + view_h, tl_x:tl_x + view_w]
        if disp_map.size != 0:
            disp_map = cv2.resize(disp_map, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)

        # 2. RENDER CAM VIEW
        disp_cam = get_cam_view(frame)

        # 3. OVERLAY POINTS
        for i, (w_pt, c_pt) in enumerate(PAIRS):
            vx, vy = world_to_view_pixels(w_pt[0], w_pt[1])
            if 0 <= vx < MAP_SIZE and 0 <= vy < MAP_SIZE:
                cv2.circle(disp_map, (vx, vy), 6, (0, 255, 0), -1)
                cv2.putText(disp_map, str(i + 1), (vx + 8, vy), 0, 0.6, (0, 255, 0), 2)
            cx, cy = real_to_cam_view(c_pt[0], c_pt[1], CAM_W, CAM_H)
            if 0 <= cx < CAM_W and 0 <= cy < CAM_H:
                cv2.circle(disp_cam, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(disp_cam, str(i + 1), (cx + 10, cy), 0, 1, (0, 255, 0), 2)

        if TEMP_WORLD:
            vx, vy = world_to_view_pixels(TEMP_WORLD[0], TEMP_WORLD[1])
            if 0 <= vx < MAP_SIZE and 0 <= vy < MAP_SIZE:
                cv2.circle(disp_map, (vx, vy), 6, (0, 255, 255), -1)
            cv2.putText(disp_cam, "CLICK MATCH", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 4. OVERLAY UI
        col_map = (0, 255, 0) if ACTIVE_MODE == "MAP" else (100, 100, 100)
        col_cam = (0, 255, 0) if ACTIVE_MODE == "CAM" else (100, 100, 100)
        cv2.rectangle(disp_map, (0, 0), (MAP_SIZE, MAP_SIZE), col_map, 10 if ACTIVE_MODE == "MAP" else 2)
        cv2.rectangle(disp_cam, (0, 0), (CAM_W, CAM_H), col_cam, 10 if ACTIVE_MODE == "CAM" else 2)
        cv2.putText(disp_map, f"MAP [{'ACTIVE' if ACTIVE_MODE == 'MAP' else 'TAB'}]", (20, 40), 0, 0.8, col_map, 2)
        cv2.putText(disp_cam, f"CAM [{'ACTIVE' if ACTIVE_MODE == 'CAM' else 'TAB'}] (x{CAM_ZOOM:.1f})", (20, 50), 0, 1,
                    col_cam, 2)

        cv2.imshow("Map", disp_map)
        cv2.imshow("Cam", disp_cam)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == 9:
            ACTIVE_MODE = "CAM" if ACTIVE_MODE == "MAP" else "MAP"
        elif k == ord('r'):
            MAP_ROTATION = (MAP_ROTATION + 1) % 4
        elif k == ord('z'):
            if STATE == "WAIT_CAM":
                TEMP_WORLD = None; STATE = "WAIT_MAP"
            elif len(PAIRS) > 0:
                PAIRS.pop()
        elif k == ord('='):
            if ACTIVE_MODE == "MAP":
                VIEW_ZOOM = min(10.0, VIEW_ZOOM + 0.2)
            else:
                CAM_ZOOM = min(10.0, CAM_ZOOM + 0.5)
        elif k == ord('-'):
            if ACTIVE_MODE == "MAP":
                VIEW_ZOOM = max(1.0, VIEW_ZOOM - 0.2)
            else:
                CAM_ZOOM = max(1.0, CAM_ZOOM - 0.5)
        elif k == ord('w'):
            if ACTIVE_MODE == "MAP":
                VIEW_PAN[1] -= 20 / VIEW_ZOOM
            else:
                CAM_PAN[1] -= 50 / CAM_ZOOM
        elif k == ord('s'):
            if ACTIVE_MODE == "MAP":
                VIEW_PAN[1] += 20 / VIEW_ZOOM
            else:
                CAM_PAN[1] += 50 / CAM_ZOOM
        elif k == ord('a'):
            if ACTIVE_MODE == "MAP":
                VIEW_PAN[0] -= 20 / VIEW_ZOOM
            else:
                CAM_PAN[0] -= 50 / CAM_ZOOM
        elif k == ord('d'):
            if ACTIVE_MODE == "MAP":
                VIEW_PAN[0] += 20 / VIEW_ZOOM
            else:
                CAM_PAN[0] += 50 / CAM_ZOOM
        elif k == ord('c'):
            if len(PAIRS) < 4: messagebox.showwarning("Warning", "Need 4+ pairs"); continue
            src = np.array([p[1] for p in PAIRS], dtype=np.float32)
            dst = np.array([p[0] for p in PAIRS], dtype=np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            final_map = rotate_image(schematic_img, MAP_ROTATION)
            cv2.imwrite(get_config_path("room_map_debug.png"), final_map)
            np.savez(get_config_path(CALIB_FILE), H=H, scale=MAP_SCALE, off_x=MAP_OFF_X, off_y=MAP_OFF_Y,
                     rotation=MAP_ROTATION, map_size=MAP_SIZE)
            with open(get_config_path(SHAPE_FILE), 'w') as f:
                json.dump(raw_shape, f, indent=4)
            messagebox.showinfo("Success", "Calibration Saved!");
            break

    cap.release()
    cv2.destroyAllWindows()