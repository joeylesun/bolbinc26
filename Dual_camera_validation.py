import cv2
import numpy as np
import open3d as o3d
import sys
import os

# ================= CONFIGURATION =================
# Files
PTS_FILE = "room_cali.pts"
CALIB_FILE_0 = "calibration_data_cam0.npz"
CALIB_FILE_1 = "calibration_data_cam1.npz"

# Hardware
CAM_INDICES = [0, 1]
# =================================================

# Global State
GLOBAL_WORLD_POS = None  # (x_meters, y_meters)
CAM1_PROJECTED_PX = None  # (u, v) on Cam 1
CAM0_CLICK_PX = None  # (u, v) on Cam 0

# Map View State (Same as Step 1)
MAP_ROTATION = 0
VIEW_ZOOM = 1.0
VIEW_PAN = [0, 0]


def load_calibration(filename):
    """Loads H matrix and Map params from .npz"""
    if not os.path.exists(filename):
        print(f"[Error] Cannot find {filename}. Run Step 1 first!")
        sys.exit(1)
    data = np.load(filename)
    return data


# Load Data
print("[Init] Loading Calibration Data...")
data0 = load_calibration(CALIB_FILE_0)
H0 = data0['H']
# Use map params from Cam0 (assuming they are consistent since same .pts file)
MAP_SCALE = data0['scale']
MAP_OFF_X = data0['off_x']
MAP_OFF_Y = data0['off_y']
MAP_SIZE = int(data0['map_size'])

data1 = load_calibration(CALIB_FILE_1)
H1 = data1['H']
H1_inv = np.linalg.inv(H1)  # Pre-calculate Inverse for World -> Cam1 projection


# --- MAP GENERATION (Reused) ---
def read_and_level_scan(filepath):
    print(f"[IO] Reading {filepath}...")
    try:
        data = np.loadtxt(filepath, skiprows=1)
        xyz = data[:, 0:3]
        rgb = data[:, 4:7] / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    except Exception as e:
        print(f"Error loading PTS: {e}");
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


def generate_map_visual(points, colors, size):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    px_x = ((points[:, 0] - MAP_OFF_X) * MAP_SCALE).astype(int)
    px_y = size - ((points[:, 1] - MAP_OFF_Y) * MAP_SCALE).astype(int)
    valid = (px_x >= 0) & (px_x < size) & (px_y >= 0) & (px_y < size)
    bgr = (colors[:, [2, 1, 0]] * 255).astype(np.uint8)
    img[px_y[valid], px_x[valid]] = bgr[valid]
    return img


def rotate_image(image, rotation):
    if rotation == 1: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 2: return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 3: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# --- COORDINATE MATH ---

def world_to_view_pixels(wx, wy):
    """World Meters -> Zoomed Map Pixels"""
    mx = (wx - MAP_OFF_X) * MAP_SCALE
    my = MAP_SIZE - ((wy - MAP_OFF_Y) * MAP_SCALE)

    # Rotation Logic (Must match Step 1 exactly)
    if MAP_ROTATION == 0:
        rx, ry = mx, my
    elif MAP_ROTATION == 1:
        rx, ry = MAP_SIZE - 1 - my, mx
    elif MAP_ROTATION == 2:
        rx, ry = MAP_SIZE - 1 - mx, MAP_SIZE - 1 - my
    elif MAP_ROTATION == 3:
        rx, ry = my, MAP_SIZE - 1 - mx

    view_w = MAP_SIZE / VIEW_ZOOM
    tl_x = (MAP_SIZE / 2 + VIEW_PAN[0]) - (view_w / 2)
    tl_y = (MAP_SIZE / 2 + VIEW_PAN[1]) - (view_w / 2)
    return int((rx - tl_x) * VIEW_ZOOM), int((ry - tl_y) * VIEW_ZOOM)


def get_view_image(rotated_map):
    """Crops map based on zoom"""
    h, w = rotated_map.shape[:2]
    view_w = int(w / VIEW_ZOOM)
    view_h = int(h / VIEW_ZOOM)
    cx = int(w / 2 + VIEW_PAN[0])
    cy = int(h / 2 + VIEW_PAN[1])
    x1, y1 = max(0, cx - view_w // 2), max(0, cy - view_h // 2)
    x2, y2 = min(w, x1 + view_w), min(h, y1 + view_h)
    crop = rotated_map[y1:y2, x1:x2]
    if crop.size == 0: return rotated_map
    return cv2.resize(crop, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)


# --- MOUSE CALLBACKS ---

def mouse_cam0(event, x, y, flags, param):
    global GLOBAL_WORLD_POS, CAM1_PROJECTED_PX, CAM0_CLICK_PX

    if event == cv2.EVENT_LBUTTONDOWN:
        CAM0_CLICK_PX = (x, y)

        # 1. Cam0 Pixel -> World Coordinate (Using H0)
        # H maps Pixel -> World directly
        p_cam = np.array([x, y, 1.0])
        w_vec = H0 @ p_cam

        if abs(w_vec[2]) > 1e-9:
            wx = w_vec[0] / w_vec[2]
            wy = w_vec[1] / w_vec[2]
            GLOBAL_WORLD_POS = (wx, wy)
            print(f"[Calc] World Pos: ({wx:.2f}m, {wy:.2f}m)")

            # 2. World Coordinate -> Cam1 Pixel (Using Inverse H1)
            # Standard Homography: World = H1 @ Cam1
            # Therefore: Cam1 = Inv(H1) @ World
            p_world = np.array([wx, wy, 1.0])
            c1_vec = H1_inv @ p_world

            if abs(c1_vec[2]) > 1e-9:
                c1_x = int(c1_vec[0] / c1_vec[2])
                c1_y = int(c1_vec[1] / c1_vec[2])
                CAM1_PROJECTED_PX = (c1_x, c1_y)
                print(f"[Calc] Projected to Cam1: ({c1_x}, {c1_y})")


def main():
    global MAP_ROTATION, VIEW_ZOOM, VIEW_PAN

    # 1. Setup Map
    pts, cols = read_and_level_scan(PTS_FILE)
    if pts is None: return
    base_map_img = generate_map_visual(pts, cols, MAP_SIZE)

    # 2. Setup Cameras
    cap0 = cv2.VideoCapture(CAM_INDICES[0])
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap1 = cv2.VideoCapture(CAM_INDICES[1])
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 3. Windows
    cv2.namedWindow("Map")
    cv2.namedWindow("Cam0 (Master)")
    cv2.namedWindow("Cam1 (Slave)")

    # Only Cam0 gets the click listener for now (Master)
    cv2.setMouseCallback("Cam0 (Master)", mouse_cam0)

    print("--- DUAL CAMERA VALIDATION ---")
    print("INSTRUCTIONS:")
    print("1. Click any object in 'Cam0 (Master)'.")
    print("2. System calculates Real World position (shows on Map).")
    print("3. System predicts where that object is on 'Cam1'.")
    print("KEYS: [r]=Rotate Map | [+/-]=Zoom Map | [WASD]=Pan Map | [q]=Quit")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Error reading cameras");
            break

        # --- Draw Map ---
        rot_map = rotate_image(base_map_img.copy(), MAP_ROTATION)
        disp_map = get_view_image(rot_map)

        # --- Visualizations ---

        # 1. Draw Click on Cam0 (Yellow)
        if CAM0_CLICK_PX:
            cv2.circle(frame0, CAM0_CLICK_PX, 10, (0, 255, 255), -1)
            cv2.circle(frame0, CAM0_CLICK_PX, 12, (0, 0, 0), 2)
            cv2.putText(frame0, "SOURCE", (CAM0_CLICK_PX[0] + 15, CAM0_CLICK_PX[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 2. Draw Prediction on Map (Cyan)
        if GLOBAL_WORLD_POS:
            vx, vy = world_to_view_pixels(GLOBAL_WORLD_POS[0], GLOBAL_WORLD_POS[1])
            if 0 <= vx < MAP_SIZE and 0 <= vy < MAP_SIZE:
                cv2.circle(disp_map, (vx, vy), 8, (255, 255, 0), -1)
                cv2.putText(disp_map, f"Pos: {GLOBAL_WORLD_POS[0]:.1f},{GLOBAL_WORLD_POS[1]:.1f}m",
                            (vx + 10, vy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Draw Prediction on Cam1 (Magenta)
        if CAM1_PROJECTED_PX:
            cv2.circle(frame1, CAM1_PROJECTED_PX, 10, (255, 0, 255), -1)
            cv2.circle(frame1, CAM1_PROJECTED_PX, 12, (0, 0, 0), 2)
            cv2.putText(frame1, "PREDICTED", (CAM1_PROJECTED_PX[0] + 15, CAM1_PROJECTED_PX[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # --- UI Overlays ---
        cv2.putText(frame0, "CAM 0 (CLICK HERE)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame1, "CAM 1 (VERIFICATION)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Show Windows
        cv2.imshow("Map", disp_map)
        cv2.imshow("Cam0 (Master)", frame0)
        cv2.imshow("Cam1 (Slave)", frame1)

        # Controls
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('r'):
            MAP_ROTATION = (MAP_ROTATION + 1) % 4
        elif k == ord('='):
            VIEW_ZOOM = min(10.0, VIEW_ZOOM + 0.2)
        elif k == ord('-'):
            VIEW_ZOOM = max(1.0, VIEW_ZOOM - 0.2)
        elif k == ord('w'):
            VIEW_PAN[1] -= 20 / VIEW_ZOOM
        elif k == ord('s'):
            VIEW_PAN[1] += 20 / VIEW_ZOOM
        elif k == ord('a'):
            VIEW_PAN[0] -= 20 / VIEW_ZOOM
        elif k == ord('d'):
            VIEW_PAN[0] += 20 / VIEW_ZOOM

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()