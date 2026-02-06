#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np
import os

# CONFIG
WS_HOST = "localhost"
WS_PORT = 8765
CALIB_FILE = "calibration_data_cam0.npz"
SHAPE_FILE = "room_shape.json"


async def handler(websocket):
    print(f"[TEST] Frontend Connected!")

    # 1. Load Map Config (Scale/Offset)
    if not os.path.exists(CALIB_FILE):
        print(f"[ERR] Missing {CALIB_FILE}. Run Step 1 first.")
        return

    d = np.load(CALIB_FILE)
    config = {
        'scale': float(d['scale']),
        'off_x': float(d['off_x']),
        'off_y': float(d['off_y']),
        'map_size': int(d['map_size'])
    }

    # 2. Load Room Shape
    if os.path.exists(SHAPE_FILE):
        with open(SHAPE_FILE, 'r') as f:
            shape = json.load(f)
        print(f"[TEST] Loaded {len(shape)} points from {SHAPE_FILE}")
    else:
        print(f"[ERR] Missing {SHAPE_FILE}. Run Step 1 first.")
        shape = []

    # 3. Create Dummy Zones (Just to verify they draw too)
    dummy_zones = [
        {"id": 0, "x": shape[0]['x'] + 1.0, "y": shape[0]['y'] + 1.0},
        {"id": 1, "x": shape[1]['x'] - 1.0, "y": shape[1]['y'] + 1.0}
    ] if len(shape) > 1 else []

    # 4. Send Packet
    packet = {
        "type": "INIT_CONFIG",
        "config": config,
        "room_shape": shape,
        "fixed_zones": dummy_zones
    }

    await websocket.send(json.dumps(packet))
    print("[TEST] Handshake Sent. Check your browser.")

    # Keep connection alive
    await websocket.wait_closed()


async def main():
    print(f"--- ROOM RENDER TEST SERVER ---")
    print(f"Serving on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass