# bolbinc26
```
SUVOS_System/
│
├── config/                  <-- Stores all JSONs and Reference Images
│   ├── dynamic_zones.json   <-- (From Step 1)
│   └── ref_bump_frame.jpg   <-- (NEW: For Bump Detection)
│
├── suvos_utils.py           <-- The "spine" connecting everything
│
├── step1_calibrate_dual.py  <-- [Source: Demo_dual_camera_calibration.py]
│                                TODO: Saves 'ref_bump_frame.jpg' when done.
│
├── step2_validate_tool.py   <-- [Source: Dual_camera_validation.py]
│                                TODO: Uses suvos_utils to load zones.
│
├── step3_run_system.py      <-- [Source: Demo_dual_camera_detection.py]
│                                + Integrated: YOLO Loop
│                                + Integrated: Hardware/Relay Logic
│                                + TODO: WebSocket Server
│                                + TODO: Camera Bump Check (YOLO_self_check.py)
│
└── interface/
    ├── index.html           <-- [Source: SUVOS unified eACH_V18.html]
    │                            + TODO: Listen for WebSocket JSON for the live update 2D visualization
    └── assets/
```

