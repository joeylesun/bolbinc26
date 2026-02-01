# bolbinc26
```
SUVOS_System/
│
├── suvos_utils.py           <-- The Shared Library (Paths, Logging, Vector Math)
│
├── step1_calibrate_dual.py  <-- The Map Maker and Camera Calibration (Demo_dual_camera_calibration.py)
│
├── step2_validate_tool.py   <-- Quick calibration Check (Demo_dual_camera_validation.py)
│
├── step3_run_system.py      <-- Backend (YOLO + WebSocket + Hardware Logic). Yolo and Hardware Logic are implemented in Demo_dual_camera_detection.py
│
└── interface/
    ├── index.html           <-- Frontend (SUVOS unified eACH_V18.html)
    ├── assets/              <-- (Images for the offline mode)
```

TODO:
