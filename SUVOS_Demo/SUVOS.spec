# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# 1. FORCE IMPORTS
# We explicitly tell PyInstaller to find these libraries
hidden_imports = [
    'ultralytics',
    'websockets',
    'serial',
    'numpy',
    'cv2',
    'PIL'
]
# Collect all sub-modules for YOLO so it doesn't break
hidden_imports += collect_submodules('ultralytics')

a = Analysis(
    ['suvos_backend_single.py'],
    pathex=[],
    binaries=[],
    # 2. INCLUDE YOUR DATA FILES
    # Format: ('Source Path', 'Destination Path inside App')
    datas=[
        ('interface', 'interface'),
        ('calibration_data_cam0.npz', '.'),
        ('led_zones.json', '.'),
        ('room_shape.json', '.'),
        ('yolov8n.pt', '.')
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SUVOS_Monitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # False = No terminal window (background app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SUVOS_Monitor',
)

app = BUNDLE(
    coll,
    name='SUVOS_Monitor.app',
    icon=None,
    bundle_identifier='com.suvos.monitor',
    # 3. CRITICAL: CAMERA PERMISSIONS
    info_plist={
        'NSCameraUsageDescription': 'SUVOS needs camera access to detect people in zones.',
        'NSHighResolutionCapable': 'True'
    },
)