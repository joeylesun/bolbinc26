# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['suvos_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), ('interface', 'interface'), ('yolov8n.pt', '.')],
    hiddenimports=['PIL._tkinter_finder'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/logo.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SUVOS_Monitor',
)
app = BUNDLE(
    coll,
    name='SUVOS_Monitor.app',
    icon='assets/logo.png',
    bundle_identifier=None,
)
