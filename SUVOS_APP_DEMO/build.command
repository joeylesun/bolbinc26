#!/bin/bash

# ==========================================
#   SUVOS MONITOR - AUTOMATED BUILDER
#   (One-Click Install & Pack for macOS)
# ==========================================

# 1. Navigate to the script's directory (ensures relative paths work)
cd "$(dirname "$0")"
echo "Working Directory: $(pwd)"

# 2. Cleanup Old Builds
echo "[1/6] Cleaning up previous build artifacts..."
rm -rf build dist venv *.spec

# 3. Create Virtual Environment (Using system python3)
echo "[2/6] Creating isolated Python environment..."
python3 -m venv venv
source venv/bin/activate

# 4. Install Dependencies
echo "[3/6] Installing dependencies..."
# Upgrade pip first
pip install --upgrade pip
# Install exact versions from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# 5. Build the App with PyInstaller
echo "[4/6] Packaging Application..."

# We construct the PyInstaller command dynamically to include assets
# --noconfirm: Overwrite output directory
# --clean: Clear cache
# --windowed: No terminal window for the user
# --icon: Sets the app icon (uses your logo)
# --add-data: Includes the 'assets' folder
# --add-data: Includes the 'interface' folder (for web dashboard)
# --hidden-import: Ensures PIL and Tkinter work together

python -m PyInstaller --noconfirm --clean --windowed \
    --name "SUVOS_Monitor" \
    --icon "assets/logo.png" \
    --add-data "assets:assets" \
    --add-data "interface:interface" \
    --add-data "yolov8n.pt:." \
    --hidden-import "PIL._tkinter_finder" \
    suvos_launcher.py

# 6. Sign the App (Ad-Hoc Signature)
echo "[5/6] Signing the Application..."
# Remove quarantine attribute (fixes "App is damaged" error)
xattr -cr dist/SUVOS_Monitor.app
# Deep sign the app bundle
codesign --force --deep --sign - "dist/SUVOS_Monitor.app"

# 7. Create Distribution Zip
echo "[6/6] Zipping for distribution..."
cd dist
zip -r SUVOS_Monitor_v1.0.zip SUVOS_Monitor.app
cd ..

echo "BUILD COMPLETE!"
echo "Your app is ready at: dist/SUVOS_Monitor.app"
echo "Sharable Zip: dist/SUVOS_Monitor_v1.0.zip"