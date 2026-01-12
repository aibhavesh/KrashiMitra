#!/bin/bash
# KrashiMitra - Linux/macOS Installation Script
# Run this script to set up the project automatically

echo "============================================================"
echo " KrashiMitra - Automated Installation Script"
echo "============================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[1/4] Python detected"
python3 --version

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
if [ -d "env" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv env
    echo "Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo ""
echo "[3/4] Installing dependencies..."
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
python check_installation.py

echo ""
echo "============================================================"
echo " Installation Complete!"
echo "============================================================"
echo ""
echo "To run the application:"
echo "  1. Activate environment: source env/bin/activate"
echo "  2. Start server: python app.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
