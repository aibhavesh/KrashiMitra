@echo off
REM KrashiMitra - Windows Installation Script
REM Run this script to set up the project automatically

echo ============================================================
echo  KrashiMitra - Automated Installation Script
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python detected
python --version

REM Create virtual environment
echo.
echo [2/4] Creating virtual environment...
if exist env (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv env
    echo Virtual environment created
)

REM Activate virtual environment and install dependencies
echo.
echo [3/4] Installing dependencies...
call env\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

REM Verify installation
echo.
echo [4/4] Verifying installation...
python check_installation.py

echo.
echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.
echo To run the application:
echo   1. Activate environment: env\Scripts\activate
echo   2. Start server: python app.py
echo   3. Open browser: http://localhost:5000
echo.
pause
