@echo off
REM Quick start script for HVSR Progressive Layer Stripping GUI (Windows)

echo ==========================================
echo HVSR Progressive Layer Stripping GUI
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo Warning: PySide6 not found. Installing dependencies...
    pip install -r requirements.txt
)

REM Run the application
echo Starting GUI application...
python app.py

echo.
echo Application closed
pause
