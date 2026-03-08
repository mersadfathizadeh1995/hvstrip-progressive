@echo off
REM HVSR Progressive Layer Stripping Analysis - GUI Launcher
REM This script activates the virtual environment and launches the GUI

echo ============================================
echo HVSR Progressive Layer Stripping Analysis
echo ============================================
echo.

REM Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"

REM Check if venv exists
if not exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting GUI...
echo.

REM Activate venv and run GUI
cd /d "%SCRIPT_DIR%"
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Set PYTHONPATH to include the project root
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

REM Run the GUI module
python -m hvstrip_progressive.gui

REM Check if there was an error
if errorlevel 1 (
    echo.
    echo ERROR: GUI failed to start. Check the error messages above.
    pause
)
