@echo off
REM Quick start script for HVSR Progressive Layer Stripping GUI (Windows)

echo ==========================================
echo HVSR Progressive Layer Stripping GUI
echo ==========================================
echo.

REM Set the project root directory (two levels up from gui folder)
set "PROJECT_ROOT=%~dp0..\.."
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

REM Check if virtual environment exists
if not exist "%VENV_PYTHON%" (
    echo Error: Virtual environment not found at %VENV_PYTHON%
    echo Please create a virtual environment first using:
    echo python -m venv .venv
    pause
    exit /b 1
)

REM Check Python version
echo Using Python from virtual environment...
"%VENV_PYTHON%" --version

REM Check if required packages are installed
echo Checking dependencies...
"%VENV_PYTHON%" -c "import PySide6" 2>nul
if errorlevel 1 (
    echo Warning: PySide6 not found. Installing dependencies...
    "%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt"
)

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

REM Run the application as a module
echo Starting GUI application...
"%VENV_PYTHON%" -m hvstrip_progressive.gui

echo.
echo Application closed
pause
