@echo off
REM Launcher for HVSR Progressive GUI with virtual environment (Windows)

setlocal EnableDelayedExpansion

REM Determine repository root (directory of this script)
set "ROOT=%~dp0"
set "PKG_DIR=%ROOT%new package"
set "VENV_DIR=%PKG_DIR%\.venv"

echo ==============================================
echo HVSR Progressive Layer Stripping - GUI Launcher
echo ==============================================
echo.

REM Check for Python
where python >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not on PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create venv if missing
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip
python -m pip install --upgrade pip >nul 2>&1

REM Install/refresh package and GUI deps
pushd "%PKG_DIR%"
echo Ensuring package and GUI dependencies are installed...
pip install -e .
if errorlevel 1 (
    echo Failed to install hvstrip_progressive package.
    popd
    pause
    exit /b 1
)
pip install -r hvstrip_progressive\gui\requirements.txt
if errorlevel 1 (
    echo Failed to install GUI requirements.
    popd
    pause
    exit /b 1
)
popd

REM Launch the GUI as a module
echo Starting GUI application...
python -m hvstrip_progressive.gui

echo.
echo Application closed.
pause

